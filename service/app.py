"""
FastAPI-сервис recsys-hm (Фаза 10 Master Execution Plan).

Production-моделью, как зафиксировано в Фазах 6-9, остаётся ALS +
автоматический fallback на Popularity для неизвестных пользователей —
CatBoostRanker (Фаза 8) в сервис сознательно не подключается (обучен,
задокументирован, но не превосходит retrieval-этап, см. Доклад,
раздел 13 "Фаза 9").

Архитектурные решения:

    1. Обучение при старте сервиса, без сохранённых на диск ALS-артефактов.
       ALS на полном train обучается ~2 минуты (см. BRANCH_HANDOFF_PHASE_7_8.md,
       "Известные технические ограничения" — не проблема на используемом
       железе), а Popularity — секунды. Один явный, предсказуемый шаг
       инициализации при старте (startup event) проще для этого масштаба
       проекта, чем отдельный пайплайн сохранения/версионирования
       ALS-артефактов (векторы юзеров/товаров, mapping) — тот пайплайн
       стоит вводить, когда время холодного старта или частота
       передеплоя реально станут проблемой, не раньше.

    2. Наружу сервис принимает и возвращает исходный hex customer_id, не
       внутренний customer_idx (Int64). Маппинг между ними делается внутри
       эндпоинта через customer_id_mapping.parquet — см. докстринг
       service/schemas.py.

    3. Fallback на Popularity для неизвестных customer_id — тот же
       принцип, что использовался при офлайн-оценке (Фаза 6,
       run_als_eval.py: build_predictions_with_fallback) и что теперь
       измерен количественно по сегментам (Фаза 9): персонализация
       работает только для юзеров, встречавшихся в train, для всех
       остальных — единый Popularity top-k без персонализации. Источник
       рекомендаций (`source` в ответе) возвращается явно, чтобы вызывающая
       сторона могла отличить персонализированный результат от fallback.

    4. customer_id, которого нет вообще ни в customer_id_mapping (совсем
       не встречался в датасете ни разу), также получает Popularity
       fallback, а не 404 — с точки зрения сервиса это тот же случай
       "холодного" пользователя, что и customer_id из train_end..сегодня,
       которого ALS не видел; различать их на уровне HTTP-ответа не имеет
       смысла для потребителя API.

Запуск:
    uvicorn service.app:app --host 0.0.0.0 --port 8000
    (host/port берутся из configs/config.yaml: service.host/service.port,
    если сервис поднимается через `python -m service.app`)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import polars as pl
from fastapi import FastAPI, HTTPException, Query

from src.config import CONFIG
from src.recommenders.als import (
    ALSCandidateGenerator,
    build_article_id_mapping,
    build_customer_matrix_mapping,
    build_interaction_matrix,
    fit_als_model,
)
from src.recommenders.popularity import PopularityRecommender
from service.schemas import HealthResponse, RecommendationResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """
    Держатель обученных моделей и id-маппинга — заполняется один раз при
    старте (lifespan) и переиспользуется во всех запросах. Не Pydantic,
    потому что это не сетевой контракт, а внутреннее состояние процесса.
    """

    als_generator: ALSCandidateGenerator
    popularity_recommender: PopularityRecommender
    customer_id_to_idx: dict[str, int]
    top_k: int
    n_als_users: int  # для /health — ALSCandidateGenerator не хранит это публично


model_state: ModelState | None = None


def _train_models() -> ModelState:
    """
    Обучает Popularity и ALS на train ровно тем же способом, что и
    src/evaluation/run_als_eval.py (Фаза 6) / run_segmented_eval.py
    (Фаза 9) — не вводит новый способ сборки моделей для сервиса.
    """
    processed_dir = CONFIG.paths.data_processed

    logger.info("Читаю train и customer_id_mapping из %s", processed_dir)
    train = pl.read_parquet(processed_dir / "train.parquet")
    id_mapping = pl.read_parquet(processed_dir / "customer_id_mapping.parquet")

    customer_id_to_idx = dict(
        zip(id_mapping["customer_id"].to_list(), id_mapping["customer_idx"].to_list())
    )

    popularity_recommender = PopularityRecommender().fit(train)

    user_mapping = build_customer_matrix_mapping(train)
    item_mapping = build_article_id_mapping(train)
    als_config = CONFIG.als

    interaction_matrix = build_interaction_matrix(
        train, user_mapping, item_mapping, als_config.confidence_alpha
    )
    als_model = fit_als_model(
        interaction_matrix,
        factors=als_config.factors,
        iterations=als_config.iterations,
        regularization=als_config.regularization,
        random_state=als_config.random_state,
    )
    als_generator = ALSCandidateGenerator(
        als_model, interaction_matrix, user_mapping, item_mapping, top_k=als_config.top_k_candidates
    )

    logger.info(
        "Модели готовы: ALS покрывает %d юзеров, Popularity top-%d товаров, "
        "customer_id_mapping — %d записей",
        len(user_mapping), popularity_recommender.top_n, len(customer_id_to_idx),
    )

    return ModelState(
        als_generator=als_generator,
        popularity_recommender=popularity_recommender,
        customer_id_to_idx=customer_id_to_idx,
        top_k=CONFIG.service.default_top_k,
        n_als_users=user_mapping.height,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Обучает модели один раз при старте сервиса (см. пункт 1 докстринга
    модуля) и кладёт их в module-level model_state, доступный эндпоинтам.
    """
    global model_state
    logger.info("Старт сервиса: обучаю ALS + Popularity...")
    model_state = _train_models()
    logger.info("Сервис готов принимать запросы.")
    yield
    model_state = None


app = FastAPI(
    title="recsys-hm",
    description="Two-stage recommender (ALS retrieval + Popularity fallback) for H&M dataset",
    lifespan=lifespan,
)


def _get_model_state() -> ModelState:
    if model_state is None:
        raise HTTPException(status_code=503, detail="Модели ещё не загружены, попробуйте позже")
    return model_state


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Проверка готовности сервиса — сколько юзеров/товаров реально загружено."""
    state = _get_model_state()
    return HealthResponse(
        status="ok",
        als_users_loaded=state.n_als_users,
        popularity_items_loaded=state.popularity_recommender.top_n,
    )


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    user_id: str = Query(..., description="ID покупателя"),
    top_k: int | None = Query(None, description="Количество рекомендаций"),
) -> RecommendationResponse:
    """
    Возвращает top-k article_id для customer_id: персонализированные
    ALS-кандидаты, если customer_id встречался в train, иначе Popularity
    fallback (см. пункт 3-4 докстринга модуля — сценарий неизвестного
    customer_id обрабатывается тем же fallback-путём, не как ошибка).
    """
    state = _get_model_state()
    k = top_k if top_k is not None else state.top_k

    customer_idx = state.customer_id_to_idx.get(user_id)

    # Заранее проверяем, знает ли ALS этого customer_idx, вместо того чтобы
    # звать recommend_for_users() с потенциально холодным юзером — при
    # пустом known_customer_idxs внутри als.py recommend_for_users()
    # это не документированный и не гарантированный путь (см. докстринг
    # метода: холодные юзеры там просто "пропускаются", но пустой входной
    # список — отдельный случай, не проверенный явно), поэтому сервис
    # сам решает эту развилку на своей стороне, не полагаясь на
    # implicit-обработку пустого батча внутри чужого метода.
    if customer_idx is not None and customer_idx in state.als_generator._customer_idx_to_matrix_idx:
        als_predictions = state.als_generator.recommend_for_users([customer_idx])
        return RecommendationResponse(
            customer_id=user_id,
            recommendations=als_predictions[customer_idx][:k],
            source="als",
        )

    # customer_id не в mapping вообще, или ALS не покрыл его (холодный
    # старт внутри train) — оба случая закрываются одним и тем же
    # Popularity fallback, см. пункт 4 докстринга модуля.
    fallback = state.popularity_recommender.recommend(k=k)
    return RecommendationResponse(
        customer_id=user_id,
        recommendations=fallback,
        source="popularity_fallback",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service.app:app",
        host=CONFIG.service.host,
        port=CONFIG.service.port,
    )
