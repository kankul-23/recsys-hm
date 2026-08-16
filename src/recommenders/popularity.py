"""
Popularity Baseline для recsys-hm.

Реализует Фазу 4 Master Execution Plan:
    - глобальный (не персонализированный) топ-N популярных товаров,
      посчитанный СТРОГО на train
    - единая точка отсчёта для оценки прироста от ALS/CatBoost на
      последующих фазах

Ключевые архитектурные решения (зафиксированы по итогам обсуждения):

    1. Метрика популярности — count УНИКАЛЬНЫХ покупателей, а не count
       транзакций. Обоснование напрямую связано с находкой из Фазы 2 EDA
       про дубликаты транзакций (см. notebooks/01_eda_temporal.ipynb,
       раздел про 9.36% "лишнего" объёма): если считать по transaction
       count, один покупатель, купивший 5 единиц одного артикула,
       перевешивает 5 разных покупателей, купивших по одной штуке. Для
       Popularity-баейзлайна нужен сигнал охвата аудитории ("скольким
       разным людям товар вообще понравился"), а не сигнал объёма продаж.

    2. Список фиксированный, БЕЗ персонализации и без исключения товаров,
       которые пользователь уже купил в train. Это осознанное упрощение
       для первого, самого простого baseline — персонализированная
       фильтрация уже купленного добавляется в Фазе 5 (ALS), где она
       обязательна по плану ("Top-100 кандидатов с гарантированным
       исключением ранее купленных товаров").

    3. top-N фиксирован на уровне 100 (совпадает с top_k_candidates у ALS
       в config.yaml, чтобы модели можно было сравнивать на одном
       горизонте кандидатов). Метрики @K режутся из этого единого списка
       на этапе оценки (src/evaluation/metrics.py), а не пересчитываются
       под каждое K отдельно — экономит вычисления и гарантирует, что
       Recall@10 и Recall@100 считаются на согласованных данных.

    4. Category-level срезы популярности (упомянуты в Master Execution
       Plan для Фазы 4) сознательно ОТЛОЖЕНЫ. Они относятся к сценарию
       fallback для холодных пользователей в проде (Фаза 9), где нужен
       контекст категории интереса юзера — то есть это уже вопрос
       архитектуры сервинга, а не офлайн-baseline. TODO: вернуться при
       реализации Фазы 9 (service/app.py, fallback на Popularity).

Запуск:
    python -m src.recommenders.popularity
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Фиксированный размер топа — совпадает с als.top_k_candidates в config.yaml,
# чтобы Popularity и ALS были сравнимы на одном горизонте кандидатов.
TOP_N = 100


class PopularityRecommender:
    """
    Непесонализированный baseline: один и тот же топ-N товаров для всех
    пользователей, отсортированный по числу уникальных покупателей в train.

    Не фильтрует товары, которые пользователь уже купил — это осознанное
    упрощение первого baseline (см. модульный docstring, пункт 2).
    """

    def __init__(self, top_n: int = TOP_N) -> None:
        self.top_n = top_n
        self._popular_items: pl.DataFrame | None = None  # article_id, n_unique_customers

    def fit(self, train: pl.DataFrame) -> "PopularityRecommender":
        """
        Считает популярность каждого article_id на train как число уникальных
        customer_idx, купивших этот товар хотя бы раз, и сохраняет top_n
        товаров по убыванию популярности.

        train ожидается уже отфильтрованным от аномальных покупателей
        (src/data/split.py: filter_anomalous_customers) — эта функция сама
        по себе фильтрацию не делает и не должна её дублировать.
        """
        popularity = (
            train.group_by("article_id")
            .agg(pl.col("customer_idx").n_unique().alias("n_unique_customers"))
            .sort("n_unique_customers", descending=True)
        )

        self._popular_items = popularity.head(self.top_n)

        logger.info(
            "Popularity: посчитано на %d строках train, %d уникальных article_id, "
            "top-%d сохранён (от %d до %d уникальных покупателей)",
            train.height,
            popularity.height,
            self.top_n,
            self._popular_items["n_unique_customers"].max(),
            self._popular_items["n_unique_customers"].min(),
        )

        return self

    def recommend(self, k: int | None = None) -> list[int]:
        """
        Возвращает топ-k article_id по убыванию популярности.

        Список одинаков для всех пользователей — Popularity не
        персонализирует рекомендации. k=None возвращает весь top_n список,
        посчитанный в fit().
        """
        if self._popular_items is None:
            raise RuntimeError("Сначала нужно вызвать fit() — популярность не посчитана.")

        items = self._popular_items["article_id"].to_list()
        return items if k is None else items[:k]

    def recommend_for_users(self, user_ids: list[int], k: int | None = None) -> dict[int, list[int]]:
        """
        Удобная обёртка для оценки на valid/test: отдаёт один и тот же
        top-k список каждому user_id из переданного списка.

        Формат словаря user_id -> список рекомендаций совпадает с тем,
        что ожидает src/evaluation/metrics.py от любого рекомендера
        (Popularity, ALS, CatBoost) — единый контракт между этапами.
        """
        recs = self.recommend(k)
        return {user_id: recs for user_id in user_ids}


def main() -> None:
    train_path = CONFIG.paths.data_processed / "train.parquet"
    logger.info("Читаю train: %s", train_path)
    train = pl.read_parquet(train_path)

    recommender = PopularityRecommender(top_n=TOP_N).fit(train)

    top_10 = recommender.recommend(k=10)
    logger.info("Топ-10 популярных article_id: %s", top_10)


if __name__ == "__main__":
    main()
