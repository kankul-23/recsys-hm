"""
Диагностика Фазы 8: почему CatBoostRanker хуже голого ALS на valid.

Наблюдение (прогон 12% train, iterations=500, early stopping на 155):
    ALS (Фаза 6, + Popularity fallback):  recall@10=0.0110  ndcg@100=0.0170
    CatBoostRanker (тот же кандидат-сет): recall@10=0.0072  ndcg@100=0.0144

Recall@100 у ranker'а идентичен ALS (0.0443) — ranker не меняет НАБОР
кандидатов (это всё тот же top-100 ALS+Popularity из valid_features),
только их ПОРЯДОК. Значит переупорядочивание ухудшает верхушку списка
относительно естественного ALS-ранжирования.

Этот скрипт проверяет, что именно доминирует в перестановке: ранжирует
valid_features ТРЕМЯ способами БЕЗ переобучения модели (только сортировка
готовых колонок + инференс уже обученной модели) —

    1. по сырому als_score напрямую (как ранжировал бы "чистый" ALS)
    2. по item_popularity_count напрямую (как ранжировал бы Popularity)
    3. предсказанием уже обученного CatBoostRanker (models/catboost_ranker.cbm)

и сравнивает Recall@10/NDCG@100 для всех трёх на одном и том же наборе
кандидатов (valid_features, все 187783 юзера). Если (1) близко
воспроизводит ALS-цифры Фазы 6, а (3) хуже — доказывает, что ranker
переоценивает популярность в ущерб персонализации, а не что-то другое
(баг в данных, holes в фичах и т.д.).

Дёшево: только инференс на готовых данных, без обучения — минуты, а не
часы, в отличие от grid search по гиперпараметрам.

Запуск:
    python -m src.evaluation.diagnose_ranking_order
"""

from __future__ import annotations

import logging

import polars as pl
from catboost import CatBoostRanker

from src.config import CONFIG
from src.evaluation.metrics import build_ground_truth, evaluate
from src.ranking.ranker import FEATURE_COLUMNS, prepare_for_pool, restore_eval_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rank_by_column(df: pl.DataFrame, score_col: str) -> dict[int, list[int]]:
    """
    Ранжирует df по убыванию score_col внутри каждого customer_idx —
    тот же принцип сортировки, что predictions_from_pool в ranker.py,
    но без обращения к модели (сортировка по уже существующей колонке).
    """
    ranked = (
        df.select(["customer_idx", "article_id", score_col])
        .sort(["customer_idx", score_col], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_id").alias("_ranked_items"))
    )
    return dict(zip(ranked["customer_idx"].to_list(), ranked["_ranked_items"].to_list()))


def rank_by_model(df: pl.DataFrame, model: CatBoostRanker) -> dict[int, list[int]]:
    """Ранжирует df предсказанием уже обученной модели — та же логика, что predictions_from_pool."""
    scores = model.predict(df.select(FEATURE_COLUMNS).to_pandas())
    ranked = (
        df.select(["customer_idx", "article_id"])
        .with_columns(pl.Series("_score", scores))
        .sort(["customer_idx", "_score"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_id").alias("_ranked_items"))
    )
    return dict(zip(ranked["customer_idx"].to_list(), ranked["_ranked_items"].to_list()))


def main() -> None:
    processed_dir = CONFIG.paths.data_processed
    valid_features_path = processed_dir / "valid_features" / "part_0001.parquet"
    valid_raw_path = processed_dir / "valid.parquet"
    model_path = CONFIG.paths.models / "catboost_ranker.cbm"

    logger.info("Читаю valid-признаки: %s", valid_features_path)
    valid_features = pl.read_parquet(valid_features_path)
    valid_labeled = restore_eval_labels(valid_features, valid_raw_path)
    valid_labeled = prepare_for_pool(valid_labeled)

    ground_truth = build_ground_truth(pl.read_parquet(valid_raw_path))
    k_values = CONFIG.evaluation.k_values

    # --- 1. Голый als_score ---
    logger.info("=== Ранжирование по сырому als_score ===")
    predictions_als = rank_by_column(valid_labeled, "als_score")
    results_als = evaluate(predictions_als, ground_truth, k_values)
    logger.info("als_score: %s", results_als)

    # --- 2. Голая популярность ---
    logger.info("=== Ранжирование по item_popularity_count ===")
    predictions_pop = rank_by_column(valid_labeled, "item_popularity_count")
    results_pop = evaluate(predictions_pop, ground_truth, k_values)
    logger.info("item_popularity_count: %s", results_pop)

    # --- 3. Обученный ranker ---
    logger.info("=== Ранжирование обученным CatBoostRanker (%s) ===", model_path)
    model = CatBoostRanker()
    model.load_model(str(model_path))
    predictions_model = rank_by_model(valid_labeled, model)
    results_model = evaluate(predictions_model, ground_truth, k_values)
    logger.info("ranker: %s", results_model)

    # --- Итог ---
    logger.info("=== Сравнение ===")
    logger.info("als_score (голый):        %s", results_als)
    logger.info("item_popularity (голый):  %s", results_pop)
    logger.info("ranker (обученный):       %s", results_model)


if __name__ == "__main__":
    main()
