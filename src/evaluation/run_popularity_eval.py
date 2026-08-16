"""
Оценка Popularity baseline на valid и test.

Связывает src/recommenders/popularity.py и src/evaluation/metrics.py:
считает популярность на train, оценивает Recall@K/NDCG@K на valid и
отдельно на test (K берутся из configs/config.yaml: evaluation.k_values).

Оценка идёт по ВСЕМ юзерам из оценочного окна, включая холодных
(тех, кого не было в train) — см. обоснование в docstring
src/evaluation/metrics.py, пункт 2. Predictions для холодных юзеров —
тот же глобальный Popularity-список, что и для остальных: у Popularity
нет персонализации, поэтому для него "холодный" юзер не требует
отдельной логики, в отличие от ALS/CatBoost в следующих фазах.

Зачем валидировать и на valid, и на test отдельно: valid — то, на чём
в будущем можно будет подбирать гиперпараметры ALS/CatBoost без утечки
в test; test — финальная, "нетронутая" оценка. Для Popularity
гиперпараметров нет, но прогон на обоих окнах уже сейчас даёт
референсные числа для сравнения с последующими моделями на одинаковых
срезах данных.

Запуск:
    python -m src.evaluation.run_popularity_eval
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG
from src.evaluation.metrics import build_ground_truth, evaluate
from src.recommenders.popularity import PopularityRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_on_split(
    recommender: PopularityRecommender,
    split_df: pl.DataFrame,
    split_name: str,
    k_values: list[int],
) -> dict[str, float]:
    """Строит ground truth из split_df и считает метрики для recommender."""
    logger.info("=== Оценка на %s ===", split_name)

    ground_truth = build_ground_truth(split_df)
    predictions = recommender.recommend_for_users(list(ground_truth.keys()))
    results = evaluate(predictions, ground_truth, k_values)

    return results


def main() -> None:
    processed_dir = CONFIG.paths.data_processed

    logger.info("Читаю train/valid/test из %s", processed_dir)
    train = pl.read_parquet(processed_dir / "train.parquet")
    valid = pl.read_parquet(processed_dir / "valid.parquet")
    test = pl.read_parquet(processed_dir / "test.parquet")

    recommender = PopularityRecommender().fit(train)

    k_values = CONFIG.evaluation.k_values

    valid_results = evaluate_on_split(recommender, valid, "valid", k_values)
    test_results = evaluate_on_split(recommender, test, "test", k_values)

    logger.info("=== Итог: Popularity baseline ===")
    logger.info("valid: %s", valid_results)
    logger.info("test:  %s", test_results)


if __name__ == "__main__":
    main()
