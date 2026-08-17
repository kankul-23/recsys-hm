"""
Оценка ALS Candidate Generator на valid и test (Фаза 6, часть 1).

Связывает src/recommenders/als.py, src/recommenders/popularity.py и
src/evaluation/metrics.py: считает Candidate Recall@K/NDCG@K для ALS и
сравнивает с уже посчитанным Popularity baseline (Фаза 4).

Ключевое решение — честность сравнения с Popularity:

    ALSCandidateGenerator.recommend_for_users() молча пропускает юзеров,
    которых не было в train (холодный старт) — у него просто нет для них
    вектора. Если оценивать ALS "как есть", evaluate() в metrics.py
    исключит этих юзеров из усреднения (нет предсказаний = пропуск), и
    Recall@K для ALS будет считаться только по тёплым юзерам — более лёгкому
    подмножеству, чем то, на котором считался Popularity (который покрывает
    вообще всех, включая холодных, единым списком).

    Чтобы сравнение было корректным (обе модели оцениваются на ОДНОМ и том
    же множестве юзеров), для юзеров, которых ALS не покрыл, здесь
    подставляется fallback-список от Popularity — тот же принцип, что
    заложен в Фазу 9 плана (Production Serving: "автоматический Fallback
    на Popularity" для неизвестных пользователей), только применённый уже
    сейчас, на этапе офлайн-оценки, а не в проде.

Запуск:
    python -m src.evaluation.run_als_eval
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG
from src.evaluation.metrics import build_ground_truth, evaluate
from src.recommenders.als import (
    ALSCandidateGenerator,
    build_article_id_mapping,
    build_customer_matrix_mapping,
    build_interaction_matrix,
    fit_als_model,
)
from src.recommenders.popularity import PopularityRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_predictions_with_fallback(
    als_generator: ALSCandidateGenerator,
    popularity_recommender: PopularityRecommender,
    customer_idxs: list[int],
) -> dict[int, list[int]]:
    """
    Строит predictions для evaluate(): персональные ALS-кандидаты для
    юзеров, которых ALS знает, и Popularity-фоллбэк для остальных.

    Fallback подставляется ЦЕЛИКОМ (весь top-100 Popularity), а не частично
    дополняет короткий список ALS — на этом этапе холодный юзер либо
    полностью персонализирован (ALS), либо полностью нет (Popularity),
    без смешивания источников в одном списке. Смешивание кандидатов
    из разных моделей внутри одного списка — отдельное архитектурное
    решение (актуально для Фазы 9), которое здесь сознательно не берётся.
    """
    als_predictions = als_generator.recommend_for_users(customer_idxs)
    popularity_fallback = popularity_recommender.recommend()

    predictions: dict[int, list[int]] = {}
    n_fallback = 0

    for customer_idx in customer_idxs:
        if customer_idx in als_predictions:
            predictions[customer_idx] = als_predictions[customer_idx]
        else:
            predictions[customer_idx] = popularity_fallback
            n_fallback += 1

    logger.info(
        "Predictions собраны: %d юзеров через ALS, %d через Popularity fallback (холодный старт)",
        len(als_predictions), n_fallback,
    )

    return predictions


def evaluate_on_split(
    als_generator: ALSCandidateGenerator,
    popularity_recommender: PopularityRecommender,
    split_df: pl.DataFrame,
    split_name: str,
    k_values: list[int],
) -> dict[str, float]:
    """Строит ground truth из split_df и считает метрики ALS+fallback."""
    logger.info("=== Оценка ALS на %s ===", split_name)

    ground_truth = build_ground_truth(split_df)
    predictions = build_predictions_with_fallback(
        als_generator, popularity_recommender, list(ground_truth.keys())
    )
    results = evaluate(predictions, ground_truth, k_values)

    return results


def main() -> None:
    processed_dir = CONFIG.paths.data_processed

    logger.info("Читаю train/valid/test из %s", processed_dir)
    train = pl.read_parquet(processed_dir / "train.parquet")
    valid = pl.read_parquet(processed_dir / "valid.parquet")
    test = pl.read_parquet(processed_dir / "test.parquet")

    # --- Обучение Popularity (нужен для fallback) ---
    popularity_recommender = PopularityRecommender().fit(train)

    # --- Обучение ALS (та же последовательность, что в src/recommenders/als.py) ---
    user_mapping = build_customer_matrix_mapping(train)
    item_mapping = build_article_id_mapping(train)

    als_config = CONFIG.als
    interaction_matrix = build_interaction_matrix(
        train, user_mapping, item_mapping, als_config.confidence_alpha
    )
    model = fit_als_model(
        interaction_matrix,
        factors=als_config.factors,
        iterations=als_config.iterations,
        regularization=als_config.regularization,
        random_state=als_config.random_state,
    )
    als_generator = ALSCandidateGenerator(
        model, interaction_matrix, user_mapping, item_mapping, top_k=als_config.top_k_candidates
    )

    # --- Оценка ---
    k_values = CONFIG.evaluation.k_values

    valid_results = evaluate_on_split(als_generator, popularity_recommender, valid, "valid", k_values)
    test_results = evaluate_on_split(als_generator, popularity_recommender, test, "test", k_values)

    logger.info("=== Итог: ALS (+ Popularity fallback для холодных) ===")
    logger.info("valid: %s", valid_results)
    logger.info("test:  %s", test_results)


if __name__ == "__main__":
    main()
