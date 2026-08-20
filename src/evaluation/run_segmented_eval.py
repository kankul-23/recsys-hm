"""
Сегментированная оценка (Active/Warm vs Cold Users) для recsys-hm.

Реализует Шаги 1-2 Фазы 9 Master Execution Plan (см.
BRANCH_HANDOFF_PHASE_7_8.md, раздел "Следующий шаг по плану — конкретные
действия для Фазы 9"): прогоняет Popularity, ALS+fallback и CatBoostRanker
через evaluate_by_segment() (metrics.py) на valid И на test, с разбивкой
на cold (customer_idx не встречался в train) и warm юзеров.

Это не исследовательский, а завершающий прогон — задача зафиксировать
цифрами то, что было предположением в хэндоффе Фазы 8 ("ALS должен заметно
проседать именно на warm-сегменте относительно cold, там результат
целиком зависит от Popularity fallback"), а не переоткрывать выводы Фазы 8
(тюнинг ranker'а и новые признаки здесь сознательно не делаются — см.
докстринг ranker.py и раздел "Что НЕ делать в Фазе 9" хэндоффа).

Архитектурные решения:

    1. cold_user_ids определяется ОДИНАКОВО для всех трёх моделей и для
       обоих сплитов (valid и test) — как customer_idx из ground truth
       оценочного окна, которых НЕ было среди customer_idx в train. Это
       тот же критерий "холодности", что ALS уже применяет неявно
       (ALSCandidateGenerator пропускает юзеров без вектора) — здесь он
       считается явно и заранее, чтобы одна и та же разбивка на cold/warm
       использовалась для всех трёх моделей без риска рассинхронизации.

    2. Predictions для ALS собираются тем же build_predictions_with_fallback()
       из run_als_eval.py (Фаза 6) — не переизобретается заново. На cold
       юзерах ALS+fallback по построению вырождается в чистый Popularity
       (ALS не покрывает их вообще), так что ожидаемая находка — метрики
       ALS+fallback и Popularity на cold-сегменте должны почти совпасть;
       расхождение говорило бы об ошибке в сборке predictions, а не о
       реальном различии моделей.

    3. Predictions для CatBoostRanker строятся ТОЛЬКО на valid — ranker
       никогда не обучался и не оценивался на test (см. хэндофф: "test
       трогался только в Фазе 6 для честного финального сравнения
       Popularity/ALS — CatBoostRanker и вся диагностика Фазы 8 шли
       исключительно на valid"). Прогонять ranker на test сейчас означало
       бы использовать "нетронутый" сплит для модели, которая и так уже
       не идёт в прод (см. Фазу 8) — оценка на test оставлена только для
       Popularity и ALS+fallback (Шаг 2 плана: "финальное сравнение
       Popularity/ALS", ranker явно не упомянут в этом шаге).

    4. Ranker-признаки на valid читаются из уже собранного
       valid_features/part_0001.parquet (Фаза 7) — та же таблица, что
       использует ranker.py, не пересобирается заново.

Запуск:
    python -m src.evaluation.run_segmented_eval
"""

from __future__ import annotations

import logging

import polars as pl
from catboost import CatBoostRanker

from src.config import CONFIG
from src.evaluation.metrics import build_ground_truth, evaluate_by_segment
from src.evaluation.run_als_eval import build_predictions_with_fallback
from src.ranking.ranker import prepare_for_pool, predictions_from_pool, restore_eval_labels
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


# =============================================================================
# Определение cold_user_ids
# =============================================================================

def get_cold_user_ids(eval_customer_idxs: set[int], train_customer_idxs: set[int]) -> set[int]:
    """
    Холодный юзер оценочного окна — customer_idx, которого не было среди
    customer_idx в train. Тот же критерий, по которому ALSCandidateGenerator
    неявно пропускает юзеров (нет строки в customer_idx_to_matrix_idx) —
    здесь он считается явно и заранее (см. пункт 1 докстринга модуля),
    чтобы разбивка была единой для всех моделей и обоих сплитов.
    """
    cold = eval_customer_idxs - train_customer_idxs
    logger.info(
        "Cold/warm: %d холодных из %d юзеров оценочного окна (%.1f%%)",
        len(cold), len(eval_customer_idxs), 100 * len(cold) / len(eval_customer_idxs),
    )
    return cold


# =============================================================================
# Predictions для CatBoostRanker (только valid — см. пункт 3 докстринга)
# =============================================================================

def build_ranker_predictions_valid(model: CatBoostRanker) -> dict[int, list[int]]:
    """
    Собирает predictions ranker'а на полном valid — тот же способ, что
    main() в ranker.py (restore_eval_labels + prepare_for_pool +
    predictions_from_pool), без обучения новой модели: model передаётся
    уже обученной (загруженной из models/catboost_ranker.cbm).
    """
    processed_dir = CONFIG.paths.data_processed
    valid_features_path = processed_dir / "valid_features" / "part_0001.parquet"
    valid_raw_path = processed_dir / "valid.parquet"

    valid_features = pl.read_parquet(valid_features_path)
    valid_labeled_full = restore_eval_labels(valid_features, valid_raw_path)
    valid_labeled_full = prepare_for_pool(valid_labeled_full)

    return predictions_from_pool(model, valid_labeled_full)


# =============================================================================
# Печать результатов
# =============================================================================

def log_segmented_results(model_name: str, split_name: str, segmented: dict[str, dict[str, float]]) -> None:
    logger.info("--- %s на %s ---", model_name, split_name)
    for segment in ("cold", "warm"):
        logger.info("  %s: %s", segment, segmented[segment])


# =============================================================================
# Точка входа
# =============================================================================

def main() -> None:
    processed_dir = CONFIG.paths.data_processed
    k_values = CONFIG.evaluation.k_values

    logger.info("Читаю train/valid/test из %s", processed_dir)
    train = pl.read_parquet(processed_dir / "train.parquet")
    valid = pl.read_parquet(processed_dir / "valid.parquet")
    test = pl.read_parquet(processed_dir / "test.parquet")

    train_customer_idxs = set(train["customer_idx"].unique().to_list())

    # --- Обучение Popularity и ALS (та же последовательность, что в
    # run_als_eval.py — не переизобретается заново) ---
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

    # --- Загрузка обученного CatBoostRanker (models/catboost_ranker.cbm,
    # см. пункт 3 докстринга — не переобучается здесь) ---
    ranker_model_path = CONFIG.paths.models / "catboost_ranker.cbm"
    ranker_model = CatBoostRanker()
    ranker_model.load_model(str(ranker_model_path))
    logger.info("CatBoostRanker загружен: %s", ranker_model_path)

    # =========================================================================
    # VALID — все три модели (Шаг 1 плана)
    # =========================================================================
    valid_ground_truth = build_ground_truth(valid)
    valid_customer_idxs = set(valid_ground_truth.keys())
    valid_cold_ids = get_cold_user_ids(valid_customer_idxs, train_customer_idxs)

    popularity_predictions_valid = popularity_recommender.recommend_for_users(
        list(valid_customer_idxs)
    )
    als_predictions_valid = build_predictions_with_fallback(
        als_generator, popularity_recommender, list(valid_customer_idxs)
    )
    ranker_predictions_valid = build_ranker_predictions_valid(ranker_model)

    popularity_segmented_valid = evaluate_by_segment(
        popularity_predictions_valid, valid_ground_truth, k_values, valid_cold_ids
    )
    als_segmented_valid = evaluate_by_segment(
        als_predictions_valid, valid_ground_truth, k_values, valid_cold_ids
    )
    ranker_segmented_valid = evaluate_by_segment(
        ranker_predictions_valid, valid_ground_truth, k_values, valid_cold_ids
    )

    # =========================================================================
    # TEST — только Popularity и ALS+fallback (Шаг 2 плана; ranker на test
    # намеренно не считается, см. пункт 3 докстринга модуля)
    # =========================================================================
    test_ground_truth = build_ground_truth(test)
    test_customer_idxs = set(test_ground_truth.keys())
    test_cold_ids = get_cold_user_ids(test_customer_idxs, train_customer_idxs)

    popularity_predictions_test = popularity_recommender.recommend_for_users(
        list(test_customer_idxs)
    )
    als_predictions_test = build_predictions_with_fallback(
        als_generator, popularity_recommender, list(test_customer_idxs)
    )

    popularity_segmented_test = evaluate_by_segment(
        popularity_predictions_test, test_ground_truth, k_values, test_cold_ids
    )
    als_segmented_test = evaluate_by_segment(
        als_predictions_test, test_ground_truth, k_values, test_cold_ids
    )

    # =========================================================================
    # Итог
    # =========================================================================
    logger.info("=== Итог Фазы 9: сегментированная оценка (cold vs warm) ===")
    log_segmented_results("Popularity", "valid", popularity_segmented_valid)
    log_segmented_results("ALS + fallback", "valid", als_segmented_valid)
    log_segmented_results("CatBoostRanker", "valid", ranker_segmented_valid)
    log_segmented_results("Popularity", "test", popularity_segmented_test)
    log_segmented_results("ALS + fallback", "test", als_segmented_test)


if __name__ == "__main__":
    main()
