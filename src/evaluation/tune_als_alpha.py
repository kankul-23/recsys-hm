"""
Точечный перебор confidence_alpha для Implicit ALS (Фаза 6, разбор
результата).

Причина, почему тюним именно confidence_alpha, а не полный grid search
по factors/regularization/confidence_alpha: confidence_alpha сильнее всего
завязан на природу конкретного датасета (см. Hu, Koren, Volinsky, 2008 —
исходная статья тюнила его под Last.fm, где сотни прослушиваний на пару
юзер-трек; в H&M среднее число покупок одной пары (юзер, товар) — 1-2, то
есть совсем другой масштаб сигнала). factors=64 и regularization=0.05 —
более стандартные, устойчивые значения, менее вероятный источник
проблемы, замеченной в src/evaluation/run_als_eval.py (ALS почти не
отличим от Popularity, местами хуже на test).

Перебираются только 3 значения (15, 40, 100) — узкий, целенаправленный
чек-ап, а не исчерпывающий тюнинг: полный grid search по нескольким
гиперпараметрам сразу стоил бы кратно дороже по времени обучения
(~2 минуты на один прогон ALS), не будучи оправданным на этом этапе.

Оценка — ТОЛЬКО на valid (не на test), чтобы не тратить test раньше
времени: test должен оставаться "нетронутым" до финального сравнения
моделей в Фазе 8, а valid — ровно то место, где допустимо принимать
решения о гиперпараметрах.

Оценка — БЕЗ Popularity fallback для холодных юзеров (в отличие от
run_als_eval.py): здесь цель — сравнить сами варианты ALS между собой на
одинаковом (только тёплые юзеры) подмножестве, а не воспроизводить
абсолютное сравнение с Popularity. Fallback добавил бы одинаковый сдвиг
ко всем трём вариантам и не помог бы увидеть разницу между ними.

Запуск:
    python -m src.evaluation.tune_als_alpha
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_ALPHAS = [15, 40, 100]


def main() -> None:
    processed_dir = CONFIG.paths.data_processed

    logger.info("Читаю train/valid из %s", processed_dir)
    train = pl.read_parquet(processed_dir / "train.parquet")
    valid = pl.read_parquet(processed_dir / "valid.parquet")

    user_mapping = build_customer_matrix_mapping(train)
    item_mapping = build_article_id_mapping(train)

    ground_truth = build_ground_truth(valid)

    als_config = CONFIG.als
    k_values = CONFIG.evaluation.k_values

    all_results: dict[int, dict[str, float]] = {}

    for alpha in CANDIDATE_ALPHAS:
        logger.info("=== confidence_alpha=%d ===", alpha)

        interaction_matrix = build_interaction_matrix(train, user_mapping, item_mapping, alpha)
        model = fit_als_model(
            interaction_matrix,
            factors=als_config.factors,
            iterations=als_config.iterations,
            regularization=als_config.regularization,
            random_state=als_config.random_state,
        )
        generator = ALSCandidateGenerator(
            model, interaction_matrix, user_mapping, item_mapping, top_k=als_config.top_k_candidates
        )

        # Без fallback: только юзеры, которых ALS реально знает (см. докстринг).
        predictions = generator.recommend_for_users(list(ground_truth.keys()))
        results = evaluate(predictions, ground_truth, k_values)

        all_results[alpha] = results

    logger.info("=== Сводка по confidence_alpha (valid, только тёплые юзеры) ===")
    for alpha, results in all_results.items():
        logger.info("alpha=%d: %s", alpha, results)

    best_alpha = max(all_results, key=lambda a: all_results[a][f"recall@{k_values[-1]}"])
    logger.info(
        "Лучший alpha по recall@%d: %d (%.4f)",
        k_values[-1], best_alpha, all_results[best_alpha][f"recall@{k_values[-1]}"],
    )


if __name__ == "__main__":
    main()
