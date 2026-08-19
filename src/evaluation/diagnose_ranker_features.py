"""
Диагностика Фазы 8: почему CatBoostRanker не превзошёл голый ALS.

Два прогона (5% и 12% train) дали ПОЧТИ ИДЕНТИЧНЫЙ результат — feature
importance одинаковый с точностью до сотых, Recall@10 не улучшился.
Это опровергает гипотезу "мало данных" (иначе прирост в 2.4x данных
дал бы хоть какое-то отличие) и указывает на структурную причину: сами
признаки могут быть недостаточно вариативны ВНУТРИ top-100 кандидатов
одного юзера, чтобы ranker'у было что использовать для различения
"купит именно этот" — даже если те же признаки хорошо различают товары
на масштабе всего каталога (как показала разведка в
02_feature_exploration.ipynb).

Этот скрипт считает variance/range признаков ВНУТРИ группы (customer_idx),
а не по всему датасету — то, что не проверялось в исходной разведке.

Запуск:
    python -m src.evaluation.diagnose_ranker_features
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_FEATURES = [
    "item_popularity_count",
    "item_avg_price",
    "item_price_volatility",
    "item_days_since_first_sale",
    "item_age_bucket_affinity",
    "als_score",
    "price_diff_from_user_avg",
    "user_age_diff_from_item_typical_buyer",
]


def main() -> None:
    processed_dir = CONFIG.paths.data_processed
    train_features_dir = processed_dir / "train_features"

    logger.info("Читаю первый чанк train_features (part_0001.parquet) для диагностики")
    sample = pl.read_parquet(train_features_dir / "part_0001.parquet")

    # Внутригрупповая (по customer_idx) вариация каждого признака — это
    # то, что реально видит CatBoost при сравнении кандидатов ОДНОГО
    # юзера между собой, а не разброс по всему датасету (который уже
    # проверялся в 02_feature_exploration.ipynb и был в порядке).
    logger.info("Внутригрупповая (per-customer_idx) статистика по кандидатам:")

    within_group_stats = (
        sample.group_by("customer_idx")
        .agg([
            *[pl.col(c).std().alias(f"{c}_std") for c in CANDIDATE_FEATURES],
            *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in CANDIDATE_FEATURES],
            pl.len().alias("n_candidates"),
        ])
    )

    logger.info("Медианный размер группы (кандидатов на юзера): %s",
                within_group_stats["n_candidates"].median())

    for feature in CANDIDATE_FEATURES:
        std_col = f"{feature}_std"
        range_col = f"{feature}_range"
        median_std = within_group_stats[std_col].median()
        median_range = within_group_stats[range_col].median()
        pct_zero_variance = (
            within_group_stats.filter(pl.col(std_col) == 0).height
            / within_group_stats.height
        )
        logger.info(
            "  %s: медианный std=%.4f, медианный range=%.4f, "
            "%.1f%% юзеров с НУЛЕВОЙ вариацией (все кандидаты одинаковы)",
            feature, median_std or 0.0, median_range or 0.0, 100 * pct_zero_variance,
        )

    # Отдельно: same_product_group_as_history и is_cold_* — булевы,
    # проверяем долю юзеров, у которых признак вообще принимает оба
    # значения (True и False) среди их кандидатов — если почти всегда
    # константа внутри группы, ranker не может её использовать для
    # сравнения ВНУТРИ группы (только между группами, что не то же самое
    # для groupwise loss).
    for bool_feature in ["same_product_group_as_history", "is_cold_user", "is_cold_item"]:
        variation = (
            sample.group_by("customer_idx")
            .agg(pl.col(bool_feature).n_unique().alias("_n_unique"))
        )
        pct_varying = (variation.filter(pl.col("_n_unique") > 1).height / variation.height)
        logger.info(
            "  %s: %.1f%% юзеров имеют ОБА значения среди кандидатов (иначе константа в группе)",
            bool_feature, 100 * pct_varying,
        )


if __name__ == "__main__":
    main()
