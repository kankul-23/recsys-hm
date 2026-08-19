"""
Диагностика Фазы 8, продолжение: проверка гипотезы "ranker учится
тавтологии" — что label на train тривиально предсказывается через
item_popularity_count / user_total_purchases, вместо того чтобы нести
дополнительный сигнал сверх того, что эти признаки и так кодируют.

Контекст (diagnose_ranking_order.py): голый als_score (recall@10=0.0106)
и голая item_popularity_count (recall@10=0.0101) ПООТДЕЛЬНОСТИ ранжируют
лучше, чем обученный CatBoostRanker (recall@10=0.0072), использующий
оба этих признака плюс ещё 11. Значит комбинация признаков через
обучение УХУДШАЕТ то, что каждый признак даёт по отдельности — это не
объясняется нехваткой сигнала, это объясняется тем, что обучение находит
не тот сигнал.

Рабочая гипотеза: item_popularity_count и user_total_purchases считаются
по ТОМУ ЖЕ train-окну, что и сам label (см. src/features/builder.py:
positive_pairs строится из train, item_features/user_features — тоже
агрегаты по train). Товар, купленный юзером в train (label=1), при этом
статистически чаще оказывается популярным товаром (item_popularity_count
высокий) и товаром юзера с длинной историей (user_total_purchases
высокий) — не потому что это ПРЕДСКАЗЫВАЕТ покупку, а потому что
популярные товары в принципе чаще покупают, включая ЭТУ САМУЮ покупку,
которая вошла в подсчёт популярности. CatBoost, оптимизируя loss на
label, находит эту тавтологию как самый сильный доступный сигнал и
переобучается на неё в ущерб als_score (единственному признаку, не
привязанному напрямую к train-статистике покупок тем же образом).

Проверяется здесь БЕЗ обучения модели — чистая статистика на train:
    1. Средний item_popularity_count / user_total_purchases для
       label=1 vs label=0 — если для label=1 систематически и сильно
       выше, тавтология подтверждается.
    2. AUC-подобная проверка: если ранжировать train ТОЛЬКО по
       item_popularity_count, какой Recall@10 получится НА САМОМ TRAIN
       (не valid) — если он аномально высокий (существенно выше, чем
       "случайный" бы дал), это подтверждает, что признак почти
       тождественен целевой переменной на train, хотя и не является
       ей по построению (см. builder.py — item_popularity_count не
       включает сам label напрямую, но коррелирует с ним по конструкции
       данных).

Запуск:
    python -m src.evaluation.diagnose_label_tautology
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    processed_dir = CONFIG.paths.data_processed
    train_features_dir = processed_dir / "train_features"

    logger.info("Читаю train_features (лениво, все 44 чанка): %s", train_features_dir)
    train_lazy = pl.scan_parquet(str(train_features_dir / "*.parquet"))

    # --- 1. Средние значения по группам label=1 / label=0 ---
    logger.info("=== Средние значения признаков по label (весь train) ===")
    stats = (
        train_lazy
        .group_by("label")
        .agg([
            pl.col("item_popularity_count").mean().alias("item_popularity_count_mean"),
            pl.col("item_popularity_count").median().alias("item_popularity_count_median"),
            pl.col("user_total_purchases").mean().alias("user_total_purchases_mean"),
            pl.col("als_score").mean().alias("als_score_mean"),
            pl.len().alias("n_rows"),
        ])
        .sort("label")
        .collect(engine="streaming")
    )
    logger.info("%s", stats)

    # --- 2. Отношение средних: если для label=1 популярность в разы
    # выше, чем для label=0, это прямая улика тавтологии — CatBoost
    # видит "высокая популярность -> вероятно label=1" почти как
    # детерминированное правило, а не слабую корреляцию.
    row_pos = stats.filter(pl.col("label") == 1)
    row_neg = stats.filter(pl.col("label") == 0)
    if row_pos.height and row_neg.height:
        ratio_pop = row_pos["item_popularity_count_mean"][0] / row_neg["item_popularity_count_mean"][0]
        ratio_purch = row_pos["user_total_purchases_mean"][0] / row_neg["user_total_purchases_mean"][0]
        ratio_als = row_pos["als_score_mean"][0] / row_neg["als_score_mean"][0]
        logger.info(
            "Отношение label=1 / label=0: item_popularity_count=%.2fx, "
            "user_total_purchases=%.2fx, als_score=%.2fx",
            ratio_pop, ratio_purch, ratio_als,
        )

    # --- 3. Recall@10 НА САМОМ TRAIN, ранжируя только по
    # item_popularity_count — если аномально высок, популярность почти
    # тождественна label на train (по конструкции данных, не по смыслу
    # признака) и модель просто её заучивает.
    logger.info("Считаю Recall@10 на train при ранжировании по item_popularity_count...")
    train_full = train_lazy.select(
        ["customer_idx", "article_id", "label", "item_popularity_count"]
    ).collect(engine="streaming")

    ranked = (
        train_full
        .sort(["customer_idx", "item_popularity_count"], descending=[False, True])
        .with_columns(
            pl.int_range(0, pl.len()).over("customer_idx").alias("_rank")
        )
        .filter(pl.col("_rank") < 10)
    )
    hits_per_user = (
        ranked.group_by("customer_idx")
        .agg(pl.col("label").sum().alias("hits_in_top10"))
    )
    positives_per_user = (
        train_full.filter(pl.col("label") == 1)
        .group_by("customer_idx")
        .agg(pl.len().alias("n_positives"))
    )
    recall_frame = hits_per_user.join(positives_per_user, on="customer_idx", how="inner").filter(
        pl.col("n_positives") > 0
    ).with_columns(
        (pl.col("hits_in_top10") / pl.min_horizontal(pl.col("n_positives"), pl.lit(10))).alias("recall_at_10")
    )
    mean_recall = recall_frame["recall_at_10"].mean()
    logger.info(
        "Recall@10 НА TRAIN (ранжируя внутри train-кандидатов по item_popularity_count): %.4f "
        "(для сравнения: тот же голый item_popularity_count на VALID дал 0.0101 — "
        "diagnose_ranking_order.py)",
        mean_recall,
    )
    logger.info(
        "Если это число намного выше 0.0101 (valid) — популярность почти тождественна "
        "label НА TRAIN конкретно (тавтология конструкции данных), и не обобщается на valid."
    )


if __name__ == "__main__":
    main()
