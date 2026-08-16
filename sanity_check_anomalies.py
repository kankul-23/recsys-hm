"""
Sanity-check для Фазы 3: распределение числа транзакций среди покупателей,
отфильтрованных как аномальные (порог 99-го перцентиля по train).

Не часть основного пайплайна — разовый диагностический скрипт.
Запуск: python sanity_check_anomalies.py
"""

import polars as pl
from datetime import date

from src.config import CONFIG

transactions = pl.read_parquet(CONFIG.paths.data_processed / "transactions_train.parquet")

train_end = date.fromisoformat(CONFIG.temporal_split.train_end)
train = transactions.filter(pl.col("t_dat") <= train_end)

counts = train.group_by("customer_idx").agg(pl.len().alias("n_transactions"))

threshold = int(
    counts.select(pl.col("n_transactions").quantile(0.99, interpolation="nearest")).item()
)
print(f"Порог (p99): {threshold} транзакций")
print(f"Всего покупателей в train: {counts.height}")

anomalous = counts.filter(pl.col("n_transactions") > threshold)
print(f"Аномальных покупателей: {anomalous.height}")
print()

# Распределение аномальных по бакетам
print("=== Распределение аномальных по бакетам n_transactions ===")
buckets = anomalous.with_columns(
    pl.when(pl.col("n_transactions") <= 200).then(pl.lit("181-200"))
    .when(pl.col("n_transactions") <= 300).then(pl.lit("201-300"))
    .when(pl.col("n_transactions") <= 500).then(pl.lit("301-500"))
    .when(pl.col("n_transactions") <= 1000).then(pl.lit("501-1000"))
    .otherwise(pl.lit("1000+"))
    .alias("bucket")
).group_by("bucket").agg(pl.len().alias("n_customers")).sort("n_customers", descending=True)
print(buckets)
print()

# Топ-20 самых активных
print("=== Топ-20 по числу транзакций ===")
print(anomalous.sort("n_transactions", descending=True).head(20))
print()

# Насколько "плавный" переход через порог — смотрим окрестность p95-p99.9
print("=== Перцентили в окрестности порога ===")
for p in [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]:
    val = counts.select(pl.col("n_transactions").quantile(p, interpolation="nearest")).item()
    print(f"  p{p*100:.1f}: {val} транзакций")
