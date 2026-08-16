"""
Temporal Train/Valid/Test Split для recsys-hm.

Реализует Фазу 3 Master Execution Plan:
    - нарезка transactions_train.parquet на train/valid/test по датам
      из configs/config.yaml (temporal_split) — защита от Temporal Leakage
    - фильтрация аномальных покупателей (опт/корпоративные аккаунты),
      порог считается СТРОГО по train, чтобы не подглядывать в будущее

Ключевое архитектурное решение (зафиксировано по итогам обсуждения):
    Порог фильтрации — фиксированное число транзакций на покупателя
    (configs/config.yaml: anomaly_filter.max_transactions), подобранное по
    итогам визуального анализа распределения (см. sanity_check_anomalies.py),
    а не "чистый" процентиль. Чистый 99-й перцентиль на этом датасете резал
    бы 79% отсечённых из диапазона правдоподобного поведения (0.5-1
    покупка/день) — фиксированный порог целится точнее в явные выбросы.

    Порог применяется ТОЛЬКО к train — valid и test остаются нетронутыми,
    чтобы метрики на них измеряли качество на реалистичном,
    неотфильтрованном распределении покупателей.

Запуск:
    python -m src.data.split
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Temporal split
# =============================================================================

def temporal_split(
    df: pl.DataFrame,
    train_end: str,
    valid_end: str,
    test_end: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Режет transactions на train/valid/test по датам t_dat.

    Границы (по итогам EDA, Фаза 2):
        train: t_dat <= train_end
        valid: train_end < t_dat <= valid_end
        test:  valid_end < t_dat <= test_end

    Верхняя граница каждого окна включительна (<=), нижняя — исключительна (>),
    так что периоды не пересекаются и не теряют ни одного дня на стыках.
    """
    train_end_d = date.fromisoformat(train_end)
    valid_end_d = date.fromisoformat(valid_end)
    test_end_d = date.fromisoformat(test_end)

    train = df.filter(pl.col("t_dat") <= train_end_d)
    valid = df.filter((pl.col("t_dat") > train_end_d) & (pl.col("t_dat") <= valid_end_d))
    test = df.filter((pl.col("t_dat") > valid_end_d) & (pl.col("t_dat") <= test_end_d))

    logger.info(
        "Split: train=%d строк (<= %s), valid=%d строк (%s, %s], test=%d строк (%s, %s]",
        train.height, train_end,
        valid.height, train_end, valid_end,
        test.height, valid_end, test_end,
    )

    return train, valid, test


# =============================================================================
# Фильтрация аномальных покупателей (только на train)
# =============================================================================

def filter_anomalous_customers(train: pl.DataFrame, threshold: int) -> pl.DataFrame:
    """
    Убирает из train покупателей, чьё число транзакций внутри train
    превышает threshold (вероятно опт/корпоративные аккаунты, а не
    обычные розничные покупатели — см. Фаза 2 EDA, раздел 5, и
    sanity_check_anomalies.py для обоснования конкретного значения).

    Применяется ТОЛЬКО к train. valid/test не фильтруются.
    """
    counts = train.group_by("customer_idx").agg(pl.len().alias("n_transactions"))
    normal_customers = counts.filter(pl.col("n_transactions") <= threshold).select("customer_idx")

    n_before = train.height
    filtered = train.join(normal_customers, on="customer_idx", how="inner")
    n_after = filtered.height

    logger.info(
        "Фильтрация train: %d -> %d строк (убрано %d строк, %.2f%%)",
        n_before, n_after, n_before - n_after, 100 * (n_before - n_after) / n_before,
    )

    return filtered


# =============================================================================
# Запись в Parquet
# =============================================================================

def write_split(df: pl.DataFrame, name: str) -> None:
    """Пишет train/valid/test в data/processed/{name}.parquet со сжатием Snappy."""
    out_dir = CONFIG.paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"

    df.write_parquet(out_path, compression="snappy")
    logger.info("Записано: %s (%.1f MB)", out_path, out_path.stat().st_size / 1024**2)


# =============================================================================
# Точка входа
# =============================================================================

def main() -> None:
    transactions_path = CONFIG.paths.data_processed / "transactions_train.parquet"
    logger.info("Читаю transactions: %s", transactions_path)
    transactions = pl.read_parquet(transactions_path)

    ts = CONFIG.temporal_split
    train, valid, test = temporal_split(
        transactions,
        train_end=ts.train_end,
        valid_end=ts.valid_end,
        test_end=ts.test_end,
    )

    threshold = CONFIG.anomaly_filter.max_transactions
    train_clean = filter_anomalous_customers(train, threshold)

    write_split(train_clean, "train")
    write_split(valid, "valid")
    write_split(test, "test")

    logger.info("Split завершён.")


if __name__ == "__main__":
    main()
