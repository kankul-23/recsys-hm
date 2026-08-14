"""
ETL-пайплайн recsys-hm: data/raw/*.csv -> data/processed/*.parquet

Реализует Фазу 1 Master Execution Plan:
    - сканирование CSV через pl.scan_csv (lazy, без загрузки всего файла в RAM)
    - приведение типов согласно зафиксированной схеме
    - customer_id (64-символьный hex) -> компактный Int64 customer_idx
      через отдельную таблицу-маппинг (нужна для сервинга по реальному ID
      и как готовый numeric-индекс для матрицы User x Item в Фазе 5)
    - сохранение в Parquet (Snappy)

Запуск:
    python -m src.data.loader
    (или: python src/data/loader.py, если пути настроены через sys.path)
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Схема типов. Зафиксирована по итогам обсуждения размеров файлов:
#   articles.csv            ~4.3 MB   (~105k строк)
#   customers.csv           ~100 MB   (~1.37M строк)
#   transactions_train.csv  ~598 MB   (~28M+ строк)
# =============================================================================

ARTICLES_DTYPES: dict[str, pl.DataType] = {
    "article_id": pl.Int32,
    "product_code": pl.Int32,
    "prod_name": pl.Categorical,
    "product_type_no": pl.Int32,
    "product_type_name": pl.Categorical,
    "product_group_name": pl.Categorical,
    "graphical_appearance_no": pl.Int32,
    "graphical_appearance_name": pl.Categorical,
    "colour_group_code": pl.Int32,
    "colour_group_name": pl.Categorical,
    "perceived_colour_value_id": pl.Int32,
    "perceived_colour_value_name": pl.Categorical,
    "perceived_colour_master_id": pl.Int32,
    "perceived_colour_master_name": pl.Categorical,
    "department_no": pl.Int32,
    "department_name": pl.Categorical,
    "index_code": pl.Categorical,
    "index_name": pl.Categorical,
    "index_group_no": pl.Int32,
    "index_group_name": pl.Categorical,
    "section_no": pl.Int32,
    "section_name": pl.Categorical,
    "garment_group_no": pl.Int32,
    "garment_group_name": pl.Categorical,
    "detail_desc": pl.Utf8,
}

CUSTOMERS_DTYPES: dict[str, pl.DataType] = {
    "customer_id": pl.Utf8,  # 64-символьный hex, до маппинга в Int64
    "FN": pl.Float32,        # 1.0 / null -> обрабатывается ниже как boolean-флаг
    "Active": pl.Float32,    # 1.0 / null
    "club_member_status": pl.Categorical,
    "fashion_news_frequency": pl.Categorical,
    "age": pl.Float32,       # с пропусками, поэтому не Int8 на чтении
    "postal_code": pl.Utf8,  # хэш-строка, высокая кардинальность -> не Categorical
}

TRANSACTIONS_DTYPES: dict[str, pl.DataType] = {
    "t_dat": pl.Utf8,  # парсится в pl.Date отдельным шагом (строгий формат)
    "customer_id": pl.Utf8,
    "article_id": pl.Int32,
    "price": pl.Float32,
    "sales_channel_id": pl.Int8,
}


# =============================================================================
# customer_id: hex -> compact Int64
# =============================================================================

def build_customer_id_mapping(customers_raw_path) -> pl.DataFrame:
    """
    Строит таблицу-маппинг customer_id (hex, Utf8) -> customer_idx (Int64).

    customer_idx назначается по стабильной сортировке hex-строк, чтобы
    маппинг был детерминирован между перезапусками (не зависит от порядка
    появления строк в CSV).
    """
    logger.info("Строю маппинг customer_id -> customer_idx из %s", customers_raw_path)

    mapping = (
        pl.scan_csv(customers_raw_path, schema_overrides={"customer_id": pl.Utf8})
        .select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index(name="customer_idx")
        .select(["customer_id", pl.col("customer_idx").cast(pl.Int64)])
        .collect()
    )

    logger.info("Маппинг построен: %d уникальных customer_id", mapping.height)
    return mapping


# =============================================================================
# Загрузчики по каждой таблице
# =============================================================================

def load_articles() -> pl.DataFrame:
    """Читает articles.csv, приводит типы, возвращает материализованный DataFrame."""
    path = CONFIG.paths.data_raw / "articles.csv"
    logger.info("Читаю articles: %s", path)

    df = (
        pl.scan_csv(path, schema_overrides=ARTICLES_DTYPES)
        .collect()
    )

    logger.info("articles: %d строк, %d колонок", df.height, df.width)
    return df


def load_customers(id_mapping: pl.DataFrame) -> pl.DataFrame:
    """
    Читает customers.csv, приводит типы, заменяет customer_id (hex) на
    customer_idx (Int64) через джойн с id_mapping.

    FN и Active приводятся к Boolean: 1.0 -> True, null -> False.
    Исходная семантика в датасете H&M — это флаг с пропусками, где пропуск
    фактически означает "не активен / не подписан", поэтому null -> False
    осознанно, а не как потеря информации.
    """
    path = CONFIG.paths.data_raw / "customers.csv"
    logger.info("Читаю customers: %s", path)

    df = (
        pl.scan_csv(path, schema_overrides=CUSTOMERS_DTYPES)
        .join(id_mapping.lazy(), on="customer_id", how="left", coalesce=True)
        .with_columns(
            pl.col("FN").fill_null(0.0).cast(pl.Boolean).alias("FN"),
            pl.col("Active").fill_null(0.0).cast(pl.Boolean).alias("Active"),
            pl.col("age").cast(pl.Int8),
        )
        .drop("customer_id")
        .select(["customer_idx", *[c for c in CUSTOMERS_DTYPES if c != "customer_id"]])
        .collect()
    )

    n_unmapped = df.filter(pl.col("customer_idx").is_null()).height
    if n_unmapped:
        logger.warning("customers: %d строк не нашли пару в id_mapping", n_unmapped)

    logger.info("customers: %d строк, %d колонок", df.height, df.width)
    return df


def load_transactions(id_mapping: pl.DataFrame) -> pl.DataFrame:
    """
    Читает transactions_train.csv, приводит типы, заменяет customer_id (hex)
    на customer_idx (Int64) через джойн с id_mapping, парсит t_dat в pl.Date.

    Это самая большая таблица (~28M+ строк) — вся обработка идёт через
    lazy-план (scan_csv -> join -> with_columns -> collect), materializация
    происходит один раз в конце, а не на каждом промежуточном шаге.
    """
    path = CONFIG.paths.data_raw / "transactions_train.csv"
    logger.info("Читаю transactions_train: %s", path)

    df = (
        pl.scan_csv(path, schema_overrides=TRANSACTIONS_DTYPES)
        .join(id_mapping.lazy(), on="customer_id", how="left", coalesce=True)
        .with_columns(
            pl.col("t_dat").str.to_date(format="%Y-%m-%d").alias("t_dat"),
        )
        .drop("customer_id")
        .select(["t_dat", "customer_idx", "article_id", "price", "sales_channel_id"])
        .collect()
    )

    n_unmapped = df.filter(pl.col("customer_idx").is_null()).height
    if n_unmapped:
        logger.warning("transactions: %d строк не нашли пару в id_mapping", n_unmapped)

    logger.info("transactions_train: %d строк, %d колонок", df.height, df.width)
    return df


# =============================================================================
# Запись в Parquet
# =============================================================================

def write_parquet(df: pl.DataFrame, name: str) -> None:
    """Пишет DataFrame в data/processed/{name}.parquet со сжатием Snappy."""
    out_dir = CONFIG.paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"

    df.write_parquet(out_path, compression="snappy")
    logger.info("Записано: %s (%.1f MB)", out_path, out_path.stat().st_size / 1024**2)


# =============================================================================
# Точка входа
# =============================================================================

def main() -> None:
    customers_raw_path = CONFIG.paths.data_raw / "customers.csv"

    id_mapping = build_customer_id_mapping(customers_raw_path)
    write_parquet(id_mapping, "customer_id_mapping")

    articles = load_articles()
    write_parquet(articles, "articles")

    customers = load_customers(id_mapping)
    write_parquet(customers, "customers")

    transactions = load_transactions(id_mapping)
    write_parquet(transactions, "transactions_train")

    logger.info("ETL завершён.")


if __name__ == "__main__":
    main()
