"""
Feature Engineering для recsys-hm.

Реализует Фазу 7 Master Execution Plan: формирует признаки для пары
(User, Candidate_Item), которые дальше идут в CatBoostRanker (Фаза 8).

Продолжение разведки из notebooks/02_feature_exploration.ipynb — все
архитектурные решения ниже подтверждены цифрами оттуда, а не только
теорией:

    1. NaN у холодных юзеров/товаров — оставляются как есть, НЕ
       заполняются средним/нулём (подмена отсутствия истории выдуманным
       значением была бы тихой деградацией качества). Дополняются явными
       бинарными is_cold_user / is_cold_item, чтобы CatBoost видел режим
       явно, а не восстанавливал его по NaN-паттернам в разных колонках.
       Актуальные цифры cold start на train/valid: 8.3% юзеров, 9.4%
       товаров (см. ноутбук, пункт 6 — расхождение с более ранним EDA
       Фазы 2 объясняется применением filter_anomalous_customers).

    2. als_score — сырой скор ALS для пары (юзер, товар), из
       ALSCandidateGenerator.recommend_with_scores_for_users()
       (src/recommenders/als.py). Разведка показала: разброс скора
       внутри top-100 одного юзера узкий (std~0.06, range~0.30) —
       рабочая гипотеза, что als_score ценен как ОДИН из признаков для
       CatBoost, а не как самостоятельный ранжировщик (см. Фазу 6),
       здесь не переоценивается, а просто передаётся моделью дальше.

    3. Все user_*/item_* агрегаты считаются ОДИН РАЗ по train
       (build_user_features/build_item_features) и передаются как
       параметр в build_features() — не пересчитываются отдельно под
       valid/test. Это защита от temporal leakage: агрегаты видят только
       прошлое относительно train_end, независимо от того, для какого
       сплита строятся кандидаты.

    4. Negative sampling — только для train (обучающий датасет ранкера).
       Негативы берутся из top-100 кандидатов ALS для юзера, ЗА ВЫЧЕТОМ
       его реальных покупок в этом же окне — это "похоже, но не купил",
       содержательный негатив, а не случайный товар из каталога, который
       модель отличила бы тривиально. Соотношение 1:4 (positive:negative)
       берётся из configs/config.yaml (negative_sampling.ratio).
       Для valid/test negative sampling НЕ применяется — там строится
       полный список кандидатов ALS на юзера (Фаза 8 будет ранжировать
       именно его, а не искусственно урезанный набор).

    5. ВСЕ join-подобные операции между ALS-кандидатами и остальными
       таблицами реализованы через Polars (join / anti-join / ранговая
       выборка), а не через Python-циклы или словари на миллионы ключей.
       Первая версия модуля строила Python dict[int, list[tuple]] на
       1.33M юзеров и наполняла список словарей построчно (rows.append)
       для негативов — это привело к MemoryError на полном train
       (~1.33M юзеров x top-100 кандидатов x ratio=4 негативов). Оверхед
       Python-объектов (int/tuple/dict на каждую запись) на такой
       кардинальности на порядок дороже, чем типизированное колоночное
       хранение Polars/Arrow.

Запуск:
    python -m src.features.builder
"""

from __future__ import annotations

import logging

import polars as pl

from src.config import CONFIG
from src.recommenders.als import ALSCandidateGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# User-признаки (считаются один раз по train)
# =============================================================================

def build_user_features(train: pl.DataFrame, articles: pl.DataFrame) -> pl.DataFrame:
    """
    Строит user_total_purchases, user_unique_articles, user_avg_price,
    user_days_since_last_purchase, user_favorite_product_group — строго
    по train (см. пункт 3 модульного докстринга).

    train_end берётся из CONFIG.temporal_split.train_end — та же граница,
    что использует src/data/split.py, чтобы recency-признак был согласован
    с тем, как именно train был отрезан от valid/test.
    """
    train_end_date = pl.Series([CONFIG.temporal_split.train_end]).str.to_date().item()

    base_features = (
        train.group_by("customer_idx")
        .agg([
            pl.len().alias("user_total_purchases"),
            pl.col("article_id").n_unique().alias("user_unique_articles"),
            pl.col("price").mean().alias("user_avg_price"),
            pl.col("t_dat").max().alias("_last_purchase_date"),
        ])
        .with_columns(
            (train_end_date - pl.col("_last_purchase_date")).dt.total_days()
            .alias("user_days_since_last_purchase")
        )
        .drop("_last_purchase_date")
    )

    favorite_group = (
        train.join(articles.select(["article_id", "product_group_name"]), on="article_id", how="left")
        .group_by(["customer_idx", "product_group_name"])
        .agg(pl.len().alias("_n"))
        .sort(["customer_idx", "_n"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .first()
        .select(["customer_idx", pl.col("product_group_name").alias("user_favorite_product_group")])
    )

    user_features = base_features.join(favorite_group, on="customer_idx", how="left")

    logger.info("user_features построены: %d юзеров", user_features.height)
    return user_features


# =============================================================================
# Item-признаки (считаются один раз по train)
# =============================================================================

def build_item_features(train: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:
    """
    Строит item_popularity_count, item_avg_price, item_price_volatility,
    item_days_since_first_sale, item_age_bucket_affinity — строго по train.

    item_popularity_count — count УНИКАЛЬНЫХ покупателей (та же логика,
    что PopularityRecommender.fit(), не count транзакций — см. находку
    Фазы 4 про дубликаты).

    item_price_volatility — std цены; будет null для товаров с
    единственной транзакцией в train (недостаточно данных для дисперсии,
    это честный NaN, не ошибка — подтверждено разведкой: 4280 из 99338
    товаров на прогоне пользователя).
    """
    train_end_date = pl.Series([CONFIG.temporal_split.train_end]).str.to_date().item()

    base_features = (
        train.group_by("article_id")
        .agg([
            pl.col("customer_idx").n_unique().alias("item_popularity_count"),
            pl.col("price").mean().alias("item_avg_price"),
            pl.col("price").std().alias("item_price_volatility"),
            pl.col("t_dat").min().alias("_first_sale_date"),
        ])
        .with_columns(
            (train_end_date - pl.col("_first_sale_date")).dt.total_days()
            .alias("item_days_since_first_sale")
        )
        .drop("_first_sale_date")
    )

    age_affinity = (
        train.join(customers.select(["customer_idx", "age"]), on="customer_idx", how="left")
        .group_by("article_id")
        .agg(pl.col("age").median().alias("item_age_bucket_affinity"))
    )

    item_features = base_features.join(age_affinity, on="article_id", how="left")

    logger.info("item_features построены: %d товаров", item_features.height)
    return item_features


# =============================================================================
# ALS-кандидаты: генерация батчами + разворот в плоскую таблицу
# =============================================================================

def als_candidates_to_frame(
    als_candidates: dict[int, list[tuple[int, float]]]
) -> pl.DataFrame:
    """
    Разворачивает {customer_idx: [(article_id, score), ...]} (то, что
    отдаёт ALSCandidateGenerator.recommend_with_scores_for_users()) в
    плоский Polars DataFrame (customer_idx, article_id, als_score).

    Вызывается на уровне одного батча (см. generate_candidates_to_parquet),
    а не на всех 1.33M юзеров разом — так промежуточный словарь Python
    остаётся ограниченного размера (batch_size юзеров), а не растёт на
    весь train (см. пункт 5 модульного докстринга).
    """
    customer_idxs: list[int] = []
    article_ids: list[int] = []
    scores: list[float] = []
    for customer_idx, pairs in als_candidates.items():
        for article_id, score in pairs:
            customer_idxs.append(customer_idx)
            article_ids.append(article_id)
            scores.append(score)

    frame = pl.DataFrame({
        "customer_idx": pl.Series(customer_idxs, dtype=pl.Int64),
        "article_id": pl.Series(article_ids, dtype=pl.Int32),
        "als_score": pl.Series(scores, dtype=pl.Float32),
    })
    return frame


def generate_candidates_to_parquet(
    als_generator: ALSCandidateGenerator,
    customer_idxs: list[int],
    out_dir,
    batch_size: int = 50_000,
) -> "Path":
    """
    Генерирует ALS-кандидатов для всех customer_idxs батчами и пишет
    КАЖДЫЙ батч сразу в свой parquet-файл внутри out_dir — вместо того,
    чтобы копить все батчи в списке и склеивать их в один pl.DataFrame
    в памяти (прежняя generate_candidates_in_batches).

    Причина замены: на полном train кандидатов получается ~133M строк.
    Даже собранные батчами (не построчным Python dict — та проблема была
    решена раньше), финальный pl.concat() всё равно материализует все
    133M строк как один DataFrame, который потом ЕЩЁ РАЗ участвует целиком
    в каждом из 44 join-ов при сборке признаков (build_features_chunk) —
    итоговая точка отказа: build_features_chunk() падал с MemoryError на
    самом первом чанке, то есть до чанкинга признаков дело даже не
    доходило в стабильном режиме. 133M-строчная таблица кандидатов сама
    по себе плюс параллельный join уже превышали доступную RAM (16 GB на
    машине, где воспроизвели падение).

    Здесь batch_size — тот же принцип, что и раньше (прогресс виден по
    батчам, implicit.recommend() не даёт собственный progress bar), но
    каждый батч сразу пишется на диск и не задерживается в памяти.
    Результат читается обратно лениво через pl.scan_parquet(out_dir /
    "*.parquet") тем, что вызывает эту функцию — см. sample_negatives()
    и build_features_streaming() ниже.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_users = len(customer_idxs)
    n_batches = (n_users + batch_size - 1) // batch_size
    logger.info(
        "Генерирую ALS-кандидатов для %d юзеров батчами по %d (%d батчей), пишу в %s",
        n_users, batch_size, n_batches, out_dir,
    )

    total_rows = 0
    for batch_num, start in enumerate(range(0, n_users, batch_size), start=1):
        batch = customer_idxs[start:start + batch_size]
        batch_candidates = als_generator.recommend_with_scores_for_users(batch)
        batch_frame = als_candidates_to_frame(batch_candidates)

        part_path = out_dir / f"candidates_{batch_num:04d}.parquet"
        batch_frame.write_parquet(part_path, compression="snappy")
        total_rows += batch_frame.height

        done = min(start + batch_size, n_users)
        logger.info(
            "Батч %d/%d готов и записан (%d / %d юзеров, %.1f%%)",
            batch_num, n_batches, done, n_users, 100 * done / n_users,
        )

    logger.info("ALS-кандидаты собраны: %d строк в %d файлах (%s)", total_rows, n_batches, out_dir)
    return out_dir


# =============================================================================
# Negative sampling (только для train)
# =============================================================================

def sample_negatives(
    positive_pairs: pl.DataFrame,
    als_candidates_dir,
    ratio: int,
    top_k: int,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Для каждого юзера сэмплирует негативы из его top-K ALS-кандидатов, за
    вычетом товаров, которые он реально купил (positive_pairs) — см.
    пункт 4 модульного докстринга.

    Размер группы (positive + negative) на юзера = max(ratio * n_positives,
    top_k - n_positives), а не просто ratio * n_positives — см. находку
    Фазы 8 (diagnose_ranking_order.py + diagnose_label_tautology.py):
    train-группа при чистом ratio=4 имела МЕДИАННЫЙ размер 40 (медианный
    юзер с ~8 позитивами), тогда как valid/test-группа — фиксированные
    ~100 кандидатов (полный top-K ALS, без сэмплирования, см.
    build_eval_candidates). CatBoostRanker, обученный ранжировать внутри
    маленьких "плотных" train-групп, давал НА VALID Recall@10 хуже, чем
    голый als_score или голая item_popularity_count по отдельности —
    прямое следствие train/inference distribution skew: модель учится
    паттернам, специфичным для маленьких групп, которые не переносятся
    на большие разреженные группы (0.154% relevant) реального инференса.

    top_k — то же значение, что configs/config.yaml: als.top_k_candidates,
    используется здесь, чтобы train-группа была сопоставима по размеру с
    тем, что видит модель на valid/test (build_eval_candidates), а не
    произвольное число.

    ratio остаётся нижней границей (не убран из сигнатуры и из
    config.yaml) — для юзеров с большим числом позитивов (например,
    100+, у "тяжёлых" покупателей, не отфильтрованных anomaly_filter)
    max() всё равно даёt ratio*n_positives, если это больше top_k -
    n_positives — то есть у активных юзеров группа может честно
    превышать top_k, что нормально: она отражает их реальную историю,
    а не искусственно урезается.

    als_candidates_dir — директория с parquet-файлами от
    generate_candidates_to_parquet(), а НЕ материализованный DataFrame.
    Читается через pl.scan_parquet (lazy): join, filter и ранговая
    выборка строятся как один ленивый план и выполняются потоково при
    .collect() в конце, вместо того чтобы держать все ~133M строк
    кандидатов как единый DataFrame в памяти на всём протяжении функции —
    именно это (не сам чанкинг признаков) было фактической причиной
    MemoryError на первом же чанке build_features_chunk.

    Логика выборки не изменилась: anti-join против позитивов + случайный
    ранг внутри группы customer_idx (Polars не поддерживает sample с
    разным n на группу одним вызовом).
    """
    from pathlib import Path

    als_candidates_dir = Path(als_candidates_dir)
    candidates_lazy = pl.scan_parquet(als_candidates_dir / "*.parquet")

    n_users_total = positive_pairs["customer_idx"].n_unique()
    n_users_with_candidates = (
        candidates_lazy.select("customer_idx").unique().collect().height
    )
    if n_users_with_candidates < n_users_total:
        logger.warning(
            "%d юзеров из train без ALS-кандидатов (из %d) — не получат негативов "
            "(неожиданно для train, стоит проверить build_interaction_matrix)",
            n_users_total - n_users_with_candidates, n_users_total,
        )

    # Пул кандидатов на негативы = ALS top-K минус реально купленное этим юзером.
    negative_pool = candidates_lazy.join(
        positive_pairs.lazy()
        .select(["customer_idx", "article_id"])
        .with_columns(pl.lit(True).alias("_purchased")),
        on=["customer_idx", "article_id"],
        how="left",
    ).filter(pl.col("_purchased").is_null()).drop("_purchased")

    # Сколько негативов нужно каждому юзеру:
    #   max(ratio * n_positives, top_k - n_positives)
    # Левая часть — прежняя логика (баланс классов пропорционально
    # активности юзера). Правая часть — новая: гарантирует, что итоговая
    # группа (n_positives + n_negatives) дотягивает до top_k, где пул
    # кандидатов это позволяет (natural cap — inner join ниже всё равно
    # ограничивает выборку реальным размером пула ALS-кандидатов).
    n_positives_per_user = (
        positive_pairs.lazy()
        .group_by("customer_idx")
        .agg(pl.len().alias("_n_positives"))
        .with_columns(
            pl.max_horizontal(
                pl.col("_n_positives") * ratio,
                (pl.lit(top_k) - pl.col("_n_positives")).clip(lower_bound=0),
            ).alias("_n_needed")
        )
    )

    negative_pool = negative_pool.join(n_positives_per_user, on="customer_idx", how="inner")

    # group_by + sample с разным n на группу: Polars не поддерживает это
    # одним вызовом .sample() (n общий на всю таблицу), поэтому сэмплируем
    # через случайный ранг внутри группы (shuffle + int_range) и берём
    # первые _n_needed по этому рангу — эквивалент sample без замены,
    # полностью на уровне Polars-выражений, без Python-цикла.
    rng_col = pl.int_range(0, pl.len()).shuffle(seed=random_state).over("customer_idx")
    negatives = (
        negative_pool
        .with_columns(rng_col.alias("_rank"))
        .filter(pl.col("_rank") < pl.col("_n_needed"))
        .select(["customer_idx", "article_id"])
        .with_columns(pl.lit(0).alias("label"))
        .collect()
    )

    logger.info("Negative sampling: %d негативных пар (ratio=%d)", negatives.height, ratio)
    return negatives


# =============================================================================
# Сборка признаков для пары (User, Candidate_Item)
# =============================================================================

def build_features_chunk(
    pairs_chunk: pl.DataFrame,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
    articles: pl.DataFrame,
    customers: pl.DataFrame,
    als_candidates_lazy: pl.LazyFrame,
    als_generator: "ALSCandidateGenerator | None" = None,
) -> pl.DataFrame:
    """
    Собирает признаки для ОДНОГО чанка пар (см. build_features_streaming
    ниже — почему это чанками, а не одним DataFrame на весь train).

    als_candidates_lazy — pl.scan_parquet(candidates_dir / "*.parquet"),
    а НЕ материализованный DataFrame. Ранее сюда передавался целиком
    собранный в памяти als_candidates_df (~133M строк на полном train):
    даже при чанкинге pairs_chunk (маленький, 2M строк) правая таблица
    join'а держалась в RAM целиком все 44 итерации — и именно это, а не
    сам чанк, было источником MemoryError на самом первом чанке. Передача
    LazyFrame позволяет Polars выполнить join потоково относительно
    диска, не материализуя все 133M строк кандидатов разом.

    als_generator (опционален): используется ТОЛЬКО для строк, где
    als_score вышел null после джойна с als_candidates_lazy. Это не
    "холодный юзер/товар" в общем случае — als_candidates_lazy содержит
    top-K РЕКОМЕНДАЦИЙ (с filter_already_liked_items=True), а не скор
    для любой пары. Для позитивных пар (реальная покупка) als_score
    ВСЕГДА null после этого джойна: купленный товар в принципе не может
    попасть в top-K рекомендаций того же юзера. Без фикса ranker получал
    идеальный, но бессмысленный сигнал "als_score is null <=> label=1"
    (importance=0.65 на als_score, всё остальное ~0 — обнаружено
    smoke-тестом Фазы 8). als_generator.score_pairs() досчитывает честный
    dot-product для именно этих null-строк — если генератор не передан
    (например, для eval-фич, где такой утечки нет — см. build_eval_features,
    там als_score изначально не null для тёплых юзеров), функция работает
    как раньше, без досчёта.

    is_cold_user / is_cold_item строятся через anti-join с user_features /
    item_features (пара холодная, если её customer_idx / article_id не
    нашёл пару в агрегатах, посчитанных по train), а НЕ через
    `col.is_in(list(train_customers))` — is_in с Python-списком на
    миллион+ элементов пересчитывается для каждой строки чанка и на
    масштабе всего train создавал бы ту же нагрузку, что и предыдущие
    MemoryError-паттерны в этом модуле. join использует ту же
    hash-структуру, что и остальные джойны в функции — единообразно и
    дешевле.
    """
    features = (
        pairs_chunk.lazy()
        .join(user_features.lazy(), on="customer_idx", how="left")
        .join(item_features.lazy(), on="article_id", how="left")
        .join(articles.lazy().select(["article_id", "product_group_name"]), on="article_id", how="left")
        .join(customers.lazy().select(["customer_idx", "age"]), on="customer_idx", how="left")
        .join(als_candidates_lazy, on=["customer_idx", "article_id"], how="left")
    )

    features = features.with_columns([
        (pl.col("product_group_name") == pl.col("user_favorite_product_group"))
            .alias("same_product_group_as_history"),
        (pl.col("item_avg_price") - pl.col("user_avg_price")).alias("price_diff_from_user_avg"),
        (pl.col("age") - pl.col("item_age_bucket_affinity")).alias("user_age_diff_from_item_typical_buyer"),
        # user_total_purchases null <=> customer_idx не нашёлся в user_features
        # <=> customer_idx не встречался в train <=> холодный юзер (тот же
        # принцип для item_popularity_count / is_cold_item).
        pl.col("user_total_purchases").is_null().alias("is_cold_user"),
        pl.col("item_popularity_count").is_null().alias("is_cold_item"),
    ])

    # collect() один раз здесь — весь план (5 join + with_columns) выполняется
    # потоково относительно als_candidates_lazy (читается с диска), а не
    # держит 133M строк кандидатов материализованными в момент построения плана.
    # engine="streaming" — актуальный параметр в Polars >=1.25 (streaming=True
    # был deprecated в пользу этого аргумента).
    result = features.drop(["product_group_name", "age"]).collect(engine="streaming")

    if als_generator is not None:
        null_mask = result["als_score"].is_null()
        n_null = null_mask.sum()
        if n_null > 0:
            null_rows = result.filter(null_mask)
            recomputed = als_generator.score_pairs(
                null_rows["customer_idx"].to_list(),
                null_rows["article_id"].to_list(),
            )
            # Досчитанный скор проставляется обратно только в null-позиции
            # (with_row_index + join по индексу строки — избегаем join по
            # (customer_idx, article_id), который мог бы задвоить строки,
            # если один и тот же article_id встречается у юзера дважды по
            # какой-то причине).
            result = (
                result
                .with_row_index("_row_idx")
                .join(
                    pl.DataFrame({
                        "_row_idx": result.with_row_index("_row_idx").filter(null_mask)["_row_idx"],
                        "_recomputed_score": pl.Series(recomputed, dtype=pl.Float32),
                    }),
                    on="_row_idx",
                    how="left",
                )
                .with_columns(
                    pl.coalesce(["als_score", "_recomputed_score"]).alias("als_score")
                )
                .drop(["_row_idx", "_recomputed_score"])
            )

    return result


def build_features_streaming(
    pairs: pl.DataFrame,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
    articles: pl.DataFrame,
    customers: pl.DataFrame,
    als_candidates_dir,
    out_dir,
    als_generator: "ALSCandidateGenerator | None" = None,
    chunk_size: int = 2_000_000,
) -> None:
    """
    То же самое, что build_features_chunk(), но по чанкам pairs, с записью
    каждого чанка в СВОЙ parquet-файл внутри out_dir — вместо того, чтобы
    держать весь результат (десятки-сотни миллионов строк x ~20 колонок)
    в памяти одновременно.

    Причина: build_features() как единый join на всём train (~91M пар
    positive+negative) упал с MemoryError — Polars материализует
    промежуточный результат каждого из 5 последовательных join сразу для
    всех строк, и на этом объёме суммарная память превысила доступную.
    Чанкинг по pairs (не по als_candidates — та таблица теперь читается
    лениво через pl.scan_parquet и join выполняется потоково относительно
    диска, см. build_features_chunk) держит пиковую память ограниченной
    размером chunk_size и потоковым чтением кандидатов, независимо от
    общего числа пар.

    Один файл на чанк (part_0000.parquet, part_0001.parquet, ...), а не
    ручной ParquetWriter поверх pyarrow: pyarrow не был подтверждён как
    зависимость проекта (Polars умеет писать parquet сам, без него), а
    добавлять непроверенную зависимость ради append-записи — лишний риск.
    Директория читается обратно единым датасетом через
    pl.read_parquet(out_dir / "*.parquet") — Polars поддерживает
    glob-паттерны нативно, склейка происходит лениво при чтении.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    als_candidates_lazy = pl.scan_parquet(Path(als_candidates_dir) / "*.parquet")

    n_pairs = pairs.height
    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    logger.info(
        "Строю признаки для %d пар чанками по %d (%d чанков), пишу в %s",
        n_pairs, chunk_size, n_chunks, out_dir,
    )

    for chunk_num, start in enumerate(range(0, n_pairs, chunk_size), start=1):
        chunk = pairs.slice(start, chunk_size)

        # Проверка на разрыв группы customer_idx между чанками: pairs должен
        # быть заранее отсортирован по customer_idx (см. main() — sort перед
        # вызовом), тогда единственный риск разрыва — юзер оказался ровно на
        # границе chunk_size. Если последний customer_idx этого чанка
        # совпадает с первым customer_idx следующего чанка, часть строк
        # этого юзера ушла в другой чанк — для YetiRank это означает, что
        # ни одна из двух половин не содержит обе метки (0 и 1) для этого
        # юзера. На полном train таких юзеров единицы (граница чанка
        # попадает внутрь ~100-500 строк одного юзера из 2M чанка) — не
        # блокирует прогон, но логируется явно, чтобы не быть неожиданностью
        # при дальнейшем разборе метрик ranker'а.
        if chunk_num < n_chunks:
            next_start = start + chunk_size
            if next_start < n_pairs:
                last_customer_in_chunk = chunk["customer_idx"][-1]
                first_customer_next = pairs.slice(next_start, 1)["customer_idx"][0]
                if last_customer_in_chunk == first_customer_next:
                    logger.warning(
                        "customer_idx=%d разорван между чанком %d и %d (граница chunk_size) — "
                        "часть его пар окажется в другом файле",
                        last_customer_in_chunk, chunk_num, chunk_num + 1,
                    )

        features_chunk = build_features_chunk(
            chunk, user_features, item_features, articles, customers, als_candidates_lazy,
            als_generator=als_generator,
        )

        part_path = out_dir / f"part_{chunk_num:04d}.parquet"
        features_chunk.write_parquet(part_path, compression="snappy")

        done = min(start + chunk_size, n_pairs)
        logger.info(
            "Чанк %d/%d записан в %s (%d / %d пар, %.1f%%)",
            chunk_num, n_chunks, part_path.name, done, n_pairs, 100 * done / n_pairs,
        )

    logger.info("Готово: признаки записаны в %s (%d файлов)", out_dir, n_chunks)


# =============================================================================
# Сборка признаков для valid/test (без negative sampling, с Popularity
# fallback для холодных юзеров) — используется для оценки CatBoostRanker
# в Фазе 8/9, переиспользует user_features/item_features, посчитанные
# по train (см. пункт 3 модульного докстринга — защита от leakage).
# =============================================================================

def build_eval_candidates(
    als_generator: ALSCandidateGenerator,
    popularity_recommender,
    eval_customer_idxs: list[int],
    batch_size: int = 50_000,
) -> pl.DataFrame:
    """
    Строит полный список кандидатов (article_id, als_score) для оценочного
    сплита (valid/test) — БЕЗ negative sampling: в отличие от train, здесь
    CatBoost будет ранжировать именно этот список целиком (см. пункт 4
    модульного докстринга), не искусственно урезанный набор.

    Два источника кандидатов:
      - тёплые юзеры (были в train) — через als_generator, тот же top_k
        и то же исключение уже купленного (filter_already_liked_items),
        что и при построении train-кандидатов;
      - холодные юзеры (в eval_customer_idxs, но не в train) — фиксированный
        Popularity top-N список (popularity_recommender), с als_score=null,
        поскольку ALS для них вектора не строил. Это тот же fallback,
        что уже был обоснован в Фазе 6 (run_als_eval.py) для честного
        сравнения ALS с Popularity — здесь применяется тем же способом,
        чтобы у ranker'а было ЧТО ранжировать для каждого юзера из
        оценочного окна, а не только для тёплого подмножества.

    is_cold_user для этих строк потом естественно проставится в
    build_features_chunk через null в user_features (холодный юзер там
    и не появлялся) — отдельно здесь не размечается.
    """
    warm_candidates = als_generator.recommend_with_scores_for_users(eval_customer_idxs)
    warm_customer_idxs = set(warm_candidates.keys())

    cold_customer_idxs = [c for c in eval_customer_idxs if c not in warm_customer_idxs]
    logger.info(
        "Eval-кандидаты: %d тёплых юзеров (ALS), %d холодных юзеров (Popularity fallback)",
        len(warm_customer_idxs), len(cold_customer_idxs),
    )

    warm_frame = als_candidates_to_frame(warm_candidates)

    # Popularity fallback: один и тот же список article_id для всех холодных
    # юзеров этого сплита, als_score=null (ALS не считал скор для них).
    popular_items = popularity_recommender.recommend()
    cold_frame = pl.DataFrame({
        "customer_idx": pl.Series(cold_customer_idxs, dtype=pl.Int64),
    }).join(
        pl.DataFrame({"article_id": pl.Series(popular_items, dtype=pl.Int32)}),
        how="cross",
    ).with_columns(pl.lit(None, dtype=pl.Float32).alias("als_score"))

    candidates = pl.concat([warm_frame, cold_frame], how="vertical")
    logger.info("Eval-кандидаты собраны: %d строк (%d юзеров)", candidates.height, len(eval_customer_idxs))
    return candidates


def build_eval_features(
    split_name: str,
    user_features: pl.DataFrame,
    item_features: pl.DataFrame,
    articles: pl.DataFrame,
    customers: pl.DataFrame,
    als_generator: ALSCandidateGenerator,
    popularity_recommender,
) -> None:
    """
    Собирает признаки для valid или test (split_name — "valid"/"test") и
    пишет в data/processed/{split_name}_features/.

    В отличие от build_features_streaming (train): здесь НЕТ negative
    sampling — pairs = полный список кандидатов на юзера (build_eval_candidates),
    ranker (Фаза 8) ранжирует именно его целиком. label не проставляется
    здесь — соответствие купил/не купил для оценки берётся отдельно из
    build_ground_truth (src/evaluation/metrics.py) на данных самого
    valid/test, а не встраивается в фичи (иначе это была бы утечка метки
    в признаковую таблицу, которую использует и инференс, и обучение).

    user_features/item_features передаются параметром, а не пересчитываются —
    те же агрегаты по train, что и для build_features_streaming (см. пункт
    3 модульного докстринга: leakage-защита не зависит от того, для какого
    сплита строятся кандидаты).
    """
    processed_dir = CONFIG.paths.data_processed
    eval_df = pl.read_parquet(processed_dir / f"{split_name}.parquet")
    eval_customer_idxs = eval_df["customer_idx"].unique().to_list()

    logger.info(
        "Строю eval-признаки для '%s': %d уникальных юзеров в сплите",
        split_name, len(eval_customer_idxs),
    )

    candidates = build_eval_candidates(
        als_generator, popularity_recommender, eval_customer_idxs
    )

    # Тот же путь сборки признаков, что и для train (build_features_chunk),
    # но als_candidates передаются напрямую как LazyFrame из уже готового
    # candidates DataFrame — отдельного parquet на диске здесь не нужно,
    # объём на порядок меньше train (юзеры eval-окна x top_k, а не все
    # 1.33M юзеров train x top_k).
    features = build_features_chunk(
        candidates, user_features, item_features, articles, customers, candidates.lazy()
    )

    out_dir = processed_dir / f"{split_name}_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part_0001.parquet"
    features.write_parquet(out_path, compression="snappy")

    logger.info("Готово: eval-признаки для '%s' записаны в %s", split_name, out_path)


# =============================================================================
# Точка входа — пример полного прогона для train
# =============================================================================

def main() -> None:
    from src.recommenders.als import (
        build_article_id_mapping,
        build_customer_matrix_mapping,
        build_interaction_matrix,
        fit_als_model,
    )

    processed_dir = CONFIG.paths.data_processed
    logger.info("Читаю train/articles/customers из %s", processed_dir)

    train = pl.read_parquet(processed_dir / "train.parquet")
    articles = pl.read_parquet(processed_dir / "articles.parquet")
    customers = pl.read_parquet(processed_dir / "customers.parquet")

    user_features = build_user_features(train, articles)
    item_features = build_item_features(train, customers)

    # --- ALS: нужен для als_score и для пула негативов ---
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

    all_train_users = user_mapping["customer_idx"].to_list()
    als_candidates_dir = CONFIG.paths.data_processed / "als_candidates"
    generate_candidates_to_parquet(als_generator, all_train_users, als_candidates_dir)

    # --- Positive pairs (реальные покупки в train) + negatives ---
    positive_pairs = (
        train.select(["customer_idx", "article_id"])
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )
    negative_pairs = sample_negatives(
        positive_pairs, als_candidates_dir,
        ratio=CONFIG.negative_sampling.ratio,
        top_k=als_config.top_k_candidates,
    )

    # ВАЖНО: сортировка по customer_idx перед конкатенацией+нарезкой на чанки.
    # positive_pairs (25.7M) и negative_pairs (61.3M) идут раздельными блоками
    # в pl.concat — без сортировки первые ~13 чанков build_features_streaming
    # состояли бы ИСКЛЮЧИТЕЛЬНО из позитивов (все label=1), а негативы того
    # же юзера лежали бы в чанке 20+. Обнаружено smoke-тестом ranker'а:
    # CatBoost упал с "All train targets are equal" на part_0001.parquet,
    # где действительно было 2M строк подряд с label=1.
    #
    # Для YetiRank это не просто неудобство, а фатальная ошибка: группа
    # (customer_idx) должна целиком помещаться в один чанк/Pool, чтобы
    # ranker вообще мог сравнивать позитив с негативом внутри юзера — если
    # юзер разорван между чанками, часть его группы всегда состоит из
    # одной метки, а значит бесполезна для обучения ранжированию.
    #
    # Сортировка (не shuffle) гарантирует смежность строк одного юзера
    # физически на диске — при разбиении на чанки по 2M строк подряд юзер
    # целиком попадёт в один чанк, если только не окажется ровно на
    # границе чанка (см. проверку в build_features_streaming).
    all_pairs = pl.concat([positive_pairs, negative_pairs], how="vertical").sort("customer_idx")

    # --- Финальная сборка признаков — потоково, чанками, сразу на диск ---
    # als_generator передаётся, чтобы build_features_chunk мог досчитать
    # честный als_score для позитивных пар (см. докстринг build_features_chunk —
    # без этого все позитивы получали als_score=null, что ranker
    # использовал как утечку метки вместо реального сигнала).
    out_dir = CONFIG.paths.data_processed / "train_features"
    build_features_streaming(
        all_pairs, user_features, item_features, articles, customers,
        als_candidates_dir, out_dir, als_generator=als_generator,
    )
    logger.info(
        "Для чтения результата целиком: pl.read_parquet('%s/*.parquet')", out_dir
    )

    # --- Признаки для valid/test — без negative sampling, с Popularity
    # fallback для холодных юзеров (см. build_eval_features) ---
    from src.recommenders.popularity import PopularityRecommender

    popularity_recommender = PopularityRecommender().fit(train)

    for split_name in ("valid", "test"):
        build_eval_features(
            split_name, user_features, item_features, articles, customers,
            als_generator, popularity_recommender,
        )


if __name__ == "__main__":
    main()
