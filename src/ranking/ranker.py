"""
CatBoostRanker (Stage 2: Reranking) для recsys-hm.

Реализует Фазу 8 Master Execution Plan: обучает CatBoostRanker на
признаках, собранных в Фазе 7 (src/features/builder.py), поверх
кандидатов ALS (Фаза 5), оценивает на valid через
src/evaluation/metrics.py.

Опирается на паттерны, уже проверенные smoke_test_ranker.py (тот
прогон нашёл и подтвердил фикс реальной утечки — als_score был null
ровно для label=1 до того, как als.py стал досчитывать честный скор
через score_pairs() для позитивов):

    1. Sort by customer_idx перед Pool — обязательно для YetiRank
       (group_id должен идти монотонно неубывающими блоками).
    2. Categorical -> Utf8 + fill_null("missing") для cat_features.
    3. Label для valid восстанавливается ОТДЕЛЬНО через
       build_ground_truth (metrics.py), не хранится в самих
       признаках — та же таблица используется и для инференса, где
       label заведомо неизвестен.

Ключевое решение этого модуля — как учиться на 87M строк (44 чанка
train_features), не упираясь в память тем же способом, что уже дважды
падал в Фазе 7:

    Обучение CatBoostRanker на всём train_features разом (87M строк в
    памяти как pandas DataFrame для Pool) — не тот же риск, что был у
    Polars-джойнов (там проблема была в повторной материализации
    133M-строчной таблицы на каждой из 44 итераций), но сопоставимого
    порядка по объёму: 87M строк x 16 признаков в pandas/numpy на
    машине, где уже случались два MemoryError на меньших объёмах — риск
    неоправданный, когда есть дешевая альтернатива.

    Вместо потокового/инкрементального обучения (init_model= на каждый
    чанк) — это НЕ эквивалент обучения на полных данных для groupwise
    loss (YetiRank строит деревья на попарных сравнениях внутри группы
    по всему датасету сразу; обучение по кускам меняло бы результат в
    зависимости от порядка чанков, это не содержательный способ
    обучать ranking-модель) — используется сэмплирование ПО ЮЗЕРАМ
    (не по строкам: разбиение по строкам разорвало бы группы для
    YetiRank) до заданной доли train_sample_fraction.

    ПЕРВАЯ попытка (25% юзеров, ~22M строк train + 18.8M строк valid
    одновременно в памяти) упала с CatBoostError: bad allocation —
    внутри самого CatBoost, не в Python/Polars. Причина не в размере
    датасета самом по себе, а в том, что YetiRank строит на старте
    матрицу попарных сравнений ВНУТРИ каждой группы (юзера) — при
    top_k_candidates=100 кандидатов на юзера это до ~100^2/2 пар
    сравнения НА юзера, дополнительно к самим строкам; при 332K юзеров
    это заметно превышает объём, который предполагает голое "строк x
    колонок". Снижено до 5% (TRAIN_SAMPLE_FRACTION) — на этом объёме
    smoke_test_ranker.py уже показал осмысленный feature importance на
    ЕЩЁ меньшей доле (2.3% train), так что 5% — разумный запас, а не
    произвольное число. VALID_SAMPLE_FRACTION дополнительно ограничивает
    eval_set (нужен только для early stopping, не для финальной
    метрики — Recall@K/NDCG@K считается отдельно через evaluate() на
    полном ranked-списке, не через сам Pool).

Запуск:
    python -m src.ranking.ranker
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from src.config import CONFIG
from src.evaluation.metrics import build_ground_truth, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "user_total_purchases",
    "user_unique_articles",
    "user_avg_price",
    "user_days_since_last_purchase",
    "user_favorite_product_group",
    "item_popularity_count",
    "item_avg_price",
    "item_price_volatility",
    "item_days_since_first_sale",
    "item_age_bucket_affinity",
    "als_score",
    "same_product_group_as_history",
    "price_diff_from_user_avg",
    "user_age_diff_from_item_typical_buyer",
]
CAT_FEATURES = ["user_favorite_product_group"]

# is_cold_user / is_cold_item убраны из FEATURE_COLUMNS (не строка удалена
# из данных — фичи остаются в train_features/valid_features на диске,
# просто не передаются в Pool). Причина не эмпирическая, а архитектурная:
# диагностика (src/evaluation/diagnose_ranker_features.py) показала
# "0.0% юзеров имеют ОБА значения среди кандидатов" для обеих фич — то
# есть is_cold_user и is_cold_item КОНСТАНТНЫ внутри каждой группы
# (customer_idx). Это ожидаемо по построению: is_cold_user определён на
# уровне юзера, а группа для YetiRank — это все кандидаты ОДНОГО юзера,
# так что внутри группы юзер не может быть то холодным, то нет.
# YetiRank — groupwise loss, он ищет разбиения, которые переставляют
# кандидатов МЕСТАМИ внутри группы; признак-константа внутри группы даёт
# нулевой прирост для любого такого разбиения вне зависимости от
# гиперпараметров (depth, learning_rate, iterations) — это не то, что
# лечится тюнингом или дополнительными данными, поэтому раньше эти два
# признака стабильно получали importance=0.00 на любом прогоне.
#
# Остальные 13 фич (item_avg_price, price_diff_from_user_avg,
# user_age_diff_from_item_typical_buyer и т.д.) диагностика подтвердила
# как ИМЕЮЩИЕ реальную вариацию внутри групп (0.0% юзеров с нулевой
# вариацией) — их низкая importance на прогоне 12%/iterations=181
# (early stopping) — открытый вопрос, а не архитектурный тупик; отсюда
# LEARNING_RATE_GRID ниже, а не удаление этих фич.

# Первый прогон на 5% (bad_alloc заставил снизить с 25%) обучился успешно
# (early stopping на итерации 209/500, used_ram_limit сработал корректно
# без падения), но дал Recall@10 ХУЖЕ, чем голый ALS, и feature importance
# практически нулевой у 13 из 16 признаков — сильная улика недообучения на
# слишком узкой выборке, а не архитектурная проблема. Поднято до 12% —
# промежуточное значение между заведомо недостаточными 5% и упавшими 25%,
# проверяет гипотезу "мало данных", не рискуя повторить bad_alloc.
#
# Прогон на 12% (early stopping на итерации 181/500, best NDCG=0.0142)
# подтвердил: is_cold_user/is_cold_item архитектурно нулевые (см. выше),
# но у остальных слабых фич вариация внутри групп есть — значит "мало
# данных" не главная причина. Следующая проверяемая гипотеза — early
# stopping (patience=50) останавливает обучение раньше, чем более слабые
# признаки успевают дать вклад, который замаскирован тремя доминирующими
# (item_popularity_count, als_score, user_total_purchases) на первых
# ~180 итерациях. LEARNING_RATE_GRID проверяет это точечно: меньший LR
# делает шаги "осторожнее", давая слабым признакам больше практических
# итераций до срабатывания overfitting detector, вместо немедленного
# перехода к перебору depth (который сам по себе не создаёт сигнал там,
# где его не видно — см. обсуждение перед этим прогоном).
#
# ПОСЛЕ фикса train/valid candidate-pool mismatch в builder.py
# (sample_negatives: max(ratio*n_positives, top_k-n_positives) вместо
# чистого ratio*n_positives) медианный размер train-группы вырос с ~40
# до ~100 кандидатов на юзера (подтверждено логом пересборки: 152M пар
# вместо 87M). На ТОМ ЖЕ 12%, который раньше держался на грани (превышал
# used_ram_limit=6gb, 6.48GiB, но не падал), это привело к повторному
# CatBoostError: bad allocation — причина та же, что и в первом падении
# (25%->5%): YetiRank строит попарные сравнения ВНУТРИ каждой группы, и
# при росте медианной группы в ~2.5x (40->100) память под эти сравнения
# растёт КВАДРАТИЧНО (~(100/40)^2 ≈ 6x на юзера), а не линейно от числа
# строк — снижение TRAIN_SAMPLE_FRACTION пропорционально ЭТОМУ росту
# (0.12 / 2.5 ≈ 0.05), а не наугад до прежнего "безопасного" 5%: тот
# уровень был безопасен для групп размера ~40, а не ~100, так что даже
# формально совпадающее число означает разный физический объём работы.
DEFAULT_TRAIN_SAMPLE_FRACTION = 0.05
DEFAULT_VALID_SAMPLE_FRACTION = 0.10

# Узкий grid по learning_rate — не полный depth x learning_rate (дорого:
# ~55 минут на прогон при 200 итерациях на 12% train), а один параметр за
# раз, тем же принципом, что и tune_als_alpha.py в Фазе 6: сначала LR при
# depth=6 (дефолт из config.yaml), затем — если понадобится — depth при
# лучшем LR. Значения меньше дефолтного (0.05), поскольку гипотеза именно
# в "модель делает слишком большие шаги и не успевает довериться слабым
# признакам" — не берём значения БОЛЬШЕ дефолта, что проверяло бы
# противоположную (менее вероятную по симптомам) гипотезу.
LEARNING_RATE_GRID = [0.03, 0.05, 0.1]


# =============================================================================
# Подготовка данных для CatBoost Pool
# =============================================================================

def prepare_for_pool(df: pl.DataFrame) -> pl.DataFrame:
    """
    Приводит датафрейм к виду, готовому для CatBoost Pool — тот же
    контракт, что в smoke_test_ranker.py (не переизобретается заново,
    чтобы поведение smoke-теста и финального обучения не расходилось):

      - group_id (customer_idx) отсортирован по возрастанию — ОБЯЗАТЕЛЬНО
        для YetiRank, иначе CatBoost либо упадёт, либо тихо пересчитает
        группы неправильно.
      - Categorical -> Utf8: Polars Categorical хранит внутренний числовой
        код + словарь; CatBoost cat_features ожидает питоновские строки.
      - null в cat_features -> строка "missing".
      - Boolean -> Int8 (is_cold_user/is_cold_item) — не все версии
        catboost одинаково стабильно едят чистый Boolean.
    """
    return (
        df.sort("customer_idx")
        .with_columns([
            pl.col("user_favorite_product_group").cast(pl.Utf8).fill_null("missing"),
            pl.col("is_cold_user").cast(pl.Int8),
            pl.col("is_cold_item").cast(pl.Int8),
        ])
    )


def sample_by_user(
    lazy: pl.LazyFrame,
    sample_fraction: float,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Отбирает sample_fraction юзеров ЦЕЛИКОМ (все их строки — позитивы и
    негативы вместе для train, весь список кандидатов для valid), а не
    sample_fraction строк напрямую.

    Разница критична для YetiRank: если сэмплировать по строкам, часть
    юзеров потеряла бы либо все позитивы, либо все негативы (группа
    развалилась бы на бесполезные для groupwise loss обрывки) —
    сэмплирование по customer_idx гарантирует, что каждый отобранный
    юзер приходит в Pool с полной группой.

    Принимает LazyFrame (pl.scan_parquet), а не путь — используется и
    для train_features (44 файла), и для valid_features (1 файл), без
    дублирования логики между ними.

    Отбор через .filter(customer_idx.is_in(...)) на LazyFrame, не через
    .collect() всех строк с последующим фильтром в памяти — план
    выполняется потоково относительно диска.
    """
    all_customer_idxs = (
        lazy.select("customer_idx").unique().collect(engine="streaming")["customer_idx"]
    )
    rng = np.random.default_rng(random_state)
    n_sample = max(1, int(len(all_customer_idxs) * sample_fraction))
    sampled_customer_idxs = rng.choice(all_customer_idxs.to_numpy(), size=n_sample, replace=False)

    sample = (
        lazy.filter(pl.col("customer_idx").is_in(sampled_customer_idxs.tolist()))
        .collect(engine="streaming")
    )

    logger.info(
        "Сэмпл: %d юзеров из %d (%.0f%%), %d строк",
        n_sample, len(all_customer_idxs), 100 * sample_fraction, sample.height,
    )
    return sample


def restore_eval_labels(eval_features: pl.DataFrame, eval_raw_path: Path) -> pl.DataFrame:
    """
    eval_features (valid/test, из build_eval_features в builder.py)
    писались БЕЗ label намеренно — та же таблица признаков используется
    и для оценки, и представляет собой то, что видел бы инференс в
    проде (где label неизвестен по определению). Здесь label
    восстанавливается отдельно через build_ground_truth (metrics.py) —
    тот же контракт "релевантно = купил в оценочном окне", что и
    офлайн-метрики Recall@K/NDCG@K, чтобы определение не разъезжалось
    между обучением ranker'а и его последующей оценкой.
    """
    eval_raw = pl.read_parquet(eval_raw_path)
    ground_truth = build_ground_truth(eval_raw)  # {customer_idx: {article_id, ...}}

    gt_customer_idxs = []
    gt_article_ids = []
    for customer_idx, items in ground_truth.items():
        for article_id in items:
            gt_customer_idxs.append(customer_idx)
            gt_article_ids.append(article_id)

    gt_frame = pl.DataFrame({
        "customer_idx": pl.Series(gt_customer_idxs, dtype=pl.Int64),
        "article_id": pl.Series(gt_article_ids, dtype=pl.Int32),
        "_relevant": True,
    })

    labeled = (
        eval_features
        .join(gt_frame, on=["customer_idx", "article_id"], how="left")
        .with_columns(pl.col("_relevant").fill_null(False).cast(pl.Int8).alias("label"))
        .drop("_relevant")
    )
    return labeled


def build_pool(df: pl.DataFrame) -> Pool:
    """Строит CatBoost Pool из подготовленного (prepare_for_pool) DataFrame."""
    return Pool(
        data=df.select(FEATURE_COLUMNS).to_pandas(),
        label=df["label"].to_list(),
        group_id=df["customer_idx"].to_list(),
        cat_features=CAT_FEATURES,
    )


# =============================================================================
# Обучение
# =============================================================================

def train_ranker(train_pool: Pool, valid_pool: Pool) -> CatBoostRanker:
    """
    Обучает CatBoostRanker с гиперпараметрами из configs/config.yaml
    (ranker.*) — не подобранными под датасет отдельно (см. BRANCH_HANDOFF:
    "Гиперпараметры CatBoostRanker в config.yaml — пока значения по
    умолчанию" — тот открытый вопрос из Фазы 6 всё ещё открыт и здесь
    не закрывается, тюнинг гиперпараметров ranker'а — не предмет этого
    модуля).

    early_stopping_rounds — не было в smoke-тесте (там фиксированные
    50 итераций для быстрой проверки), но на iterations=500 из конфига
    уместно остановиться раньше, если valid-метрика перестала расти,
    вместо того чтобы тратить время на заведомо избыточные деревья.

    used_ram_limit — прямой рычаг CatBoost на случай повторного
    bad_alloc (первая попытка обучения на 25% train упала именно так):
    без этого параметра CatBoost сам решает, сколько памяти выделить под
    внутренние структуры YetiRank (попарные сравнения внутри группы), и
    может попытаться выделить больше, чем реально доступно, вместо
    контролируемой деградации. 6gb — консервативная граница с запасом
    под остальной процесс Python (Polars-датафреймы, pandas-копии для
    Pool), не жёстко привязана к объёму RAM машины — при повторном сбое
    первое, что стоит понизить, это именно это число.
    """
    ranker_config = CONFIG.ranker

    model = CatBoostRanker(
        loss_function=ranker_config.loss_function,
        iterations=ranker_config.iterations,
        learning_rate=ranker_config.learning_rate,
        depth=ranker_config.depth,
        random_state=ranker_config.random_state,
        early_stopping_rounds=50,
        used_ram_limit="6gb",
        verbose=50,
    )
    model.fit(train_pool, eval_set=valid_pool)

    return model


def predictions_from_pool(model: CatBoostRanker, df: pl.DataFrame) -> dict[int, list[int]]:
    """
    Ранжирует df (уже подготовленный prepare_for_pool, отсортированный по
    customer_idx) моделью и собирает predictions в контракте
    {customer_idx: [article_id, ...]} — том же, что ожидает evaluate()
    из src/evaluation/metrics.py (единый контракт для Popularity/ALS/
    CatBoost, см. docstring metrics.py).

    Ранжирование делается внутри каждой группы (customer_idx) отдельно:
    CatBoostRanker.predict() отдаёт скор на строку, не готовый порядок —
    сортировка по (customer_idx, -score) через Polars, без Python-цикла
    по группам на масштабе valid (~19M строк).
    """
    scores = model.predict(df.select(FEATURE_COLUMNS).to_pandas())

    ranked = (
        df.select(["customer_idx", "article_id"])
        .with_columns(pl.Series("_score", scores))
        .sort(["customer_idx", "_score"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_id").alias("_ranked_items"))
    )

    return dict(zip(ranked["customer_idx"].to_list(), ranked["_ranked_items"].to_list()))


def tune_learning_rate(
    train_pool: Pool,
    valid_pool: Pool,
    learning_rates: list[float] = LEARNING_RATE_GRID,
) -> dict[float, dict]:
    """
    Точечный перебор learning_rate на ОДНОМ фиксированном train_pool/
    valid_pool — тот же принцип, что tune_als_alpha.py в Фазе 6: несколько
    значений одного гиперпараметра на одинаковых данных, а не полный grid
    (depth x learning_rate), который на iterations=500/12% train стоил бы
    ~3x дороже по времени без ясной причины пробовать все комбинации сразу.

    depth фиксирован на значении из config.yaml (ranker.depth) — сначала
    проверяется гипотеза "early stopping срабатывает раньше, чем слабые
    признаки успевают дать вклад" (см. комментарий у LEARNING_RATE_GRID),
    depth не варьируется в этом прогоне.

    train_pool/valid_pool передаются готовыми (не пересобираются внутри
    цикла) — тот же train-сэмпл и то же разбиение на eval_set для каждого
    значения learning_rate, иначе сравнение между вариантами было бы
    нечестным (см. Фазу 6: "прогон трёх значений... на одинаковом
    подмножестве тёплых юзеров").

    Возвращает {learning_rate: {"best_iteration": ..., "best_score": ...,
    "feature_importance": {...}}} — не сохраняет модели на диск (это
    диагностический прогон, не финальное обучение) и не считает
    Recall@K/NDCG@K на полном valid для каждого варианта — та часть в разы
    дороже (полный инференс на 18.8M строк на КАЖДОЕ значение grid), а
    для выбора learning_rate достаточно valid-лосса CatBoost и feature
    importance, которые интересуют именно эту диагностику.
    """
    ranker_config = CONFIG.ranker
    results: dict[float, dict] = {}

    for lr in learning_rates:
        logger.info("=== tune_learning_rate: learning_rate=%.3f ===", lr)

        model = CatBoostRanker(
            loss_function=ranker_config.loss_function,
            iterations=ranker_config.iterations,
            learning_rate=lr,
            depth=ranker_config.depth,
            random_state=ranker_config.random_state,
            early_stopping_rounds=50,
            used_ram_limit="6gb",
            verbose=50,
        )
        model.fit(train_pool, eval_set=valid_pool)

        importances = dict(zip(FEATURE_COLUMNS, model.get_feature_importance(train_pool)))
        results[lr] = {
            "best_iteration": model.get_best_iteration(),
            "best_score": model.get_best_score()["validation"][ranker_config.loss_function],
            "feature_importance": importances,
        }

        logger.info(
            "learning_rate=%.3f: best_iteration=%d, best_score=%.4f",
            lr, results[lr]["best_iteration"], results[lr]["best_score"],
        )

    logger.info("=== Итог tune_learning_rate ===")
    for lr, r in sorted(results.items(), key=lambda x: -x[1]["best_score"]):
        top_features = sorted(r["feature_importance"].items(), key=lambda x: -x[1])[:5]
        logger.info(
            "learning_rate=%.3f: best_score=%.4f (iter=%d), топ-5 фич: %s",
            lr, r["best_score"], r["best_iteration"], top_features,
        )

    return results


# =============================================================================
# Точка входа
# =============================================================================

def main() -> None:
    processed_dir = CONFIG.paths.data_processed
    train_features_dir = processed_dir / "train_features"
    valid_features_path = processed_dir / "valid_features" / "part_0001.parquet"
    valid_raw_path = processed_dir / "valid.parquet"

    logger.info("Загружаю train-сэмпл (%.0f%% юзеров) из %s",
                100 * DEFAULT_TRAIN_SAMPLE_FRACTION, train_features_dir)
    train_lazy = pl.scan_parquet(str(train_features_dir / "*.parquet"))
    train_sample = sample_by_user(train_lazy, DEFAULT_TRAIN_SAMPLE_FRACTION)
    train_sample = prepare_for_pool(train_sample)

    logger.info("Читаю valid-признаки из %s", valid_features_path)
    valid_features = pl.read_parquet(valid_features_path)
    valid_labeled_full = restore_eval_labels(valid_features, valid_raw_path)
    valid_labeled_full = prepare_for_pool(valid_labeled_full)

    n_positive = valid_labeled_full["label"].sum()
    logger.info(
        "valid (полный): %d строк, %d положительных (%.3f%%)",
        valid_labeled_full.height, n_positive, 100 * n_positive / valid_labeled_full.height,
    )

    # Для eval_set (early stopping во время обучения) используем подсэмпл —
    # экономит память на этапе, который иначе держит valid_pool живым весь
    # прогон обучения одновременно с train_pool (это и было частью
    # bad_alloc в первой попытке). ФИНАЛЬНАЯ метрика Recall@K/NDCG@K ниже
    # считается на valid_labeled_full (ВСЕ юзеры valid), не на этом
    # подсэмпле — иначе цифра стала бы несравнимой с Popularity (Фаза 4)
    # и ALS (Фаза 6), которые считались на полном оценочном окне.
    valid_lazy_for_eval_set = valid_labeled_full.lazy()
    valid_sample_for_training = sample_by_user(valid_lazy_for_eval_set, DEFAULT_VALID_SAMPLE_FRACTION)
    valid_sample_for_training = prepare_for_pool(valid_sample_for_training)

    train_pool = build_pool(train_sample)
    valid_pool_for_training = build_pool(valid_sample_for_training)

    logger.info("Обучаю CatBoostRanker (iterations=%d)", CONFIG.ranker.iterations)
    model = train_ranker(train_pool, valid_pool_for_training)

    logger.info("Feature importance:")
    importances = model.get_feature_importance(train_pool)
    for name, importance in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        logger.info("  %s: %.2f", name, importance)

    # --- Офлайн-оценка Recall@K/NDCG@K на ПОЛНОМ valid — тот же контракт
    # и та же метрика, что Popularity (Фаза 4) и ALS (Фаза 6), чтобы
    # цифры были сравнимы между всеми тремя моделями напрямую. Не через
    # Pool (не нужна повторная материализация в CatBoost-формате для
    # инференса) — predictions_from_pool сама делает model.predict() на
    # pandas-срезе нужных колонок. ---
    ground_truth = build_ground_truth(pl.read_parquet(valid_raw_path))
    predictions = predictions_from_pool(model, valid_labeled_full)
    results = evaluate(predictions, ground_truth, CONFIG.evaluation.k_values)

    logger.info("=== Итог: CatBoostRanker на valid (полный, все юзеры) ===")
    logger.info("%s", results)

    models_dir = CONFIG.paths.models
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "catboost_ranker.cbm"
    model.save_model(str(model_path))
    logger.info("Модель сохранена: %s", model_path)


if __name__ == "__main__":
    import sys

    # python -m src.ranking.ranker            -> обучение финальной модели (main)
    # python -m src.ranking.ranker --tune-lr  -> тюнинг learning_rate (Фаза 8,
    #   диагностика плоского feature importance — см. LEARNING_RATE_GRID
    #   и tune_learning_rate() выше), тот же принцип отдельного
    #   диагностического режима, что tune_als_alpha.py в Фазе 6.
    if "--tune-lr" in sys.argv:
        processed_dir = CONFIG.paths.data_processed
        train_features_dir = processed_dir / "train_features"
        valid_features_path = processed_dir / "valid_features" / "part_0001.parquet"
        valid_raw_path = processed_dir / "valid.parquet"

        logger.info("tune_learning_rate: загружаю тот же train/valid сэмпл, что и main()")
        train_lazy = pl.scan_parquet(str(train_features_dir / "*.parquet"))
        train_sample = sample_by_user(train_lazy, DEFAULT_TRAIN_SAMPLE_FRACTION)
        train_sample = prepare_for_pool(train_sample)

        valid_features = pl.read_parquet(valid_features_path)
        valid_labeled_full = restore_eval_labels(valid_features, valid_raw_path)
        valid_labeled_full = prepare_for_pool(valid_labeled_full)

        valid_sample_for_training = sample_by_user(
            valid_labeled_full.lazy(), DEFAULT_VALID_SAMPLE_FRACTION
        )
        valid_sample_for_training = prepare_for_pool(valid_sample_for_training)

        train_pool = build_pool(train_sample)
        valid_pool_for_training = build_pool(valid_sample_for_training)

        tune_learning_rate(train_pool, valid_pool_for_training)
    else:
        main()
