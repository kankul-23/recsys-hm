"""
Candidate Generation — Implicit ALS для recsys-hm.

Реализует Фазу 5 Master Execution Plan:
    - построение разреженной матрицы взаимодействий User x Item на train
    - обучение Implicit ALS (F=64, Iter=15, Reg=0.05 — configs/config.yaml)
    - генерация Top-100 персональных кандидатов на юзера с гарантированным
      исключением ранее купленных товаров

Это Retrieval-этап двухуровневой архитектуры: ALS быстро отбирает широкий
пул из 100 потенциально релевантных товаров на юзера из полного каталога
(~99k article_id), а точное ранжирование этого пула — задача CatBoost
в Фазе 7. ALS должен быть быстрым и достаточно полным (высокий Recall@100),
а не точным на топ-10 — это ответственность следующего этапа.

Ключевые архитектурные решения:

    1. article_idx — отдельный компактный Int64-индекс (0..N-1) для товаров,
       построенный тем же принципом, что и customer_idx в Фазе 1
       (src/data/loader.py): детерминированная сортировка по article_id,
       чтобы индекс не зависел от порядка появления строк в train.
       customer_idx уже существует и переиспользуется как есть — но он
       был построен по ПОЛНОМУ customers.csv (1 371 980 строк), тогда как
       для ALS нужна матрица только по юзерам, встретившимся в train.
       Поэтому для юзеров используется НЕ customer_idx напрямую, а
       собственный плотный индекс матрицы (matrix_user_idx), построенный
       по уникальным customer_idx из train, — иначе матрица User x Item
       содержала бы 1.37M строк вместо реального числа активных юзеров
       в train, большинство из которых — только нули (лишний расход
       памяти без пользы, поскольку ALS не даёт значимых векторов для
       юзеров без единого взаимодействия).

    2. Confidence-вес взаимодействия — 1 + confidence_alpha * count, где
       count — число покупок пары (юзер, товар) на train (Hu, Koren,
       Volinsky, 2008). confidence_alpha=40 — стандартное значение по
       умолчанию, не подобрано под датасет отдельно (configs/config.yaml).
       Это единственное место в проекте, где повторные покупки одного
       товара несут дополнительный сигнал — в отличие от Popularity
       (Фаза 4), где сознательно использовался только факт покупки
       (count уникальных покупателей), здесь частота повторных покупок
       поднимает уверенность модели в паре (юзер, товар), а не саму
       предпочтительность (она в implicit ALS всегда бинарна: 0 или 1).

    3. Исключение уже купленного — реализовано ВРУЧНУЮ после получения
       сырых рекомендаций от библиотеки, а не через встроенный параметр
       filter_already_liked_items у AlternatingLeastSquares.recommend().
       Причина: встроенный фильтр библиотеки работает по УЖЕ переданной
       юзер-айтемной матрице взаимодействий (тем же train, что и при
       обучении) и не позволяет так же прозрачно логировать, сколько
       кандидатов было отфильтровано — здесь это делается явно и
       проверяемо, что важно для Фазы 10 (автотест "нет купленных товаров
       в кандидатах").

    4. GPU опционален. Библиотека implicit поддерживает CUDA-ускорение
       (значимо на матрицах такого размера), но обучение не должно падать
       на машине без GPU или без нужной сборки CUDA — поэтому используется
       try/except с осознанным откатом на CPU и явным логом о том, что
       используется.

Запуск:
    python -m src.recommenders.als
"""

from __future__ import annotations

import logging

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Индексация товаров (article_idx) — тем же принципом, что customer_id_mapping
# в src/data/loader.py, но по уникальным article_id именно из train.
# =============================================================================

def build_article_id_mapping(train: pl.DataFrame) -> pl.DataFrame:
    """
    Строит таблицу-маппинг article_id -> article_idx (Int64) по уникальным
    товарам, встретившимся в train. Детерминированная сортировка по
    article_id — маппинг не зависит от порядка появления строк.
    """
    mapping = (
        train.select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index(name="article_idx")
        .select(["article_id", pl.col("article_idx").cast(pl.Int64)])
    )

    logger.info("article_id_mapping построен: %d уникальных товаров в train", mapping.height)
    return mapping


def build_customer_matrix_mapping(train: pl.DataFrame) -> pl.DataFrame:
    """
    Строит плотный matrix_user_idx (0..N-1) по уникальным customer_idx,
    встретившимся в train — НЕ то же самое, что сам customer_idx (который
    построен по полному customers.csv и может содержать "дыры" относительно
    train). См. модульный docstring, пункт 1.
    """
    mapping = (
        train.select("customer_idx")
        .unique()
        .sort("customer_idx")
        .with_row_index(name="matrix_user_idx")
        .select(["customer_idx", pl.col("matrix_user_idx").cast(pl.Int64)])
    )

    logger.info("matrix_user_idx построен: %d уникальных юзеров в train", mapping.height)
    return mapping


# =============================================================================
# Построение разреженной матрицы User x Item
# =============================================================================

def build_interaction_matrix(
    train: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
    confidence_alpha: int,
) -> sp.csr_matrix:
    """
    Строит разреженную CSR-матрицу (matrix_user_idx x article_idx) со
    значениями confidence = 1 + confidence_alpha * count, где count —
    число покупок пары (юзер, товар) на train (см. пункт 2 докстринга
    модуля).

    CSR (Compressed Sparse Row) выбран, а не COO/CSC, потому что
    implicit.als.AlternatingLeastSquares ожидает на вход именно CSR —
    формат оптимален для построчного доступа, который использует ALS
    при пересчёте юзерских векторов на каждой итерации.
    """
    counts = (
        train.group_by(["customer_idx", "article_id"])
        .agg(pl.len().alias("count"))
        .join(user_mapping, on="customer_idx", how="inner")
        .join(item_mapping, on="article_id", how="inner")
        .with_columns(
            (1 + confidence_alpha * pl.col("count")).alias("confidence")
        )
    )

    n_users = user_mapping.height
    n_items = item_mapping.height

    matrix = sp.csr_matrix(
        (
            counts["confidence"].to_numpy().astype(np.float32),
            (
                counts["matrix_user_idx"].to_numpy(),
                counts["article_idx"].to_numpy(),
            ),
        ),
        shape=(n_users, n_items),
    )

    logger.info(
        "Матрица User x Item построена: %d x %d, %d ненулевых элементов (плотность %.4f%%)",
        n_users, n_items, matrix.nnz, 100 * matrix.nnz / (n_users * n_items),
    )

    return matrix


# =============================================================================
# Обучение ALS
# =============================================================================

def fit_als_model(
    interaction_matrix: sp.csr_matrix,
    factors: int,
    iterations: int,
    regularization: float,
    random_state: int,
) -> AlternatingLeastSquares:
    """
    Обучает AlternatingLeastSquares на разреженной матрице взаимодействий.

    Пытается использовать GPU (CUDA) — значимое ускорение на матрицах
    такого размера. При недоступности GPU (нет CUDA-сборки библиотеки
    или нет видеокарты) откатывается на CPU без падения (см. пункт 4
    докстринга модуля).
    """
    try:
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
            use_gpu=True,
        )
        model.fit(interaction_matrix)
        logger.info("ALS обучен на GPU (factors=%d, iterations=%d)", factors, iterations)
    except Exception as exc:
        logger.warning("GPU недоступен (%s) — обучаю на CPU", exc)
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
            use_gpu=False,
        )
        model.fit(interaction_matrix)
        logger.info("ALS обучен на CPU (factors=%d, iterations=%d)", factors, iterations)

    return model


# =============================================================================
# Генератор кандидатов
# =============================================================================

class ALSCandidateGenerator:
    """
    Обёртка над обученной ALS-моделью: переводит рекомендации из
    внутренних индексов матрицы (matrix_user_idx, article_idx) обратно
    в исходные ID (customer_idx, article_id) и гарантирует исключение
    уже купленных в train товаров (см. пункт 3 докстринга модуля).
    """

    def __init__(
        self,
        model: AlternatingLeastSquares,
        interaction_matrix: sp.csr_matrix,
        user_mapping: pl.DataFrame,
        item_mapping: pl.DataFrame,
        top_k: int,
    ) -> None:
        self.model = model
        self.interaction_matrix = interaction_matrix
        self.top_k = top_k

        # Быстрые словари для перевода ID <-> внутренний индекс матрицы.
        self._customer_idx_to_matrix_idx = dict(
            zip(user_mapping["customer_idx"].to_list(), user_mapping["matrix_user_idx"].to_list())
        )
        self._article_idx_to_article_id = dict(
            zip(item_mapping["article_idx"].to_list(), item_mapping["article_id"].to_list())
        )
        # Обратный словарь article_id -> article_idx — нужен score_pairs()
        # для перевода произвольных пар (customer_idx, article_id) во
        # внутренние индексы матрицы, в дополнение к уже существующему
        # article_idx -> article_id, который используют recommend_*.
        self._article_id_to_article_idx = dict(
            zip(item_mapping["article_id"].to_list(), item_mapping["article_idx"].to_list())
        )

    def recommend_for_users(self, customer_idxs: list[int]) -> dict[int, list[int]]:
        """
        Возвращает top_k персональных article_id-кандидатов для каждого
        customer_idx из списка.

        Юзеры, которых не было в train (нет строки в customer_idx_to_matrix_idx),
        пропускаются — для них ALS не строил вектор и не может дать
        персональную рекомендацию; такие холодные юзеры обрабатываются
        отдельно, через fallback на Popularity (см. Фазу 9 плана).
        """
        known_customer_idxs = [c for c in customer_idxs if c in self._customer_idx_to_matrix_idx]
        skipped = len(customer_idxs) - len(known_customer_idxs)
        if skipped:
            logger.info(
                "%d юзеров из %d не встречались в train — пропущены (холодный старт, fallback на Popularity)",
                skipped, len(customer_idxs),
            )

        matrix_user_idxs = np.array(
            [self._customer_idx_to_matrix_idx[c] for c in known_customer_idxs]
        )

        # implicit.recommend с filter_already_liked_items=True исключает
        # товары, которые есть в interaction_matrix для этого юзера
        # (т.е. уже купленные в train) — это и есть гарантированное
        # исключение из требования Фазы 5.
        item_idxs, scores = self.model.recommend(
            matrix_user_idxs,
            self.interaction_matrix[matrix_user_idxs],
            N=self.top_k,
            filter_already_liked_items=True,
        )

        recommendations: dict[int, list[int]] = {}
        for customer_idx, row_item_idxs in zip(known_customer_idxs, item_idxs):
            article_ids = [
                self._article_idx_to_article_id[idx]
                for idx in row_item_idxs
                if idx in self._article_idx_to_article_id  # implicit паддинг: -1 при нехватке кандидатов
            ]
            recommendations[customer_idx] = article_ids

        return recommendations

    def recommend_with_scores_for_users(
        self, customer_idxs: list[int]
    ) -> dict[int, list[tuple[int, float]]]:
        """
        То же самое, что recommend_for_users(), но дополнительно возвращает
        сырой als_score для каждого кандидата — нужно для Фазы 7 (Feature
        Engineering), где als_score используется как один из признаков пары
        (User, Candidate_Item) для CatBoost.

        Не заменяет recommend_for_users(): тот метод используется в оценке
        (run_als_eval.py, tune_als_alpha.py), где скор не нужен — там
        предсказания сравниваются с ground truth только по факту попадания
        article_id в top-K. Разделение методов сохраняет старый контракт
        нетронутым и не расширяет его опциональным флагом, который усложнил
        бы сигнатуру ради одного нового потребителя (builder.py).

        Логика идентична recommend_for_users() (тот же фильтр холодных
        юзеров, тот же filter_already_liked_items=True) — только на выходе
        список пар (article_id, als_score) вместо голых article_id.
        """
        known_customer_idxs = [c for c in customer_idxs if c in self._customer_idx_to_matrix_idx]
        skipped = len(customer_idxs) - len(known_customer_idxs)
        if skipped:
            logger.info(
                "%d юзеров из %d не встречались в train — пропущены (холодный старт, fallback на Popularity)",
                skipped, len(customer_idxs),
            )

        matrix_user_idxs = np.array(
            [self._customer_idx_to_matrix_idx[c] for c in known_customer_idxs]
        )

        item_idxs, scores = self.model.recommend(
            matrix_user_idxs,
            self.interaction_matrix[matrix_user_idxs],
            N=self.top_k,
            filter_already_liked_items=True,
        )

        recommendations: dict[int, list[tuple[int, float]]] = {}
        for customer_idx, row_item_idxs, row_scores in zip(known_customer_idxs, item_idxs, scores):
            article_scores = [
                (self._article_idx_to_article_id[idx], float(score))
                for idx, score in zip(row_item_idxs, row_scores)
                if idx in self._article_idx_to_article_id  # implicit паддинг: -1 при нехватке кандидатов
            ]
            recommendations[customer_idx] = article_scores

        return recommendations

    def score_pairs(self, customer_idxs: list[int], article_ids: list[int]) -> list[float | None]:
        """
        Считает "сырой" ALS-скор (dot-product user_factors x item_factors)
        для ПРОИЗВОЛЬНОГО списка пар (customer_idx, article_id) — включая
        пары, где article_id уже куплен этим юзером в train.

        Отличие от recommend_with_scores_for_users(): тот метод — top-K
        РЕКОМЕНДАЦИЙ модели с обязательным filter_already_liked_items=True,
        то есть купленные товары туда принципиально попасть не могут.
        Этот метод — скор для КОНКРЕТНОЙ заданной пары, без какой-либо
        фильтрации, тем же способом, каким сам ALS ранжирует кандидатов
        внутри model.recommend() (dot-product факторов).

        Нужен для Фазы 7 (builder.py): позитивные пары (реальные покупки)
        обязаны получить содержательный als_score, а не null. Если считать
        als_score только через recommend_with_scores_for_users (top-K с
        фильтром), для ВСЕХ позитивов als_score будет null по построению —
        товар, который юзер купил, туда не попадёт никогда, поскольку
        именно он и исключается фильтром. Модель тогда обучается отличать
        null/not-null als_score вместо реального сигнала — обнаружено
        smoke-тестом ranker'а (Фаза 8): als_score получал importance=0.65
        при том, что null в нём идеально совпадал с label=1.

        Векторизовано через поэлементное произведение + сумму по оси
        факторов (эквивалент батча dot-product), а не Python-цикл с
        np.dot на пару — на позитивах train (~25.7M строк) построчный
        Python-цикл был бы на порядки медленнее и стал бы новым узким
        местом сборки признаков.

        Возвращает None для пар, где customer_idx или article_id не
        встречались в train (холодный юзер/товар — ALS не строил для них
        вектор, скор физически не из чего посчитать).
        """
        item_idxs = np.array(
            [self._article_id_to_article_idx.get(a, -1) for a in article_ids]
        )
        matrix_idxs = np.array(
            [self._customer_idx_to_matrix_idx.get(c, -1) for c in customer_idxs]
        )

        valid_mask = (item_idxs >= 0) & (matrix_idxs >= 0)

        # -1 как временная заглушка для невалидных индексов, чтобы не
        # выйти за границы factors при батчевой индексации; реальные
        # значения для этих позиций всё равно отбрасываются valid_mask
        # ниже и заменяются на None.
        safe_matrix_idxs = np.where(valid_mask, matrix_idxs, 0)
        safe_item_idxs = np.where(valid_mask, item_idxs, 0)

        user_vecs = self.model.user_factors[safe_matrix_idxs]
        item_vecs = self.model.item_factors[safe_item_idxs]
        raw_scores = np.sum(user_vecs * item_vecs, axis=1)

        return [
            float(score) if is_valid else None
            for score, is_valid in zip(raw_scores, valid_mask)
        ]


# =============================================================================
# Точка входа
# =============================================================================

def main() -> None:
    train_path = CONFIG.paths.data_processed / "train.parquet"
    logger.info("Читаю train: %s", train_path)
    train = pl.read_parquet(train_path)

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

    generator = ALSCandidateGenerator(
        model, interaction_matrix, user_mapping, item_mapping, top_k=als_config.top_k_candidates
    )

    # Быстрая проверка на первых 5 юзерах train.
    sample_users = user_mapping["customer_idx"].to_list()[:5]
    sample_recs = generator.recommend_for_users(sample_users)
    for customer_idx, article_ids in sample_recs.items():
        logger.info("customer_idx=%d: топ-5 кандидатов %s", customer_idx, article_ids[:5])


if __name__ == "__main__":
    main()
