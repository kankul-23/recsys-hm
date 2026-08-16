"""
Offline Evaluation Metrics для recsys-hm.

Реализует метрический слой Фазы 4 (первая точка использования — Popularity
baseline) и Фазы 8 (Offline Evaluation) Master Execution Plan.

Спроектирован как переиспользуемый модуль, не привязанный к конкретному
рекомендеру: принимает predictions + ground truth в едином формате
{user_id: [article_id, ...]} и не знает, откуда взялись предсказания —
Popularity, ALS или CatBoost. Это тот же контракт, что отдаёт
PopularityRecommender.recommend_for_users() (src/recommenders/popularity.py).

Ключевые решения (зафиксированы по итогам обсуждения):

    1. Ground truth — множество УНИКАЛЬНЫХ article_id, купленных юзером в
       оценочном окне (valid/test), бинарная релевантность (купил/не
       купил), БЕЗ градации по числу покупок. Стандартная формулировка
       Recall@K/NDCG@K для implicit-feedback рекомендательных систем.
       Введение graded relevance (вес по количеству покупок) добавило бы
       сложность без обоснования на этом этапе — тем более что EDA уже
       показал: повторная покупка одного товара — это "сила предпочтения
       одного человека", а не сигнал о том, что товар нужнее для ранжирования.

    2. Агрегация метрик — среднее по ВСЕМ юзерам из оценочного окна,
       ВКЛЮЧАЯ холодных (тех, кого не было в train). Это принципиально:
       по EDA Фазы 2, холодные юзеры — 10.4% от финального split, и это
       не шум для отбрасывания, а часть реального распределения, на
       котором система будет работать в проде (см. Фазу 9 — fallback на
       Popularity именно для этого сценария). Если считать метрики только
       по юзерам из train, результат будет искусственно завышен и
       скроет ровно ту проблему, под которую спроектирован fallback.
       Юзеры без покупок в оценочном окне вообще (у которых ground truth
       пуст) исключаются из усреднения — для них Recall/NDCG математически
       не определены, а не равны нулю.

    3. Cold vs Warm разбивка (Active Users vs Cold/Low-Activity Users)
       предусмотрена как отдельная функция evaluate_by_segment(), но не
       единственный способ смотреть на метрики — общее среднее по всем
       юзерам остаётся основной цифрой для сравнения Popularity/ALS/CatBoost
       между собой. Это соответствует Фазе 8 плана (разбор ошибок отдельно
       по сегментам, но после сквозного замера).

Запуск (пример использования, не самостоятельный скрипт):
    from src.evaluation.metrics import build_ground_truth, evaluate
"""

from __future__ import annotations

import logging
import math

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Ground truth
# =============================================================================

def build_ground_truth(eval_df: pl.DataFrame) -> dict[int, set[int]]:
    """
    Строит ground truth из valid/test: для каждого customer_idx — множество
    уникальных article_id, купленных им в этом окне.

    Бинарная релевантность (см. модульный docstring, пункт 1) — количество
    покупок одного товара не учитывается, только сам факт покупки.
    """
    grouped = (
        eval_df.group_by("customer_idx")
        .agg(pl.col("article_id").unique().alias("items"))
    )

    ground_truth = {
        row["customer_idx"]: set(row["items"])
        for row in grouped.iter_rows(named=True)
    }

    logger.info("Ground truth построен: %d юзеров с покупками в оценочном окне", len(ground_truth))
    return ground_truth


# =============================================================================
# Метрики на уровне одного юзера
# =============================================================================

def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Recall@K для одного юзера: доля релевантных товаров, попавших в топ-K
    рекомендаций, от общего числа релевантных товаров у юзера.

    relevant не должен быть пустым — вызывающий код (evaluate()) отвечает
    за исключение юзеров без ground truth до вызова этой функции.
    """
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    NDCG@K для одного юзера. При бинарной релевантности (см. пункт 1)
    DCG сводится к сумме 1/log2(position+1) по позициям попаданий, IDCG —
    та же формула для "идеального" ранжирования, где все релевантные
    товары стоят на первых местах (в пределах k и len(relevant)).
    """
    top_k = recommended[:k]

    dcg = sum(
        1.0 / math.log2(position + 2)  # +2: position с 0, log2(1)=0 избегаем
        for position, item in enumerate(top_k)
        if item in relevant
    )

    n_ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(n_ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Оценка на всём eval-множестве
# =============================================================================

def evaluate(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k_values: list[int],
) -> dict[str, float]:
    """
    Считает Recall@K и NDCG@K для каждого K из k_values, усредняя по всем
    юзерам из ground_truth (см. пункт 2 — включая холодных, если для них
    в predictions передан осмысленный список, например Popularity fallback).

    Юзеры из ground_truth, для которых нет предсказаний в predictions,
    пропускаются с предупреждением в лог — это сигнал о проблеме в
    связывающем коде (predictions должны покрывать всех юзеров из
    ground_truth), а не ожидаемая ситуация.

    Возвращает плоский словарь вида {"recall@10": ..., "ndcg@10": ...,
    "recall@100": ..., "ndcg@100": ...}.
    """
    missing_predictions = set(ground_truth) - set(predictions)
    if missing_predictions:
        logger.warning(
            "%d юзеров из ground_truth не имеют предсказаний в predictions — "
            "пропущены при усреднении (пример: %s)",
            len(missing_predictions),
            list(missing_predictions)[:5],
        )

    results: dict[str, float] = {}

    for k in k_values:
        recalls = []
        ndcgs = []

        for user_id, relevant in ground_truth.items():
            if user_id not in predictions:
                continue

            recommended = predictions[user_id]
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        results[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        results[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

        logger.info(
            "K=%d: Recall@%d=%.4f, NDCG@%d=%.4f (по %d юзерам)",
            k, k, results[f"recall@{k}"], k, results[f"ndcg@{k}"], len(recalls),
        )

    return results


# =============================================================================
# Разбивка по сегментам (Фаза 8: Active vs Cold/Low-Activity)
# =============================================================================

def evaluate_by_segment(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k_values: list[int],
    cold_user_ids: set[int],
) -> dict[str, dict[str, float]]:
    """
    Считает те же метрики отдельно для холодных юзеров (cold_user_ids —
    те, кого не было в train) и для остальных (warm/active).

    Не заменяет evaluate() — используется дополнительно, для разбора
    ошибок по сегментам (см. Фазу 8 плана), когда общее среднее уже
    посчитано и нужно понять, за счёт какого сегмента модель теряет качество.
    """
    cold_gt = {uid: items for uid, items in ground_truth.items() if uid in cold_user_ids}
    warm_gt = {uid: items for uid, items in ground_truth.items() if uid not in cold_user_ids}

    logger.info("Разбивка по сегментам: %d холодных, %d тёплых юзеров", len(cold_gt), len(warm_gt))

    return {
        "cold": evaluate(predictions, cold_gt, k_values),
        "warm": evaluate(predictions, warm_gt, k_values),
    }
