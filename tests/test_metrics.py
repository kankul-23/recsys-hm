"""
Unit-тесты для src/evaluation/metrics.py (Фаза 11 Master Execution Plan).

Проверяет математику метрик на маленьких, полностью контролируемых
примерах, а не на реальном датасете — цель этих тестов - зафиксировать
поведение recall_at_k/ndcg_at_k/evaluate/evaluate_by_segment на понятных
руками просчитанных случаях, чтобы будущие правки в metrics.py не
сломали их незаметно.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    build_ground_truth,
    evaluate,
    evaluate_by_segment,
    ndcg_at_k,
    recall_at_k,
)


# =============================================================================
# recall_at_k
# =============================================================================

class TestRecallAtK:
    def test_all_relevant_items_at_top(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        assert recall_at_k(recommended, relevant, k=5) == 1.0

    def test_no_hits(self):
        recommended = [1, 2, 3]
        relevant = {99, 100}
        assert recall_at_k(recommended, relevant, k=3) == 0.0

    def test_partial_hits(self):
        # 1 из 2 релевантных попал в топ-3
        recommended = [1, 99, 100]
        relevant = {1, 2}
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(0.5)

    def test_relevant_item_outside_k_not_counted(self):
        # relevant=2 стоит на позиции 4 (индекс 3), за пределами k=2
        recommended = [1, 99, 100, 2]
        relevant = {1, 2}
        assert recall_at_k(recommended, relevant, k=2) == pytest.approx(0.5)

    def test_k_larger_than_recommended_list(self):
        # k превышает длину recommended — recall_at_k не должен падать,
        # top_k просто берёт весь список
        recommended = [1]
        relevant = {1}
        assert recall_at_k(recommended, relevant, k=100) == 1.0


# =============================================================================
# ndcg_at_k
# =============================================================================

class TestNdcgAtK:
    def test_perfect_ranking_gives_ndcg_1(self):
        # Все релевантные товары стоят на первых местах — DCG == IDCG
        recommended = [1, 2, 3]
        relevant = {1, 2}
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_no_hits_gives_ndcg_0(self):
        recommended = [1, 2, 3]
        relevant = {99}
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0

    def test_worse_position_gives_lower_ndcg_than_perfect(self):
        relevant = {1, 2}
        perfect = ndcg_at_k([1, 2, 99], relevant, k=3)
        worse = ndcg_at_k([99, 1, 2], relevant, k=3)
        assert worse < perfect
        assert perfect == pytest.approx(1.0)

    def test_manual_dcg_calculation(self):
        # relevant=1 стоит на позиции 2 (индекс 1) -> DCG = 1/log2(1+2) = 1/log2(3)
        # IDCG (n_ideal_hits=1) = 1/log2(0+2) = 1/log2(2) = 1
        recommended = [99, 1, 100]
        relevant = {1}
        expected = (1.0 / math.log2(3)) / 1.0
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(expected)

    def test_idcg_capped_by_k_not_only_by_relevant_count(self):
        # 3 релевантных, но k=2 -> n_ideal_hits = min(3, 2) = 2, не 3
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, k=2) == pytest.approx(1.0)


# =============================================================================
# build_ground_truth
# =============================================================================

class TestBuildGroundTruth:
    def test_groups_unique_articles_per_customer(self, mock_polars_df):
        # customer 1 купил article 10 дважды и article 20 один раз ->
        # ground truth должен схлопнуть дубликат в множество {10, 20}
        eval_df = mock_polars_df(
            {"customer_idx": [1, 1, 2], "article_id": [10, 10, 30]}
        )
        gt = build_ground_truth(eval_df)
        assert gt[1] == {10}
        assert gt[2] == {30}

    def test_binary_relevance_ignores_repeat_purchase_count(self, mock_polars_df):
        # Одна покупка и пять покупок одного и того же товара дают
        # одинаковый ground truth — бинарная релевантность (см. docstring
        # metrics.py, пункт 1)
        eval_df_once = mock_polars_df({"customer_idx": [1], "article_id": [10]})
        eval_df_many = mock_polars_df(
            {"customer_idx": [1] * 5, "article_id": [10] * 5}
        )
        assert build_ground_truth(eval_df_once) == build_ground_truth(eval_df_many)


# =============================================================================
# evaluate
# =============================================================================

class TestEvaluate:
    def test_averages_across_users(self):
        predictions = {1: [10, 20], 2: [30, 40]}
        ground_truth = {1: {10}, 2: {99}}  # user 1 полный hit, user 2 полный miss
        results = evaluate(predictions, ground_truth, k_values=[2])
        assert results["recall@2"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    def test_user_missing_from_predictions_is_skipped_not_zero(self):
        # user 2 есть в ground_truth, но не в predictions — evaluate()
        # должен пропустить его, а не засчитать как 0.0 (см. docstring)
        predictions = {1: [10]}
        ground_truth = {1: {10}, 2: {20}}
        results = evaluate(predictions, ground_truth, k_values=[1])
        # если бы user 2 засчитался как 0, среднее было бы 0.5, а не 1.0
        assert results["recall@1"] == pytest.approx(1.0)

    def test_empty_ground_truth_gives_zero_not_error(self):
        results = evaluate({}, {}, k_values=[10])
        assert results["recall@10"] == 0.0
        assert results["ndcg@10"] == 0.0

    def test_multiple_k_values_computed_independently(self):
        predictions = {1: [10, 20, 30]}
        ground_truth = {1: {30}}  # релевантный товар только на 3-й позиции
        results = evaluate(predictions, ground_truth, k_values=[1, 3])
        assert results["recall@1"] == 0.0
        assert results["recall@3"] == 1.0


# =============================================================================
# evaluate_by_segment
# =============================================================================

class TestEvaluateBySegment:
    def test_splits_ground_truth_by_cold_user_ids(self):
        predictions = {1: [10], 2: [10], 3: [10]}
        ground_truth = {1: {10}, 2: {99}, 3: {10}}
        cold_ids = {2, 3}

        segmented = evaluate_by_segment(predictions, ground_truth, k_values=[1], cold_user_ids=cold_ids)

        assert set(segmented.keys()) == {"cold", "warm"}
        # cold: users 2 (miss) и 3 (hit) -> recall = 0.5
        assert segmented["cold"]["recall@1"] == pytest.approx(0.5)
        # warm: только user 1 (hit) -> recall = 1.0
        assert segmented["warm"]["recall@1"] == pytest.approx(1.0)

    def test_all_users_cold_gives_empty_warm_segment(self):
        predictions = {1: [10]}
        ground_truth = {1: {10}}
        segmented = evaluate_by_segment(predictions, ground_truth, k_values=[1], cold_user_ids={1})
        assert segmented["warm"]["recall@1"] == 0.0  # пустой сегмент -> evaluate() отдаёт 0.0, не ошибку
        assert segmented["cold"]["recall@1"] == pytest.approx(1.0)
