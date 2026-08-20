"""
Интеграционные тесты для service/app.py (Фаза 11 Master Execution Plan).

Не запускают lifespan (который читает train.parquet/customer_id_mapping.parquet
с диска и обучает ALS на полном датасете — недопустимо в тестах). Вместо
этого подменяют module-level service.app.model_state готовым ModelState,
построенным на тех же маленьких синтетических данных и тех же реальных
классах (PopularityRecommender, ALSCandidateGenerator), что в
test_recommenders.py — то есть тестируется реальная логика эндпоинтов
поверх настоящих (просто маленьких) моделей, а не заглушек, имитирующих
их поведение.
"""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

import service.app as app_module
from src.recommenders.als import (
    ALSCandidateGenerator,
    build_article_id_mapping,
    build_customer_matrix_mapping,
    build_interaction_matrix,
    fit_als_model,
)
from src.recommenders.popularity import PopularityRecommender


@pytest.fixture
def client(monkeypatch):
    """
    Собирает ModelState на маленьком синтетическом train (тот же принцип,
    что tiny_als_setup в test_recommenders.py) и подменяет им
    service.app.model_state напрямую — обходит lifespan/чтение с диска.

    customer_idx=1 -> known hex id "known_hex_id_1" (встречался в train,
    ALS должен дать персональные рекомендации).
    "unknown_hex_id" намеренно отсутствует в customer_id_to_idx —
    проверяет ветку "id вообще не в mapping".
    """
    train = pl.DataFrame(
        {
            "customer_idx": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6],
            "article_id": [10, 20, 30, 10, 40, 20, 30, 50, 10, 60, 40, 70, 50, 80],
        }
    )
    user_mapping = build_customer_matrix_mapping(train)
    item_mapping = build_article_id_mapping(train)
    interaction_matrix = build_interaction_matrix(
        train, user_mapping, item_mapping, confidence_alpha=40
    )
    als_model = fit_als_model(
        interaction_matrix, factors=4, iterations=3, regularization=0.05, random_state=42
    )
    als_generator = ALSCandidateGenerator(
        als_model, interaction_matrix, user_mapping, item_mapping, top_k=3
    )
    popularity_recommender = PopularityRecommender(top_n=10).fit(train)

    state = app_module.ModelState(
        als_generator=als_generator,
        popularity_recommender=popularity_recommender,
        customer_id_to_idx={"known_hex_id_1": 1, "known_hex_id_2": 2},
        top_k=5,
        n_als_users=user_mapping.height,
    )
    monkeypatch.setattr(app_module, "model_state", state)

    return TestClient(app_module.app)


class TestHealthEndpoint:
    def test_returns_200_with_loaded_counts(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["als_users_loaded"] == 6
        assert body["popularity_items_loaded"] == 10

    def test_returns_503_when_models_not_loaded(self, monkeypatch):
        # model_state=None имитирует запрос, пришедший до завершения lifespan
        monkeypatch.setattr(app_module, "model_state", None)
        client = TestClient(app_module.app)
        response = client.get("/health")
        assert response.status_code == 503


class TestRecommendEndpoint:
    def test_known_customer_gets_als_source(self, client):
        response = client.get("/recommend", params={"user_id": "known_hex_id_1"})
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "als"
        assert body["customer_id"] == "known_hex_id_1"
        assert len(body["recommendations"]) > 0

    def test_known_customer_als_recs_exclude_purchased(self, client):
        # customer_idx=1 купил {10, 20, 30} в фикстуре train — ни один
        # не должен встретиться в ответе (сквозная проверка через HTTP,
        # то же свойство, что test_excludes_already_purchased_items
        # в test_recommenders.py, но через реальный эндпоинт)
        response = client.get("/recommend", params={"user_id": "known_hex_id_1"})
        recs = set(response.json()["recommendations"])
        assert recs.isdisjoint({10, 20, 30})

    def test_unknown_customer_id_gets_popularity_fallback(self, client):
        # id вообще не в customer_id_to_idx — не 404, а fallback
        # (см. докстринг app.py, пункт 4)
        response = client.get("/recommend", params={"user_id": "never_seen_this_id"})
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "popularity_fallback"

    def test_missing_user_id_returns_422(self, client):
        # user_id обязателен (Query(...)) — без него FastAPI отдаёт 422,
        # не 500 и не пропускает запрос дальше
        response = client.get("/recommend")
        assert response.status_code == 422

    def test_top_k_query_param_is_respected(self, client):
        response = client.get(
            "/recommend", params={"user_id": "never_seen_this_id", "top_k": 2}
        )
        assert len(response.json()["recommendations"]) == 2

    def test_default_top_k_used_when_not_specified(self, client):
        # ModelState.top_k=5 в фикстуре client — без явного top_k в запросе
        # эндпоинт должен использовать это значение по умолчанию
        response = client.get("/recommend", params={"user_id": "never_seen_this_id"})
        assert len(response.json()["recommendations"]) == 5

    def test_returns_503_when_models_not_loaded(self, monkeypatch):
        monkeypatch.setattr(app_module, "model_state", None)
        client = TestClient(app_module.app)
        response = client.get("/recommend", params={"user_id": "anything"})
        assert response.status_code == 503
