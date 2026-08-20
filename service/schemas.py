"""
Pydantic-схемы для FastAPI-сервиса recsys-hm (Фаза 10).

Контракт эндпоинта /recommend: снаружи сервис принимает и возвращает
исходный hex customer_id (Utf8) — тот же формат, что был в customers.csv /
transactions_train.csv до Фазы 1. Внутренний Int64 customer_idx (маппинг из
customer_id_mapping.parquet) остаётся деталью реализации и наружу не течёт —
это осознанное решение, а не полумера: потребитель API (мобильное приложение,
внутренний сервис заказов) не должен знать о существовании внутреннего
индекса, и смена схемы индексации внутри проекта не должна ломать контракт.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Ответ эндпоинта GET /recommend."""

    customer_id: str = Field(..., description="Исходный hex customer_id, как в запросе")
    recommendations: list[int] = Field(..., description="article_id по убыванию релевантности")
    source: str = Field(
        ...,
        description=(
            "Откуда взяты рекомендации: 'als' — персонализированный ALS-кандидат "
            "(customer_id встречался в train), 'popularity_fallback' — глобальный "
            "Popularity-fallback (холодный customer_id, ALS не строил для него вектор)."
        ),
    )


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /health."""

    status: str
    als_users_loaded: int
    popularity_items_loaded: int
