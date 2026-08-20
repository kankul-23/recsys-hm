"""
Общие pytest-фикстуры для тестов recsys-hm (Фаза 11).
"""

from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def mock_polars_df():
    """
    Фабрика для маленьких polars.DataFrame по словарю колонок — избавляет
    тесты metrics.py/als.py от повторения `pl.DataFrame({...})` в каждом
    кейсе и делает вызов явным (mock_polars_df({"a": [...], "b": [...]})).
    """

    def _make(columns: dict[str, list]) -> pl.DataFrame:
        return pl.DataFrame(columns)

    return _make
