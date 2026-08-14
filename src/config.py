"""
Централизованная загрузка конфигурации проекта recsys-hm.

BASE_DIR вычисляется относительно расположения этого файла (src/config.py),
а не текущей рабочей директории — так конфиг работает одинаково что из
ноутбука, что из FastAPI-сервиса, что из pytest.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

# src/config.py -> parents[0] = src/, parents[1] = корень проекта recsys-hm
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


class Paths(BaseModel):
    data_raw: Path
    data_processed: Path
    models: Path


class TemporalSplit(BaseModel):
    train_end: Optional[str] = None
    valid_end: Optional[str] = None
    test_end: Optional[str] = None


class ALSConfig(BaseModel):
    factors: int
    iterations: int
    regularization: float
    top_k_candidates: int
    random_state: int


class NegativeSamplingConfig(BaseModel):
    ratio: int


class RankerConfig(BaseModel):
    loss_function: str
    iterations: int
    learning_rate: float
    depth: int
    random_state: int


class EvaluationConfig(BaseModel):
    k_values: list[int]


class ServiceConfig(BaseModel):
    default_top_k: int
    host: str
    port: int


class Config(BaseModel):
    paths: Paths
    temporal_split: TemporalSplit
    als: ALSConfig
    negative_sampling: NegativeSamplingConfig
    ranker: RankerConfig
    evaluation: EvaluationConfig
    service: ServiceConfig

    def resolve_paths(self) -> "Config":
        """Делает пути в paths абсолютными относительно BASE_DIR."""
        self.paths.data_raw = BASE_DIR / self.paths.data_raw
        self.paths.data_processed = BASE_DIR / self.paths.data_processed
        self.paths.models = BASE_DIR / self.paths.models
        return self


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Читает configs/config.yaml и возвращает провалидированный Config."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw).resolve_paths()


# Готовый к использованию синглтон — просто `from src.config import CONFIG`
CONFIG = load_config()


if __name__ == "__main__":
    # Быстрая проверка: python -m src.config
    print(CONFIG.model_dump_json(indent=2))
