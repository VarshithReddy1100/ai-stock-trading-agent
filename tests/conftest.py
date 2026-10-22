import pytest

from src.config import ModelConfig, PipelineConfig
from src.synthetic import generate_synthetic_ohlcv


@pytest.fixture(scope="session")
def ohlcv():
    return generate_synthetic_ohlcv(n=900, seed=11)


@pytest.fixture()
def tiny_model_cfg():
    return ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=4, batch_size=32, patience=3)


@pytest.fixture()
def tiny_pipeline_cfg(tiny_model_cfg):
    cfg = PipelineConfig()
    cfg.model = tiny_model_cfg
    return cfg
