import importlib

import numpy as np
import pytest

from donkeycar.parts.tensorrt_pilot import TensorRTPilot
from donkeycar.utils import normalize_image


def test_preprocess_matches_training_normalization():
    image = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
    expected = np.ascontiguousarray(normalize_image(image), dtype=np.float32)
    actual = TensorRTPilot.preprocess(image)
    np.testing.assert_array_equal(actual, expected)


def test_preprocess_crop_matches_training_shape():
    image = np.zeros((165, 160, 3), dtype=np.uint8)
    result = TensorRTPilot.preprocess(image, crop=(45, 0, 0, 0))
    assert result.shape == (120, 160, 3)


def test_module_imports_without_tensor_rt(monkeypatch):
    module = importlib.import_module("donkeycar.parts.tensorrt_pilot")
    assert module.TensorRTPilot


def test_missing_engine_is_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        TensorRTPilot(tmp_path / "missing.engine")


def test_config_defaults_are_additive():
    config = importlib.import_module("donkeycar.templates.cfg_complete")
    assert config.DONKEY_INFERENCE_BACKEND == ""
    assert config.TENSORRT_NORMALIZE is True
