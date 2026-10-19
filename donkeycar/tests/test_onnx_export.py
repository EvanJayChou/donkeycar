import os

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow',
                         reason='TensorFlow not installed, skipping onnx tests')

from donkeycar.config import Config
from donkeycar.parts.interpreter import KerasInterpreter
from donkeycar.parts.keras import KerasLinear, KerasCategorical
from donkeycar.pipeline.training import export_onnx


def _cfg():
    cfg = Config()
    cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH = 120, 160, 3
    return cfg


def _save(pilot_cls, tmp_path):
    interpreter = KerasInterpreter()
    kl = pilot_cls(interpreter=interpreter)
    keras_path = str(tmp_path / 'model.keras')
    interpreter.model.save(keras_path)
    return kl, keras_path


def test_export_skips_unsupported_model(tmp_path):
    kl, keras_path = _save(KerasCategorical, tmp_path)
    onnx_path = str(tmp_path / 'model.onnx')
    export_onnx(_cfg(), kl, keras_path, onnx_path)
    assert not os.path.exists(onnx_path)


def test_export_linear_matches_keras(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('tf2onnx')
    kl, keras_path = _save(KerasLinear, tmp_path)
    onnx_path = str(tmp_path / 'model.onnx')
    export_onnx(_cfg(), kl, keras_path, onnx_path)
    assert os.path.exists(onnx_path)

    session = ort.InferenceSession(onnx_path,
                                   providers=['CPUExecutionProvider'])
    image = np.random.default_rng(1).random((1, 120, 160, 3),
                                            dtype=np.float32)
    angle, throttle = session.run(None,
                                  {session.get_inputs()[0].name: image})
    expected = kl.interpreter.model(image, training=False)
    np.testing.assert_allclose(angle.reshape(-1),
                               np.asarray(expected[0]).reshape(-1), atol=1e-3)
    np.testing.assert_allclose(throttle.reshape(-1),
                               np.asarray(expected[1]).reshape(-1), atol=1e-3)
