"""Export trained Keras pilots to ONNX for TensorRT inference on Jetson.

Uses Keras 3's native ONNX export (``model.export(..., format="onnx")``) and
falls back to tf2onnx on older Keras. The export uses a fixed batch size of 1
so the TensorRT engine built from it has static shapes, and validates the
ONNX outputs against the Keras model with ONNX Runtime on CPU.

Requires (on the training machine): tensorflow, onnx, onnxruntime, tf2onnx.
tf2onnx (also used internally by Keras' TF-backend ONNX export) is not
compatible with numpy 2.x; use ``numpy<2`` in the training environment.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _as_list(result):
    if isinstance(result, dict):
        result = list(result.values())
    return list(result) if isinstance(result, (tuple, list)) else [result]


def _export_native(model, output_path, signature):
    if not hasattr(model, "export"):
        raise AttributeError("model.export unavailable")
    model.export(str(output_path), format="onnx", input_signature=signature)


def _export_tf2onnx(model, output_path, signature, opset):
    import tensorflow as tf
    import tf2onnx

    @tf.function(input_signature=signature)
    def serving(img_in):
        result = _as_list(model(img_in, training=False))
        # zero-padded keys keep output order stable after key sorting
        return {f"output_{i:02d}": value for i, value in enumerate(result)}

    tf2onnx.convert.from_function(
        serving, input_signature=signature, opset=opset,
        output_path=str(output_path))


def keras_model_to_onnx(model_path, output_path, input_shape=(120, 160, 3),
                        image_path=None, opset=17, atol=1e-3,
                        force_tf2onnx=False):
    """Convert a saved .keras/.h5 model to ONNX and validate it.

    :return: maximum absolute difference between Keras and ONNX outputs
    :raises AssertionError: if the outputs differ by more than ``atol``
    """
    import tensorflow as tf
    import onnxruntime as ort

    model = tf.keras.models.load_model(model_path, compile=False)
    shape = tuple(int(dim) for dim in input_shape)
    signature = [tf.TensorSpec((1, *shape), tf.float32, name="img_in")]

    if force_tf2onnx:
        _export_tf2onnx(model, output_path, signature, opset)
    else:
        try:
            _export_native(model, output_path, signature)
            logger.info("Exported with Keras native ONNX export")
        except Exception as exc:  # older Keras or unsupported layer
            logger.info(f"Keras native ONNX export failed ({exc}); "
                        f"using tf2onnx")
            _export_tf2onnx(model, output_path, signature, opset)

    session = ort.InferenceSession(str(output_path),
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    sample = rng.random((1, *shape), dtype=np.float32)
    if image_path:
        from PIL import Image
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape != shape:
            raise ValueError(
                f"Test image shape {image.shape} does not match {shape}")
        sample = (image.astype(np.float32) / 255.0)[None]
    original = _as_list(model(sample, training=False))
    converted = session.run(None, {session.get_inputs()[0].name: sample})
    # TensorRTPilot reads output 0 as angle and output 1 as throttle
    logger.info(f"ONNX outputs: "
                f"{[(o.name, o.shape) for o in session.get_outputs()]}")
    if len(converted) != len(original):
        raise AssertionError(
            f"ONNX has {len(converted)} outputs, Keras has {len(original)}")
    max_diff = max(
        float(np.max(np.abs(np.asarray(a).reshape(-1) -
                            np.asarray(b).reshape(-1))))
        for a, b in zip(original, converted))
    logger.info(f"Wrote {output_path}; max abs diff vs Keras={max_diff:.8g}")
    if max_diff > atol:
        raise AssertionError(f"ONNX validation failed: {max_diff} > {atol}")
    return max_diff
