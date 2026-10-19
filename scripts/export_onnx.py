#!/usr/bin/env python3
"""Export a Donkeycar Keras linear model to ONNX and compare CPU outputs.

Uses Keras 3's native ONNX export (``model.export(..., format="onnx")``,
Keras >= 3.8) and falls back to tf2onnx on older Keras. The export uses a
fixed batch size of 1 so the TensorRT engine built from it has static shapes.

Requires (on the training PC): tensorflow, onnx, onnxruntime, tf2onnx. Note
tf2onnx (also used internally by Keras' TF-backend ONNX export) is not
compatible with numpy 2.x; use ``numpy<2`` in the export environment.
"""
import argparse

import numpy as np


def _export_native(model, output_path, signature):
    if not hasattr(model, "export"):
        raise AttributeError("model.export unavailable")
    model.export(str(output_path), format="onnx", input_signature=signature)


def _export_tf2onnx(model, output_path, signature, opset):
    import tensorflow as tf
    import tf2onnx

    @tf.function(input_signature=signature)
    def serving(img_in):
        result = model(img_in, training=False)
        if isinstance(result, dict):
            result = list(result.values())
        if isinstance(result, (tuple, list)):
            # zero-padded keys keep output order stable after key sorting
            return {f"output_{i:02d}": value for i, value in enumerate(result)}
        return {"output_00": result}

    tf2onnx.convert.from_function(
        serving, input_signature=signature, opset=opset,
        output_path=str(output_path))


def export_keras(model_path, output_path, image_path=None,
                 input_shape=(120, 160, 3), opset=17, atol=1e-3,
                 force_tf2onnx=False):
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
            print("exported with Keras native ONNX export")
        except Exception as exc:  # older Keras or unsupported layer
            print(f"Keras native ONNX export failed ({exc}); using tf2onnx")
            _export_tf2onnx(model, output_path, signature, opset)

    session = ort.InferenceSession(str(output_path),
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    sample = rng.random((1, *shape), dtype=np.float32)
    if image_path:
        from PIL import Image
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape != shape:
            raise ValueError(f"Test image shape {image.shape} does not match {shape}")
        sample = (image.astype(np.float32) / 255.0)[None]
    original = model(sample, training=False)
    if isinstance(original, dict):
        original = list(original.values())
    original = list(original) if isinstance(original, (tuple, list)) else [original]
    converted = session.run(None, {session.get_inputs()[0].name: sample})
    print("ONNX outputs (TensorRTPilot reads them in this order as "
          "angle, throttle):",
          [(o.name, o.shape) for o in session.get_outputs()])
    if len(converted) != len(original):
        raise AssertionError(
            f"ONNX has {len(converted)} outputs, Keras has {len(original)}")
    max_diff = max(
        float(np.max(np.abs(np.asarray(a).reshape(-1) - np.asarray(b).reshape(-1))))
        for a, b in zip(original, converted))
    print(f"wrote {output_path}; max abs diff={max_diff:.8g}")
    if max_diff > atol:
        raise AssertionError(f"ONNX validation failed: {max_diff} > {atol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="trained .keras (or .h5) model")
    parser.add_argument("output", help="output .onnx path")
    parser.add_argument("--image")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--tf2onnx", action="store_true",
                        help="skip Keras native export and use tf2onnx")
    args = parser.parse_args()
    export_keras(args.model, args.output, args.image,
                 (args.height, args.width, args.depth), args.opset, args.atol,
                 args.tf2onnx)


if __name__ == "__main__":
    main()
