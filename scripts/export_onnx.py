#!/usr/bin/env python3
"""Export a Donkeycar Keras linear model to ONNX and compare CPU outputs."""
import argparse
from pathlib import Path

import numpy as np


def export_keras(model_path, output_path, image_path=None, input_shape=(120, 160, 3),
                 opset=17, atol=1e-3):
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    model = tf.keras.models.load_model(model_path, compile=False)
    shape = tuple(int(dim) for dim in input_shape)
    signature = (tf.TensorSpec((1, *shape), tf.float32, name="img_in"),)

    @tf.function(input_signature=signature)
    def serving(img_in):
        result = model(img_in, training=False)
        if isinstance(result, (tuple, list)):
            return {f"output_{i}": value for i, value in enumerate(result)}
        return {"output_0": result}

    onnx_model, _ = tf2onnx.convert.from_function(
        serving, input_signature=signature, opset=opset,
        output_path=str(output_path))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    sample = (rng.random((1, *shape), dtype=np.float32))
    if image_path:
        from PIL import Image
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape != shape:
            raise ValueError(f"Test image shape {image.shape} does not match {shape}")
        sample = (image.astype(np.float32) / 255.0)[None]
    original = model(sample, training=False)
    original = list(original) if isinstance(original, (tuple, list)) else [original]
    converted = session.run(None, {session.get_inputs()[0].name: sample})
    original_flat = np.concatenate([np.asarray(v).reshape(-1) for v in original])
    converted_flat = np.concatenate([np.asarray(v).reshape(-1) for v in converted])
    max_diff = float(np.max(np.abs(original_flat - converted_flat)))
    print(f"wrote {output_path}; max abs diff={max_diff:.8g}")
    if max_diff > atol:
        raise AssertionError(f"ONNX validation failed: {max_diff} > {atol}")
    return onnx_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--image")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    export_keras(args.model, args.output, args.image,
                 (args.height, args.width, args.depth), args.opset, args.atol)


if __name__ == "__main__":
    main()
