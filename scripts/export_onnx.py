#!/usr/bin/env python3
"""Export a Donkeycar Keras linear model to ONNX and compare CPU outputs.

Training already does this automatically when CREATE_ONNX = True; use this
script for models trained earlier. See donkeycar/parts/onnx_export.py.
"""
import argparse
import logging

from donkeycar.parts.onnx_export import keras_model_to_onnx


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
    logging.getLogger().setLevel(logging.INFO)
    keras_model_to_onnx(args.model, args.output,
                        (args.height, args.width, args.depth),
                        image_path=args.image, opset=args.opset,
                        atol=args.atol, force_tf2onnx=args.tf2onnx)


if __name__ == "__main__":
    main()
