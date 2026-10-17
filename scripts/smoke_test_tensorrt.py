#!/usr/bin/env python3
"""Jetson-only TensorRT pilot smoke test on a recorded image."""
import argparse
import time
import numpy as np
from PIL import Image
from donkeycar.parts.tensorrt_pilot import TensorRTPilot

parser = argparse.ArgumentParser()
parser.add_argument("engine")
parser.add_argument("image")
parser.add_argument("--count", type=int, default=100)
args = parser.parse_args()
pilot = TensorRTPilot(args.engine)
image = np.asarray(Image.open(args.image).convert("RGB"))
times = []
output = None
for _ in range(args.count):
    start = time.perf_counter()
    output = pilot.run(image)
    times.append((time.perf_counter() - start) * 1000.0)
print(f"mean_ms={np.mean(times):.3f} p95_ms={np.percentile(times, 95):.3f} output={output}")
