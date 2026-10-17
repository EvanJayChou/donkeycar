#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:?usage: $0 model.onnx [model.engine]}"
ENGINE_PATH="${2:-${ONNX_PATH%.onnx}.engine}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec is not installed or is not on PATH" >&2
  exit 1
fi
echo "TensorRT: $(trtexec --version 2>&1 | head -n 1)"
trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16 \
  --memPoolSize=workspace:1024 --builderOptimizationLevel=3
echo "Wrote $ENGINE_PATH"
