# Jetson Docker inference

This path is inference-only. Train on a PC/cloud GPU using the normal
Donkeycar pipeline, export the Keras `.keras` model to ONNX on that machine,
then build the TensorRT engine on the Jetson. The model input is one
`(IMAGE_H, IMAGE_W, IMAGE_DEPTH)` image, normally `(120, 160, 3)`, converted
from uint8 `[0,255]` to float32 `[0,1]`. Deterministic configured image
transformations run in the vehicle before the pilot; random augmentations are
training-only. Do not install TensorFlow or Torch in this image.

The Dockerfile currently uses `nvcr.io/nvidia/l4t-jetpack:r36.4.0` as an
example matched JetPack 6/L4T r36 tag. Check the host's `cat
/etc/nv_tegra_release` and select the corresponding NVIDIA tag before use;
the tag, TensorRT Python binding, CUDA, and `trtexec` checks need verification
on the Jetson.

## Build and run

```bash
docker build -f Dockerfile.jetson -t donkeycar-jetson .
docker run -it --rm --runtime nvidia --network host --privileged \
  -v /dev:/dev -v "$(pwd)/mycar:/root/mycar" \
  -v /tmp/argus_socket:/tmp/argus_socket donkeycar-jetson
```

For a USB camera, prefer a narrower device mapping such as
`--device=/dev/video0` (and the needed serial/I2C/input devices) instead of
`--privileged`. CSI cameras need the Argus socket mount and a GStreamer camera
pipeline; start with `CAMERA_TYPE = 'V4L'` and a USB camera first.

## PC export and Jetson engine build

The export runs in the PC training environment (Python 3.12+, TF 2.19 /
Keras 3). It uses Keras' native ONNX export and falls back to tf2onnx
(`--tf2onnx` forces it). It needs `onnx`, `onnxruntime` and `tf2onnx`
installed, and `numpy<2`, because tf2onnx is not numpy-2 compatible. The
export uses a fixed batch size of 1 and prints the output order. The pilot
reads output 0 as angle and output 1 as throttle.

```bash
pip install onnx onnxruntime tf2onnx "numpy<2"
python scripts/export_onnx.py model.keras model.onnx --image test.jpg
# in the Jetson container, after copying model.onnx:
bash scripts/build_trt_engine.sh model.onnx model.engine
trtexec --version
```

Serialized engines are not portable across TensorRT versions or GPU
architectures. Rebuild the engine on each Jetson after changing the image,
TensorRT, or JetPack version.

Set these in the car's `myconfig.py` and pass the engine as the model argument
when using the complete template:

```python
DONKEY_INFERENCE_BACKEND = "tensorrt"
TENSORRT_ENGINE_PATH = "/root/mycar/models/model.engine"
```

This backend supports only image-only `linear` models. Behavior, lidar,
odometry, IMU, and localizer setups raise an error at startup. Cropping and
masking come from the normal `TRANSFORMATIONS` config, which runs before
the pilot just as in training. This backend is separate from upstream's
`tensorrt_linear` model type, which uses TF-TRT and needs TensorFlow on the
car.

The pilot can also fall back to a sibling `.onnx` file only when ONNX Runtime
has a TensorRT or CUDA execution provider; a serialized `.engine` is otherwise
loaded through TensorRT Python bindings.

## Verification order

1. Build the container and confirm the Dockerfile import check passes.
2. Run `python3 -c 'import tensorrt, pycuda.driver; print(tensorrt.__version__)'`, `trtexec --version`, and `nvidia-smi`/`tegrastats` as available.
3. Start the drive loop with a USB camera and no model in user mode.
4. Export ONNX on the PC and confirm the script's max absolute difference.
5. Build the engine on the Jetson with `build_trt_engine.sh`.
6. Run the Jetson smoke script on recorded tub images and inspect mean/p95 latency.
7. Start closed-loop driving with wheels off the ground first.
