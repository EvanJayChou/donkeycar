"""TensorRT serialized-engine pilot for Jetson inference.

The module intentionally imports TensorRT/CUDA lazily so the PC training and
test paths remain usable without either vendor runtime.
"""
import logging
import time
from pathlib import Path

import numpy as np

from donkeycar.utils import normalize_image

logger = logging.getLogger(__name__)


class TensorRTPilot:
    """Run a linear Donkeycar model from a serialized TensorRT engine."""

    def __init__(self, engine_path, input_shape=(120, 160, 3),
                 crop=None, normalize=True, warmup_frames=20):
        self.engine_path = str(Path(engine_path).expanduser())
        self.input_shape = tuple(input_shape)
        self.crop = crop
        self.normalize = normalize
        self.warmup_frames = int(warmup_frames)
        self._latencies = []
        self._frame_count = 0
        self._backend = None
        self._load()

    @staticmethod
    def preprocess(img_array, input_shape=(120, 160, 3), crop=None,
                   normalize=True):
        """Match the linear Keras training path: crop, shape, normalize."""
        image = np.asarray(img_array)
        if crop is not None:
            top, bottom, left, right = crop
            end = image.shape[0] - bottom if bottom else image.shape[0]
            end_x = image.shape[1] - right if right else image.shape[1]
            image = image[top:end, left:end_x]
        if tuple(image.shape) != tuple(input_shape):
            raise ValueError(
                f"Input image shape {image.shape} does not match {input_shape}; "
                "configure camera/transforms to produce the model shape.")
        if normalize:
            image = normalize_image(image)
        return np.ascontiguousarray(image, dtype=np.float32)

    def _load(self):
        if not Path(self.engine_path).is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        try:
            self._load_pycuda()
            self._backend = "tensorrt"
            return
        except ImportError as exc:
            logger.warning("TensorRT Python/CUDA bindings unavailable: %s", exc)
        except Exception:
            logger.exception("Could not load TensorRT engine with pycuda")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "TensorRT bindings could not load the engine and onnxruntime "
                "is not installed; install pycuda/tensorrt or onnxruntime-gpu"
            ) from exc
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            raise ImportError(
                "Neither TensorRT Python bindings nor an ONNX Runtime CUDA "
                "execution provider is available")
        # ORT does not load a serialized TensorRT .engine directly. This
        # fallback is therefore for a same-stem ONNX model beside the engine.
        onnx_path = str(Path(self.engine_path).with_suffix(".onnx"))
        if not Path(onnx_path).is_file():
            raise RuntimeError(
                "onnxruntime fallback requires a sibling .onnx file; "
                f"missing {onnx_path}")
        self._ort = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._ort.get_inputs()[0].name
        self._backend = "onnxruntime"

    def _load_pycuda(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 - creates the CUDA context

        self._trt = trt
        self._cuda = cuda
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as engine_file:
            runtime = trt.Runtime(self._logger)
            self._engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self._engine is None:
            raise RuntimeError("TensorRT failed to deserialize the engine")
        self._context = self._engine.create_execution_context()
        if hasattr(self._engine, "num_io_tensors"):
            self._load_pycuda_tensor_api(trt, cuda)
            return
        self._bindings = []
        self._host_outputs = []
        self._device_outputs = []
        self._input_index = None
        for index in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                shape = (1, *self.input_shape)
                self._context.set_binding_shape(index, shape)
            dtype = trt.nptype(self._engine.get_binding_dtype(index))
            size = int(np.prod(shape))
            device = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self._bindings.append(int(device))
            if self._engine.binding_is_input(index):
                self._input_index = index
                self._input_shape = shape
                self._input_dtype = dtype
                self._input_device = device
            else:
                host = cuda.pagelocked_empty(size, dtype)
                self._host_outputs.append((host, shape))
                self._device_outputs.append(device)
        if self._input_index is None or len(self._host_outputs) < 2:
            raise RuntimeError("Expected one input and two TensorRT outputs")
        self._stream = cuda.Stream()

    def _load_pycuda_tensor_api(self, trt, cuda):
        """Allocate buffers using the TensorRT 10 name-based API."""
        self._tensor_names = [self._engine.get_tensor_name(i)
                              for i in range(self._engine.num_io_tensors)]
        self._tensor_outputs = []
        self._tensor_input = None
        self._stream = cuda.Stream()
        for name in self._tensor_names:
            shape = tuple(self._engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                shape = (1, *self.input_shape)
                self._context.set_input_shape(name, shape)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            device = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self._context.set_tensor_address(name, int(device))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._tensor_input = (name, device, shape, dtype)
            else:
                self._tensor_outputs.append((device, shape, dtype))
        if self._tensor_input is None or len(self._tensor_outputs) < 2:
            raise RuntimeError("Expected one input and two TensorRT outputs")

    def _infer_trt(self, image):
        if hasattr(self._engine, "num_io_tensors"):
            name, input_device, _, _ = self._tensor_input
            self._cuda.memcpy_htod_async(input_device, image, self._stream)
            self._context.execute_async_v3(self._stream.handle)
            outputs = []
            for device, shape, dtype in self._tensor_outputs:
                host = self._cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                self._cuda.memcpy_dtoh_async(host, device, self._stream)
                outputs.append((host, shape))
            self._stream.synchronize()
            return [np.array(host).reshape(shape).copy() for host, shape in outputs]
        self._cuda.memcpy_htod_async(self._input_device, image, self._stream)
        self._context.execute_async_v2(self._bindings, self._stream.handle)
        for (host, shape), device in zip(self._host_outputs, self._device_outputs):
            self._cuda.memcpy_dtoh_async(host, device, self._stream)
        self._stream.synchronize()
        return [np.array(host).reshape(shape).copy()
                for host, _ in self._host_outputs]

    def run(self, img_array):
        image = self.preprocess(img_array, self.input_shape, self.crop,
                                self.normalize)
        start = time.perf_counter()
        if self._backend == "tensorrt":
            outputs = self._infer_trt(image)
        else:
            outputs = self._ort.run(None, {self._input_name: image[None]})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._frame_count += 1
        if self._frame_count <= self.warmup_frames:
            self._latencies.append(elapsed_ms)
            if self._frame_count == self.warmup_frames:
                logger.info("TensorRT pilot startup latency: mean=%.2f ms p95=%.2f ms",
                            np.mean(self._latencies), np.percentile(self._latencies, 95))
        values = [float(np.asarray(output).reshape(-1)[0]) for output in outputs]
        return values[0], values[1]

    def shutdown(self):
        self._latencies.clear()
        self._context = None

    def load(self, model_path):
        """Compatibility with the normal Donkeycar pilot loading hook."""
        if str(Path(model_path).expanduser()) != self.engine_path:
            raise ValueError("TensorRTPilot engine path cannot change at runtime")
