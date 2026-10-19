# Donkeycar on Jetson Orin Nano (Docker + TensorRT)

Two machines are involved:

- **Training machine** (laptop or GPU cluster): trains the model and exports
  it to ONNX.
- **Jetson** (on the car): runs everything in Docker and converts the ONNX
  model into a TensorRT `.engine` file for fast inference.

```
 Training machine                          Jetson (Docker)
 ────────────────                          ───────────────
 donkey train ──► pilot_1.keras            ./jetson.sh engine models/pilot_1.onnx
              └─► pilot_1.onnx ── copy ──►   └─► pilot_1.engine
                                           ./jetson.sh drive models/pilot_1.engine
```

Requires JetPack 6 (L4T r36.x) on the Jetson.

---

## 1. Jetson: first-time setup

```bash
git clone git@github.com:EvanJayChou/donkeycar.git
cd donkeycar
./jetson.sh build      # builds the Docker image (takes a while the first time)
./jetson.sh setup      # creates your car folder at ~/mycar
./jetson.sh check      # confirms the GPU and TensorRT work
```

Then edit `~/mycar/myconfig.py` for your camera, steering and throttle
hardware, as for any Donkeycar. Use a USB camera (`CAMERA_TYPE = "V4L"`)
first, because it is the simplest to get working in Docker.

> If `docker` says *permission denied*, run `sudo usermod -aG docker $USER`
> once, then log out and back in.

## 2. Jetson: drive and collect data

```bash
./jetson.sh drive
```

Open `http://<jetson-ip>:8887` in a browser to drive. Recorded data goes to
`~/mycar/data/`. Press `Ctrl+C` to stop.

## 3. Training machine: train and export to ONNX

First, install once, from the repo folder (Python 3.12):

```bash
pip install -e ".[pc,onnx]"
```

Copy the data from the car, then train:

```bash
scp -r <user>@<jetson-ip>:~/mycar/data ./mycar/
cd mycar
donkey train --tub ./data --model ./models/pilot_1.keras
```

Because `CREATE_ONNX = True` is set in the config, training writes
`models/pilot_1.onnx` next to the `.keras` model. It also checks that the
ONNX model gives the same outputs as the Keras model. For models trained
earlier, export by hand:

```bash
python scripts/export_onnx.py models/pilot_1.keras models/pilot_1.onnx
```

Copy the ONNX file to the car:

```bash
scp models/pilot_1.onnx <user>@<jetson-ip>:~/mycar/models/
```

## 4. Jetson: build the engine and drive on autopilot

```bash
./jetson.sh engine models/pilot_1.onnx
./jetson.sh drive models/pilot_1.engine
```

In the web UI, switch the mode to *Local Pilot* (or *Local Angle*). Test
with the wheels off the ground first.

There is no config to change. Passing a `.engine` file automatically uses
the TensorRT pilot.

---

## Command reference

| Command | What it does |
|---|---|
| `./jetson.sh build` | Build the image. Re-run after `git pull`. |
| `./jetson.sh setup` | Create `~/mycar` (first time only). |
| `./jetson.sh check` | Print the TensorRT version and GPU name. |
| `./jetson.sh engine FILE.onnx` | Build `FILE.engine` next to it. |
| `./jetson.sh drive [FILE.engine]` | Drive manually, or with the TensorRT autopilot. |
| `./jetson.sh shell` | Open a shell in the container, with the car folder at `/root/mycar`. |

Optional settings go in front of the command, for example
`CAR_DIR=~/car2 ./jetson.sh drive`:

- `CAR_DIR` sets the car folder. The default is `~/mycar`.
- `L4T_TAG` overrides the base image tag. By default it is detected from
  `/etc/nv_tegra_release`, so for example R36 revision 4.x uses `r36.4.0`.
- `IMAGE` sets the Docker image name.

## Good to know

- **Rebuild engines on the car.** Engines only work with the TensorRT
  version and GPU they were built on. After a JetPack or image update,
  re-run `./jetson.sh engine`.
- **Model support.** Only image-only `linear` models are supported.
  Behavior, lidar, odometry, IMU and localizer setups stop at startup with
  an error, and training skips the ONNX export for them.
- **Image preprocessing.** The pilot scales images from 0–255 to 0–1, as in
  training. Cropping and masking come from the normal `TRANSFORMATIONS`
  setting, which runs before the pilot just as in training.
- **Not the same as `tensorrt_linear`.** That upstream model type uses TF-TRT
  and needs TensorFlow on the car. This setup does not.
- **CSI cameras.** The Argus socket is mounted automatically when present.
  CSI cameras need a GStreamer camera setup in `myconfig.py`.
- **ONNX export packages.** The export needs `numpy<2` because of tf2onnx.
  `pip install -e ".[onnx]"` takes care of that.
- **ONNX Runtime fallback.** If the TensorRT Python bindings cannot load the
  engine, the pilot can fall back to ONNX Runtime. This needs the matching
  `.onnx` file next to the engine and an ONNX Runtime GPU build.

## Verification order (first time on a new Jetson)

1. `./jetson.sh build` finishes. The image build includes an import check
   for TensorRT, pycuda and trtexec.
2. `./jetson.sh check` prints the TensorRT version and the GPU name.
3. `./jetson.sh drive` with a USB camera, no model, in user mode.
4. Export ONNX on the training machine and confirm the reported max
   absolute difference is small.
5. `./jetson.sh engine models/pilot_1.onnx`.
6. Smoke-test latency on a recorded image: `./jetson.sh shell`, then
   `python3 /root/donkeycar/scripts/smoke_test_tensorrt.py
   models/pilot_1.engine data/images/<some>.jpg`.
7. `./jetson.sh drive models/pilot_1.engine` with the wheels off the ground,
   then on the track.
