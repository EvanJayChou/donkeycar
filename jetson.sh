#!/usr/bin/env bash
# One-stop helper for running Donkeycar in Docker on a Jetson (JetPack 6).
# Run ./jetson.sh with no arguments for usage. See docs/JETSON_DOCKER.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-donkeycar-jetson}"
CAR_DIR="${CAR_DIR:-$HOME/mycar}"
DOCKER="${DOCKER:-docker}"

usage() {
  cat <<EOF
Usage: ./jetson.sh <command> [args]

  build            Build the Docker image (do this once, and after git pull)
  setup            Create the car folder at $CAR_DIR (first time only)
  check            Verify CUDA / TensorRT work inside the container
  engine FILE.onnx Turn an ONNX model from training into a TensorRT .engine
  drive [FILE]     Drive; with FILE.engine the TensorRT autopilot is used
  shell            Open a bash shell in the container (car folder mounted)

Settings (optional environment variables):
  CAR_DIR=$CAR_DIR   IMAGE=$IMAGE   L4T_TAG=<auto>   DOCKER=$DOCKER
EOF
}

# Map the host's L4T release (e.g. "R36 (release), REVISION: 4.3") to the
# matching NGC l4t-jetpack tag (r36.4.0).
detect_l4t_tag() {
  if [[ -n "${L4T_TAG:-}" ]]; then echo "$L4T_TAG"; return; fi
  local release=/etc/nv_tegra_release
  if [[ -r $release ]]; then
    local major minor
    major=$(sed -n 's/^# R\([0-9]*\) .*/\1/p' "$release")
    minor=$(sed -n 's/.*REVISION: \([0-9]*\)\..*/\1/p' "$release")
    if [[ -n $major && -n $minor ]]; then
      echo "r${major}.${minor}.0"; return
    fi
  fi
  echo "r36.4.0"
}

# Run a command in the container with GPU, camera, I2C/serial and the car
# folder mounted at /root/mycar. Web UI: http://<jetson-ip>:8887
run_container() {
  mkdir -p "$CAR_DIR"
  local tty=()
  [[ -t 0 ]] && tty=(-it)
  local argus=()
  [[ -e /tmp/argus_socket ]] && argus=(-v /tmp/argus_socket:/tmp/argus_socket)
  # The container runs as root; hand files it creates (config, tubs, engines)
  # back to the host user afterwards so they can be edited without sudo.
  "$DOCKER" run "${tty[@]}" --rm --runtime nvidia --network host --privileged \
    -v /dev:/dev "${argus[@]}" -v "$CAR_DIR:/root/mycar" \
    -w /root/mycar "$IMAGE" -lc \
    "$*; status=\$?; chown -R $(id -u):$(id -g) /root/mycar; exit \$status"
}

require_car() {
  if [[ ! -f "$CAR_DIR/manage.py" ]]; then
    echo "No car folder at $CAR_DIR yet; run: ./jetson.sh setup" >&2
    exit 1
  fi
}

# Accept a path on the host inside CAR_DIR, or one relative to CAR_DIR, and
# print it relative to CAR_DIR for use inside the container.
car_relative() {
  local file="$1"
  if [[ "$file" == "$CAR_DIR"/* ]]; then file="${file#"$CAR_DIR"/}"; fi
  if [[ "$file" == /* ]]; then
    echo "$1 must be inside $CAR_DIR (e.g. $CAR_DIR/models/)" >&2
    exit 1
  fi
  if [[ ! -f "$CAR_DIR/$file" ]]; then
    echo "File not found: $CAR_DIR/$file" >&2
    exit 1
  fi
  echo "$file"
}

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  build)
    tag=$(detect_l4t_tag)
    echo "Building $IMAGE on nvcr.io/nvidia/l4t-jetpack:$tag"
    "$DOCKER" build -f "$REPO_DIR/Dockerfile.jetson" \
      --build-arg "L4T_TAG=$tag" -t "$IMAGE" "$REPO_DIR"
    ;;
  setup)
    if [[ -f "$CAR_DIR/manage.py" ]]; then
      echo "Car folder already exists at $CAR_DIR"
    else
      run_container "donkey createcar --path /root/mycar --template complete"
      echo "Car created at $CAR_DIR. Edit $CAR_DIR/myconfig.py for your hardware."
    fi
    ;;
  check)
    run_container "python3 -c 'import tensorrt, pycuda.autoinit, pycuda.driver as d; \
print(\"TensorRT\", tensorrt.__version__); print(\"GPU\", d.Device(0).name())' \
      && trtexec --version 2>&1 | head -n 1"
    ;;
  engine)
    [[ $# -ge 1 ]] || { echo "usage: ./jetson.sh engine models/pilot_1.onnx" >&2; exit 1; }
    onnx=$(car_relative "$1")
    run_container "bash /root/donkeycar/scripts/build_trt_engine.sh '$onnx'"
    echo "Now drive with: ./jetson.sh drive ${onnx%.onnx}.engine"
    ;;
  drive)
    require_car
    if [[ $# -ge 1 ]]; then
      model=$(car_relative "$1")
      run_container "python3 manage.py drive --model '$model'"
    else
      run_container "python3 manage.py drive"
    fi
    ;;
  shell)
    run_container "bash"
    ;;
  *)
    usage
    [[ -z "$cmd" || "$cmd" == "help" || "$cmd" == "-h" ]] || exit 1
    ;;
esac
