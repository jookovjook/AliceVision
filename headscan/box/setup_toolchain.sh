#!/usr/bin/env bash
# Toolchain for building AliceVision on a plain Ubuntu 22.04 + CUDA 12.4 devel box (vast.ai base image).
# Mirrors docker/Dockerfile_ubuntu_deps (Kitware CMake, GCC 12, Python 3.11) plus the tools the headscan scripts need.
#   bash headscan/box/setup_toolchain.sh
# SKIP_PY311=1: no deadsnakes Python 3.11 (only the AliceVision build needs it; the venv gets Python from uv), for when
# the PPA is down (503) and the prebuilt AliceVision is restored instead of built.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
CMAKE_VERSION=3.31.11
UBU=$(. /etc/os-release && echo "$VERSION_ID")
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")

apt-get update
apt-get install -y --no-install-recommends ca-certificates gpg wget software-properties-common
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor > /usr/share/keyrings/kitware-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $CODENAME main" > /etc/apt/sources.list.d/kitware.list
PY311=(python3.11 python3.11-dev python3.11-distutils)
if [ -n "${SKIP_PY311:-}" ]; then PY311=(); else add-apt-repository -y ppa:deadsnakes/ppa; fi
apt-get update
apt-get install -y --no-install-recommends "${PY311[@]}" \
    automake bison build-essential \
    cmake=${CMAKE_VERSION}-0kitware1ubuntu${UBU}.1 cmake-data=${CMAKE_VERSION}-0kitware1ubuntu${UBU}.1 \
    cpp-12 curl file g++-12 gcc-12 gfortran-12 git libpcre2-dev libssl-dev libtool \
    libxerces-c-dev libyaml-cpp-dev nasm pkg-config \
    texinfo unzip yasm ninja-build \
    ffmpeg rsync htop tmux libgl1 libegl1 libglib2.0-0 libglu1-mesa libosmesa6
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12 --slave /usr/bin/gcov gcov /usr/bin/gcov-12 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-12

if [ -z "${SKIP_PY311:-}" ]; then
  curl -sSfL -o /tmp/pip.pyz https://bootstrap.pypa.io/pip/zipapp/pip-25.2.pyz
  python3.11 /tmp/pip.pyz install -q "pip==25.2" "numpy==1.26.4"
fi

# Python env for the headscan scripts (separate from the build Python)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 /workspace/hsvenv -q
uv pip install --python /workspace/hsvenv/bin/python -q numpy scipy opencv-python-headless pillow matplotlib trimesh open3d pupil-apriltags pyrender "PyOpenGL==3.1.7" "networkx>=3" OpenEXR

gcc --version | head -1; cmake --version | head -1; /workspace/hsvenv/bin/python -c "import cv2, open3d, pupil_apriltags; print('py ok', cv2.__version__, open3d.__version__)"
