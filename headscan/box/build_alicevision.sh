#!/usr/bin/env bash
# Build this AliceVision fork on the vast.ai box (after setup_toolchain.sh).
# Two stages, as docker/Dockerfile_ubuntu_deps + docker/Dockerfile:
#   1. deps-only superbuild (Unix Makefiles: SuiteSparse uses $(MAKE)) -> $DEPS
#   2. AliceVision itself (Ninja) against $DEPS -> $AV
#   bash headscan/box/build_alicevision.sh [deps|av|all]      (default all)
# Rebuild after a code change: bash headscan/box/build_alicevision.sh av
set -euo pipefail
STAGE="${1:-all}"
SRC=/workspace/AliceVision
DEPS_BUILD=/workspace/av_deps_build
DEPS=/opt/av/deps
AV_BUILD=/workspace/av_build
AV=/opt/av/install
CUDA_ARCHS="${CUDA_ARCHS:-86;89}"      # 3090 = 8.6, 4090 = 8.9
PY=/usr/bin/python3.11
export PATH=/usr/local/cuda/bin:$PATH CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 FC=/usr/bin/gfortran-12
CORES=$(bash "$SRC/docker/check-cpu.sh")
QUOTA=$(awk '{print int($1/100000)}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 0)
[ "$QUOTA" -gt 0 ] && [ "$CORES" -gt "$QUOTA" ] && CORES=$QUOTA     # nproc lies on vast.ai: use the cgroup quota
echo "cores: $CORES"

if [ "$STAGE" = deps ] || [ "$STAGE" = all ]; then
  mkdir -p "$DEPS/bin" "$DEPS/lib" && ln -sfn lib "$DEPS/lib64"
  export JOBS=$CORES
  cmake -S "$SRC" -B "$DEPS_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DALICEVISION_BUILD_DEPENDENCIES:BOOL=ON -DAV_BUILD_ALICEVISION:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX="$DEPS" -DAV_BUILD_DEPENDENCIES_PARALLEL="$CORES" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY" \
    -DAV_CUDA_CC_LIST="$CUDA_ARCHS" \
    -DAV_BUILD_USD:BOOL=OFF -DAV_BUILD_OPENSUBDIV:BOOL=OFF \
    -DAV_BUILD_CCTAG:BOOL=OFF -DAV_BUILD_POPSIFT:BOOL=OFF \
    -DAV_BUILD_E57FORMAT:BOOL=OFF -DAV_BUILD_XERCESC:BOOL=OFF \
    -DAV_BUILD_PCL:BOOL=OFF -DAV_BUILD_SWIG:BOOL=OFF
  cmake --build "$DEPS_BUILD" --parallel 1
  ls "$DEPS/lib/cmake/opencv4" "$DEPS/lib/cmake/Ceres" >/dev/null && echo "DEPS OK"
fi

if [ "$STAGE" = av ] || [ "$STAGE" = all ]; then
  APRILTAG_DIR=$(dirname "$(find "$DEPS" -name 'apriltagConfig.cmake' | head -1)")
  cmake -S "$SRC" -B "$AV_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS:BOOL=ON -DTARGET_ARCHITECTURE=core \
    -DALICEVISION_BUILD_DEPENDENCIES:BOOL=OFF \
    -DCMAKE_PREFIX_PATH:PATH="$DEPS" -DCMAKE_INSTALL_PREFIX:PATH="$AV" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DALICEVISION_USE_CUDA=ON -DALICEVISION_USE_OPENCV=ON -DALICEVISION_USE_APRILTAG=ON \
    -DALICEVISION_USE_ALEMBIC=ON -DALICEVISION_USE_ONNX=ON -DALICEVISION_USE_ONNX_GPU=OFF \
    -DALICEVISION_USE_CCTAG=OFF -DALICEVISION_USE_POPSIFT=OFF -DALICEVISION_USE_USD=OFF \
    -DALICEVISION_BUILD_LIDAR=OFF -DALICEVISION_BUILD_DOC=OFF -DALICEVISION_BUILD_TESTS=OFF \
    -DALICEVISION_BUILD_SWIG_BINDING=OFF -DPython3_EXECUTABLE="$PY" \
    -DOpenCV_DIR="$DEPS/lib/cmake/opencv4" -DCeres_DIR="$DEPS/lib/cmake/Ceres" \
    -DAlembic_DIR="$DEPS/lib/cmake/Alembic" -Dapriltag_DIR="$APRILTAG_DIR" \
    -DGEOGRAM_INSTALL_PREFIX="$DEPS" -DGEOGRAM_INCLUDE_DIR="$DEPS/include/geogram1" \
    -DSUITESPARSE_INCLUDE_DIR_HINTS="$DEPS/include" -DSUITESPARSE_LIBRARY_DIR_HINTS="$DEPS/lib" \
    -DBLAS_LIBRARIES="$DEPS/lib/libblas.so" -DLAPACK_LIBRARIES="$DEPS/lib/liblapack.so" \
    | tee /workspace/logs/av_configure.log
  grep -E "^\*\* " /workspace/logs/av_configure.log || true
  cmake --build "$AV_BUILD" --target install --parallel "$CORES"
  # data files the build does not fetch
  mkdir -p "$AV/share/aliceVision" && cd "$AV/share/aliceVision"
  [ -f vlfeat_K80L3.SIFT.tree ] || cp /workspace/opt/Meshroom-2023.3.0/aliceVision/share/aliceVision/vlfeat_K80L3.SIFT.tree . 2>/dev/null \
    || wget -q https://gitlab.com/alicevision/trainedVocabularyTreeData/raw/master/vlfeat_K80L3.SIFT.tree
  [ -f fcn_resnet50.onnx ] || cp /workspace/opt/Meshroom-2023.3.0/aliceVision/share/aliceVision/fcn_resnet50.onnx . 2>/dev/null || true
  cat > /opt/av/env.sh <<EOF
export ALICEVISION_ROOT=$AV
export PATH=$AV/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=$AV/lib:$DEPS/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export ALICEVISION_SENSOR_DB=$AV/share/aliceVision/cameraSensors.db
export ALICEVISION_VOCTREE=$AV/share/aliceVision/vlfeat_K80L3.SIFT.tree
export ALICEVISION_SEMANTIC_SEGMENTATION_MODEL=$AV/share/aliceVision/fcn_resnet50.onnx
EOF
  . /opt/av/env.sh && aliceVision_featureExtraction --help 2>&1 | grep -iE "tag16h5|tag36h11" | head -3
  echo "AV OK"
fi
