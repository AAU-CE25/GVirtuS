#!/bin/bash
# One-time setup for opencv-dnn: download the 4 ONNX models referenced in main.cu.
# The ImageNet test set (1000 images with <classid>_*.jpg names) is NOT downloaded
# here because ImageNet validation is not redistributable; place it manually as
# ./imagenet_test_1000/ . A 5-image smoke set can be derived from images already
# in this repo (see fallback at the bottom).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

echo "=== downloading ONNX models from github.com/onnx/models ==="
BASE=https://github.com/onnx/models/raw/main/validated/vision/classification

declare -A MODELS=(
  [mobilenetv2-10.onnx]="${BASE}/mobilenet/model/mobilenetv2-10.onnx"
  [squeezenet1.1-7.onnx]="${BASE}/squeezenet/model/squeezenet1.1-7.onnx"
  [resnet18-v1-7.onnx]="${BASE}/resnet/model/resnet18-v1-7.onnx"
  [vgg16-7.onnx]="${BASE}/vgg/model/vgg16-7.onnx"
)

for name in "${!MODELS[@]}"; do
  if [ -f "$name" ] && [ -s "$name" ]; then
    echo "  [skip] $name already present ($(du -h "$name" | cut -f1))"
    continue
  fi
  echo "  [get ] $name"
  wget -q --show-progress -O "$name" "${MODELS[$name]}" || {
    echo "  [WARN] failed to download $name from primary URL, trying mirror..."
    rm -f "$name"
    # Fallback: huggingface mirrors keep these too
    wget -q --show-progress -O "$name" "https://huggingface.co/onnx/${name%.onnx}/resolve/main/${name}" || \
      echo "  [FAIL] could not fetch $name - check URL or fetch manually"
  }
done

echo ""
echo "=== checking imagenet_test_1000/ ==="
if [ -d imagenet_test_1000 ] && [ "$(ls -A imagenet_test_1000 2>/dev/null | wc -l)" -gt 0 ]; then
  N=$(find imagenet_test_1000 -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
  echo "  found $N images"
else
  echo "  MISSING - ImageNet validation set not bundled (not redistributable)."
  echo "  Place 1000 images in ./imagenet_test_1000/ with names matching"
  echo "  <classid>_<anything>.<ext> (e.g.,  207_cat.jpg  for class 207)."
  echo "  Or grab the ILSVRC2012 validation set from a mirror you have access to."
fi

echo ""
echo "=== summary ==="
ls -lh *.onnx 2>/dev/null
