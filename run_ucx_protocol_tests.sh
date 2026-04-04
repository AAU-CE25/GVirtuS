#!/usr/bin/env bash
set -euo pipefail

apt-get update -y >/dev/null
apt-get install -y libucx-dev libucx0 zlib1g-dev >/dev/null

cmake -S /work -B /work/build-ucx-tests
cmake --build /work/build-ucx-tests --target test_framed_stream test_stream_timeout -j1

export LD_LIBRARY_PATH=/work/build-ucx-tests:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

echo "===== RUN test_framed_stream ====="
/work/build-ucx-tests/tests/test_framed_stream --gtest_color=no

echo "===== RUN test_stream_timeout ====="
/work/build-ucx-tests/tests/test_stream_timeout --gtest_color=no
