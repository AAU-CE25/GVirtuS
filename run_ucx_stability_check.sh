#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
apt-get install -y -qq libucx0 libucx-dev zlib1g-dev >/dev/null
cmake -S /work -B /work/build-ucx-stability >/dev/null
cmake --build /work/build-ucx-stability --target test_framed_stream test_stream_timeout -j1 >/dev/null
export LD_LIBRARY_PATH=/work/build-ucx-stability:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

fails=0
run_case() {
  local bin="$1"
  local filter="$2"
  if timeout 40s "$bin" --gtest_color=no --gtest_filter="$filter" >/dev/null 2>&1; then
    echo "PASS $filter"
  else
    echo "FAIL $filter"
    fails=$((fails+1))
  fi
}

run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.ContiguousStreamRoundTripWorks
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.SendAndRecvFrameRoundTrip
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.JunkPrefixResyncsAndRecovers
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.ErrorFrameRoundTripCarriesCudaCodeAndRequestSeq
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.ResponseFrameEchoesRequestSeqAndResultLen
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.ResponseSeqMismatchThrows
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.Fault*ExecutionError*
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.Fault*CorruptedByte*
run_case /work/build-ucx-stability/tests/test_framed_stream UcxLoopbackFixture.Fault*TruncatedPayload*

run_case /work/build-ucx-stability/tests/test_stream_timeout StreamTimeoutFixture.SendCompletesWithoutTimeout
run_case /work/build-ucx-stability/tests/test_stream_timeout StreamTimeoutFixture.RecvTimeoutWhenNoPeerData
run_case /work/build-ucx-stability/tests/test_stream_timeout StreamTimeoutFixture.RecvCompletesWithinTimeout
run_case /work/build-ucx-stability/tests/test_stream_timeout StreamTimeoutFixture.NoIndefiniteHangOnMissingPeer

if timeout 40s /work/build-ucx-stability/tests/test_framed_stream --gtest_color=no >/dev/null 2>&1; then
  echo "PASS ORDER test_framed_stream"
else
  echo "FAIL ORDER test_framed_stream"
  fails=$((fails+1))
fi

if timeout 40s /work/build-ucx-stability/tests/test_stream_timeout --gtest_color=no >/dev/null 2>&1; then
  echo "PASS ORDER test_stream_timeout"
else
  echo "FAIL ORDER test_stream_timeout"
  fails=$((fails+1))
fi

echo "TOTAL_FAILS $fails"
exit $fails
