#!/bin/bash
set -e

echo "Configuring build..."
cd /work/build-ucx-run
cmake .. > /dev/null 2>&1

echo "Building tests..."
make test_framed_stream test_phase2_timeout > /dev/null 2>&1

echo ""
echo "========================================================================"
echo "                    PHASE 1 TEST RESULTS"
echo "========================================================================"
./tests/test_framed_stream

echo ""
echo "========================================================================"
echo "                    PHASE 2 TEST RESULTS"
echo "========================================================================"
./tests/test_phase2_timeout

echo ""
echo "========================================================================"
echo "                    ALL TESTS COMPLETED"
echo "========================================================================"
