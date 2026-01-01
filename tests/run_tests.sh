#!/bin/bash
# ============================================
#  Football Analysis - Test Runner
# ============================================

set -e

# Move to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo ""
echo "============================================"
echo "  Running Tests"
echo "============================================"
echo ""

python -m pytest tests/ -v --tb=short "$@"

echo ""
echo "============================================"
echo "  All tests passed!"
echo "============================================"
echo ""
