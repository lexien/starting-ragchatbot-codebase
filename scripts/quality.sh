#!/bin/bash
# Run code quality checks for the project.
# Usage: ./scripts/quality.sh [--fix]
#   --fix   Apply formatting fixes automatically (default: check only)

set -e

FIX=false
if [[ "$1" == "--fix" ]]; then
    FIX=true
fi

echo "=== Code Quality Checks ==="

if $FIX; then
    echo ""
    echo ">> Formatting with black..."
    uv run black backend/
else
    echo ""
    echo ">> Checking formatting with black..."
    uv run black --check backend/
fi

echo ""
echo "All checks passed."
