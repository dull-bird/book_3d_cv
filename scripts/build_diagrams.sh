#!/bin/bash
# Build all diagrams (TikZ + Python) for the book.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building TikZ diagrams ==="
for tex in "$PROJECT_DIR"/assets/diagrams/tikz/*.tex; do
    [ -f "$tex" ] || continue
    echo "  $(basename "$tex")"
    "$SCRIPT_DIR/tikz2svg.sh" "$tex"
done

echo "=== Building Python charts ==="
source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
for py in "$PROJECT_DIR"/assets/diagrams/python/*.py; do
    [ -f "$py" ] || continue
    echo "  $(basename "$py")"
    python "$py"
done

echo "=== Done ==="
