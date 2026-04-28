#!/bin/bash
# Compile TikZ .tex file to SVG for mdBook embedding.
# Requires: xelatex (with xeCJK), pdftocairo

TEX_FILE="$1"
if [ -z "$TEX_FILE" ] || [ ! -f "$TEX_FILE" ]; then
    echo "Usage: tikz2svg.sh <path/to/file.tex>"
    exit 1
fi

# Resolve absolute paths
TEX_FILE="$(cd "$(dirname "$TEX_FILE")" && pwd)/$(basename "$TEX_FILE")"
DIR=$(dirname "$TEX_FILE")
NAME=$(basename "$TEX_FILE" .tex)
WORKDIR="/tmp/tikz_$$"

mkdir -p "$WORKDIR"
cp "$TEX_FILE" "$WORKDIR/"

cd "$WORKDIR"
if ! xelatex -interaction=nonstopmode "$NAME.tex" > /dev/null 2>&1; then
    echo "ERROR: xelatex failed for $NAME"
    cat "$NAME.log" 2>/dev/null | grep -i "error" | head -5
    rm -rf "$WORKDIR"
    exit 1
fi

if ! pdftocairo -svg "$WORKDIR/$NAME.pdf" "$DIR/$NAME.svg" 2>/dev/null; then
    echo "ERROR: pdftocairo failed for $NAME"
    rm -rf "$WORKDIR"
    exit 1
fi

cp "$WORKDIR/$NAME.pdf" "$DIR/$NAME.pdf"
rm -rf "$WORKDIR"
echo "Done: $DIR/$NAME.svg"
