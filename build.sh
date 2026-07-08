#!/usr/bin/env bash
# Build the thesis PDF.
#
# Usage:  ./build.sh          # build main.pdf
#         ./build.sh clean    # remove all build artifacts
#
# Notes on why the env vars below are needed:
#   * PATH      - use the full TeX Live install at /Library/TeX/texbin instead
#                 of the minimal TinyTeX that ships first on PATH (which is
#                 missing packages like fancyhdr).
#   * TEXINPUTS - the project is laid out for Overleaf, which searches every
#                 folder. main.tex uses root-relative \include{thesis/...} paths
#                 while the class file and plots/ live inside thesis/, so we add
#                 thesis/ (and its subfolders) to the search path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS=".//:./thesis//:"
export BSTINPUTS=".:./thesis:"
export BIBINPUTS=".:./thesis:"

if [[ "${1:-}" == "clean" ]]; then
    latexmk -C thesis/main.tex
    rm -rf build
    echo "Cleaned build artifacts."
    exit 0
fi

# \include{thesis/...} makes pdflatex write .aux files into matching subfolders
# of the output dir (e.g. build/thesis/0_preamble/preamble.aux). pdflatex cannot
# create these nested dirs itself and latexmk only makes one level at a time, so
# we pre-create every subfolder referenced by an \include in main.tex.
mkdir -p build/thesis/0_preamble \
         build/thesis/1_Background \
         build/thesis/appendices \
         build/thesis

# -f forces latexmk to push through compile errors so a full PDF (with TOC and
# resolved citations) is still produced for previewing. latexmk still returns
# nonzero when it hit errors, so we don't treat that as fatal here and instead
# judge success on whether the PDF was actually produced.
latexmk -pdf -f -interaction=nonstopmode -outdir=build thesis/main.tex || true

echo
if [[ -f build/main.pdf ]]; then
    echo "Built: build/main.pdf"
    echo "(latexmk may report errors above from missing images / unresolved macros;"
    echo " the PDF is still produced for previewing.)"
else
    echo "Build FAILED: no PDF produced. See output above." >&2
    exit 1
fi
