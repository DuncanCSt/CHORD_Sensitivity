#!/usr/bin/env bash
# Build the beamer presentation PDF.
#
# Usage:  ./build.sh          # build build/main.pdf + ../presentation.pdf
#         ./build.sh clean    # remove all build artifacts
#
# Notes on why the env vars below are needed:
#   * PATH      - use the full TeX Live install at /Library/TeX/texbin instead
#                 of the minimal TinyTeX that ships first on PATH (which may be
#                 missing packages the beamer theme depends on).
#   * TEXINPUTS - so pdflatex finds beamerthemecookie.sty (and any assets) that
#                 live alongside main.tex in this folder.
#   * BIBINPUTS / BSTINPUTS - the presentation shares the thesis bibliography
#                 (thesis/references.bib) and its BibTeX style (thesis/apj.bst),
#                 so we add ../thesis to the search path instead of copying them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# The finished PDF is also published to the repo root as presentation.pdf so it
# sits alongside the other top-level deliverables instead of being buried in the
# gitignored build/ dir, and into docs/ so the landing page can embed it -- docs/
# is the GitHub Pages publishing root, so an ../ reference would not resolve once
# the site is deployed. Both copies must be committed to appear on the site.
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
PUBLISHED_PDF="$REPO_ROOT/presentation.pdf"
DOCS_PDF="$REPO_ROOT/docs/presentation.pdf"

export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS=".//:"
export BSTINPUTS=".:../thesis:"
export BIBINPUTS=".:../thesis:"

if [[ "${1:-}" == "clean" ]]; then
    latexmk -C -outdir=build main.tex
    rm -rf build
    rm -f "$PUBLISHED_PDF" "$DOCS_PDF"
    echo "Cleaned build artifacts."
    exit 0
fi

mkdir -p build

# -f forces latexmk to push through compile errors so a full PDF is still
# produced for previewing. latexmk still returns nonzero when it hit errors, so
# we don't treat that as fatal here and instead judge success on whether the PDF
# was actually produced.
latexmk -pdf -f -interaction=nonstopmode -outdir=build main.tex || true

echo
if [[ -f build/main.pdf ]]; then
    cp build/main.pdf "$PUBLISHED_PDF"
    cp build/main.pdf "$DOCS_PDF"
    echo "Built: build/main.pdf"
    echo "Copied to: ${PUBLISHED_PDF#"$REPO_ROOT"/} and ${DOCS_PDF#"$REPO_ROOT"/}"
    echo "(latexmk may report errors above from missing images / unresolved macros;"
    echo " the PDF is still produced for previewing.)"
else
    echo "Build FAILED: no PDF produced. See output above." >&2
    exit 1
fi
