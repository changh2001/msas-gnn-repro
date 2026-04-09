#!/usr/bin/env bash
set -euo pipefail

# High-fidelity LaTeX -> DOCX pipeline for MSAS-GNN thesis.
#
# Usage:
#   bash scripts/tex2docx.sh
#   bash scripts/tex2docx.sh --reference-doc word-build/reference.docx
#   bash scripts/tex2docx.sh --main main.tex --out word-build/main-final.docx

MAIN_TEX="main.tex"
OUT_DOCX="word-build/main-final.docx"
REFERENCE_DOC=""
KEEP_INTERMEDIATE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --main)
      MAIN_TEX="$2"
      shift 2
      ;;
    --out)
      OUT_DOCX="$2"
      shift 2
      ;;
    --reference-doc)
      REFERENCE_DOC="$2"
      shift 2
      ;;
    --keep-intermediate)
      KEEP_INTERMEDIATE="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$MAIN_TEX" ]]; then
  echo "Cannot find main tex: $MAIN_TEX" >&2
  exit 1
fi

mkdir -p "word-build"

EXPANDED_TEX="word-build/main-expanded.tex"
CLEAN_TEX="word-build/main-clean.tex"
WARN_LOG="word-build/pandoc-warnings.log"

echo "[1/4] Expanding multi-file LaTeX..."
latexpand --empty-comments "$MAIN_TEX" > "$EXPANDED_TEX"

echo "[2/4] Cleaning TeX for Pandoc math parser..."
python3 "scripts/tex2docx_clean.py" --infile "$EXPANDED_TEX" --outfile "$CLEAN_TEX"

echo "[3/4] Converting to DOCX via Pandoc..."
PANDOC_ARGS=(
  "$CLEAN_TEX"
  --from=latex+raw_tex
  --to=docx
  --output="$OUT_DOCX"
  --resource-path=".:figures:data"
  --citeproc
  --bibliography="ref/refs.bib"
  --number-sections
  --top-level-division=chapter
)

if [[ -n "$REFERENCE_DOC" ]]; then
  if [[ ! -f "$REFERENCE_DOC" ]]; then
    echo "Cannot find reference doc: $REFERENCE_DOC" >&2
    exit 1
  fi
  PANDOC_ARGS+=(--reference-doc="$REFERENCE_DOC")
fi

# Capture warnings for later checklist-based proofreading.
# Avoid process substitution for better shell/sandbox compatibility.
pandoc "${PANDOC_ARGS[@]}" 2> "$WARN_LOG"

if [[ -s "$WARN_LOG" ]]; then
  echo "Pandoc warnings:"
  sed 's/^/  /' "$WARN_LOG" >&2
fi

echo "[4/4] Done."
echo "Output: $OUT_DOCX"
echo "Warnings: $WARN_LOG"

if [[ "$KEEP_INTERMEDIATE" != "1" ]]; then
  rm -f "$EXPANDED_TEX" "$CLEAN_TEX"
  echo "Intermediate files removed."
else
  echo "Intermediate files kept:"
  echo "  - $EXPANDED_TEX"
  echo "  - $CLEAN_TEX"
fi
