#!/usr/bin/env bash
# Rebuild the static site end-to-end against the newest model.
#
# Strategy:
#   1. Pick the best seed from the latest multiseed_* run (lowest val_loss).
#   2. Promote it to a canonical run_multiseed_seed{N}/ directory so all
#      single-run scripts (interpret, embeddings, pseudotime) pick it up.
#   3. Regenerate evaluation, interpretability, embeddings, pseudotime.
#   4. Run cross-seed stability against the multiseed dir.
#   5. Build the predictions site into site/ (links to all the above).
#
# Run from the gat-project/ directory:
#   bash scripts/rebuild_site.sh
#
# Override the multiseed dir or output dir:
#   MULTISEED=data/processed/multiseed_20260530_145425 bash scripts/rebuild_site.sh
#   SITE_OUT=docs bash scripts/rebuild_site.sh

set -euo pipefail

# ---------- Configuration ----------------------------------------------------

PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
SITE_OUT="${SITE_OUT:-site}"

# Resolve SITE_OUT relative to the repo root if it is a bare name like
# "docs" or "site". build_static_site.py defaults to <repo-root>/docs/, so
# normalize all output paths to that same root to keep trees in sync.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
case "${SITE_OUT}" in
  /*) ;;  # already absolute
  *)
    if [[ -d "${PARENT_ROOT}/${SITE_OUT}" ]]; then
      SITE_OUT="${PARENT_ROOT}/${SITE_OUT}"
    else
      SITE_OUT="${REPO_ROOT}/${SITE_OUT}"
    fi
    ;;
esac
echo "==> resolved SITE_OUT: ${SITE_OUT}"
mkdir -p "${SITE_OUT}"

# Locate the multiseed dir (latest by mtime if not provided).
if [[ -z "${MULTISEED:-}" ]]; then
  MULTISEED=$(ls -dt "${PROCESSED_DIR}"/multiseed_* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "${MULTISEED}" || ! -d "${MULTISEED}" ]]; then
  echo "error: no multiseed_* directory found in ${PROCESSED_DIR}/"
  echo "       set MULTISEED=path/to/multiseed_<timestamp> to override"
  exit 1
fi

echo "==> multiseed dir: ${MULTISEED}"

# ---------- 1. Pick best seed by val_loss ------------------------------------

if [[ ! -f "${MULTISEED}/results.json" ]]; then
  echo "error: ${MULTISEED}/results.json not found — cannot pick best seed"
  exit 1
fi

BEST_SEED=$(python -c "
import json, sys
with open('${MULTISEED}/results.json') as f:
    rows = json.load(f)
best = min(rows, key=lambda r: r['val_loss'])
print(best['seed'])
")
echo "==> best seed (lowest val_loss): seed_${BEST_SEED}"

SRC_SEED_DIR="${MULTISEED}/seed_${BEST_SEED}"
DEST_RUN_DIR="${PROCESSED_DIR}/run_multiseed_seed${BEST_SEED}"

# ---------- 2. Promote that seed to a canonical run dir ----------------------

mkdir -p "${DEST_RUN_DIR}"
echo "==> copying ${SRC_SEED_DIR}/ -> ${DEST_RUN_DIR}/"
for f in best_model.pt best_model.json splits.npz history.npz; do
  if [[ -f "${SRC_SEED_DIR}/${f}" ]]; then
    cp -f "${SRC_SEED_DIR}/${f}" "${DEST_RUN_DIR}/${f}"
  fi
done

# Touch the dir so it's the most-recent run_* (latest-mtime auto-pick works)
touch "${DEST_RUN_DIR}"

# ---------- 3. Regenerate single-run artifacts -------------------------------

echo
echo "==> [1/7] python -m src.evaluate"
python -m src.evaluate

echo
echo "==> [2/7] python -m src.interpret"
python -m src.interpret

echo
echo "==> [3/7] python -m src.interpret_viz --html"
python -m src.interpret_viz --html --site-dir "${SITE_OUT}/interpret"

echo
echo "==> [4/7] python -m src.embeddings"
python -m src.embeddings --site-dir "${SITE_OUT}"

echo
echo "==> [5/7] python -m src.pseudotime"
python -m src.pseudotime --site-dir "${SITE_OUT}"

echo
echo "==> [6/7] python -m src.dataset_page"
python -m src.dataset_page --site-dir "${SITE_OUT}"

echo
echo "==> [7/7] python -m src.about_gat_page (en + pt-br)"
python -m src.about_gat_page --site-dir "${SITE_OUT}" --lang en
python -m src.about_gat_page --site-dir "${SITE_OUT}" --lang pt-br

# ---------- 4. Cross-seed stability ------------------------------------------

echo
echo "==> python -m src.multiseed_stability --dir ${MULTISEED}"
python -m src.multiseed_stability --dir "${MULTISEED}" --site-dir "${SITE_OUT}"

echo
echo "==> python -m scripts.biomarker_report --run ${MULTISEED}"
python -m scripts.biomarker_report --run "${MULTISEED}" --out "${SITE_OUT}/biomarkers.html"


# ---------- 5. Build the predictions site ------------------------------------

echo
echo "==> python -m scripts.build_static_site --out ${SITE_OUT}"
python -m scripts.build_static_site --out "${SITE_OUT}"

# ---------- 6. Generate the Portuguese (pt-br) mirror ------------------------

echo
echo "==> python -m src.translate_site --site ${SITE_OUT}"
python -m src.translate_site --site "${SITE_OUT}"

# ---------- Done -------------------------------------------------------------

echo
echo "==> rebuild complete"
echo "    canonical run dir: ${DEST_RUN_DIR}"
echo "    site root:         ${SITE_OUT}/index.html"
echo
echo "    open in browser:"
echo "      python -m http.server --directory ${SITE_OUT} 8000"
echo "      then visit http://localhost:8000"
