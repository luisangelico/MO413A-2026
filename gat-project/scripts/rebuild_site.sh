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
echo "==> [1/5] python -m src.evaluate"
python -m src.evaluate

echo
echo "==> [2/5] python -m src.interpret"
python -m src.interpret

echo
echo "==> [3/5] python -m src.interpret_viz --html"
python -m src.interpret_viz --html --site-dir "${SITE_OUT}/interpret"

echo
echo "==> [4/5] python -m src.embeddings"
python -m src.embeddings --site-dir "${SITE_OUT}"

echo
echo "==> [5/5] python -m src.pseudotime"
python -m src.pseudotime --site-dir "${SITE_OUT}"

# ---------- 4. Cross-seed stability ------------------------------------------

echo
echo "==> python -m src.multiseed_stability --dir ${MULTISEED}"
python -m src.multiseed_stability --dir "${MULTISEED}"

# (multiseed_stability writes site/stability.html into <repo>/site by default;
#  if SITE_OUT is different, copy it across so all links resolve in one place.)
if [[ "${SITE_OUT}" != "site" && -f "site/stability.html" ]]; then
  mkdir -p "${SITE_OUT}"
  cp -f site/stability.html "${SITE_OUT}/stability.html"
  echo "==> copied site/stability.html -> ${SITE_OUT}/stability.html"
fi

# ---------- 5. Build the predictions site ------------------------------------

echo
echo "==> python -m scripts.build_static_site --out ${SITE_OUT}"
python -m scripts.build_static_site --out "${SITE_OUT}"

# ---------- Done -------------------------------------------------------------

echo
echo "==> rebuild complete"
echo "    canonical run dir: ${DEST_RUN_DIR}"
echo "    site root:         ${SITE_OUT}/index.html"
echo
echo "    open in browser:"
echo "      python -m http.server --directory ${SITE_OUT} 8000"
echo "      then visit http://localhost:8000"
