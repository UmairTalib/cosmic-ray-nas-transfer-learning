#!/usr/bin/env bash
# =============================================================================
# Step 2 — Reassemble the split tar and extract ONLY the 18 needed files.
#
# We do NOT need all 15,625 architecture files — only the 18 our predictor
# selected (listed in needed_indices.txt). Their internal names inside the tar
# are 000137.pickle.pbz2, 002352.pickle.pbz2, ... (6-digit index).
#
# Run from the same dir as 01_download.sh output ($DEST/nats_full by default).
# =============================================================================
set -euo pipefail

DEST="${DEST:-$PWD/nats_full}"
HANDOFF_DIR="${HANDOFF_DIR:-$PWD}"          # where needed_indices.txt lives
OUT="${OUT:-$DEST/needed_full}"             # extracted 18 files land here
cd "$DEST"

IDX_FILE="$HANDOFF_DIR/needed_indices.txt"
[[ -s "$IDX_FILE" ]] || { echo "Missing $IDX_FILE"; exit 1; }
mkdir -p "$OUT"

# Build a grep pattern like: (000137|002352|...)
PATTERN="$(paste -sd'|' "$IDX_FILE")"
echo "Looking for indices: $PATTERN"

# --- Option 1 (default): stream the parts through tar, no 2nd 360 GB copy ----
# List members first to discover the exact internal paths, then extract them.
echo "== Listing tar to locate the 18 members (streamed) =="
cat NATS-tss-v1_0-3ffb9-full.tara* | tar -tf - \
    | grep -E "(${PATTERN})\.pickle\.pbz2$" | tee members.txt

COUNT=$(wc -l < members.txt)
echo "Found $COUNT / 18 members."
[[ "$COUNT" -ge 1 ]] || { echo "No members matched — inspect 'tar -tf' output naming."; exit 1; }

echo "== Extracting those members =="
cat NATS-tss-v1_0-3ffb9-full.tara* | tar -xvf - -C "$OUT" -T members.txt
echo "Extracted into: $OUT"

# --- Option 2 (if streaming twice is slow): concatenate once, then extract ---
# cat NATS-tss-v1_0-3ffb9-full.tara* > full.tar     # needs ~360 GB more space
# tar -tf full.tar | grep -E "(${PATTERN})\.pickle\.pbz2$" > members.txt
# tar -xvf full.tar -C "$OUT" -T members.txt

echo "Done. Next: python 03_extract_state_dicts.py --full-dir \"$OUT\""
