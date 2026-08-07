#!/usr/bin/env bash
# =============================================================================
# Step 1 — Download the NATS-tss FULL archive (with trained weights) on Omni.
#
# The archive `NATS-tss-v1_0-3ffb9-full.tar` is split into 12 parts (~30 GB
# each, ~360 GB total). We only need 18 architecture files out of 15,625, but
# the split-tar forces downloading everything to reach them.
#
# RUN THIS ON A LOGIN / TRANSFER NODE (Omni compute nodes usually have NO
# internet). Point $DEST at a scratch/work filesystem with >= 750 GB free
# (360 GB parts + up to 360 GB concatenated tar).
#
# Google Drive rate-limits ANONYMOUS downloads of these popular large files
# ("Cannot retrieve the public link ... had many accesses"). Two methods are
# provided; METHOD B (rclone, authenticated) is the robust one.
# =============================================================================
set -euo pipefail

DEST="${DEST:-$PWD/nats_full}"
mkdir -p "$DEST"
cd "$DEST"
echo "Downloading into: $DEST"

FOLDER_ID="17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB"

# 12 split parts + md5 manifest (Google Drive file IDs)
declare -A FILES=(
  ["md5sum.txt"]="1jWnJfLrzBivwszeMU12rxiL4ZZ0m64Ke"
  ["NATS-tss-v1_0-3ffb9-full.taraa"]="1Prrj_BRRL_ePAOgWzB4UNH0154jwCja6"
  ["NATS-tss-v1_0-3ffb9-full.tarab"]="1ky49Q7ORKPZBJheT4QQ7KU4m4UyGo49T"
  ["NATS-tss-v1_0-3ffb9-full.tarac"]="1uZk_7I56t7xWhS_Z6TF0cEtmGJpSlsUx"
  ["NATS-tss-v1_0-3ffb9-full.tarad"]="1OyRDTB3LlZ_1dYkJqFZlgl_7U_pUFNF3"
  ["NATS-tss-v1_0-3ffb9-full.tarae"]="1c9lqiZ6R5l66fJHdLx_cM4HnPG2lvCLX"
  ["NATS-tss-v1_0-3ffb9-full.taraf"]="1XFHGkBu8SGg32bKXMVClPD5VaWN6csPG"
  ["NATS-tss-v1_0-3ffb9-full.tarag"]="107tr9zodkIUVjpF6_VQsZT4oCmNb5XAw"
  ["NATS-tss-v1_0-3ffb9-full.tarah"]="1M6y0wGvN5I0jp8tpDdIiE4vGMUADKNC6"
  ["NATS-tss-v1_0-3ffb9-full.tarai"]="1ysjCOogC5MBcJoxkmagLjza9kERIL5AM"
  ["NATS-tss-v1_0-3ffb9-full.taraj"]="1J-14xWUpqpr4gsQAMPYOUqEKIVQsWrHw"
  ["NATS-tss-v1_0-3ffb9-full.tarak"]="1ndznkqEa9rFF678aYeDmJIqDIuXw7bhk"
  ["NATS-tss-v1_0-3ffb9-full.taral"]="1UmOdaWQ-gSCclUKhQNyuRs6uqr_gj49K"
)

METHOD="${METHOD:-A}"   # A = gdown, B = rclone

if [[ "$METHOD" == "A" ]]; then
  # -------- METHOD A: gdown (simplest; may hit anonymous quota) --------------
  # pip install --user gdown
  echo "== METHOD A: gdown =="
  # Try grabbing the whole folder first (also fetches md5sum.txt):
  gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O . || true
  # Fall back to per-file with resume for anything missing:
  for name in "${!FILES[@]}"; do
    if [[ ! -s "$name" ]]; then
      echo "-- fetching $name"
      gdown --continue "${FILES[$name]}" -O "$name" || \
        echo "!! FAILED $name (quota?) — use METHOD B (rclone)"
    fi
  done

elif [[ "$METHOD" == "B" ]]; then
  # -------- METHOD B: rclone (authenticated; beats the quota) ----------------
  # One-time on a machine with a browser:
  #   rclone config           # create a remote called 'gdrive' (Google Drive)
  # Then copy rclone.conf to the cluster (~/.config/rclone/rclone.conf).
  #
  # Give your Google account access: open the shared folder in the browser and
  # "Add shortcut to Drive", then it appears under your drive root.
  REMOTE="${REMOTE:-gdrive}"
  echo "== METHOD B: rclone from ${REMOTE} =="
  # If you added a shortcut named 'NATS-tss-v1_0-3ffb9-full' to My Drive:
  rclone copy "${REMOTE}:NATS-tss-v1_0-3ffb9-full" . \
      --drive-shared-with-me -P --transfers 4 --checkers 8
fi

echo
echo "== Verifying md5 checksums =="
if [[ -s md5sum.txt ]]; then
  # md5sum.txt lines look like:  <name> : <md5>
  awk '/full.tar/ {print $3"  "$1}' md5sum.txt > .check.md5
  md5sum -c .check.md5 || { echo "MD5 MISMATCH — re-download failed parts"; exit 1; }
  echo "All parts verified OK."
else
  echo "WARNING: md5sum.txt missing; cannot verify."
fi

echo "Done. Next: bash 02_extract.sh"
