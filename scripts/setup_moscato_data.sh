#!/usr/bin/env bash
# Place the MOSCATO / CMU-MMAC inputs this experiment needs.
#
# Target layout (all of data/ is gitignored):
#
#   data/moscato/
#   ├── videos/       S12_Sandwich_7150991-2470.avi        <- CMU-MMAC, manual download
#   ├── annotations/  gt_annotations_cmu.json              <- MOSCATO release
#   ├── vocabulary/   state_dict.json, files_dict.json     <- MOSCATO release
#   ├── metadata/     inspect_video.json                   <- generated
#   └── work/         decoded frames, clip MP4, features   <- generated
#
# Usage:
#   scripts/setup_moscato_data.sh                 # source dirs default to ~/Downloads
#   MOSCATO_SRC=/path/to/unzipped scripts/setup_moscato_data.sh
#   CMU_AVI=/path/to/S12_Sandwich_7150991-2470.avi scripts/setup_moscato_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO_ROOT/data/moscato"
MOSCATO_SRC="${MOSCATO_SRC:-$HOME/Downloads}"
VIDEO_ID="S12_Sandwich_7150991-2470"

mkdir -p "$DATA"/{videos,annotations,vocabulary,metadata,work}

copy_if_missing() {  # copy_if_missing <src> <dst> <description>
  local src="$1" dst="$2" what="$3"
  if [[ -f "$dst" ]]; then
    echo "  ok       $what -> $dst (already present)"
  elif [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "  copied   $what -> $dst"
  else
    echo "  MISSING  $what: expected at $src"
    return 1
  fi
}

echo "MOSCATO annotations and vocabulary (source: $MOSCATO_SRC)"
missing=0
copy_if_missing \
  "$MOSCATO_SRC/annotations/ground_truth/CMU/gt_annotations_cmu.json" \
  "$DATA/annotations/gt_annotations_cmu.json" \
  "gt_annotations_cmu.json" || missing=1
copy_if_missing \
  "$MOSCATO_SRC/vocabulary/CMU/state_dict.json" \
  "$DATA/vocabulary/state_dict.json" \
  "CMU state_dict.json" || missing=1
copy_if_missing \
  "$MOSCATO_SRC/vocabulary/CMU/files_dict.json" \
  "$DATA/vocabulary/files_dict.json" \
  "CMU files_dict.json" || missing=1

echo
echo "CMU-MMAC source video"
AVI_DST="$DATA/videos/$VIDEO_ID.avi"
if [[ -f "$AVI_DST" ]]; then
  echo "  ok       $VIDEO_ID.avi -> $AVI_DST (already present)"
elif [[ -n "${CMU_AVI:-}" && -f "$CMU_AVI" ]]; then
  cp "$CMU_AVI" "$AVI_DST"
  echo "  copied   $VIDEO_ID.avi -> $AVI_DST"
else
  missing=1
  cat <<EOF
  MISSING  $VIDEO_ID.avi

  This file is NOT part of the MOSCATO release -- MOSCATO only stores the path
  to it. Get it from CMU-MMAC directly:

      http://kitchen.cs.cmu.edu/main.php          (dataset home; registration)

  MOSCATO's files_dict.json records the original location as:

      /mnt/raptor/datasets/egoprocel/video/data/CMU-MMAC/Sandwich/S12_Sandwich_Video/$VIDEO_ID.avi

  so within CMU-MMAC it is subject S12, the Sandwich activity, camera 7150991.
  It must be the untrimmed original: 6714 frames, 30 fps, 800x600. Then either
  drop it at $AVI_DST or re-run with:

      CMU_AVI=/path/to/$VIDEO_ID.avi scripts/setup_moscato_data.sh
EOF
fi

echo
echo "Not downloaded, deliberately: MOSCATO's pre-extracted .npy features."
echo "VidOSC extracts its own features from the video, so they are unused here."
echo "(files_dict feature_len=2239 is MOSCATO's feature stride, NOT a frame count.)"

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Setup incomplete -- see MISSING entries above." >&2
  exit 1
fi

echo
echo "All inputs present. Next:"
echo "    python -m moscato.run_experiment inspect"
