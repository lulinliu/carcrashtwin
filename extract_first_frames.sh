#!/usr/bin/env bash
# save as extract_first_frames.sh
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --list VIDEO_LIST_TXT --out OUTPUT_DIR [--quality Q]
  --list     TXT file; each line is an absolute or relative path to a video
  --out      Output directory to put extracted JPGs
  --quality  JPEG quality 2..31 (lower = better). Default: 2
Examples:
  $0 --list videos.txt --out 1022VIDEO_firstframes
EOF
}

LIST=""
OUT=""
Q=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --quality) Q="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -z "$LIST" || -z "$OUT" ]] && { usage; exit 1; }

# Check dependencies
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not found in PATH"; exit 1; }

mkdir -p "$OUT"

# Read list and extract first frame
# - Use IFS= read -r to preserve spaces
# - Unique output name by replacing '/' with '__'
count=0
while IFS= read -r vid || [[ -n "$vid" ]]; do
  # skip blank lines / comments
  [[ -z "${vid// }" || "${vid:0:1}" == "#" ]] && continue

  if [[ ! -f "$vid" ]]; then
    echo "WARN: not a file -> $vid"
    continue
  fi

  # Build output filename: replace '/' with '__' and strip extension
  base="$(basename "$vid")"
  stem="${base%.*}"
  # If you prefer flattening the *entire path* to avoid duplicates:
  # flat="$(echo "$vid" | sed 's/[\/\\]/__/g')"
  # stem="${flat%.*}"

  out_jpg="$OUT/$stem.jpg"

  # Extract first frame
  # -frames:v 1 grabs the first frame
  # -q:v $Q sets JPEG quality (2 is high quality)
  ffmpeg -hide_banner -loglevel error -y -i "$vid" -frames:v 1 -q:v "$Q" "$out_jpg" \
    && echo "OK  -> $out_jpg" \
    || echo "FAIL-> $vid"

  ((count++)) || true
done < "$LIST"

echo "Done. Extracted $count frame(s) into: $OUT"
