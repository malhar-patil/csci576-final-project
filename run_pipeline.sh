#!/usr/bin/env bash
# End-to-end pipeline driver.
#
# Takes one input video, runs the video segmenter (Justin), the audio
# pipelines (Bhuvan v1 + SSM), fuses their predictions, and writes a
# .seg file next to the video so the Malhar Qt player can load it.
#
# Usage:
#   ./run_pipeline.sh <video.mp4> [output_dir]
#
# Stages and intermediate artifacts (written to output_dir, default
# video_info/pred/<stem>/):
#   1. Justin video segmenter         -> <stem>.seg                  (next to video, raw video pred)
#   2. seg_to_json.py                  -> justin.json
#   3. pipeline.py (audio v1)          -> audio_v1.json
#   4. pipeline_ssm.py (audio SSM)     -> audio_ssm.json
#   5. fuse_smart.py (3-way fuse)      -> fused.json
#   6. json_to_seg.py                  -> <stem>_merged_final.seg     (next to video, merged result the player loads)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <video.mp4> [output_dir]" >&2
  exit 1
fi

INPUT_VIDEO_PATH="$1"
if [[ ! -f "$INPUT_VIDEO_PATH" ]]; then
  echo "error: video not found: $INPUT_VIDEO_PATH" >&2
  exit 1
fi

REPO_ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_DIR="$(cd "$(dirname "$INPUT_VIDEO_PATH")" && pwd)"
VIDEO_BASE_NAME="$(basename "$INPUT_VIDEO_PATH")"
VIDEO_STEM="${VIDEO_BASE_NAME%.*}"
ABSOLUTE_VIDEO_PATH="$VIDEO_DIR/$VIDEO_BASE_NAME"

INTERMEDIATE_DIR="${2:-$REPO_ROOT_DIR/video_info/pred/$VIDEO_STEM}"
mkdir -p "$INTERMEDIATE_DIR"

# Pick a Python interpreter — prefer the repo venv if present.
PYTHON_BIN="python3"
if [[ -x "$REPO_ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT_DIR/venv/bin/python"
fi

JUSTIN_BINARY_PATH="$REPO_ROOT_DIR/src/segment/segment"
if [[ ! -x "$JUSTIN_BINARY_PATH" ]]; then
  echo "error: justin segmenter binary not built: $JUSTIN_BINARY_PATH" >&2
  echo "       build it with: (cd src/segment && make)" >&2
  exit 1
fi

JUSTIN_RAW_SEG_PATH="$VIDEO_DIR/$VIDEO_STEM.seg"
JUSTIN_PRED_JSON_PATH="$INTERMEDIATE_DIR/justin.json"
AUDIO_V1_JSON_PATH="$INTERMEDIATE_DIR/audio_v1.json"
AUDIO_SSM_JSON_PATH="$INTERMEDIATE_DIR/audio_ssm.json"
FUSED_JSON_PATH="$INTERMEDIATE_DIR/fused.json"
FINAL_SEG_PATH="$VIDEO_DIR/${VIDEO_STEM}_merged_final.seg"

echo "[1/6] Running Justin video segmenter on $ABSOLUTE_VIDEO_PATH" >&2
"$JUSTIN_BINARY_PATH" "$ABSOLUTE_VIDEO_PATH"

echo "[2/6] Converting .seg -> unified JSON" >&2
"$PYTHON_BIN" "$REPO_ROOT_DIR/seg_to_json.py" "$JUSTIN_RAW_SEG_PATH" --output "$JUSTIN_PRED_JSON_PATH"

echo "[3/6] Running audio pipeline v1" >&2
"$PYTHON_BIN" "$REPO_ROOT_DIR/pipeline.py" "$ABSOLUTE_VIDEO_PATH" --output "$AUDIO_V1_JSON_PATH"

echo "[4/6] Running audio pipeline SSM" >&2
"$PYTHON_BIN" "$REPO_ROOT_DIR/pipeline_ssm.py" "$ABSOLUTE_VIDEO_PATH" --output "$AUDIO_SSM_JSON_PATH"

echo "[5/6] Fusing predictions" >&2
"$PYTHON_BIN" "$REPO_ROOT_DIR/fuse_smart.py" \
  --video "$JUSTIN_PRED_JSON_PATH" \
  --detector "$AUDIO_V1_JSON_PATH" \
  --detector "$AUDIO_SSM_JSON_PATH" \
  --output "$FUSED_JSON_PATH"

echo "[6/6] Writing fused .seg next to video for Malhar player: $FINAL_SEG_PATH" >&2
"$PYTHON_BIN" "$REPO_ROOT_DIR/json_to_seg.py" "$FUSED_JSON_PATH" --output "$FINAL_SEG_PATH"

echo >&2
echo "done." >&2
echo "  video : $ABSOLUTE_VIDEO_PATH" >&2
echo "  seg   : $FINAL_SEG_PATH" >&2
echo "  intermediates: $INTERMEDIATE_DIR" >&2
