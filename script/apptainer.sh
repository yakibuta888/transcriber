#!/bin/bash
# usage: ./transcriber.sh <workdir> <model> <audiofile> <outname> [diarize]

if [ "$#" -lt 4 ]; then
  echo "使い方: $0 <ディレクトリ名> <model名> <音声ファイル名> <出力名> [diarize]"
  exit 1
fi

WORKDIR="$1"
MODEL="$2"
AUDIOFILE="$3"
OUTNAME="$4"
OPTIONAL="$5"
SHARED_DIR="/mnt/data/$WORKDIR"

# 現在のホストユーザー名を取得
HOST_USER=$(whoami)

# diarizeオプション判定
DIARIZE_ARG=""
if [ "$OPTIONAL" = "diarize" ]; then
  DIARIZE_ARG="--diarize"
fi

export APPTAINER_CACHEDIR="$SHARED_DIR/apptainer_cache"
export APPTAINER_TMPDIR="$SHARED_DIR/tmp"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer cache clean --force

export APPTAINERENV_PYTHONPATH="/app/src/models/DeepFilterNet-0.5.6/DeepFilterNet:/app/vendor:/app:$PYTHONPATH"

nohup apptainer exec --nv --net --network none \
  --no-mount home,cwd \
  --contain --env MPLCONFIGDIR=/tmp/matplotlib \
  --env DEEPFILTER_LOG_FILE=/tmp/enhance.log \
  --env NUMBA_CACHE_DIR=/tmp/numba_cache \
  --bind "$SHARED_DIR:/files" \
  --pwd /app \
  /mnt/data/container/transcriber.sif \
  bash -c "python src/cui/main_cui.py --model $MODEL --infile /files/$AUDIOFILE --outname \"$OUTNAME\" $DIARIZE_ARG" \
  2>&1 | tee "$SHARED_DIR/progress.log" &
