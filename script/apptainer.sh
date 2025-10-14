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

export APPTAINER_CACHEDIR=/mnt/data/apptainer_cache
export TMPDIR=/mnt/data/large_dtmp
apptainer cache clean --force

apptainer exec --nv --net --network none \
  --bind "$SHARED_DIR:/home/$HOST_USER/app/files" \
  --pwd /home/$HOST_USER/app \
  /mnt/data/container/transcriber.sif \
  bash -c "python src/cui/main_cui.py --model $MODEL --infile ./files/$AUDIOFILE --outname \"$OUTNAME\" $DIARIZE_ARG && cp /home/$HOST_USER/app/src/logs/transcriber.log /home/$HOST_USER/app/files"
