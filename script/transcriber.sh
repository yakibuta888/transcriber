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
USERDIR="/home/$USER/$WORKDIR"

# diarizeオプション判定
DIARIZE_ARG=""
if [ "$OPTIONAL" = "diarize" ]; then
  DIARIZE_ARG="--diarize"
fi

docker run --rm \
  --network=none \
  --gpus all \
  --hostname transcriber \
  --add-host transcriber:127.0.0.1 \
  --name app-transcriber_$(date +"%Y%m%d_%H%M%S") \
  -v "$USERDIR:/home/client/app/files" \
  transcriber \
  bash -c "sudo chmod -R 777 /home/client/app/files && python src/cui/main_cui.py --model $MODEL --infile ./files/$AUDIOFILE --outname \"$OUTNAME\" $DIARIZE_ARG && sudo cp /home/client/app/src/logs/transcriber.log /home/client/app/files"