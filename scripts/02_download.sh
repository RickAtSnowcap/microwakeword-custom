#!/bin/bash
# Step 2: augmentation audio + pre-generated negative features (same sources as the notebook).
# datasets>=4 needs torchcodec for audio decoding, so files are fetched directly and converted
# with soundfile/ffmpeg (16 kHz, mono, 16-bit PCM) instead of datasets.Audio casting.
# AudioSet: the HF repo is now parquet shards; data/bal_train/09.parquet replaces bal_train09.tar.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
PY="$WW/venvs/mww/bin/python -I"
HF=https://huggingface.co/datasets
D=$WW/data; cd $D

rirs() {
  [ -d mit_rirs ] && return
  $PY -c "from huggingface_hub import snapshot_download as s; s('davidscripka/MIT_environmental_impulse_responses', repo_type='dataset', local_dir='$D/mit_rirs_raw', allow_patterns=['16khz/*'], max_workers=4)" || return 1
  mkdir -p mit_rirs.tmp && $PY $WW/scripts/to_int16.py $D/mit_rirs_raw/16khz $D/mit_rirs.tmp && mv mit_rirs.tmp mit_rirs
}
audioset() {
  [ -d audioset_16k ] && return
  mkdir -p audioset && wget -q -c -O audioset/bal_train_09.parquet $HF/agkphysics/AudioSet/resolve/main/data/bal_train/09.parquet || return 1
  $PY $WW/scripts/audioset_from_parquet.py audioset/bal_train_09.parquet audioset_16k.tmp && mv audioset_16k.tmp audioset_16k
}
fma() {
  [ -d fma_16k ] && return
  mkdir -p fma && wget -q -c -O fma/fma_xs.zip $HF/mchl914/fma_xsmall/resolve/main/fma_xs.zip || return 1
  (cd fma && unzip -q -o fma_xs.zip)
  rm -rf fma_16k.tmp; mkdir -p fma_16k.tmp
  find fma/fma_small -name '*.mp3' -print0 | xargs -0 -P 16 -I{} sh -c 'ffmpeg -nostdin -loglevel error -y -i "$1" -ac 1 -ar 16000 -c:a pcm_s16le "fma_16k.tmp/$(basename "$1" .mp3).wav" || echo "skip $1"' _ {}
  mv fma_16k.tmp fma_16k
}
neg() {
  [ -d negative_datasets/$1 ] && return
  wget -q -c -O negative_datasets/$1.zip $HF/kahrendt/microwakeword/resolve/main/$1.zip || return 1
  unzip -q -o negative_datasets/$1.zip -d negative_datasets
}
mkdir -p negative_datasets
pids=()
rirs > $WW/logs/02_rirs.log 2>&1 & pids+=($!)
audioset > $WW/logs/02_audioset.log 2>&1 & pids+=($!)
fma & pids+=($!)
for f in dinner_party dinner_party_eval no_speech speech; do neg $f & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] && echo DONE || echo FAILED
