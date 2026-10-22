#!/bin/bash
# v2 addition: confusable negative phrases (notebook comment: "generating negative samples similar
# to the wake word"). Same generator + variation defaults as the positives; phrases cycle per sample.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd $WW/data
$WW/venvs/piper/bin/python -I -m piper_sample_generator $WW/scripts/adversarial_phrases.txt \
  --model $WW/data/gen_model/en_US-libritts_r-medium.pt \
  --max-samples ${N:-20000} --batch-size ${BS:-50} \
  --output-dir $WW/data/adversarial_samples
