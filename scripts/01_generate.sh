#!/bin/bash
# Step 1: positive samples of your wake phrase (PHRASE="hey jasper" scripts/01_generate.sh)
# with piper-sample-generator (notebook variation defaults: slerp 0.5,
# length scales 1.0/0.75/1.25/1.4, noise scales .667-1.4, noise_w 0.8, all speakers)
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd $WW/data
$WW/venvs/piper/bin/python -I -m piper_sample_generator "${PHRASE:-hey snoopy}" \
  --model $WW/data/gen_model/en_US-libritts_r-medium.pt \
  --max-samples ${N:-20000} --batch-size ${BS:-100} \
  --output-dir $WW/data/generated_samples
