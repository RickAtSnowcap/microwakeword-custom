#!/bin/bash
# Extra eval data: near-miss phrases (piper generator, 50 each) for a confusability check.
# These are sound-alikes for "hey snoopy": replace them with ones for your own phrase.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
O=$WW/data/eval_extra/nearmiss; mkdir -p $O
for p in "snoopy" "hey snoop" "hey sophie" "hey siri" "hey jarvis" "okay nabu" "hey susie" "hey stupid" "hey soupy" "hey scooby" "hey snappy" "hey there" "happy" "hey google" "hey smoothie" "hey snippy" "hey spoony" "hey stacy" "hey sneaky" "hey sloppy"; do
  d=$O/$(echo "$p" | tr ' ' '_')
  [ -d "$d" ] && continue
  $WW/venvs/piper/bin/python -I -m piper_sample_generator "$p" --model $WW/data/gen_model/en_US-libritts_r-medium.pt \
    --max-samples 50 --batch-size 25 --output-dir "$d" 2>&1 | tail -n 1
done
