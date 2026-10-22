# microwakeword-custom

Train your own [microWakeWord](https://github.com/kahrendt/microWakeWord) model on your own GPU, then put
it on a Home Assistant Voice Preview Edition (or any ESP32-S3 running ESPHome's `micro_wake_word`)
next to the stock wake words.

These are the scripts I used to make my Voice PE answer to "Hey Snoopy". They follow the community
Colab notebook's recipe, run locally (no paid A100 runtime), and add one thing the notebook doesn't
have: a second round of training on **sound-alike phrases**, which stopped the model from waking up
to "Hey Siri". The write-up is at
[sudowren.com/how-to/hey-snoopy-wake-word](https://sudowren.com/how-to/hey-snoopy-wake-word).

No training audio or model files are included: the scripts generate and download everything.

## What you need

- Linux with an NVIDIA GPU. I used an RTX 4060 Ti 16 GB; TensorFlow is capped at 4 GB
  (`GPU_MEM_MB` in `scripts/train_launcher.py`), so the card can keep doing other work.
- About 60 GB of disk, on an SSD. Training reads features at random, so a spinning disk makes it crawl.
- [uv](https://docs.astral.sh/uv/), git, wget, unzip and ffmpeg.
- About 7 hours, mostly unattended.

## Set up

Everything lives in this folder: `scripts/env.sh` points every cache and temp folder here.

```bash
source scripts/env.sh
uv python install 3.10 3.11
uv venv -p 3.10 venvs/mww; uv venv -p 3.11 venvs/piper; uv venv -p 3.11 venvs/tflm

git clone https://github.com/kahrendt/microWakeWord src/microWakeWord
git -C src/microWakeWord checkout 4665173cd35f1cff9a61e06fc427f124766c488e
git clone https://github.com/rhasspy/piper-sample-generator src/piper-sample-generator
git -C src/piper-sample-generator checkout 2971426a55072f7d22fec416ca7800df8bd23207

VIRTUAL_ENV=venvs/piper uv pip install ./src/piper-sample-generator
echo $WW/src/piper-sample-generator > venvs/piper/lib/python3.11/site-packages/piper_train_path.pth
VIRTUAL_ENV=venvs/mww uv pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
VIRTUAL_ENV=venvs/mww uv pip install -e ./src/microWakeWord 'tensorflow[and-cuda]==2.21.0' 'datasets<4' librosa tensorboard
VIRTUAL_ENV=venvs/tflm uv pip install tflite-micro

mkdir -p data/gen_model
wget -O data/gen_model/en_US-libritts_r-medium.pt \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
cp src/piper-sample-generator/models/en_US-libritts_r-medium.pt.json data/gen_model/
```

`venv-locks/` has the exact package versions of my run, if you need to match them.

## Train

```bash
PHRASE="hey snoopy" scripts/01_generate.sh   # 20,000 synthetic clips of your phrase
scripts/02_download.sh                         # room impulse responses, background audio, negative sets
(cd data && ../venvs/mww/bin/python -I ../scripts/03_features.py)   # augment + spectrograms

VARIANT=v1 scripts/05_train.sh                 # optional: the plain notebook recipe, for comparison

# Edit scripts/adversarial_phrases.txt first: sound-alikes for YOUR phrase.
scripts/10_generate_adversarial.sh
(cd data && ../venvs/mww/bin/python -I ../scripts/03_features.py \
  $WW/data/adversarial_samples $WW/data/adversarial_augmented_features 1 2)
VARIANT=v2 scripts/05_train.sh                 # notebook recipe + sound-alike negatives
```

The model ends up at
`train_v2/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`.

## Test

```bash
scripts/06_nearmiss.sh   # 50 clips of each near-miss phrase (edit the list for your phrase)
M=train_v2/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
venvs/mww/bin/python -I scripts/07_extra_eval.py $M $(for d in data/eval_extra/nearmiss/*; do echo "$(basename $d)=$d"; done)
venvs/mww/bin/python -I scripts/08_ambient_faph.py $M data/negative_datasets/dinner_party_eval/*_ambient/*
venvs/mww/bin/python -I scripts/09_test_frr.py $M data/generated_augmented_features/testing/wakeword_mmap
venvs/tflm/bin/python -I scripts/arena_size.py $M   # minimum tensor arena; add ~10-15% in the manifest
```

`07` prints the share of clips that would wake the device at each cutoff, `08` the false accepts per
hour on long dinner-party recordings, and `09` the false-reject rate on held-out positives.

## Put it on the device

Copy the model to `esphome/wakewords/my_wake_word.tflite`, fill in `esphome/my_wake_word.json.example`
(save it as `wakewords/my_wake_word.json`), set your device's name in
`esphome/voice-pe-custom-wake-word.yaml`, then build and upload with the official ESPHome image:

```bash
cd esphome
docker run --rm -v "$PWD":/config ghcr.io/esphome/esphome compile voice-pe-custom-wake-word.yaml
docker run --rm --network host -v "$PWD":/config ghcr.io/esphome/esphome \
  upload voice-pe-custom-wake-word.yaml --device <the device's IP>
```

Then pick the new wake word in the device's settings in Home Assistant.

## Notes

- `datasets` must stay below 4: newer versions decode audio only with torchcodec.
- TensorFlow needs `LD_LIBRARY_PATH` pointed at the venv's `nvidia/*/lib` folders to see the GPU;
  `05_train.sh` does that.
- The AudioSet repo no longer has the tarball the notebook downloads, so `02_download.sh` decodes a
  parquet shard instead (`audioset_from_parquet.py`).

## License

Apache-2.0, the same as microWakeWord, whose notebook these scripts follow.
