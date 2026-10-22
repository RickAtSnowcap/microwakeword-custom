"""Step 3: augment positive clips and write training/validation/testing spectrogram features.

Direct conversion of the augmentation + feature cells of
microWakeWord/notebooks/basic_training_notebook.ipynb (same settings), with paths under
the wakeword folder's data/ (the folder is this script's parent's parent).
"""
import os
import sys
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

WW = Path(__file__).resolve().parent.parent
D = f"{WW}/data"
# usage: 03_features.py [INPUT_WAV_DIR OUTPUT_FEATURE_DIR [TRAIN_REPEAT TRAIN_SLIDE_FRAMES]]
# defaults: positives, notebook settings (repeat 2, slide_frames 10)
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{D}/generated_samples"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{D}/generated_augmented_features"
TRAIN_REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TRAIN_SLIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 10

clips = Clips(
    input_directory=IN_DIR,
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=[f"{D}/mit_rirs"],
    background_paths=[f"{D}/fma_16k", f"{D}/audioset_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

output_dir = OUT_DIR
os.makedirs(output_dir, exist_ok=True)

for split in ["training", "validation", "testing"]:
    out_dir = os.path.join(output_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    split_name = "train"
    repetition = TRAIN_REPEAT
    spectrograms = SpectrogramGeneration(
        clips=clips, augmenter=augmenter, slide_frames=TRAIN_SLIDE, step_ms=10
    )
    if split == "validation":
        split_name = "validation"
        repetition = 1
    elif split == "testing":
        split_name = "test"
        repetition = 1
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
        )

    print(f"== {split} (split={split_name}, repeat={repetition})", flush=True)
    RaggedMmap.from_generator(
        out_dir=os.path.join(out_dir, "wakeword_mmap"),
        sample_generator=spectrograms.spectrogram_generator(
            split=split_name, repeat=repetition
        ),
        batch_size=100,
        verbose=True,
    )
print("DONE")
