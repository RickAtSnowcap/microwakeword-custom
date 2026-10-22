"""Extra sanity eval of the streaming quantized tflite on wav folders (not part of the notebook).

For each clip (16 kHz assumed; resampled otherwise), pads 1 s of silence on both sides, computes
pymicro-features spectrograms (10 ms step, same as training), runs the streaming model with
stride 3, averages probabilities over a sliding window of 5 (as ESPHome does) and records the max.
Prints the fraction of clips that would trigger at several cutoffs.

usage: 07_extra_eval.py MODEL.tflite NAME=DIR [NAME=DIR ...]
"""
import sys, pathlib
import numpy as np, soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from microwakeword.inference import Model
from microwakeword.audio.audio_utils import generate_features_for_clip

CUTOFFS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
model_path = sys.argv[1]
print("set".ljust(26) + "n".rjust(5) + "".join(f"{c:>7.2f}" for c in CUTOFFS))
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    maxes = []
    for p in sorted(pathlib.Path(d).glob("*.wav")):
        y, sr = sf.read(p, dtype="int16", always_2d=True)
        y = y[:, 0]
        if sr != 16000:
            import librosa
            y = (librosa.resample(y.astype(np.float32) / 32768, orig_sr=sr, target_sr=16000) * 32767).astype(np.int16)
        y = np.concatenate([np.zeros(16000, np.int16), y, np.zeros(16000, np.int16)])
        m = Model(model_path, stride=3)  # fresh streaming state per clip
        probs = np.array(m.predict_spectrogram(generate_features_for_clip(y, step_ms=10)))
        maxes.append(sliding_window_view(probs, 5).mean(axis=-1).max())
    maxes = np.array(maxes)
    print(name.ljust(26) + str(len(maxes)).rjust(5) + "".join(f"{(maxes > c).mean():>7.0%}" for c in CUTOFFS))
