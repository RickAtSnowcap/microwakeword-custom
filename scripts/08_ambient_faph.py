"""FA/h of the exported streaming quantized tflite on the dinner_party_eval ambient sets
(testing_ambient = DiPCo, validation_ambient = CHiME-6 dev/eval), using microWakeWord's own
compute_false_accepts_per_hour (sliding window 5, 25-slice cooldown) on whole tracks.
usage: 08_ambient_faph.py MODEL.tflite MMAP_DIR [MMAP_DIR ...]"""
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mmap_ninja.ragged import RaggedMmap
from microwakeword.inference import Model
from microwakeword.test import compute_false_accepts_per_hour

cutoffs = np.array([0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99])
model_path = sys.argv[1]
for d in sys.argv[2:]:
    probs, slices = [], 0
    for spec in RaggedMmap(d):
        m = Model(model_path, stride=3)
        p = np.array(m.predict_spectrogram(spec))
        slices += len(spec)
        probs.append(sliding_window_view(p, 5).mean(axis=-1))
    hours = slices * 0.010 / 3600
    faph = compute_false_accepts_per_hour(probs, cutoffs, 25, stride=3, step_s=0.010)
    print(f"{d.split('/')[-2]}/{d.split('/')[-1]}: {hours:.2f} h")
    for c, f in zip(cutoffs, faph):
        print(f"  cutoff {c:.2f}: faph={f:.3f}  (false accepts={round(f*hours)})")
