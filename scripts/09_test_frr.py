"""FRR of the exported streaming quantized tflite on the positive testing features, using the same
method as microwakeword.test.tflite_streaming_model_roc (skip first 25 slices, sliding window 5, max).
usage: 09_test_frr.py MODEL.tflite TESTING_MMAP_DIR"""
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mmap_ninja.ragged import RaggedMmap
from microwakeword.inference import Model

cutoffs = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]
maxes = []
for spec in RaggedMmap(sys.argv[2]):
    m = Model(sys.argv[1], stride=3)
    p = np.array(m.predict_spectrogram(spec))
    maxes.append(sliding_window_view(p[25:], 5).mean(axis=-1).max())
maxes = np.array(maxes)
print(f"positives: {len(maxes)}")
for c in cutoffs:
    print(f"  cutoff {c:.2f}: frr={(maxes <= c).mean():.4f}")
