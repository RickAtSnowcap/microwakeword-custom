"""Convert every wav in SRC to 16 kHz mono int16 wav in DST (as the notebook's array*32767 cast)."""
import sys, pathlib
import numpy as np, soundfile as sf
src, dst = map(pathlib.Path, sys.argv[1:3])
for p in sorted(src.glob("*.wav")):
    a, sr = sf.read(p, dtype="float32", always_2d=True)
    assert sr == 16000, (p, sr)
    sf.write(dst / p.name, (a[:, 0] * 32767).astype(np.int16), 16000, subtype="PCM_16")
print("converted", len(list(dst.glob("*.wav"))))
