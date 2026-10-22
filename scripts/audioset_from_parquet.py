"""Decode AudioSet clips from an agkphysics/AudioSet parquet shard to 16 kHz mono int16 wavs.
(The repo no longer has the bal_train09.tar the notebook fetches; data/bal_train/09.parquet replaces it.)"""
import io, sys, pathlib
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
pf = pq.ParquetFile(src)
print(pf.schema_arrow, flush=True)
n = bad = 0
for batch in pf.iter_batches(batch_size=64, columns=["audio"]):
    for a in batch.column("audio").to_pylist():
        name = pathlib.Path(a.get("path") or f"clip_{n}").stem + ".wav"
        try:
            y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sf.write(dst / name, (np.clip(y, -1, 1) * 32767).astype(np.int16), 16000, subtype="PCM_16")
            n += 1
        except Exception as e:  # skip undecodable clips
            bad += 1
print(f"wrote {n} clips, skipped {bad}")
