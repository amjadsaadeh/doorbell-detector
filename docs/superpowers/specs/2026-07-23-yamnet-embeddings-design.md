# YAMNet Embeddings + XGBoost Head — Design

**Date:** 2026-07-23
**Branch:** `yamnet-features`
**Status:** approved direction ("Proposal 2"), implementation on this branch

## Goal

Replace hand-crafted MFCC features with embeddings from YAMNet (MobileNet-v1
audio model pretrained on AudioSet, 521 classes incl. doorbell/bell sounds),
keeping the existing XGBoost head and the rest of the DVC pipeline unchanged.
Rationale: with only ~51 real annotated recordings, a pretrained embedding
space that already separates bell-like sounds from domestic background is the
largest accuracy-per-real-sample win available, at minimal pipeline
disruption.

## What changes

| Piece | Change |
|-------|--------|
| `src/extract_yamnet_features.py` | **New.** Per-file YAMNet frame embeddings → `data/yamnet_data/*.npy` |
| `src/draw_data.py` | Chunk slicing reads YAMNet frames (0.48 s hop) instead of MFCC frames (512-sample hop) and mean-pools to a fixed 1024-dim vector per chunk; feature column renamed `yamnet_features` |
| `dvc.yaml` | `extract_features` stage swaps script + output dir (`mfcc_data` → `yamnet_data`); `draw_data` dep updated |
| `pyproject.toml` | + `tensorflow` (CPU), + `tensorflow-hub` |
| `src/train_xgboost.py` | **Unchanged** — it auto-detects the `*_features` column; MLflow run becomes `xgboost-yamnet-<branch>` |
| `src/extract_mfcc_features.py` | Kept (still used on `main`); no longer referenced by `dvc.yaml` on this branch |

## Feature extraction (`extract_yamnet_features.py`)

- Loads YAMNet from TF Hub (`https://tfhub.dev/google/yamnet/1`), cached
  under `data/downloads/tfhub` (same git-ignored cache dir as the noise-pool
  archives) so re-runs don't need the network.
- For every `.wav` in `data/audio`, `data/augmented_audio`, and the external
  noise pools: load with `librosa.load(sr=16000, mono=True)` (defensive —
  pools are already 16 kHz mono, uploads may not be), pad to YAMNet's minimum
  0.975 s window, run the model, save `embeddings.T` — shape `(1024, T)`,
  mirroring the `(n_mfcc, T)` orientation of the MFCC path.
- Serial loop instead of the MFCC path's `multiprocessing.Pool`: TF is not
  fork-safe and each worker would re-load the model; single-process YAMNet on
  CPU is far faster than real time, which is enough here.

## Chunk features (`draw_data.py`)

YAMNet emits one 1024-dim embedding per 0.48 s hop (0.96 s window). For a
chunk `[start, end)` in ms:

- `start_frame = start // 480`, `n_frames = (end - start) // 480` (4 frames
  for the 2000 ms chunks) — fixed frame-rate slicing, same principle as the
  MFCC path's `FRAMES_PER_MS` (see the librosa +1-frame comment there).
- **Mean-pool over the sliced frames** → always a `(1024,)` vector. Pooling
  (rather than concatenating frames) makes the chunk feature
  shape-independent of frame count, which sidesteps the exact class of bug
  documented for MFCC: a 2 s augmented clip yields only 3 YAMNet frames
  (no final partial window), while a 2 s slice of a long file yields 4.
- Guard: if a slice at the very end of a file comes up empty, fall back to
  the file's last frame.

## Trade-offs accepted

- **Temporal order within a chunk is lost** by mean pooling. Standard
  practice for AudioSet-embedding transfer; revisit (mean+max concat) only
  if metrics disappoint.
- **TensorFlow (~600 MB) joins the training env.** Training-host only; the
  Pi is unaffected (it cannot run xgboost/librosa today either — inference
  placement is out of scope for this branch and stays with the deployed
  cross-correlation detector).
- **First `dvc repro` needs network** to fetch the YAMNet weights (~17 MB);
  afterwards served from the local cache.

## Success criteria

- `uv run dvc repro` completes: `extract_features → draw_data → train_model`.
- MLflow run `xgboost-yamnet-yamnet-features` logs `feature_type: yamnet`,
  `n_features: 1024`, and val precision/recall/F1 comparable to or better
  than the latest `xgboost-mfcc-main` run (dataset md5 lineage makes the
  comparison honest).
- Existing test suite still passes; new unit tests cover the frame-slice +
  mean-pool logic without needing TF or network.
