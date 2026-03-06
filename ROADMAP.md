# Doorbell Detector – Roadmap

## Current State (as of resumption, March 2026)

- 7-stage DVC pipeline (fetch → convert → download → quality → features → balance → train)
- XGBoost binary classifier on 13 MFCC coefficients → **95% F1**
- 644 labelled audio files (291 MB, tracked via DVC)
- Real-time inference in `data_collector.py` with circular buffer
- Label Studio annotation workflow with SDK integration

---

## Backlog (priority order)

### 1. Feature Enrichment – `src/extract_mfcc_features.py`
**Effort: ~2h | Impact: high**

Extend feature extraction to capture more audio signal properties:
- **Delta + delta-delta MFCCs** – temporal dynamics of the 13 existing MFCCs (×3 feature count, no inference overhead)
- **Spectral centroid, bandwidth, rolloff** – 4 values, characterise bell harmonic structure
- **Zero-crossing rate, RMS energy** – discriminate transient bell from sustained background

Expected gain: 2–5 F1 points without changing the model or deployment.

```yaml
# params.yaml additions
feature_extraction:
  n_mfcc: 13
  n_fft: 512
  include_delta: true       # add
  include_spectral: true    # add
```

---

### 2. Label Studio ML Backend – pre-annotation with existing model
**Effort: ~4h | Impact: high (saves hours per annotation session)**

Label Studio supports an ML backend that pre-labels incoming audio. Wire the XGBoost model as a backend:
- Install `label-studio-ml` package
- Serve `models/xgboost_model.json` + feature extraction via a small Flask endpoint
- New recordings get predicted labels automatically → annotator only corrects mistakes (~5 s vs. ~30 s per clip)

See: https://labelstud.io/guide/ml

---

### 3. Data Augmentation DVC Stage
**Effort: ~4h | Impact: high (addresses acoustic diversity gap)**

Add an `augment` stage to `dvc.yaml` between `download_audio` and `extract_features`.
Apply `librosa` augmentations to `bell` clips only:
- Mix bell + random background chunk at varied SNR levels
- Pitch shift ±2 semitones
- Time stretch ±10%
- Convolve with synthetic room impulse responses

Can 10× the effective positive class without re-labeling. More importantly, it covers the acoustic conditions (open window, different distances, TV noise) that the current dataset likely under-represents.

---

### 4. Passive Data Collection via systemd
**Effort: ~3h | Impact: medium**

Replace the manual Termux/SSH trigger with:
- `systemd` service file for `data_collector.py` on the Pi (auto-start, restart on crash)
- GPIO button (e.g. pin 18) wired to a momentary switch near the mic to **flag** positive moments
- Alternative: MQTT message from phone to flag, using Home Assistant or similar

This closes the collection loop without requiring SSH access every time.

---

### 5. Active Learning Logging – `src/data_collector.py`
**Effort: ~2h | Impact: medium**

Log uncertain samples (model probability in `0.4 < p < 0.6`) to a separate directory.
These are the most informative samples to annotate next. Upload batch to Label Studio periodically.

```python
# addition to data_collector.py
if 0.4 < prob < 0.6:
    save_clip(audio, uncertain_dir / f"{timestamp}.wav")
```

---

### 6. DVC Remote Migration
**Effort: ~1h | Impact: medium (fragility fix)**

Current remote: `url = ../dvc_remote` – breaks on fresh clone.

Migrate to [DagsHub](https://dagshub.com) (free Git + DVC hosting, zero config):
```bash
dvc remote modify origin url s3://dagshub/<user>/doorbell-detector
# or use DagsHub's native DVC remote support
```

Alternative: self-hosted MinIO if you want to stay local.

---

### 7. Makefile for Common Tasks
**Effort: ~1h | Impact: low (ergonomics)**

```makefile
fetch-data:
    dvc repro fetch_labeled_data convert_labeled_data download_audio

train:
    dvc repro extract_features draw_data train_model

deploy:
    bash src/deploy.sh

exp:
    dvc exp run --set-param model.n_estimators=$(N) --set-param model.max_depth=$(D)
```

---

## Model Strategy Notes

**Current: XGBoost is the right choice for Pi Zero W.**

| Model | Inference | F1 (est.) | Deploy |
|---|---|---|---|
| XGBoost (current) | <1 ms | 95% | JSON ✓ |
| + delta MFCCs | <1 ms | ~97% | JSON ✓ |
| SVM (RBF) | ~5 ms | similar | joblib |
| TFLite (Zero 2 W only) | ~50–100 ms | higher at >5k clips | TFLite runtime |

Switch to deep learning only after collecting >5 000 clips and if you are on the **Zero 2 W**.

---

## Verification Checklist

After any feature or model change:

```bash
dvc repro              # run only changed stages
dvc metrics show       # compare F1, precision, recall
dvc exp show           # compare experiments
python -m pytest       # unit tests
bash src/deploy.sh     # push to Pi
# ring doorbell, verify detection in data_collector.py log
```
