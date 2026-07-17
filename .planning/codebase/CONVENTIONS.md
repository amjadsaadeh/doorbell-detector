# Coding Conventions

**Analysis Date:** 2026-05-26

## Naming Patterns

**Files:**
- Python source scripts use `snake_case` throughout: `extract_mfcc_features.py`, `convert_labeled_data.py`, `train_xgboost.py`, `dataqualityutils.py`
- Test files are suffixed with `_test.py` (not prefixed): `convert_labeled_data_test.py`, `feature_extraction_test.py`
- Shell scripts use `snake_case` as well: `fetch_data.sh`, `deploy.sh`
- Systemd files use kebab-case: `doorbell-collector.service`, `doorbell-collector.env`

**Functions:**
- `snake_case` throughout: `extract_mfcc_features`, `process_audio_data`, `process_single_file`, `annotation_to_sample_per_row`, `get_data_quality_metrics`, `setup_mqtt`, `setup_gpio`, `parse_args`
- Top-level script entry functions are named `main()` in scripts that are not purely data transformation utilities

**Variables:**
- `snake_case` for all local and module-level variables: `audio_buffer`, `trigger_event`, `buffer_chunks`, `mqtt_client`
- Module-level constants use `UPPER_SNAKE_CASE`: `CHANNELS`, `FORMAT`, `RATE`, `AUDIO_DATA_PATH`, `MFCC_FEATURES_FILE_BASE`, `MQTT_AVAILABLE`, `GPIO_AVAILABLE`, `OLD_DATA_PATH`

**Classes:**
- Test classes use `PascalCase` prefixed with `Test`: `TestConvertLabeledData`, `TestExtractMFCCFeatures`
- No domain/service classes exist in the codebase — all logic is in module-level functions

**CLI arguments:**
- `argparse` flags use `--kebab-case`: `--model-path`, `--save-dir`, `--buffer-minutes`, `--mqtt-host`, `--mqtt-tls-ca`, `--manual-trigger-only`
- The resulting `args` namespace attributes are accessed with `args.snake_case` (argparse converts hyphens to underscores automatically)

## Code Organization Patterns

The project is split into two sub-projects with distinct purposes:

**`src/`** — ML pipeline scripts (data preparation → feature extraction → training). Each script is a standalone pipeline stage invoked by DVC. Scripts expose a `main()` guard and are orchestrated via `dvc.yaml`, not imported by each other (except `dataqualityutils.py` which is a shared utility module imported by `draw_data.py` and `extract_data_quality.py`).

**`data_collection/`** — Raspberry Pi runtime. A single self-contained script `data_collector.py` with no dependency on `src/` or `params.yaml`. Configuration is entirely via CLI flags.

**Module structure within scripts:**
- Imports at top (stdlib → third-party, no grouping separator observed)
- Module-level constants directly after imports
- Optional dependency blocks using `try/except ImportError` with boolean availability flags (e.g., `MQTT_AVAILABLE`, `GPIO_AVAILABLE`)
- Section dividers using `# ---...---` comment banners for logical grouping within large files
- All executable logic guarded by `if __name__ == "__main__":`

## Configuration Patterns

**ML pipeline configuration** is read from `params.yaml` at runtime using `yaml.safe_load`. Scripts open it themselves as the first step in `main()`:
```python
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
```
Scripts are assumed to run from the project root (relative path `"params.yaml"`). DVC tracks which params keys each stage depends on in `dvc.yaml`.

**Runtime (collector) configuration** is entirely via CLI flags, with defaults baked into `argparse`. The systemd unit sources `/etc/doorbell-collector.env` as an `EnvironmentFile` and expands variables into the `ExecStart` command. No `params.yaml` is used in the collection sub-project.

**Secrets / API credentials** are read from environment variables via `os.getenv()` — e.g., `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`. These are not committed (`.env` is gitignored).

## Error Handling Patterns

Error handling is minimal and context-specific:

**In `data_collector.py` (long-running daemon):**
- `OSError` on `stream.read()` is caught inline to reopen the audio stream gracefully:
  ```python
  except OSError:
      log.warning("Buffer overflow – reopening stream")
      stream = open_stream()
  ```
- `KeyboardInterrupt` is caught at the top-level loop to trigger a clean shutdown in a `finally` block (stops PyAudio, disconnects MQTT, cleans up GPIO)
- MQTT/GPIO setup failures are logged with `log.error()` and return `None` — the program continues without that trigger source
- Missing model file is checked explicitly with `os.path.exists()` before loading; exits on failure

**In ML pipeline scripts:**
- No explicit exception handling — errors propagate and kill the DVC stage, which is the expected behavior for batch pipeline steps
- Validation of required arguments is done manually in `main()` for `data_collector.py`:
  ```python
  if not args.manual_trigger_only and args.model_path is None:
      print("error: ...", file=sys.stderr)
      sys.exit(2)
  ```

**In tests:**
- `assertRaises` used in `feature_extraction_test.py` to verify `FileNotFoundError` and `ValueError` are raised for invalid inputs

## Logging

**In `data_collector.py` (the only file using logging):**
- Uses the stdlib `logging` module configured at module level:
  ```python
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
  )
  log = logging.getLogger(__name__)
  ```
- Uses `log.info`, `log.warning`, `log.error` — no `log.debug` usage observed
- Log messages use `%`-style format strings (not f-strings): `log.info("Saved %d chunks ...", len(frames), ...)`
- Systemd unit routes stdout/stderr to journald (`StandardOutput=journal`); logs viewable with `journalctl -u doorbell-collector -f`

**In ML pipeline scripts:**
- No logging — `print()` is not used either. Progress is shown via `tqdm` progress bars. DVC captures pipeline output.

## Comment Style and Documentation

**Module-level docstrings:** Used in some scripts as a description of the script's purpose:
- `data_collector.py` has a detailed multi-paragraph module docstring describing all trigger sources and CLI modes
- `extract_data_quality.py`, `draw_data.py`, `convert_labeled_data.py` have single-line module docstrings
- `extract_mfcc_features.py`, `train_xgboost.py`, `dataqualityutils.py` have no module docstring

**Function docstrings:** Used selectively on public-facing data transformation functions with Google-style format (Args/Returns sections):
- `annotation_to_sample_per_row()` — full docstring
- `get_mfcc_features()` — full docstring
- `get_data_quality_metrics()` — no docstring despite being a shared utility
- Helper/internal functions (e.g., `process_single_file`, `save_audio`, `open_stream`) — no docstrings

**Inline comments:** Used liberally to explain non-obvious logic, e.g.:
```python
# Warm-up: skip initial chunks so the mic settles
# Convert to binary problem
# This way the shape of the slice is more reliably the same every time
```
Section-separator banners in `data_collector.py` use `# -----------...` lines with a centered label.

**TODO comments:** One present: `# TODO try imputations` in `src/draw_data.py` line 79.

## Formatting and Linting

**Formatter:** `black==24.10.0` is listed in `requirements.txt`, indicating it is used. No `pyproject.toml` or `.black` config file exists, so defaults apply (88-character line length).

**Linter:** No `.flake8`, `.pylintrc`, `ruff.toml`, or equivalent config found. No linter is explicitly configured.

**Type hints:** Used partially:
- Function signatures sometimes include type hints: `file_path: Path | str`, `n_mfcc: int`, `-> np.ndarray`, `-> Dict[str, int | float]`
- Some parameters lack annotations: `n_fft=2048` in `extract_mfcc_features()`, `params` dict arguments throughout
- No `mypy` config present; type checking appears unenforced

**Import style:** No import sorting tool configured (no `isort` config). Imports follow stdlib-then-third-party order informally but with no enforced separator.

**VSCode configuration** (`.vscode/settings.json`) configures `unittest` as the test runner (not pytest) with pattern `*_test.py` discovered from `./tests/`.

---

*Convention analysis: 2026-05-26*
