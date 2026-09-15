"""One definition of what identifies an MLflow run, shared by every stage.

The three logging scripts each grew their own answer to "which run is this?",
and they drifted: train runs tagged `git_branch` and logged `feature_type` as
a param, quantized runs did both plus a `stage` tag, and the run *name* was
built from the branch. That name was the only thing visible in the run list,
and it was the least informative field available -- seven consecutive
`cnn-logmel-unified-pipeline` runs spanned four datasets and seven commits
with nothing in the list to tell them apart.

Two fixes live here:

**The commit names the run.** MLflow already records the commit as
`mlflow.source.git.commit` on every run for free, so nothing needed to be
captured -- it just was never surfaced. `git rev-parse --abbrev-ref HEAD`, by
contrast, returns the literal string "HEAD" inside the detached worktree
`dvc exp run --temp` uses, which is exactly where the feature/head sweep runs.
The runs most worth comparing were the ones with the least identity.

**Tags carry the axes, params carry the settings.** MLflow's UI can group and
filter by tag but not by param, so anything you want to slice runs by
(feature_type, head, stage, dataset_md5) has to be a tag. The former two were
params only, `dataset_md5` was a param on train runs and absent from quantized
ones -- so a model and its int8 counterpart could not be joined at all.
"""

import hashlib
import json
import os
import subprocess

import mlflow

from paths import DATA_FILE, MODEL_DIR

MLFLOW_EXPERIMENT_NAME = "doorbell-detector"

# The int8 graph that gets flashed. Registering it is what makes "which run
# produced the model on the device?" a lookup instead of an md5 hunt.
REGISTERED_MODEL_NAME = "doorbell-detector-int8"
CHAMPION_ALIAS = "champion"

# train_model writes its run id here so evaluate_quantized can nest under it.
# Inside MODEL_DIR on purpose: that directory is already a DVC output of
# train_model and a dependency of both downstream stages, so the handoff
# travels the existing dependency graph and needs no dvc.yaml change. It is
# also cached with the model, which means a `dvc checkout` of an old model
# restores the pointer to the run that trained it.
RUN_HANDOFF = MODEL_DIR / "mlflow_run.json"

# dvc.lock is rewritten by the very `dvc repro` invocation that runs these
# stages, so it is always modified while a stage executes. Counting it as a
# dirty working tree would mark every single run +dirty and make the marker
# meaningless.
_DIRT_PATHSPEC = [".", ":(exclude)dvc.lock"]


def _git(*args: str, default: str = "") -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else default


def git_commit() -> str:
    """Short commit sha, or 'nogit' outside a repository.

    `rev-parse HEAD` rather than `--abbrev-ref HEAD`: it resolves to a real
    sha in a detached worktree, where the symbolic form degrades to "HEAD".
    """
    return _git("rev-parse", "--short=7", "HEAD", default="nogit") or "nogit"


def is_dirty() -> bool:
    """True when tracked source differs from the commit naming the run.

    Untracked files are ignored -- data/, models/ and the scratch output of
    the pipeline itself are untracked by design and say nothing about whether
    the *code and params* that produced a run are recoverable.
    """
    return bool(
        _git("status", "--porcelain", "--untracked-files=no", "--", *_DIRT_PATHSPEC)
    )


def dvc_exp_name() -> str | None:
    """The `dvc exp run` experiment name, if this stage runs under one.

    DVC exports it into the stage environment (dvc/repo/experiments/queue/
    base.py), which is how a --temp run recovers the identity that the
    detached HEAD destroyed.
    """
    return os.environ.get("DVC_EXP_NAME") or None


def git_ref() -> str:
    """Human-readable 'where did this come from', best available source.

    Branch when attached; the experiment's baseline commit when DVC detached
    us; the commit itself as a last resort. Never the string "HEAD".
    """
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch
    baseline = os.environ.get("DVC_EXP_BASELINE_REV")
    if baseline:
        return f"detached@{baseline[:7]}"
    return f"detached@{git_commit()}"


def dataset_md5() -> str:
    """md5 of balanced_data.h5 -- the same digest DVC records for it.

    A tag rather than a param so runs can be grouped by dataset version, and
    so a training run and its quantized child can be joined on it.
    """
    return hashlib.md5(DATA_FILE.read_bytes()).hexdigest()


def run_name(head: str, feature_type: str, suffix: str = "") -> str:
    """e.g. cnn-logmel@2cb39e1, or cnn-logmel@2cb39e1+dirty-int8.

    The commit is what makes this unique in practice: across the seven
    identically-named historic runs it takes seven distinct values where the
    branch took one.
    """
    marker = "+dirty" if is_dirty() else ""
    return f"{head}-{feature_type}@{git_commit()}{marker}{suffix}"


def run_tags(head: str, feature_type: str, stage: str) -> dict[str, str]:
    """The canonical tag block. Every run gets all of it, both stages.

    `git_branch` is deliberately absent: it was the field that resolved to
    "HEAD" on 14 of 57 historic runs. `dvc_exp` and `git_ref` replace it with
    values that survive a detached worktree.
    """
    return {
        "feature_type": feature_type,
        "head": head,
        "stage": stage,
        "dataset_md5": dataset_md5(),
        "dvc_exp": dvc_exp_name() or "workspace",
        "git_ref": git_ref(),
        "dirty": str(is_dirty()).lower(),
    }


def write_run_handoff(run_id: str) -> None:
    """Record the training run id for the downstream quantization stage."""
    RUN_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    RUN_HANDOFF.write_text(json.dumps({"run_id": run_id}, indent=4))


def read_run_handoff() -> str | None:
    """The training run id, or None if the model predates the handoff file.

    None is not an error: quantization then logs a top-level run exactly as
    it used to, rather than failing over a missing pointer.
    """
    if not RUN_HANDOFF.exists():
        return None
    try:
        return json.loads(RUN_HANDOFF.read_text()).get("run_id") or None
    except json.JSONDecodeError:
        return None


def register_int8_model(run_id: str, artifact_uri: str, passed: bool, tags: dict):
    """Register the int8 graph, aliasing it `champion` only if it passed.

    The registry is what answers "which run produced the .tflite on the
    device". Aliasing on the gate verdict means the alias can only ever point
    at a model the gate approved, while failed versions stay registered and
    inspectable rather than vanishing.
    """
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        # Already exists. create_registered_model is the only way to make one
        # and it raises rather than upserting.
        pass
    version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=artifact_uri,
        run_id=run_id,
        tags={k: str(v) for k, v in tags.items()},
    )
    if passed:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
        )
    return version
