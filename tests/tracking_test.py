"""Tests for run identity.

The bug these exist to prevent already shipped once: `git rev-parse
--abbrev-ref HEAD` returns the literal string "HEAD" in the detached worktree
`dvc exp run --temp` uses, so 14 of 57 historic runs are tagged with a branch
named "HEAD" and named `cnn-mfcc-HEAD`. That is unrecoverable after the fact --
the point of naming a run is to know later which commit it came from.
"""

import json
import subprocess

import pytest

import tracking


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with one commit, as the process working directory.

    Real git rather than a mock: every failure being guarded here came from
    git's actual behaviour in a state the code did not anticipate, which a
    mock would simply reproduce the assumption of.
    """
    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "Test")
    (tmp_path / "file.txt").write_text("one\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "first")
    return tmp_path


def _detach(repo):
    """Reproduce what `dvc exp run --temp` leaves the worktree in."""
    subprocess.run(
        ["git", "checkout", "-q", "--detach", "HEAD"],
        cwd=repo, check=True, capture_output=True,
    )


def test_git_ref_never_returns_the_string_head(repo):
    """The regression that made the variant sweep untraceable."""
    _detach(repo)
    assert tracking.git_ref() != "HEAD"
    assert tracking.git_ref().startswith("detached@")


def test_git_ref_uses_dvc_baseline_when_detached(repo, monkeypatch):
    monkeypatch.setenv("DVC_EXP_BASELINE_REV", "abc1234def5678")
    _detach(repo)
    assert tracking.git_ref() == "detached@abc1234"


def test_git_ref_is_the_branch_when_attached(repo):
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.strip()
    assert tracking.git_ref() == branch


def test_commit_resolves_in_a_detached_worktree(repo):
    """The whole point of naming runs by commit rather than branch."""
    attached = tracking.git_commit()
    _detach(repo)
    assert tracking.git_commit() == attached
    assert len(attached) == 7


def test_run_name_distinguishes_runs_from_different_commits(repo):
    first = tracking.run_name("cnn", "logmel")
    (repo / "file.txt").write_text("two\n")
    subprocess.run(["git", "commit", "-qam", "second"], cwd=repo, check=True,
                   capture_output=True)
    assert tracking.run_name("cnn", "logmel") != first


def test_dvc_lock_churn_does_not_mark_a_run_dirty(repo):
    """dvc.lock is rewritten by the repro that runs the stage.

    If it counted, every run would be +dirty and the marker would carry no
    information at all.
    """
    (repo / "dvc.lock").write_text("initial\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "lock"], cwd=repo, check=True,
                   capture_output=True)
    (repo / "dvc.lock").write_text("rewritten by dvc repro\n")
    assert tracking.is_dirty() is False


def test_modified_source_marks_a_run_dirty(repo):
    (repo / "file.txt").write_text("uncommitted change\n")
    assert tracking.is_dirty() is True
    assert tracking.run_name("cnn", "logmel").endswith("+dirty")


def test_untracked_output_does_not_mark_a_run_dirty(repo):
    """data/ and models/ are untracked by design and say nothing about code."""
    (repo / "untracked_output.h5").write_text("pipeline output\n")
    assert tracking.is_dirty() is False


def test_run_tags_cover_every_comparison_axis(repo, monkeypatch):
    monkeypatch.setattr(tracking, "dataset_md5", lambda: "deadbeef")
    monkeypatch.setenv("DVC_EXP_NAME", "my-exp")
    tags = tracking.run_tags("cnn", "logmel", stage="train")
    assert tags["feature_type"] == "logmel"
    assert tags["head"] == "cnn"
    assert tags["stage"] == "train"
    assert tags["dataset_md5"] == "deadbeef"
    assert tags["dvc_exp"] == "my-exp"
    # The field that used to be "HEAD" is gone entirely.
    assert "git_branch" not in tags


def test_dvc_exp_defaults_to_workspace(repo, monkeypatch):
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    monkeypatch.setattr(tracking, "dataset_md5", lambda: "deadbeef")
    assert tracking.run_tags("cnn", "logmel", stage="train")["dvc_exp"] == "workspace"


def test_run_handoff_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "RUN_HANDOFF", tmp_path / "trained" / "run.json")
    tracking.write_run_handoff("abc123")
    assert tracking.read_run_handoff() == "abc123"


def test_missing_handoff_is_not_an_error(tmp_path, monkeypatch):
    """A model trained before the handoff existed must still quantize."""
    monkeypatch.setattr(tracking, "RUN_HANDOFF", tmp_path / "absent.json")
    assert tracking.read_run_handoff() is None


def test_corrupt_handoff_is_not_an_error(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text("{not json")
    monkeypatch.setattr(tracking, "RUN_HANDOFF", path)
    assert tracking.read_run_handoff() is None


def test_quantized_run_name_marks_the_int8_graph(repo):
    train = tracking.run_name("cnn", "logmel")
    assert tracking.run_name("cnn", "logmel", suffix="-int8") == f"{train}-int8"


def _commit_params(repo):
    (repo / "params.yaml").write_text("feature_extraction:\n  type: logmel\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "params"], cwd=repo, check=True,
                   capture_output=True)


def test_params_rewrite_under_dvc_exp_is_not_dirty(repo, monkeypatch):
    """`dvc exp run -S` applies overrides by rewriting params.yaml.

    Before this exclusion all three --temp runs of the first clean sweep were
    named +dirty with nothing actually uncommitted. DVC records the rewrite
    in the experiment, so the run is reproducible via `dvc exp`.
    """
    _commit_params(repo)
    monkeypatch.setenv("DVC_EXP_NAME", "vatic-dabs")
    (repo / "params.yaml").write_text("feature_extraction:\n  type: mfcc\n")
    assert tracking.is_dirty() is False


def test_params_edit_outside_dvc_exp_is_dirty(repo, monkeypatch):
    """In a plain repro an edited params.yaml is a real uncommitted change."""
    _commit_params(repo)
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    (repo / "params.yaml").write_text("feature_extraction:\n  type: mfcc\n")
    assert tracking.is_dirty() is True


def test_dvc_exp_does_not_hide_real_source_changes(repo, monkeypatch):
    """The params.yaml exclusion must not become a blanket pass for exp runs."""
    monkeypatch.setenv("DVC_EXP_NAME", "vatic-dabs")
    (repo / "file.txt").write_text("uncommitted change\n")
    assert tracking.is_dirty() is True


def test_passing_workspace_run_promotes_champion(monkeypatch):
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    assert tracking.promotes_champion(passed=True) is True


def test_sweep_variant_never_promotes_champion(monkeypatch):
    """The regression: every passing sweep variant took the alias in turn.

    The gate measures quantization damage, not quality, so passing it cannot
    decide what ships. On the first clean sweep that left @champion on stft
    (val_f1 0.8888) instead of the committed logmel default (0.9931).
    """
    monkeypatch.setenv("DVC_EXP_NAME", "mirky-pony")
    assert tracking.promotes_champion(passed=True) is False


def test_failed_gate_never_promotes_champion(monkeypatch):
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    assert tracking.promotes_champion(passed=False) is False
