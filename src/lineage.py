"""Trace chunks to their recordings, and recordings to the errors they cause.

provenance.py already answers "which annotation, file and offset is this
chunk". Two things were missing for hunting error-prone recordings:

* Synthetic chunks (~46% of the dataset) only knew their signal recording and
  noise *pool*. data/lineage/augmentation_lineage.csv, written by the
  augmentation stage, adds the noise recording + offset, so a mix traces to
  every recording it was built from.
* Nothing rolled chunk-level outcomes up to a recording. A recording is what
  you can actually fix: relabel it, re-record it, drop it.

Everything here reads pipeline outputs (the out-of-fold table, the manifest,
the lineage table), so every error is a *held-out* one. Not a DVC stage, like
inspect_dataset.py: it builds data/lineage_report/, a disposable cache.

  uv run python src/lineage.py report            # CSVs + index.html
  uv run python src/lineage.py trace 1234        # one chunk back to its sources
  uv run python src/lineage.py trace aug_snr-5_12_1
"""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from lineage_paths import AUGMENTATION_LINEAGE, LINEAGE_REPORT_DIR
from paths import OOF_PREDICTIONS
from provenance import chunk_provenance

LINEAGE_COLUMNS = [
    "annotation_id",
    "signal_file",
    "signal_start_ms",
    "noise_file",
    "noise_start_ms",
    "noise_roll_samples",
    "gain_db",
]

FP, FN = "false_positive", "false_negative"


def classify_kind(table: pd.DataFrame) -> pd.Series:
    """real / augmented / noise_pool, from the columns that distinguish them.

    Augmented rows carry snr_db; external noise rows carry noise_pool but no
    SNR (augmented rows carry both, hence the order).
    """
    snr = table["snr_db"] if "snr_db" in table else pd.Series(np.nan, index=table.index)
    pool = table["noise_pool"] if "noise_pool" in table else pd.Series(np.nan, index=table.index)
    kind = np.where(snr.notna(), "augmented", np.where(pool.notna(), "noise_pool", "real"))
    return pd.Series(kind, index=table.index, name="kind")


def build_chunk_lineage(provenance: pd.DataFrame, augmentation: pd.DataFrame | None) -> pd.DataFrame:
    """One row per chunk with the recording(s) it descends from.

    "Recording" columns, by kind:
      real        source_recording = the labeled file
      noise_pool  source_recording = the pool wav (ESC-50 / DEMAND clip)
      augmented   source_recording = the doorbell file (signal_file), plus
                  noise_file / noise_start_ms for the background it was mixed with
    """
    table = provenance.copy()
    table["kind"] = classify_kind(table)
    if augmentation is not None and len(augmentation):
        extra = augmentation[LINEAGE_COLUMNS].drop_duplicates("annotation_id")
        table = table.merge(extra, on="annotation_id", how="left")
    for column in LINEAGE_COLUMNS[1:]:
        if column not in table:
            table[column] = np.nan
    table["source_recording"] = np.where(
        table["kind"] == "augmented",
        table["signal_file"].fillna(table["split_group"]),
        table["audio_file_name"],
    )
    return table


def with_outcomes(lineage: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Attach held-out verdicts; chunks no fold held out are dropped."""
    verdict = oof[["row_index", "fold", "y_true", "y_score", "margin", "outcome"]]
    return lineage.merge(verdict, on="row_index", how="inner")


def _rates(group: pd.DataFrame) -> pd.Series:
    positives = int((group["y_true"] == 1).sum())
    negatives = int((group["y_true"] == 0).sum())
    fp = int((group["outcome"] == FP).sum())
    fn = int((group["outcome"] == FN).sum())
    errors = group[group["outcome"].isin([FP, FN])]
    dominant = "none" if fp + fn == 0 else (FP if fp > fn else FN if fn > fp else "both")
    return pd.Series(
        {
            "chunks": len(group),
            "positives": positives,
            "negatives": negatives,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": fp / negatives if negatives else np.nan,
            "fn_rate": fn / positives if positives else np.nan,
            "dominant_error": dominant,
            "worst_error_margin": errors["margin"].max() if len(errors) else np.nan,
        }
    )


def own_chunk_errors(scored: pd.DataFrame) -> pd.DataFrame:
    """Errors on a recording's own chunks (real recordings and pool clips)."""
    own = scored[scored["kind"] != "augmented"]
    out = own.groupby("source_recording").apply(_rates, include_groups=False)
    s3 = own.groupby("source_recording")["remote_audio_path"].first()
    out["source"] = own.groupby("source_recording")["kind"].first()
    out["remote_audio_path"] = s3
    return out.reset_index()


def signal_source_errors(scored: pd.DataFrame) -> pd.DataFrame:
    """Per doorbell recording: how often its synthetic mixes get missed, by SNR.

    Synthetic positives are all front_doorbell, so the only error is a miss.
    """
    aug = scored[scored["kind"] == "augmented"]
    if aug.empty:
        return pd.DataFrame()
    by_snr = (
        aug.assign(miss=aug["outcome"] == FN)
        .pivot_table(index="source_recording", columns="snr_db", values="miss", aggfunc="mean")
        .add_prefix("fn_rate_snr")
    )
    overall = aug.groupby("source_recording").apply(
        lambda g: pd.Series({"mixes": len(g), "missed": int((g["outcome"] == FN).sum())}),
        include_groups=False,
    )
    overall["fn_rate"] = overall["missed"] / overall["mixes"]
    return overall.join(by_snr).reset_index().sort_values(["fn_rate", "mixes"], ascending=False)


def noise_source_errors(scored: pd.DataFrame) -> pd.DataFrame:
    """Per noise recording: how often a mix built on it gets missed."""
    aug = scored[(scored["kind"] == "augmented") & scored["noise_file"].notna()]
    if aug.empty:
        return pd.DataFrame()
    out = aug.groupby(["noise_pool", "noise_file"]).apply(
        lambda g: pd.Series({"mixes": len(g), "missed": int((g["outcome"] == FN).sum())}),
        include_groups=False,
    )
    out["fn_rate"] = out["missed"] / out["mixes"]
    return out.reset_index().sort_values(["fn_rate", "mixes"], ascending=False)


def recording_profile(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per recording across every role it plays.

    The same file can be a negative of its own, the signal source of mixes,
    and the noise under other mixes; seeing the three together is what tells
    "this recording is mislabeled" apart from "this recording is just loud".
    """
    own = own_chunk_errors(scored).rename(columns={"source_recording": "recording"})
    own = own.add_prefix("own_").rename(columns={"own_recording": "recording"})
    sig = signal_source_errors(scored)
    if len(sig):
        sig = sig[["source_recording", "mixes", "missed", "fn_rate"]]
        sig.columns = ["recording", "as_signal_mixes", "as_signal_missed", "as_signal_fn_rate"]
    noise = noise_source_errors(scored)
    if len(noise):
        noise = noise.rename(columns={"noise_file": "recording"})
        noise = noise[["recording", "mixes", "missed", "fn_rate"]]
        noise.columns = ["recording", "as_noise_mixes", "as_noise_missed", "as_noise_fn_rate"]
    profile = own
    for part in (sig, noise):
        if len(part):
            profile = profile.merge(part, on="recording", how="outer")
    return profile


def error_contribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Which recordings account for the error rate, largest share first.

    Every misclassified chunk is charged to one recording: its own file, or for
    a synthetic mix the doorbell recording it was built from (the noise file is
    reported separately by noise_source_errors, since a mix has one error but
    two parents). Shares are of all errors, so they sum to 1 and `cum_share`
    reads as a Pareto curve: how few recordings explain most of the mistakes.

    `lift` is share of errors over share of chunks. A recording at 5x is
    producing five times the errors its size would predict, so a large file
    with many easy chunks does not outrank a small one that is nearly all wrong.
    `error_rate_without` is the overall error rate if the recording were
    dropped: what fixing or removing it would buy.
    """
    total_errors = int(scored["outcome"].isin([FP, FN]).sum())
    total_chunks = len(scored)
    grouped = scored.groupby("source_recording")
    out = pd.DataFrame(
        {
            "chunks": grouped.size(),
            "false_positives": grouped["outcome"].apply(lambda s: int((s == FP).sum())),
            "false_negatives": grouped["outcome"].apply(lambda s: int((s == FN).sum())),
        }
    )
    out["errors"] = out["false_positives"] + out["false_negatives"]
    out["share_of_errors"] = out["errors"] / total_errors if total_errors else np.nan
    out["share_of_chunks"] = out["chunks"] / total_chunks
    out["lift"] = out["share_of_errors"] / out["share_of_chunks"]
    out["error_rate_without"] = (total_errors - out["errors"]) / (total_chunks - out["chunks"])
    out = out[out["errors"] > 0].sort_values(["errors", "lift"], ascending=False)
    out["cum_share"] = out["share_of_errors"].cumsum()
    out["source"] = scored.groupby("source_recording")["kind"].agg(
        lambda s: "synthetic+real" if s.nunique() > 1 else s.iloc[0]
    )
    return out.reset_index()


def error_chunks(scored: pd.DataFrame) -> pd.DataFrame:
    """Every misclassified chunk with the full chain back to its sources."""
    bad = scored[scored["outcome"].isin([FP, FN])]
    columns = [
        "row_index", "outcome", "y_score", "margin", "fold", "kind", "label",
        "source_recording", "chunk_start", "chunk_end", "remote_audio_path",
        "snr_db", "noise_pool", "noise_file", "noise_start_ms", "signal_start_ms",
        "annotation_id",
    ]
    return bad[[c for c in columns if c in bad]].sort_values("margin", ascending=False)


# ---------------------------------------------------------------- HTML views

CSS = (
    "body{font:14px system-ui,sans-serif;margin:2rem auto;max-width:1200px;color:#222}"
    "h1,h2{margin-top:2rem}table{border-collapse:collapse;width:100%;margin:.5rem 0}"
    "th,td{padding:4px 8px;border-bottom:1px solid #ddd;text-align:left;font-size:13px}"
    "th{background:#f4f4f4}td.n{text-align:right;font-variant-numeric:tabular-nums}"
    "p.note{color:#555;max-width:900px}code{background:#f4f4f4;padding:0 3px}"
)


HEAT_COLUMNS = {"share_of_errors", "cum_share"}


def _cell(value, heat: bool) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return '<td class="n">–</td>'
        style = f' style="background:rgba(220,60,40,{min(value, 1) * 0.7:.2f})"' if heat else ""
        return f'<td class="n"{style}>{value:.2f}</td>' if heat or abs(value) < 10 else f'<td class="n">{value:.1f}</td>'
    if isinstance(value, (int, np.integer)):
        return f'<td class="n">{value}</td>'
    return f"<td>{html.escape(str(value))}</td>"


def html_table(frame: pd.DataFrame, heat_prefixes=("fp_rate", "fn_rate", "own_fp_rate", "own_fn_rate", "as_")) -> str:
    if frame.empty:
        return "<p><em>nothing to show</em></p>"
    heats = [c for c in frame.columns if (str(c).startswith(heat_prefixes) and "rate" in str(c)) or c in HEAT_COLUMNS]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = "".join(
        "<tr>" + "".join(_cell(v, c in heats) for c, v in zip(frame.columns, row)) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"<table><tr>{head}</tr>{rows}</table>"


def build_report(scored: pd.DataFrame, top: int = 25, min_chunks: int = 5) -> str:
    own = own_chunk_errors(scored)
    fp_view = own[own["false_positives"] > 0].sort_values(
        ["false_positives", "fp_rate"], ascending=False
    ).head(top)[["source_recording", "source", "chunks", "negatives", "false_positives", "fp_rate", "worst_error_margin", "remote_audio_path"]]
    fn_view = own[own["false_negatives"] > 0].sort_values(
        ["false_negatives", "fn_rate"], ascending=False
    ).head(top)[["source_recording", "source", "chunks", "positives", "false_negatives", "fn_rate", "worst_error_margin", "remote_audio_path"]]
    sig = signal_source_errors(scored)
    sig_view = sig[sig["mixes"] >= min_chunks].head(top)
    noise = noise_source_errors(scored)
    noise_view = noise[noise["mixes"] >= min_chunks].head(top)
    errors = error_chunks(scored)
    contrib = error_contribution(scored)
    n50 = int((contrib["cum_share"] < 0.5).sum()) + 1 if len(contrib) else 0
    n80 = int((contrib["cum_share"] < 0.8).sum()) + 1 if len(contrib) else 0
    counts = scored["outcome"].value_counts()
    both = own[(own["false_positives"] > 0) & (own["false_negatives"] > 0)]

    sections = [
        f"<h1>Recording error profile</h1><p class='note'>{len(scored)} held-out chunks · "
        f"{counts.get(FP, 0)} false positives · {counts.get(FN, 0)} false negatives · "
        f"{scored['source_recording'].nunique()} source recordings. Every verdict comes from a fold "
        f"model that never trained on the chunk's recording (grouped CV).</p>",
        f"<h2>Who contributes most to the error rate</h2><p class='note'>{n50} recordings account for "
        f"half of all errors and {n80} for 80% (of {len(contrib)} with any). Mix errors are charged to "
        "their doorbell recording. <code>lift</code> = share of errors / share of chunks; "
        "<code>error_rate_without</code> = overall error rate if that recording were dropped.</p>"
        + html_table(contrib.head(top)[["source_recording", "source", "chunks", "false_positives", "false_negatives", "errors", "share_of_errors", "cum_share", "lift", "error_rate_without"]]),
        "<h2>False-positive-prone recordings</h2><p class='note'>Background recordings (and noise-pool clips) "
        "the model calls a doorbell. High <code>fp_rate</code> with a high worst margin often means a mislabeled "
        "or genuinely bell-like clip. Listen first.</p>" + html_table(fp_view),
        "<h2>False-negative-prone recordings</h2><p class='note'>Real doorbell recordings whose chunks the model misses.</p>"
        + html_table(fn_view),
        "<h2>Doorbell recordings whose synthetic mixes get missed</h2><p class='note'>Fraction of mixes built "
        f"from each doorbell file that are missed, by SNR (min {min_chunks} mixes). Misses that persist at high SNR "
        "point at the source clip rather than the noise.</p>" + html_table(sig_view),
        "<h2>Noise sources that break mixes</h2><p class='note'>Background recordings or pool clips under mixes "
        f"that get missed (min {min_chunks} mixes).</p>" + html_table(noise_view),
        f"<h2>Recordings with both error types</h2>{html_table(both[['source_recording','chunks','false_positives','false_negatives']])}",
        f"<h2>All {len(errors)} error chunks</h2><p class='note'>Highest-margin (most confident) mistakes first. "
        "<code>row_index</code> feeds <code>lineage.py trace</code>.</p>" + html_table(errors),
    ]
    return f"<!doctype html><meta charset=utf-8><title>Recording error profile</title><style>{CSS}</style>" + "".join(sections)


# ----------------------------------------------------------------------- CLI

def load_scored(oof_path: Path = OOF_PREDICTIONS, lineage_path: Path = AUGMENTATION_LINEAGE) -> pd.DataFrame:
    augmentation = pd.read_csv(lineage_path) if Path(lineage_path).exists() else None
    if augmentation is None:
        print(f"warning: {lineage_path} missing; synthetic chunks lose their noise lineage")
    return with_outcomes(build_chunk_lineage(chunk_provenance(), augmentation), pd.read_csv(oof_path))


def write_report(out_dir: Path = LINEAGE_REPORT_DIR) -> Path:
    scored = load_scored()
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "chunk_lineage.csv", index=False)
    recording_profile(scored).to_csv(out_dir / "recording_profile.csv", index=False)
    error_chunks(scored).to_csv(out_dir / "error_chunks.csv", index=False)
    error_contribution(scored).to_csv(out_dir / "error_contribution.csv", index=False)
    (out_dir / "index.html").write_text(build_report(scored))
    return out_dir / "index.html"


def trace(key: str) -> str:
    """The chain for one chunk, by row_index or annotation_id."""
    lineage = build_chunk_lineage(
        chunk_provenance(),
        pd.read_csv(AUGMENTATION_LINEAGE) if AUGMENTATION_LINEAGE.exists() else None,
    )
    match = lineage[lineage["row_index"] == int(key)] if key.isdigit() else lineage[lineage["annotation_id"] == key]
    if match.empty:
        raise SystemExit(f"no chunk matches {key!r}")
    verdicts = pd.read_csv(OOF_PREDICTIONS).set_index("row_index") if OOF_PREDICTIONS.exists() else None
    lines = []
    for _, r in match.iterrows():
        lines.append(f"chunk {r.row_index}  [{r.kind}]  label={r.label}  window {r.chunk_start}-{r.chunk_end} ms")
        if verdicts is not None and r.row_index in verdicts.index:
            v = verdicts.loc[r.row_index]
            lines.append(f"  verdict      {v.outcome}  score={v.y_score:.3f}  margin={v.margin:.3f}  fold={v.fold}")
        if r.kind == "augmented":
            lines.append(f"  signal       {r.signal_file} @ {int(r.signal_start_ms)} ms  (doorbell)")
            noise_at = "" if pd.isna(r.noise_start_ms) else f" @ {int(r.noise_start_ms)} ms, rolled {int(r.noise_roll_samples)} samples"
            lines.append(f"  noise        [{r.noise_pool}] {r.noise_file}{noise_at}")
            lines.append(f"  mix          SNR {r.snr_db} dB, gain {r.gain_db} dB")
        else:
            lines.append(f"  recording    {r.audio_file_name}")
            if isinstance(r.remote_audio_path, str) and r.remote_audio_path:
                lines.append(f"  s3           {r.remote_audio_path}")
            lines.append(f"  annotation   {r.annotation_id}  (span {r.start}-{r.end} s)")
        lines.append(f"  split_group  {r.split_group}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="write CSVs and index.html to data/lineage_report/")
    one = sub.add_parser("trace", help="print one chunk's lineage")
    one.add_argument("key", help="row_index or annotation_id")
    args = parser.parse_args()
    if args.command == "report":
        print(f"wrote {write_report()}")
    else:
        print(trace(args.key))
