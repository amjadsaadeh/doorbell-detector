"""Tests for chunk-to-recording lineage and the error-prone-recording roll-ups.

The roll-ups decide which recording someone goes to relabel or re-record, so
what is pinned down is attribution: synthetic mixes count against the doorbell
recording (and the noise file) they were built from, never the synthetic file,
and false positives/negatives are never pooled into one error count.
"""

import unittest

import numpy as np
import pandas as pd

from src.lineage import (
    build_chunk_lineage,
    classify_kind,
    error_chunks,
    error_contribution,
    noise_source_errors,
    own_chunk_errors,
    recording_profile,
    signal_source_errors,
    with_outcomes,
)


def provenance_frame() -> pd.DataFrame:
    rows = [
        # real background, one FP-prone recording
        ("a1", "bg1.wav", "bg1.wav", np.nan, np.nan),
        ("a1", "bg1.wav", "bg1.wav", np.nan, np.nan),
        # real doorbell recording, missed once
        ("a2", "bell1.wav", "bell1.wav", np.nan, np.nan),
        ("a2", "bell1.wav", "bell1.wav", np.nan, np.nan),
        # external noise clip
        ("noise_esc50_x", "x.wav", "x.wav", np.nan, "esc50"),
        # two mixes of bell1 over bg1, one missed
        ("aug_snr0_0_0", "aug_snr0_0_0.wav", "bell1.wav", 0.0, "labeled_background"),
        ("aug_snr0_0_1", "aug_snr0_0_1.wav", "bell1.wav", 0.0, "labeled_background"),
    ]
    return pd.DataFrame(
        {
            "row_index": range(len(rows)),
            "annotation_id": [r[0] for r in rows],
            "audio_file_name": [r[1] for r in rows],
            "split_group": [r[2] for r in rows],
            "snr_db": [r[3] for r in rows],
            "noise_pool": [r[4] for r in rows],
            "remote_audio_path": "s3://b/k.wav",
            "label": ["background"] * 2 + ["front_doorbell"] * 2 + ["background"] + ["front_doorbell"] * 2,
            "chunk_start": 0,
            "chunk_end": 2000,
        }
    )


def augmentation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annotation_id": ["aug_snr0_0_0", "aug_snr0_0_1"],
            "signal_file": ["bell1.wav"] * 2,
            "signal_start_ms": [500, 500],
            "noise_file": ["bg1.wav", "bg1.wav"],
            "noise_start_ms": [0, 250],
            "noise_roll_samples": [10, 20],
            "gain_db": [1.0, -1.0],
        }
    )


def oof_frame() -> pd.DataFrame:
    # row: (y_true, outcome)
    verdicts = [
        (0, "false_positive"), (0, "true_negative"),
        (1, "false_negative"), (1, "true_positive"),
        (0, "true_negative"),
        (1, "false_negative"), (1, "true_positive"),
    ]
    return pd.DataFrame(
        {
            "row_index": range(len(verdicts)),
            "fold": 0,
            "y_true": [v[0] for v in verdicts],
            "y_score": 0.5,
            "margin": [0.4, 0.1, 0.3, 0.2, 0.1, 0.45, 0.05],
            "outcome": [v[1] for v in verdicts],
        }
    )


def scored() -> pd.DataFrame:
    lineage = build_chunk_lineage(provenance_frame(), augmentation_frame())
    return with_outcomes(lineage, oof_frame())


class ChunkLineageTest(unittest.TestCase):
    def test_kinds(self):
        kinds = classify_kind(provenance_frame()).tolist()
        self.assertEqual(kinds, ["real"] * 4 + ["noise_pool"] + ["augmented"] * 2)

    def test_mixes_resolve_to_their_doorbell_recording_and_noise_file(self):
        table = build_chunk_lineage(provenance_frame(), augmentation_frame())
        mix = table[table["annotation_id"] == "aug_snr0_0_1"].iloc[0]
        self.assertEqual(mix["source_recording"], "bell1.wav")
        self.assertEqual(mix["noise_file"], "bg1.wav")
        self.assertEqual(mix["noise_start_ms"], 250)

    def test_works_without_the_lineage_table(self):
        table = build_chunk_lineage(provenance_frame(), None)
        self.assertEqual(len(table), 7)
        self.assertTrue(table["noise_file"].isna().all())
        # falls back to split_group, which already names the signal recording
        self.assertEqual(table.loc[5, "source_recording"], "bell1.wav")

    def test_row_count_is_preserved(self):
        self.assertEqual(len(build_chunk_lineage(provenance_frame(), augmentation_frame())), 7)


class RollupTest(unittest.TestCase):
    def test_false_positives_and_negatives_are_kept_apart(self):
        own = own_chunk_errors(scored()).set_index("source_recording")
        self.assertEqual(own.loc["bg1.wav", "false_positives"], 1)
        self.assertEqual(own.loc["bg1.wav", "false_negatives"], 0)
        self.assertEqual(own.loc["bg1.wav", "fp_rate"], 0.5)
        self.assertEqual(own.loc["bg1.wav", "dominant_error"], "false_positive")
        self.assertEqual(own.loc["bell1.wav", "dominant_error"], "false_negative")
        self.assertEqual(own.loc["x.wav", "dominant_error"], "none")

    def test_mix_errors_count_against_the_signal_not_the_synthetic_file(self):
        own = own_chunk_errors(scored())
        self.assertFalse(own["source_recording"].str.startswith("aug_").any())
        sig = signal_source_errors(scored()).set_index("source_recording")
        self.assertEqual(sig.loc["bell1.wav", "mixes"], 2)
        self.assertEqual(sig.loc["bell1.wav", "missed"], 1)
        self.assertEqual(sig.loc["bell1.wav", "fn_rate_snr0.0"], 0.5)

    def test_noise_file_is_charged_for_mixes_built_on_it(self):
        noise = noise_source_errors(scored()).set_index("noise_file")
        self.assertEqual(noise.loc["bg1.wav", "mixes"], 2)
        self.assertEqual(noise.loc["bg1.wav", "fn_rate"], 0.5)

    def test_profile_shows_all_roles_of_one_file(self):
        profile = recording_profile(scored()).set_index("recording")
        self.assertEqual(profile.loc["bg1.wav", "own_false_positives"], 1)
        self.assertEqual(profile.loc["bg1.wav", "as_noise_mixes"], 2)
        self.assertEqual(profile.loc["bell1.wav", "as_signal_missed"], 1)

    def test_error_chunks_lists_only_mistakes_most_confident_first(self):
        errors = error_chunks(scored())
        self.assertEqual(set(errors["outcome"]), {"false_positive", "false_negative"})
        self.assertEqual(len(errors), 3)
        self.assertTrue(errors["margin"].is_monotonic_decreasing)

    def test_contribution_charges_mixes_to_signal_and_shares_sum_to_one(self):
        c = error_contribution(scored()).set_index("source_recording")
        # 3 errors: bg1 (FP), bell1 (FN own), bell1 (FN via mix)
        self.assertEqual(c.loc["bell1.wav", "errors"], 2)
        self.assertEqual(c.loc["bg1.wav", "errors"], 1)
        self.assertNotIn("x.wav", c.index)  # no errors, not listed
        self.assertAlmostEqual(c["share_of_errors"].sum(), 1.0)
        self.assertEqual(c.index[0], "bell1.wav")
        self.assertAlmostEqual(c["cum_share"].iloc[-1], 1.0)
        # 4 of 7 chunks are bell1's; dropping it leaves 1 error in 3 chunks
        self.assertAlmostEqual(c.loc["bell1.wav", "error_rate_without"], 1 / 3)

    def test_chunks_no_fold_held_out_are_dropped(self):
        lineage = build_chunk_lineage(provenance_frame(), augmentation_frame())
        self.assertEqual(len(with_outcomes(lineage, oof_frame().head(3))), 3)


if __name__ == "__main__":
    unittest.main()
