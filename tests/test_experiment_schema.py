"""Focused checks for the versioned experiment evidence boundary."""

import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.aggregate import aggregate, load_logs
from experiments.launch_formal_gate import (
    EXPECTED_DATA_SOURCE_SHA256,
    EXPECTED_SOURCE_SNAPSHOT_SHA256,
    build_runs,
    verify_expected_logs,
)
from experiments.official_bciciv_2a import (
    GDF_ARCHIVE_NAME,
    LABEL_ARCHIVE_NAME,
    OFFICIAL_EEG_CHANNEL_NAMES,
    OFFICIAL_EOG_CHANNEL_NAMES,
    OfficialBCICIV2aSource,
    OfficialDatasetError,
    official_eeg_picks,
)
from experiments.schema import (
    canonical_sha256,
    git_provenance,
    normalization_fit_scope,
    RUNNER_VERSION,
    SCHEMA_VERSION,
    # Aliased: pytest would otherwise collect the imported ``test_*`` name as a
    # test case.
    test_subject_used_for_normalization as norm_uses_test_subject,
    protocol_id,
    selection_mode,
    selection_split,
    validate_log,
)
from figures.gen_fig1_factorization import build_figure as build_fig1
from figures.gen_fig3_paired_effects import build_figure as build_fig3
from figures.gen_fig4_pareto import build_figure as build_fig4
from figures.gen_fig5_subject_shift import build_figure as build_fig5
from figures import gen_tables


def make_log(
    seed=0,
    *,
    dataset="BNCI2014_001",
    normalization="within_trial",
    legacy_loso=False,
):
    subjects = list(range(1, 10))
    protocol = {
        "protocol_id": protocol_id(legacy_loso, normalization),
        "selection_mode": selection_mode(legacy_loso),
        "selection_split": selection_split(legacy_loso),
        "normalization_mode": normalization,
        "normalization_fit_scope": normalization_fit_scope(normalization, legacy_loso),
        "test_subject_used_for_model_selection": legacy_loso,
        "test_subject_used_for_normalization": norm_uses_test_subject(normalization),
        "metric": "accuracy",
    }
    per_subject = {
        str(subject): {
            "main_accuracy": 0.4,
            "balanced_accuracy": 0.4,
            "task_accuracy": {"main": 0.4},
            "confusion_matrix": [[4, 6, 0, 0], [0, 4, 6, 0], [0, 0, 4, 6], [6, 0, 0, 4]],
            "n_params": 10,
            "selected_epoch": 1,
            "best_inner_val_accuracy": 0.4,
            "inner_val_trace": [
                {"epoch": 1, "train_loss": 1.0, "main_accuracy": 0.4}
            ],
            "validation_source": protocol["selection_mode"],
            "time_sec": 0.1,
        }
        for subject in subjects
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "status": "completed",
        "run_id": f"eegnet_equal_BNCI2014_001_s{seed}__{protocol['protocol_id']}",
        "variant": "eegnet",
        "weighting": "equal",
        "dataset": dataset,
        "seed": seed,
        "subjects": subjects,
        "protocol": protocol,
        "metric": "accuracy",
        "mean_accuracy": 0.4,
        "std_accuracy": 0.0,
        "mean_metric": 0.4,
        "std_metric": 0.0,
        "task_definition": {
            "dataset": dataset,
            "n_original_classes": 4,
            "definition_version": "label-map-v1",
            "tasks": [
                {
                    "name": "main",
                    "n_classes": 4,
                    "loss_weight": 1.0,
                    "mapping": "identity",
                }
            ],
        },
        "config": {"task_names": ["main"]},
        "legacy_loso": legacy_loso,
        "normalization": normalization,
        "per_subject": per_subject,
        "provenance": {
            "command": ["python", "paper_runner.py"],
            "command_text": "python paper_runner.py",
            "cwd": "D:/BCI_lab",
            "root": "D:/BCI_lab",
            "python_executable": "D:/python/python.exe",
            "packages": {
                "python": "3.8.4",
                "numpy": "1.24.0",
                "scikit_learn": "1.3.0",
                "moabb": "1.0.0",
                "torch": "2.4.1",
            },
            "git_sha": "test",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
            "untracked_files": [],
            "untracked_file_sha256": {},
            "source_snapshot": {
                "files": [{"path": "experiments/paper_runner.py", "sha256": "a" * 64}],
                "sha256": canonical_sha256({
                    "files": [{"path": "experiments/paper_runner.py", "sha256": "a" * 64}]
                }),
            },
            "data_source": {
                "kind": "synthetic-test",
                "dataset_id": dataset,
                "source_urls": ["https://example.invalid/test"],
            },
        },
    }


def set_subject_values(log, base):
    """Make a non-degenerate, internally consistent synthetic run."""

    values = []
    for index, result in enumerate(log["per_subject"].values(), start=1):
        value = base + 0.01 * index
        result["main_accuracy"] = value
        result["balanced_accuracy"] = value
        result["task_accuracy"] = {"main": value}
        result["best_inner_val_accuracy"] = min(1.0, value + 0.08)
        result["inner_val_trace"] = [
            {"epoch": 1, "train_loss": 1.0, "main_accuracy": result["best_inner_val_accuracy"]}
        ]
        diagonal = int(round(value * 100))
        off_diagonal = 100 - diagonal
        result["confusion_matrix"] = [
            [diagonal, off_diagonal, 0, 0],
            [0, diagonal, off_diagonal, 0],
            [0, 0, diagonal, off_diagonal],
            [off_diagonal, 0, 0, diagonal],
        ]
        values.append(value)
    mean_value = sum(values) / len(values)
    std_value = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
    log["mean_accuracy"] = mean_value
    log["std_accuracy"] = std_value
    log["mean_metric"] = mean_value
    log["std_metric"] = std_value


def finalize_source_descriptor(log):
    descriptor = log["provenance"]["data_source"]
    descriptor["source_sha256"] = canonical_sha256(descriptor)


class ExperimentSchemaTests(unittest.TestCase):
    def test_formal_all_matrix_has_36_unique_identities(self):
        runs = build_runs("all", [0, 1, 2])
        identities = {
            (
                run["variant"], run["weighting"], run["dataset"], run["seed"],
                run["legacy_loso"], run["normalization"],
            )
            for run in runs
        }
        self.assertEqual(len(runs), 36)
        self.assertEqual(len(identities), 36)
        self.assertEqual(sum(run["variant"] == "eegnet" for run in runs), 12)
        self.assertEqual(sum(run["variant"] != "eegnet" for run in runs), 24)
        self.assertEqual(len(build_runs("ablation", [0, 1, 2])), 24)

    def test_official_gdf_channel_contract_excludes_eog_despite_mne_type(self):
        """The GDF reader may type all 25 signals as EEG; names are authoritative."""

        names = OFFICIAL_EEG_CHANNEL_NAMES + OFFICIAL_EOG_CHANNEL_NAMES
        self.assertEqual(official_eeg_picks(names), list(range(22)))

    def test_official_gdf_channel_contract_rejects_non_official_layout(self):
        names = list(OFFICIAL_EEG_CHANNEL_NAMES + OFFICIAL_EOG_CHANNEL_NAMES)
        names[-1] = "EEG-unexpected"
        with self.assertRaises(OfficialDatasetError):
            official_eeg_picks(names)

    def test_official_gdf_channel_contract_rejects_reordered_channels(self):
        names = list(OFFICIAL_EEG_CHANNEL_NAMES + OFFICIAL_EOG_CHANNEL_NAMES)
        names[0], names[1] = names[1], names[0]
        with self.assertRaises(OfficialDatasetError):
            official_eeg_picks(names)

    def test_official_gdf_extraction_replaces_tampered_cache(self):
        payload = b"official-gdf-member-content"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / GDF_ARCHIVE_NAME
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/A01T.gdf", payload)
            source = OfficialBCICIV2aSource(root)
            extracted = source._extract_member(archive_path, "A01T.gdf")
            extracted.write_bytes(b"x" * len(payload))
            restored = source._extract_member(archive_path, "A01T.gdf")
            self.assertEqual(restored.read_bytes(), payload)
            self.assertEqual(restored.parent.name, source._archive_sha256(archive_path))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / ".cache" / "bciciv_2a" / GDF_ARCHIVE_NAME).is_file()
        and (Path(__file__).resolve().parents[1] / ".cache" / "bciciv_2a" / LABEL_ARCHIVE_NAME).is_file(),
        "official BCI Competition IV 2a archives are not available locally",
    )
    def test_real_official_gdf_a01_contract(self):
        source = OfficialBCICIV2aSource(
            Path(__file__).resolve().parents[1] / ".cache" / "bciciv_2a"
        )
        X_train, y_train = source._read_gdf(
            source._gdf_path(1, "T"), 1, "training"
        )
        X_eval, y_from_events = source._read_gdf(
            source._gdf_path(1, "E"), 1, "evaluation-events"
        )
        y_eval = source._evaluation_labels(1)
        self.assertEqual(X_train.shape, (288, 22, 1001))
        self.assertEqual(X_eval.shape, (288, 22, 1001))
        self.assertTrue(np.isfinite(X_train).all())
        self.assertTrue(np.isfinite(X_eval).all())
        self.assertEqual(np.bincount(y_train, minlength=4).tolist(), [72, 72, 72, 72])
        self.assertTrue(np.all(y_from_events == -1))
        self.assertEqual(np.bincount(y_eval, minlength=4).tolist(), [72, 72, 72, 72])

    def test_protocol_id_prevents_normalization_collision(self):
        self.assertNotEqual(
            protocol_id(False, "within_trial"),
            protocol_id(False, "inner_train_scaler"),
        )

    def test_corrected_protocol_names_pooled_trial_split_explicitly(self):
        self.assertEqual(
            selection_split(False),
            "stratified_80_20_outer_train_pooled_trials",
        )
        self.assertEqual(
            normalization_fit_scope("inner_train_scaler", False),
            "inner_train_pooled_trials_only",
        )
        self.assertEqual(
            normalization_fit_scope("outer_train_scaler", False),
            "outer_train_pooled_trials_only",
        )

    def test_provenance_ignores_runtime_and_non_source_untracked_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in (
                ".cache/raw.json",
                ".scratch/pilot.json",
                ".venv/site.py",
                "experiments/schema.py",
                "notes.txt",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            def fake_git(_root, *args):
                if args[:2] == ("ls-files", "--others"):
                    # git is invoked with -z, so paths come back NUL-separated
                    # and unescaped.
                    return "\0".join((
                        ".cache/raw.json",
                        ".scratch/pilot.json",
                        ".venv/site.py",
                        "experiments/schema.py",
                        "notes.txt",
                    ))
                return ""

            with patch("experiments.schema._git", side_effect=fake_git):
                provenance = git_provenance(root)
            self.assertEqual(provenance["untracked_files"], ["experiments/schema.py"])

    def test_partial_subject_log_is_rejected_by_default(self):
        log = make_log()
        finalize_source_descriptor(log)
        log["subjects"] = log["subjects"][:2]
        log["per_subject"] = {"1": log["per_subject"]["1"], "2": log["per_subject"]["2"]}
        errors = validate_log(log)
        self.assertTrue(any("incomplete subject set" in error for error in errors))

    def test_aggregate_writes_manifest_for_valid_v2_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_dir = root / "logs"
            out_dir = root / "results"
            log_dir.mkdir()
            log = make_log()
            finalize_source_descriptor(log)
            (log_dir / "run.json").write_text(json.dumps(log), encoding="utf-8")
            manifest = aggregate(
                log_dir=log_dir,
                out_dir=out_dir,
                protocol_filter=log["protocol"]["protocol_id"],
            )
            self.assertEqual(len(manifest["entries"]), 1)
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "audit_corrected.csv").exists())

    def test_manifest_id_is_stable_across_materialization_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_dir = root / "logs"
            log_dir.mkdir()
            log = make_log()
            finalize_source_descriptor(log)
            (log_dir / "run.json").write_text(json.dumps(log), encoding="utf-8")
            with patch(
                "experiments.aggregate._now",
                side_effect=("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
            ):
                first = aggregate(
                    log_dir=log_dir,
                    out_dir=root / "results-a",
                    protocol_filter=log["protocol"]["protocol_id"],
                )
                second = aggregate(
                    log_dir=log_dir,
                    out_dir=root / "results-b",
                    protocol_filter=log["protocol"]["protocol_id"],
                )
            self.assertNotEqual(first["generated_at"], second["generated_at"])
            self.assertEqual(first["manifest_id"], second["manifest_id"])

    def test_formal_official_gate_rejects_non_official_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            log = make_log()
            finalize_source_descriptor(log)
            (log_dir / f"{log['run_id']}.json").write_text(
                json.dumps(log), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "expected 'official_bciciv_2a_gdf'"):
                verify_expected_logs(
                    build_runs("anchor", [0])[:1],
                    log_dir,
                    data_source="official-gdf",
                )

    def test_formal_gate_checks_planned_metadata_and_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            planned = build_runs("anchor", [0])[0]
            log = make_log(seed=0, normalization="within_trial", legacy_loso=False)
            log["run_id"] = planned["run_id"]
            log["provenance"]["data_source"]["kind"] = "official_bciciv_2a_gdf"
            finalize_source_descriptor(log)
            with patch(
                "experiments.launch_formal_gate.EXPECTED_SOURCE_SNAPSHOT_SHA256",
                log["provenance"]["source_snapshot"]["sha256"],
            ), patch(
                "experiments.launch_formal_gate.EXPECTED_DATA_SOURCE_SHA256",
                log["provenance"]["data_source"]["source_sha256"],
            ):
                path = log_dir / f"{planned['run_id']}.json"
                path.write_text(json.dumps(log), encoding="utf-8")
                verify_expected_logs([planned], log_dir, data_source="official-gdf")

                log["variant"] = "minimal"
                path.write_text(json.dumps(log), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "planned"):
                    verify_expected_logs([planned], log_dir, data_source="official-gdf")

    def test_formal_gate_constants_are_frozen_release_values(self):
        self.assertEqual(len(EXPECTED_SOURCE_SNAPSHOT_SHA256), 64)
        self.assertEqual(len(EXPECTED_DATA_SOURCE_SHA256), 64)

    def test_paper_table_cli_redirects_outputs_and_skips_p300(self):
        old_figure_dir = gen_tables.FIG_DIR
        old_paper_figure_dir = gen_tables.PAPER_FIG_DIR
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_dir = root / "source-figures"
            paper_dir = root / "manuscript" / "figures"
            with patch.object(sys, "argv", [
                "gen_tables.py",
                "--results-dir", str(root / "results"),
                "--output-dir", str(output_dir),
                "--paper-fig-dir", str(paper_dir),
                "--skip-p300",
            ]), patch.object(gen_tables, "factorization_tables") as factorization, patch.object(
                gen_tables, "ablation_table"
            ) as ablation, patch.object(gen_tables, "p300_table") as p300:
                gen_tables.main()
            factorization.assert_called_once_with(str(root / "results"))
            ablation.assert_called_once_with(str(root / "results"))
            p300.assert_not_called()
            self.assertEqual(gen_tables.FIG_DIR, output_dir)
            self.assertEqual(gen_tables.PAPER_FIG_DIR, paper_dir)
        gen_tables.FIG_DIR = old_figure_dir
        gen_tables.PAPER_FIG_DIR = old_paper_figure_dir

    def test_factorization_table_rejects_unmatched_seed_sets(self):
        logs = []
        for legacy_loso in (False, True):
            for normalization in ("within_trial", "outer_train_scaler"):
                seeds = (0, 1) if (not legacy_loso and normalization == "within_trial") else (0,)
                for seed in seeds:
                    logs.append(make_log(
                        seed=seed,
                        normalization=normalization,
                        legacy_loso=legacy_loso,
                    ))
        with patch.object(gen_tables, "load_manifest", return_value={"manifest_id": "test"}), patch.object(
            gen_tables, "load_manifest_logs", return_value=({}, logs)
        ):
            with self.assertRaisesRegex(ValueError, "factorization seed sets differ"):
                gen_tables.factorization_tables("unused")

    def test_old_log_is_skipped_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            (log_dir / "old.json").write_text(json.dumps({"run_id": "old"}), encoding="utf-8")
            logs, skipped, _ = load_logs(log_dir)
            self.assertEqual(logs, [])
            self.assertEqual(len(skipped), 1)

    def test_missing_provenance_field_is_rejected(self):
        log = make_log()
        finalize_source_descriptor(log)
        del log["provenance"]["packages"]
        errors = validate_log(log)
        self.assertTrue(any("provenance field 'packages'" in error for error in errors))

    def test_missing_task_mapping_is_rejected(self):
        log = make_log()
        finalize_source_descriptor(log)
        del log["task_definition"]["tasks"][0]["mapping"]
        errors = validate_log(log)
        self.assertTrue(any("field 'mapping'" in error for error in errors))

    def test_mixed_protocol_manifest_skips_generic_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_dir = root / "logs"
            out_dir = root / "results"
            log_dir.mkdir()
            for legacy_loso in (False, True):
                for normalization in ("within_trial", "outer_train_scaler"):
                    log = make_log(
                        normalization=normalization,
                        legacy_loso=legacy_loso,
                    )
                    finalize_source_descriptor(log)
                    (log_dir / f"{log['run_id']}.json").write_text(
                        json.dumps(log), encoding="utf-8"
                    )
            aggregate(log_dir=log_dir, out_dir=out_dir, protocol_filter="all")
            self.assertTrue((out_dir / "audit_factorization.csv").exists())
            self.assertTrue((out_dir / "standard_blocks_skipped.md").exists())
            self.assertFalse((out_dir / "block_A_aux_comparison.csv").exists())

    def test_manifest_backed_figures_generate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_dir = root / "logs"
            out_dir = root / "results"
            fig_dir = root / "figures"
            log_dir.mkdir()
            for legacy_loso in (False, True):
                for normalization in ("within_trial", "outer_train_scaler"):
                    for seed in (0, 1):
                        log = make_log(
                            seed=seed,
                            normalization=normalization,
                            legacy_loso=legacy_loso,
                        )
                        finalize_source_descriptor(log)
                        set_subject_values(log, 0.30 + 0.01 * seed)
                        (log_dir / f"{log['run_id']}.json").write_text(
                            json.dumps(log), encoding="utf-8"
                        )
            for seed in (0, 1):
                log = make_log(seed=seed)
                finalize_source_descriptor(log)
                log["variant"] = "minimal"
                log["run_id"] = (
                    f"minimal_equal_BNCI2014_001_s{seed}__"
                    f"{log['protocol']['protocol_id']}"
                )
                set_subject_values(log, 0.28 + 0.01 * seed)
                (log_dir / f"{log['run_id']}.json").write_text(
                    json.dumps(log), encoding="utf-8"
                )
            aggregate(log_dir=log_dir, out_dir=out_dir, protocol_filter="all")
            for build_figure, name in [
                (build_fig1, "fig1_factorization.pdf"),
                (build_fig3, "fig3_paired_effects.pdf"),
                (build_fig4, "fig4_pareto.pdf"),
                (build_fig5, "fig5_subject_shift.pdf"),
            ]:
                build_figure(out_dir, fig_dir)
                self.assertTrue((fig_dir / name).exists())


if __name__ == "__main__":
    unittest.main()
