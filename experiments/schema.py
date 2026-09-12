"""Versioned experiment metadata and provenance helpers.

The paper tables must be derived from logs that identify the evaluation
protocol and the source used to produce them.  This module is deliberately
small and dependency-light so both the runner and the aggregator can use it
before importing the training stack.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 4
RUNNER_VERSION = "4.0.0"
PROTOCOL_FAMILY = "cross-subject-loso"

NORMALIZATION_MODES = (
    "within_trial",
    "within_trial_global",
    "outer_train_scaler",
    "inner_train_scaler",
    "pooled_all_subject_scaler",
)

# Normalization modes whose transform is fitted with samples from the held-out
# subject.  These are audit conditions, not zero-calibration procedures: the
# information boundary is crossed at the preprocessing step rather than at
# checkpoint selection.
LEAKY_NORMALIZATION_MODES = frozenset({"pooled_all_subject_scaler"})

# What the training set is allowed to contain.  Every level scores the same
# object -- one held-out subject's reserved test trials -- so the two are
# directly comparable.  ``subject`` trains on the outer pool alone; ``trial``
# adds the held-out subject's own labelled crossing trials, which is the
# coarsest information-boundary violation audited here.
#
# An earlier version cut ``trial`` folds at random over pooled trials.  Its
# points were then cohort-wide fold means rather than per-subject scores, and
# averaging over subjects shrinks variance mechanically, so its spread was never
# comparable with the clean arm's.
SPLIT_LEVELS = ("subject", "trial")

# These are the subject counts expected from the MOABB datasets used by the
# paper.  A pilot with --max_subjects is therefore rejected by the aggregator
# unless the caller explicitly opts into partial runs.
# PhysionetMI ships 109 subjects, but four recordings are unusable: S088, S089,
# S092 and S100 carry a non-standard sampling rate or inconsistent annotations,
# and are excluded by convention throughout the literature.  Excluding them here
# rather than at analysis time keeps the declared subject set and the evaluated
# subject set the same object.
PHYSIONET_EXCLUDED_SUBJECTS = (88, 89, 92, 100)
PHYSIONET_SUBJECTS = tuple(
    s for s in range(1, 110) if s not in PHYSIONET_EXCLUDED_SUBJECTS
)

DATASET_EXPECTED_SUBJECT_COUNTS = {
    "BNCI2014_001": 9,
    "BNCI2015_004": 9,
    "BNCI2014_009": 10,
    "PhysionetMI": len(PHYSIONET_SUBJECTS),
    "BNCI2014_002": 14,
}

DATASET_EXPECTED_SUBJECT_IDS = {
    "BNCI2014_001": tuple(range(1, 10)),
    "BNCI2015_004": tuple(range(1, 10)),
    "BNCI2014_009": tuple(range(1, 11)),
    "PhysionetMI": PHYSIONET_SUBJECTS,
    "BNCI2014_002": tuple(range(1, 15)),
}

DATASET_MAIN_CLASSES = {
    "BNCI2014_001": 4,
    "BNCI2015_004": 5,
    "BNCI2014_009": 2,
    # left_hand / right_hand / hands / feet.  The 'rest' class is dropped so the
    # task matches the four-class motor imagery of BNCI2014_001.
    "PhysionetMI": 4,
    # right_hand vs feet.  Two classes rather than four, so chance is 0.5 here:
    # accuracy is not comparable across datasets, though the paired differences
    # and the spread ratios are.
    "BNCI2014_002": 2,
}

REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "runner_version",
    "status",
    "run_id",
    "variant",
    "weighting",
    "dataset",
    "seed",
    "subjects",
    "protocol",
    "metric",
    "mean_accuracy",
    "std_accuracy",
    "mean_metric",
    "std_metric",
    "task_definition",
    "config",
    "legacy_loso",
    "normalization",
    "per_subject",
    "provenance",
)

REQUIRED_PROTOCOL_FIELDS = (
    "protocol_id",
    "selection_mode",
    "selection_split",
    "normalization_mode",
    "normalization_fit_scope",
    "test_subject_used_for_model_selection",
    "test_subject_used_for_normalization",
    "metric",
)

REQUIRED_PROVENANCE_FIELDS = (
    "command",
    "command_text",
    "cwd",
    "root",
    "python_executable",
    "packages",
    "git_sha",
    "git_dirty",
    "git_diff_sha256",
    "untracked_files",
    "untracked_file_sha256",
    "source_snapshot",
    "data_source",
)

# What every log must carry.  ``mne`` and ``braindecode`` are collected by
# package_versions() but deliberately absent here: requiring them would reject
# the logs already released, which were written before either was recorded.
REQUIRED_PACKAGE_VERSION_FIELDS = (
    "python",
    "numpy",
    "scikit_learn",
    "moabb",
    "torch",
)

REQUIRED_TASK_DEFINITION_FIELDS = (
    "dataset",
    "n_original_classes",
    "tasks",
    "definition_version",
)

REQUIRED_TASK_FIELDS = ("name", "n_classes", "loss_weight", "mapping")

REQUIRED_SOURCE_SNAPSHOT_FIELDS = ("sha256", "files")
REQUIRED_DATA_SOURCE_FIELDS = ("kind", "dataset_id", "source_urls", "source_sha256")

# Runtime environments, downloaded data, and intermediate logs are not source
# inputs.  Including them in untracked-file provenance makes each paper log
# enormous and changes its digest whenever a cache is touched.
PROVENANCE_EXCLUDED_TOP_LEVEL = {
    ".cache",
    ".scratch",
    ".venv",
    ".venv",
    "__pycache__",
}
PROVENANCE_SOURCE_SUFFIXES = {
    ".bib",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".yaml",
    ".yml",
}

REQUIRED_PER_SUBJECT_FIELDS = (
    "main_accuracy",
    "balanced_accuracy",
    "task_accuracy",
    "confusion_matrix",
    "n_params",
    "selected_epoch",
    "best_inner_val_accuracy",
    "inner_val_trace",
    "validation_source",
    "time_sec",
)


def selection_mode(legacy_loso: bool) -> str:
    """Return the serialized model-selection source.

    ``inner_train_trials`` deliberately names the sample level.  The corrected
    runner draws this split from trials pooled across the outer-training
    subjects; it is not a subject-level validation split.
    """

    return "heldout_subject" if legacy_loso else "inner_train_trials"


def selection_split(legacy_loso: bool) -> str:
    """Describe the exact samples used to choose the early-stop checkpoint."""

    return (
        "heldout_subject_trials"
        if legacy_loso
        else "stratified_80_20_outer_train_pooled_trials"
    )


def normalization_fit_scope(normalization_mode: str, legacy_loso: bool) -> str:
    """Describe exactly which samples can fit a normalization transform."""

    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"unknown normalization mode {normalization_mode!r}")
    if normalization_mode == "within_trial":
        return "per_trial_time_axis"
    if normalization_mode == "within_trial_global":
        return "per_trial_channel_time_axes"
    if normalization_mode == "pooled_all_subject_scaler":
        return "all_subject_pooled_trials_including_heldout"
    if normalization_mode == "inner_train_scaler":
        if legacy_loso:
            raise ValueError(
                "inner_train_scaler requires the corrected inner-train-trial "
                "selection protocol"
            )
        return "inner_train_pooled_trials_only"
    return "outer_train_pooled_trials_only"


def test_subject_used_for_normalization(normalization_mode: str) -> bool:
    """Report whether the normalization transform sees held-out-subject trials.

    This is the normalization-side counterpart of
    ``test_subject_used_for_model_selection``.  Keeping the two flags separate
    lets a manifest state exactly which step crossed the information boundary.
    """

    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"unknown normalization mode {normalization_mode!r}")
    return normalization_mode in LEAKY_NORMALIZATION_MODES


def evaluation_unit(split_level: str) -> str:
    """Name what one fold scores.

    Every level scores one held-out subject's reserved test trials, so this is
    constant.  It is kept as an explicit field because a log that reports
    anything else came from the superseded random-fold design.
    """

    if split_level not in SPLIT_LEVELS:
        raise ValueError(f"unknown split level {split_level!r}")
    return "heldout_subject_reserved_test_trials"


def test_subject_used_for_training(split_level: str) -> bool:
    """Report whether labelled trials of an evaluated subject reach training."""

    if split_level not in SPLIT_LEVELS:
        raise ValueError(f"unknown split level {split_level!r}")
    return split_level == "trial"


def protocol_id(
    legacy_loso: bool,
    normalization_mode: str,
    split_level: str = "subject",
    euclidean_alignment: bool = False,
) -> str:
    """Build a stable protocol identifier used in run IDs and tables."""

    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"unknown normalization mode {normalization_mode!r}; "
            f"expected one of {NORMALIZATION_MODES}"
        )
    if split_level not in SPLIT_LEVELS:
        raise ValueError(
            f"unknown split level {split_level!r}; expected one of {SPLIT_LEVELS}"
        )
    # Defaults are omitted from the ID so that every pre-existing subject-level,
    # unaligned protocol ID, run ID and manifest key stays byte-identical.
    suffix = "" if split_level == "subject" else f"-split-{split_level}"
    if euclidean_alignment:
        suffix += "-align-euclidean"
    return (
        f"{PROTOCOL_FAMILY}-v2-"
        f"selection-{selection_mode(legacy_loso)}-"
        f"normalization-{normalization_mode}"
        f"{suffix}"
    )


# ---------------------------------------------------------------------------
# schema-v4: the target-access factorization
#
# v3 described five "ladder rungs" that each changed one nominal dimension, but
# the arms were not otherwise matched: the held-out-subject selection arm
# trained on 25% more source trials than the clean arm, the trial-level arm
# re-split source train/validation after mixing target trials in, and the
# aligned arm fitted its whitener on the held-out subject's reserved test
# trials.  v4 fixes the source partition, the optimizer budget and the target
# sampling rate across every condition, so the only thing that varies is what
# the held-out subject's *available* half is allowed to contribute.
# ---------------------------------------------------------------------------

# How a condition may use the held-out subject's available (crossing) half.
TARGET_ACCESS_MODES = ("none", "signals", "labels", "signals_and_labels")


def target_access_mode(euclidean_alignment: bool, target_supervised: bool) -> str:
    """Name what the held-out subject's available half contributes.

    ``signals`` is label-free: Euclidean Alignment estimates a whitener from the
    available trials only.  ``labels`` puts those trials into the training set
    with their labels.  The two are orthogonal, which is what makes the four
    conditions a factorial design rather than an ordered ladder.
    """

    if euclidean_alignment and target_supervised:
        return "signals_and_labels"
    if euclidean_alignment:
        return "signals"
    if target_supervised:
        return "labels"
    return "none"


def target_reference_scope(euclidean_alignment: bool) -> str:
    """Describe which trials fit the held-out subject's alignment reference."""

    if not euclidean_alignment:
        return "not_applicable"
    return "heldout_subject_available_trials_only"


def selection_scope_v4(legacy_loso: bool) -> str:
    """Describe the samples that choose the checkpoint under the v4 controls.

    Under v4 the validation set is the same source-only index set in every
    condition, so the checkpoint rule cannot differ between arms by accident.
    """

    return (
        "heldout_subject_available_trials"
        if legacy_loso
        else "fixed_source_inner_validation_trials"
    )


def protocol_id_v4(
    euclidean_alignment: bool,
    target_supervised: bool,
    normalization_mode: str,
    legacy_loso: bool = False,
) -> str:
    """Build the v4 protocol identifier.

    The two factorial bits are always written out, even when both are off, so a
    v4 identifier can never be confused with a v3 one that merely omitted its
    defaults.
    """

    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"unknown normalization mode {normalization_mode!r}; "
            f"expected one of {NORMALIZATION_MODES}"
        )
    suffix = "-selection-heldout_subject" if legacy_loso else ""
    return (
        f"{PROTOCOL_FAMILY}-v4-"
        f"align-{int(bool(euclidean_alignment))}-"
        f"targetsup-{int(bool(target_supervised))}-"
        f"normalization-{normalization_mode}"
        f"{suffix}"
    )


def metric_name(dataset: str) -> str:
    """Return the primary reported metric for a dataset."""

    return "balanced_accuracy" if dataset == "BNCI2014_009" else "accuracy"


def task_definition(dataset: str, task_configs: Iterable[Any]) -> dict[str, Any]:
    """Serialize task heads and their label mappings for a run manifest."""

    # Read the class count from the shared constant rather than a second copy
    # of it.  A private dict here was never extended when BNCI2014_002 and
    # PhysionetMI were added, so both wrote a null into this field and the gate
    # rejected every one of their logs -- the numbers were unaffected, which is
    # exactly what makes a duplicated constant hard to notice.
    n_main = DATASET_MAIN_CLASSES.get(dataset)
    if n_main is None:
        raise ValueError(
            f"no declared class count for dataset {dataset!r}; add it to "
            "DATASET_MAIN_CLASSES rather than defaulting it here"
        )
    tasks = []
    for task in task_configs:
        name = str(task.name)
        n_classes = int(task.n_classes)
        entry: dict[str, Any] = {
            "name": name,
            "n_classes": n_classes,
            "loss_weight": float(task.loss_weight),
        }
        if name == "main" or n_classes == n_main:
            entry["mapping"] = "identity"
        elif n_classes == 2:
            entry["mapping"] = "class_0_vs_rest"
        elif n_classes == 3 and n_main == 4:
            entry["mapping"] = {"0": 0, "1": 0, "2": 1, "3": 2}
        elif n_classes == 3 and n_main == 5:
            entry["mapping"] = {"0": 0, "1": 0, "2": 1, "3": 2, "4": 2}
        else:
            entry["mapping"] = "modulo_fallback"
        tasks.append(entry)
    return {
        "dataset": dataset,
        "n_original_classes": n_main,
        "tasks": tasks,
        "definition_version": "label-map-v1",
    }


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(root: Path) -> dict[str, Any]:
    """Hash every Python source file used by the runner, including untracked edits."""

    candidates: set[Path] = set(root.glob("*.py"))
    for directory in ("experiments", "EEGNets", "configs"):
        base = root / directory
        if base.is_dir():
            candidates.update(base.rglob("*.py"))
    files = []
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        })
    payload = {"files": files}
    return {"sha256": canonical_sha256(payload), **payload}


def git_provenance(root: Path) -> dict[str, Any]:
    """Collect a compact, reproducible description of the source tree."""

    diff = _git(root, "diff", "HEAD", "--binary")
    staged = _git(root, "diff", "--cached", "--binary")
    # ``-z`` emits NUL-separated, unquoted paths.  Without it git applies
    # ``core.quotepath`` escaping to non-ASCII names, and the escaped form is
    # not a path that can be opened.
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_hashes = {}
    for relative_path in filter(None, untracked.split("\0")):
        path = root / relative_path
        parts = Path(relative_path).parts
        if (
            path.is_file()
            and parts
            and parts[0] not in PROVENANCE_EXCLUDED_TOP_LEVEL
            and path.suffix.lower() in PROVENANCE_SOURCE_SUFFIXES
        ):
            untracked_hashes[relative_path] = _sha256_file(path)
    diff_payload = "\n".join((
        diff,
        staged,
        json.dumps(untracked_hashes, sort_keys=True, separators=(",", ":")),
    ))
    return {
        "git_sha": _git(root, "rev-parse", "HEAD") or None,
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff_payload.encode("utf-8")).hexdigest(),
        "untracked_files": sorted(untracked_hashes),
        "untracked_file_sha256": untracked_hashes,
    }


def package_versions() -> dict[str, str | None]:
    """Return versions without importing optional training packages.

    ``braindecode`` and ``mne`` are recorded but not required: a run that uses
    the reference decoder is only interpretable against the version of that
    decoder, and an earlier revision of this list left it out entirely, so the
    reference-implementation logs could not say which implementation they had
    referenced.  A missing entry is ``None`` rather than an error, since most
    runs do not import either package.
    """

    names = {
        "python": platform.python_version(),
        "numpy": "numpy",
        "scikit_learn": "scikit-learn",
        "moabb": "moabb",
        "torch": "torch",
        "mne": "mne",
        "braindecode": "braindecode",
    }
    versions: dict[str, str | None] = {"python": names["python"]}
    for key, distribution in list(names.items())[1:]:
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def runtime_provenance(root: Path, command: list[str]) -> dict[str, Any]:
    """Collect command, working-directory, source, and package provenance."""

    return {
        "command": list(command),
        "command_text": subprocess.list2cmdline(command),
        "cwd": str(Path.cwd()),
        "root": str(root),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "source_snapshot": source_snapshot(root),
        **git_provenance(root),
    }


def build_run_id(
    variant: str,
    weighting: str,
    dataset: str,
    seed: int,
    protocol: str,
) -> str:
    """Include the protocol in every default filename to prevent collisions."""

    return f"{variant}_{weighting}_{dataset}_s{seed}__{protocol}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_finite_number(value: Any, *, lower: float | None = None,
                      upper: float | None = None) -> bool:
    """Return whether ``value`` is a finite numeric value within bounds."""

    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    if lower is not None and number < lower:
        return False
    if upper is not None and number > upper:
        return False
    return True


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _close(observed: Any, expected: float, *, atol: float = 1e-8) -> bool:
    return _is_finite_number(observed) and abs(float(observed) - expected) <= atol


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def validate_log(
    log: dict[str, Any],
    *,
    path: str | Path = "<memory>",
    allow_partial: bool = False,
) -> list[str]:
    """Return validation errors for a schema-v2 run log."""

    errors: list[str] = []
    if log.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version={log.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    for key in REQUIRED_TOP_LEVEL_FIELDS:
        if key not in log:
            errors.append(f"missing top-level field {key!r}")
    if log.get("status") != "completed":
        errors.append(f"status={log.get('status')!r}, expected 'completed'")
    dataset = str(log.get("dataset"))
    if dataset not in DATASET_MAIN_CLASSES:
        errors.append(f"unsupported dataset {dataset!r}")
    expected_classes = DATASET_MAIN_CLASSES.get(dataset)
    protocol = log.get("protocol")
    if isinstance(protocol, dict):
        for key in REQUIRED_PROTOCOL_FIELDS:
            if key not in protocol:
                errors.append(f"missing protocol field {key!r}")
        selection = protocol.get("selection_mode")
        normalization = protocol.get("normalization_mode")
        if selection not in {"heldout_subject", "inner_train_trials"}:
            errors.append(f"unsupported selection_mode {selection!r}")
        if normalization not in NORMALIZATION_MODES:
            errors.append(f"unsupported normalization_mode {normalization!r}")
        if selection in {"heldout_subject", "inner_train_trials"} and normalization in NORMALIZATION_MODES:
            legacy_loso = selection == "heldout_subject"
            # Logs written before the split-level axis existed carry no field
            # and are subject-level by construction.
            split_level = protocol.get("split_level", "subject")
            aligned = bool(protocol.get("euclidean_alignment", False))
            # A v4 log declares the two factorial bits explicitly and is
            # re-derived with the v4 builder; a v3 log has no such field and is
            # re-derived with the old one, so both generations stay checkable
            # against the configuration they actually declare.
            is_v4 = "target_supervised" in protocol
            try:
                if is_v4:
                    expected_protocol = protocol_id_v4(
                        aligned, bool(protocol.get("target_supervised")),
                        normalization, legacy_loso
                    )
                else:
                    expected_protocol = protocol_id(
                        legacy_loso, normalization, split_level, aligned
                    )
                expected_scope = normalization_fit_scope(normalization, legacy_loso)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if protocol.get("protocol_id") != expected_protocol:
                    errors.append(
                        f"protocol_id mismatch: {protocol.get('protocol_id')!r} != {expected_protocol!r}"
                    )
                if is_v4:
                    sup = bool(protocol.get("target_supervised"))
                    if protocol.get("target_access") != target_access_mode(
                            aligned, sup):
                        errors.append(
                            "target_access disagrees with the declared "
                            "alignment and supervision flags")
                    if protocol.get("target_reference_scope") != \
                            target_reference_scope(aligned):
                        errors.append(
                            "target_reference_scope disagrees with the "
                            "alignment flag")
                    if protocol.get("selection_scope") != selection_scope_v4(
                            legacy_loso):
                        errors.append(
                            "selection_scope disagrees with the selection mode")
                    if sup and not float(protocol.get(
                            "target_sample_weight", 0.0)) > 0.0:
                        errors.append(
                            "a target-supervised run must declare a positive "
                            "target_sample_weight")
                    if not sup and float(protocol.get(
                            "target_sample_weight", 0.0)) != 0.0:
                        errors.append(
                            "a run without target supervision must declare a "
                            "zero target_sample_weight")
                    if bool(protocol.get("test_subject_used_for_training")) != sup:
                        errors.append(
                            "test_subject_used_for_training disagrees with "
                            "target_supervised")
                if protocol.get("normalization_fit_scope") != expected_scope:
                    errors.append(
                        "normalization_fit_scope does not match the declared protocol"
                    )
                if protocol.get("selection_split") != selection_split(legacy_loso):
                    errors.append("selection_split does not match the declared protocol")
                expected_norm_leak = test_subject_used_for_normalization(normalization)
                if protocol.get("test_subject_used_for_normalization") != expected_norm_leak:
                    errors.append(
                        "test_subject_used_for_normalization does not match "
                        "protocol.normalization_mode"
                    )
                if log.get("legacy_loso") != legacy_loso:
                    errors.append("legacy_loso does not match protocol.selection_mode")
                if log.get("normalization") != normalization:
                    errors.append("normalization does not match protocol.normalization_mode")
                if protocol.get("test_subject_used_for_model_selection") != legacy_loso:
                    errors.append(
                        "test_subject_used_for_model_selection does not match protocol.selection_mode"
                    )
        if protocol.get("metric") != metric_name(dataset):
            errors.append("protocol.metric does not match the dataset metric")
    else:
        errors.append("protocol must be an object")

    provenance = log.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for key in REQUIRED_PROVENANCE_FIELDS:
            if key not in provenance:
                errors.append(f"missing provenance field {key!r}")
        packages = provenance.get("packages")
        if not isinstance(packages, dict):
            errors.append("provenance.packages must be an object")
        else:
            for key in REQUIRED_PACKAGE_VERSION_FIELDS:
                if key not in packages:
                    errors.append(f"missing provenance.packages field {key!r}")
        if not isinstance(provenance.get("command"), list) or not provenance.get("command"):
            errors.append("provenance.command must be a non-empty list")
        if not isinstance(provenance.get("command_text"), str) or not provenance.get("command_text"):
            errors.append("provenance.command_text must be a non-empty string")
        if not isinstance(provenance.get("git_diff_sha256"), str) or not provenance.get("git_diff_sha256"):
            errors.append("provenance.git_diff_sha256 must be a non-empty string")
        if not isinstance(provenance.get("untracked_file_sha256"), dict):
            errors.append("provenance.untracked_file_sha256 must be an object")

        snapshot = provenance.get("source_snapshot")
        if not isinstance(snapshot, dict):
            errors.append("provenance.source_snapshot must be an object")
        else:
            for key in REQUIRED_SOURCE_SNAPSHOT_FIELDS:
                if key not in snapshot:
                    errors.append(f"missing provenance.source_snapshot field {key!r}")
            files = snapshot.get("files")
            if not isinstance(files, list) or not files:
                errors.append("provenance.source_snapshot.files must be a non-empty list")
            else:
                paths = []
                for entry in files:
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                        errors.append("provenance.source_snapshot file entries must have a path")
                        continue
                    paths.append(entry["path"])
                    if not _is_sha256(entry.get("sha256")):
                        errors.append("provenance.source_snapshot file entries must have SHA-256 hashes")
                if len(paths) != len(set(paths)):
                    errors.append("provenance.source_snapshot contains duplicate paths")
                if _is_sha256(snapshot.get("sha256")) and snapshot.get("sha256") != canonical_sha256({"files": files}):
                    errors.append("provenance.source_snapshot SHA-256 does not match its files")
            if not _is_sha256(snapshot.get("sha256")):
                errors.append("provenance.source_snapshot.sha256 must be a SHA-256 hash")

        data_source = provenance.get("data_source")
        if not isinstance(data_source, dict):
            errors.append("provenance.data_source must be an object")
        else:
            for key in REQUIRED_DATA_SOURCE_FIELDS:
                if key not in data_source:
                    errors.append(f"missing provenance.data_source field {key!r}")
            if data_source.get("dataset_id") != dataset:
                errors.append("provenance.data_source.dataset_id does not match dataset")
            urls = data_source.get("source_urls")
            if not isinstance(urls, list) or not urls or not all(isinstance(url, str) and url for url in urls):
                errors.append("provenance.data_source.source_urls must be a non-empty string list")
            recorded_source_hash = data_source.get("source_sha256")
            source_payload = dict(data_source)
            source_payload.pop("source_sha256", None)
            if not _is_sha256(recorded_source_hash):
                errors.append("provenance.data_source.source_sha256 must be a SHA-256 hash")
            elif recorded_source_hash != canonical_sha256(source_payload):
                errors.append("provenance.data_source.source_sha256 does not match the descriptor")

    task_meta = log.get("task_definition")
    if not isinstance(task_meta, dict):
        errors.append("task_definition must be an object")
    else:
        for key in REQUIRED_TASK_DEFINITION_FIELDS:
            if key not in task_meta:
                errors.append(f"missing task_definition field {key!r}")
        if task_meta.get("dataset") != dataset:
            errors.append("task_definition.dataset does not match dataset")
        if task_meta.get("n_original_classes") != expected_classes:
            errors.append("task_definition.n_original_classes does not match dataset")
        tasks = task_meta.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            errors.append("task_definition.tasks must be a non-empty list")
        else:
            task_names = []
            for index, task in enumerate(tasks):
                if not isinstance(task, dict):
                    errors.append(f"task_definition.tasks[{index}] must be an object")
                    continue
                for key in REQUIRED_TASK_FIELDS:
                    if key not in task:
                        errors.append(f"missing task_definition.tasks[{index}] field {key!r}")
                task_names.append(task.get("name"))
            if task_names.count("main") != 1:
                errors.append("task_definition must contain exactly one main task")
            else:
                main_task = next(task for task in tasks if isinstance(task, dict) and task.get("name") == "main")
                if main_task.get("n_classes") != expected_classes:
                    errors.append("main task class count does not match dataset")

    subjects = log.get("subjects")
    per_subject = log.get("per_subject")
    if not isinstance(subjects, list) or not subjects:
        errors.append("subjects must be a non-empty list")
    if not isinstance(per_subject, dict) or not per_subject:
        errors.append("per_subject must be a non-empty object")
    if isinstance(subjects, list) and isinstance(per_subject, dict):
        subject_keys_list = [str(subject) for subject in subjects]
        subject_keys = set(subject_keys_list)
        result_keys = set(per_subject)
        if len(subject_keys_list) != len(subject_keys):
            errors.append("subjects contains duplicate subject IDs")
        if subject_keys != result_keys:
            errors.append("subjects does not match per_subject keys")
        expected_count = DATASET_EXPECTED_SUBJECT_COUNTS.get(str(log.get("dataset")))
        if expected_count and len(subjects) != expected_count and not allow_partial:
            errors.append(
                f"incomplete subject set ({len(subjects)} of expected {expected_count}); "
                "use --allow-partial for pilot logs"
            )
        expected_ids = DATASET_EXPECTED_SUBJECT_IDS.get(str(log.get("dataset")))
        if expected_ids and not allow_partial and subject_keys != {str(subject) for subject in expected_ids}:
            errors.append("subjects does not match the dataset's expected subject IDs")
    for subject, result in (per_subject or {}).items():
        if not isinstance(result, dict):
            errors.append(f"per_subject[{subject!r}] must be an object")
            continue
        for key in REQUIRED_PER_SUBJECT_FIELDS:
            if key not in result:
                errors.append(f"missing per_subject[{subject!r}] field {key!r}")
        for metric_key in ("main_accuracy", "balanced_accuracy", "best_inner_val_accuracy"):
            if metric_key in result and not _is_finite_number(result[metric_key], lower=0.0, upper=1.0):
                errors.append(f"per_subject[{subject!r}].{metric_key} is not a valid accuracy")
        if not isinstance(result.get("task_accuracy"), dict):
            errors.append(f"per_subject[{subject!r}].task_accuracy must be an object")
        confusion = result.get("confusion_matrix")
        if not isinstance(confusion, list) or not confusion:
            errors.append(f"per_subject[{subject!r}].confusion_matrix must be a non-empty list")
        elif expected_classes and (
            len(confusion) != expected_classes
            or any(not isinstance(row, list) or len(row) != expected_classes for row in confusion)
        ):
            errors.append(f"per_subject[{subject!r}].confusion_matrix has the wrong shape")
        else:
            try:
                counts = [[float(value) for value in row] for row in confusion]
            except (TypeError, ValueError):
                errors.append(f"per_subject[{subject!r}].confusion_matrix must be numeric")
            else:
                flat_counts = [value for row in counts for value in row]
                row_totals = [sum(row) for row in counts]
                total = sum(row_totals)
                if any(not math.isfinite(value) or value < 0 or value != round(value) for value in flat_counts):
                    errors.append(f"per_subject[{subject!r}].confusion_matrix must contain non-negative integers")
                elif total <= 0 or any(total_for_class <= 0 for total_for_class in row_totals):
                    errors.append(f"per_subject[{subject!r}].confusion_matrix has an empty class")
                else:
                    diagonal = sum(counts[index][index] for index in range(len(counts)))
                    expected_accuracy = diagonal / total
                    expected_balanced = sum(
                        counts[index][index] / row_totals[index]
                        for index in range(len(counts))
                    ) / len(counts)
                    if not _close(result.get("main_accuracy"), expected_accuracy):
                        errors.append(f"per_subject[{subject!r}].main_accuracy does not match confusion_matrix")
                    if not _close(result.get("balanced_accuracy"), expected_balanced):
                        errors.append(f"per_subject[{subject!r}].balanced_accuracy does not match confusion_matrix")

        trace = result.get("inner_val_trace")
        if not isinstance(trace, list) or not trace:
            errors.append(f"per_subject[{subject!r}].inner_val_trace must be a non-empty list")
        else:
            trace_epochs = []
            trace_accuracies = []
            for point in trace:
                if not isinstance(point, dict) or not isinstance(point.get("epoch"), int):
                    errors.append(f"per_subject[{subject!r}].inner_val_trace has an invalid epoch")
                    continue
                if point["epoch"] <= 0 or not _is_finite_number(point.get("main_accuracy"), lower=0.0, upper=1.0):
                    errors.append(f"per_subject[{subject!r}].inner_val_trace has invalid metrics")
                    continue
                trace_epochs.append(point["epoch"])
                trace_accuracies.append(float(point["main_accuracy"]))
            if len(trace_epochs) != len(set(trace_epochs)):
                errors.append(f"per_subject[{subject!r}].inner_val_trace has duplicate epochs")
            if trace_accuracies:
                if result.get("selected_epoch") not in trace_epochs:
                    errors.append(f"per_subject[{subject!r}].selected_epoch is absent from inner_val_trace")
                if not _close(result.get("best_inner_val_accuracy"), max(trace_accuracies)):
                    errors.append(f"per_subject[{subject!r}].best_inner_val_accuracy does not match inner_val_trace")
        if result.get("validation_source") != (protocol or {}).get("selection_mode"):
            errors.append(f"per_subject[{subject!r}].validation_source does not match protocol")
        if not isinstance(result.get("n_params"), int) or result.get("n_params", 0) <= 0:
            errors.append(f"per_subject[{subject!r}].n_params must be a positive integer")
        if not isinstance(result.get("selected_epoch"), int) or result.get("selected_epoch", 0) <= 0:
            errors.append(f"per_subject[{subject!r}].selected_epoch must be a positive integer")
        if not _is_finite_number(result.get("time_sec"), lower=0.0):
            errors.append(f"per_subject[{subject!r}].time_sec must be a non-negative number")

    if log.get("metric") != metric_name(dataset):
        errors.append("top-level metric does not match the dataset metric")
    if log.get("metric") != (protocol or {}).get("metric"):
        errors.append("top-level metric does not match protocol.metric")

    metric_field = "balanced_accuracy" if metric_name(dataset) == "balanced_accuracy" else "main_accuracy"
    if isinstance(per_subject, dict) and per_subject:
        try:
            main_values = [float(result["main_accuracy"]) for result in per_subject.values()]
            metric_values = [float(result[metric_field]) for result in per_subject.values()]
        except (KeyError, TypeError, ValueError):
            pass
        else:
            expected_values = {
                "mean_accuracy": sum(main_values) / len(main_values),
                "std_accuracy": _population_std(main_values),
                "mean_metric": sum(metric_values) / len(metric_values),
                "std_metric": _population_std(metric_values),
            }
            for key, expected_value in expected_values.items():
                if not _close(log.get(key), expected_value):
                    errors.append(f"{key} does not match the per-subject results")
    return [f"{path}: {error}" for error in errors]
