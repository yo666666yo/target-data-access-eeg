"""Validate and aggregate schema-v2 experiment logs.

The old aggregator grouped only by variant, weighting, and dataset. That can
silently combine legacy and corrected protocols, so this module treats the
run manifest as part of every paper table and requires an exact protocol
selection unless the caller explicitly asks for ``all``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.schema import (
    DATASET_EXPECTED_SUBJECT_COUNTS,
    SCHEMA_VERSION,
    canonical_sha256,
    metric_name,
    protocol_id,
    protocol_id_v4,
    validate_log,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_logs(log_dir, protocol_filter=None, allow_partial=False, include_legacy=False):
    """Load logs and return ``(accepted, skipped, duplicates)``.

    Old or malformed logs are skipped by default. ``include_legacy`` only
    admits them into the manifest with an explicit marker; they never become
    schema-v2 paper evidence by accident.
    """
    logs = []
    skipped = []
    root = Path(log_dir)
    for path in sorted(root.glob("*.json")):
        raw = _read_json(path)
        if raw is None:
            skipped.append({"path": str(path), "reason": "invalid JSON"})
            continue
        errors = validate_log(raw, path=path, allow_partial=allow_partial)
        if errors:
            if include_legacy and "schema_version" not in raw:
                raw = dict(raw)
                raw["_legacy_unversioned"] = True
                raw["schema_version"] = 1
                raw["protocol"] = {
                    "protocol_id": "legacy-unversioned",
                    "selection_mode": "unknown",
                    "normalization_mode": "unknown",
                    "metric": "accuracy",
                }
                raw["metric"] = "accuracy"
                raw["mean_metric"] = raw.get("mean_accuracy", 0.0)
                raw["std_metric"] = raw.get("std_accuracy", 0.0)
                raw["_path"] = str(path)
                logs.append(raw)
            else:
                skipped.append({"path": str(path), "reason": "; ".join(errors)})
            continue
        if protocol_filter and protocol_filter != "all":
            if raw["protocol"]["protocol_id"] != protocol_filter:
                skipped.append({
                    "path": str(path),
                    "reason": f"protocol mismatch ({raw['protocol']['protocol_id']})",
                })
                continue
        raw = dict(raw)
        raw["_path"] = str(path)
        logs.append(raw)

    seen = {}
    duplicates = []
    for log in logs:
        key = (
            log.get("protocol", {}).get("protocol_id"),
            log.get("variant"),
            log.get("weighting"),
            log.get("dataset"),
            log.get("seed"),
        )
        if key in seen:
            duplicates.append((key, seen[key], log.get("_path")))
        else:
            seen[key] = log.get("_path")
    if duplicates:
        detail = "; ".join(f"{key}: {a}, {b}" for key, a, b in duplicates)
        raise ValueError(f"duplicate run identities in manifest: {detail}")
    return logs, skipped, duplicates


def _paper_logs(logs):
    return [log for log in logs if not log.get("_legacy_unversioned")]


def _metric_value(log):
    return float(log.get("mean_metric", log.get("mean_accuracy", 0.0)))


def _summary(values):
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)


def _group(logs, keys):
    groups = defaultdict(list)
    for log in logs:
        groups[tuple(log.get(key) for key in keys)].append(_metric_value(log))
    return {key: _summary(values) for key, values in groups.items()}


def _subject_digest(log):
    subjects = sorted(str(subject) for subject in log.get("subjects", []))
    return canonical_sha256(subjects)


def _comparison_signature(
    log,
    *,
    include_dataset=True,
    include_data_source=True,
    include_protocol=True,
):
    """Fingerprint every setting that must stay fixed across a comparison arm."""

    config = log.get("config", {})
    payload = {
        "schema_version": log.get("schema_version"),
        "runner_version": log.get("runner_version"),
        "epochs": log.get("epochs"),
        "batch_size": log.get("batch_size"),
        "lr": log.get("lr"),
        "wd": log.get("wd"),
        "training_config": {
            key: config.get(key)
            for key in (
                "early_stopping_patience",
                "lr_step_size",
                "lr_gamma",
                "class_weights",
            )
        },
        "source_snapshot_sha256": log.get("provenance", {}).get("source_snapshot", {}).get("sha256"),
    }
    if include_protocol:
        payload["protocol_id"] = log.get("protocol", {}).get("protocol_id")
    if include_dataset:
        payload["dataset"] = log.get("dataset")
    if include_data_source:
        payload["data_source_sha256"] = log.get("provenance", {}).get("data_source", {}).get("source_sha256")
    return canonical_sha256(payload)


def _validate_comparison_block(
    logs,
    *,
    block_name,
    arm_keys,
    scope_keys=("dataset",),
    allow_subject_differences=False,
    allow_data_source_differences=False,
):
    """Return ``(ready, reason)`` after checking arm-level matched design.

    A standard table is a comparison, not a bag of independent means.  Each
    arm must therefore contain one common seed set, one subject set, and one
    source/configuration fingerprint.  Cross-dataset breadth tables are the
    narrow exception: their subject identities and raw source hashes differ by
    construction, but each arm is still internally coherent.
    """

    if not logs:
        return False, "no eligible logs"
    scopes = defaultdict(list)
    for log in logs:
        scopes[tuple(log.get(key) for key in scope_keys)].append(log)
    for scope, scoped_logs in scopes.items():
        arms = defaultdict(list)
        for log in scoped_logs:
            arms[tuple(log.get(key) for key in arm_keys)].append(log)
        if len(arms) < 2:
            return False, f"{block_name} scope {scope!r} has fewer than two comparison arms"

        seed_sets = {}
        subject_sets = {}
        signatures = {}
        source_sets = {}
        for arm, arm_logs in arms.items():
            seed_sets[arm] = tuple(sorted(int(log["seed"]) for log in arm_logs))
            subject_sets[arm] = {_subject_digest(log) for log in arm_logs}
            signatures[arm] = {
                _comparison_signature(
                    log,
                    include_dataset=not allow_data_source_differences,
                    include_data_source=not allow_data_source_differences,
                )
                for log in arm_logs
            }
            source_sets[arm] = {
                log.get("provenance", {}).get("data_source", {}).get("source_sha256")
                for log in arm_logs
            }
            if len(subject_sets[arm]) != 1:
                raise ValueError(
                    f"{block_name} arm {arm!r} mixes subject sets: {sorted(subject_sets[arm])}"
                )
            if len(signatures[arm]) != 1:
                raise ValueError(
                    f"{block_name} arm {arm!r} mixes training/source configurations"
                )
            if len(source_sets[arm]) != 1:
                raise ValueError(
                    f"{block_name} arm {arm!r} mixes data-source fingerprints"
                )
        if len(set(seed_sets.values())) != 1:
            raise ValueError(f"{block_name} arms have different seed sets: {seed_sets}")
        if not allow_subject_differences and len({next(iter(values)) for values in subject_sets.values()}) != 1:
            raise ValueError(f"{block_name} arms have different subject sets")
        if len({next(iter(values)) for values in signatures.values()}) != 1:
            raise ValueError(f"{block_name} arms have different training/source configurations")
    return True, ""


def _write_csv(rows, header, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def manifest_identity_payload(manifest):
    """Return the stable content used to identify a materialized manifest."""

    return {
        key: value
        for key, value in manifest.items()
        if key not in {"generated_at", "log_dir", "manifest_id"}
    }


def _manifest(logs, skipped, log_dir):
    entries = []
    root = Path(log_dir).resolve()
    for log in logs:
        path = Path(log.get("_path", "")).resolve()
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        digest = ""
        try:
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            pass
        protocol = log.get("protocol", {})
        entries.append({
            "path": relative_path,
            "sha256": digest,
            "schema_version": log.get("schema_version"),
            "runner_version": log.get("runner_version"),
            "run_id": log.get("run_id"),
            "protocol_id": protocol.get("protocol_id"),
            "selection_mode": protocol.get("selection_mode"),
            "normalization_mode": protocol.get("normalization_mode"),
            "normalization_fit_scope": protocol.get("normalization_fit_scope"),
            "dataset": log.get("dataset"),
            "variant": log.get("variant"),
            "weighting": log.get("weighting"),
            "seed": log.get("seed"),
            "metric": log.get("metric"),
            "mean_metric": _metric_value(log),
            "n_subjects": len(log.get("subjects", [])),
            "git_sha": log.get("provenance", {}).get("git_sha"),
            "git_dirty": log.get("provenance", {}).get("git_dirty"),
            "command_text": log.get("provenance", {}).get("command_text"),
            "source_snapshot_sha256": log.get("provenance", {}).get("source_snapshot", {}).get("sha256"),
            "data_source_sha256": log.get("provenance", {}).get("data_source", {}).get("source_sha256"),
            "legacy_unversioned": bool(log.get("_legacy_unversioned")),
        })
    entries.sort(key=lambda item: (str(item.get("protocol_id")), str(item.get("run_id"))))
    manifest = {
        "manifest_version": 1,
        "generated_at": _now(),
        "log_dir": str(Path(log_dir).resolve()),
        "schema_version_required": SCHEMA_VERSION,
        "entries": entries,
        "skipped": skipped,
    }
    # The release identity must survive regeneration and relocation of the
    # evidence directory.  Timestamps and absolute paths describe this
    # materialization, but are not part of the evidence content.
    manifest["manifest_id"] = canonical_sha256(manifest_identity_payload(manifest))
    return manifest


def _write_manifest(manifest, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    header = [
        "path", "sha256", "schema_version", "runner_version", "run_id",
        "protocol_id", "selection_mode", "normalization_mode",
        "normalization_fit_scope", "dataset", "variant", "weighting", "seed",
        "metric", "mean_metric", "n_subjects", "git_sha", "git_dirty",
        "command_text", "source_snapshot_sha256", "data_source_sha256", "legacy_unversioned",
    ]
    _write_csv([[entry.get(key) for key in header] for entry in manifest["entries"]],
               header, out / "run_manifest.csv")


def _blind_manifest(manifest):
    """Build a shareable manifest without paths, commands, or repository IDs."""

    allowed = (
        "schema_version", "runner_version", "run_id", "protocol_id", "selection_mode",
        "normalization_mode", "normalization_fit_scope", "dataset", "variant", "weighting",
        "seed", "metric", "mean_metric", "n_subjects", "source_snapshot_sha256",
        "data_source_sha256", "legacy_unversioned",
    )
    blind = {
        "manifest_version": 1,
        "schema_version_required": manifest["schema_version_required"],
        "source_manifest_id": manifest["manifest_id"],
        "entries": [
            {key: entry.get(key) for key in allowed}
            for entry in manifest["entries"]
        ],
    }
    blind["blind_manifest_id"] = canonical_sha256(blind)
    return blind


def _write_blind_manifest(manifest, out_dir):
    blind = _blind_manifest(manifest)
    with open(Path(out_dir) / "blind_run_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(blind, stream, indent=2, ensure_ascii=True)
    return blind


def _common_header():
    return ["protocol_id", "manifest_id", "schema_version", "metric"]


def _write_standard_blocks(logs, out_dir, manifest_id):
    logs = _paper_logs(logs)
    if not logs:
        return
    out = Path(out_dir)
    common = _common_header()
    skipped = []

    def write_group(
        filename,
        logs_for_block,
        keys,
        value_names,
        *,
        block_name,
        arm_keys,
        scope_keys=("dataset",),
        allow_subject_differences=False,
        allow_data_source_differences=False,
    ):
        ready, reason = _validate_comparison_block(
            logs_for_block,
            block_name=block_name,
            arm_keys=arm_keys,
            scope_keys=scope_keys,
            allow_subject_differences=allow_subject_differences,
            allow_data_source_differences=allow_data_source_differences,
        )
        if not ready:
            skipped.append(f"- `{filename}`: {reason}")
            return
        grouped = _group(logs_for_block, keys)
        rows = []
        for key, (m, s, n) in sorted(grouped.items(), key=lambda item: str(item[0])):
            representative = next(
                log for log in logs_for_block
                if tuple(log.get(k) for k in keys) == key
            )
            row = [
                representative["protocol"]["protocol_id"], manifest_id,
                SCHEMA_VERSION, representative["metric"],
                *key, f"{m:.6f}", f"{s:.6f}", n,
            ]
            rows.append(row)
        _write_csv(rows, common + list(value_names) + ["mean_metric", "std_metric", "n_runs"],
                   out / filename)

    aux = [log for log in logs if log.get("weighting") in ("kendall", "equal")
           and log.get("variant") in ("difficulty-mtl", "single", "recon", "metric")]
    weight = [log for log in logs if log.get("variant") == "difficulty-mtl"]
    attention = [log for log in logs if log.get("variant") in (
        "difficulty-mtl", "no-channel-attn", "shared-channel-attn")]
    transformer = [log for log in logs if log.get("variant") in ("difficulty-mtl", "no-transformer")]
    flagship = [log for log in logs if log.get("variant") == "difficulty-mtl"
                and log.get("weighting") == "kendall"]
    write_group(
        "block_A_aux_comparison.csv", aux, ["dataset", "variant"], ["dataset", "variant"],
        block_name="Block A", arm_keys=("variant",),
    )
    write_group(
        "block_B_weighting.csv", weight, ["dataset", "weighting"], ["dataset", "weighting"],
        block_name="Block B", arm_keys=("weighting",),
    )
    write_group(
        "block_C_attention.csv", attention, ["dataset", "variant"], ["dataset", "attention_variant"],
        block_name="Block C", arm_keys=("variant",),
    )
    write_group(
        "block_D_transformer.csv", transformer, ["dataset", "variant"], ["dataset", "transformer_variant"],
        block_name="Block D", arm_keys=("variant",),
    )
    write_group(
        "block_E_cross_dataset.csv", flagship, ["dataset"], ["dataset"],
        block_name="Block E", arm_keys=("dataset",), scope_keys=(),
        allow_subject_differences=True, allow_data_source_differences=True,
    )

    if skipped:
        with open(out / "standard_blocks_skipped.md", "w", encoding="utf-8") as stream:
            stream.write("# Standard comparison tables skipped\n\n")
            stream.write("The matching design gate rejected the following incomplete blocks.\n\n")
            stream.write("\n".join(skipped) + "\n")

    params = {}
    for log in logs:
        for result in log.get("per_subject", {}).values():
            if result.get("n_params"):
                params[log.get("variant")] = result["n_params"]
                break
    _write_csv(
        [[logs[0]["protocol"]["protocol_id"], manifest_id, SCHEMA_VERSION,
          logs[0]["metric"], variant, count]
         for variant, count in sorted(params.items())],
        common + ["variant", "n_params"], out / "params.csv"
    )


def _write_corrected_table(logs, out_dir, manifest_id):
    corrected = [
        log for log in _paper_logs(logs)
        if log.get("protocol", {}).get("protocol_id") == protocol_id(False, "within_trial")
        and log.get("dataset") == "BNCI2014_001"
    ]
    if not corrected:
        return
    if len({(log.get("variant"), log.get("weighting")) for log in corrected}) > 1:
        _validate_comparison_block(
            corrected,
            block_name="corrected BNCI2014_001 table",
            arm_keys=("variant", "weighting"),
            scope_keys=(),
        )
    grouped = _group(corrected, ["variant", "weighting"])
    rows = []
    for (variant, weighting), (m, s, n) in sorted(grouped.items()):
        first = next(log for log in corrected if log.get("variant") == variant and log.get("weighting") == weighting)
        first_subject = next(iter(first["per_subject"].values()))
        rows.append([
            variant, weighting, "BNCI2014_001", f"{m:.6f}", f"{s:.6f}", n,
            first_subject.get("n_params"), first.get("metric"),
            first["protocol"]["protocol_id"], manifest_id, SCHEMA_VERSION,
        ])
    _write_csv(rows, [
        "variant", "weighting", "dataset", "mean_acc", "std_acc", "n_seeds",
        "n_params", "metric", "protocol_id", "manifest_id", "schema_version",
    ], Path(out_dir) / "audit_corrected.csv")


def _factorization_axes(logs):
    """Choose which 2x2 a set of logs describes.

    schema-v3 factorized selection source against normalization regime.
    schema-v4 factorizes alignment against target supervision, which is a
    different pair of axes over the same held-out subjects, so the axis names,
    the cell key and the effect labels all have to come from the logs rather
    than be hard-coded.
    """

    is_v4 = any("target_supervised" in log.get("protocol", {})
                for log in _paper_logs(logs))
    if is_v4:
        def eligible(log):
            p = log.get("protocol", {})
            # POOL and SEL are deliberately off the factorial grid: each
            # changes a third thing, so averaging one into a cell would
            # reintroduce exactly the confound the v4 controls remove.
            return (p.get("normalization_mode") == "outer_train_scaler"
                    and not p.get("test_subject_used_for_model_selection"))

        def cell_of(log):
            p = log["protocol"]
            return (bool(p.get("euclidean_alignment")),
                    bool(p.get("target_supervised")))

        return dict(
            is_v4=True,
            eligible=eligible,
            cell_of=cell_of,
            columns=("euclidean_alignment", "target_supervised"),
            effects=("Alignment (target signals)",
                     "Target supervision (target labels)",
                     "Interaction"),
            protocol_of=lambda a, s: protocol_id_v4(a, s, "outer_train_scaler"),
            note=("Axis A is Euclidean Alignment, which reads the held-out "
                  "subject's available trials but none of their labels. "
                  "Axis B is target supervision, which adds those trials with "
                  "their labels at a declared batch share.\n\n"),
            unavailable=("No EEGNet / BNCI2014_001 runs on the alignment x "
                         "target-supervision grid were present."),
        )

    def eligible_v3(log):
        p = log.get("protocol", {})
        # The v3 2x2 is defined over leave-one-subject-out folds only.  A
        # trial-split audit run carries the same selection_mode and
        # normalization_mode as a clean run, so without this guard cell_of()
        # would average an intentionally leaky arm into the clean cell.
        return (p.get("normalization_mode") in {"within_trial", "outer_train_scaler"}
                and p.get("split_level", "subject") == "subject")

    def cell_of_v3(log):
        p = log["protocol"]
        return (p["selection_mode"] == "heldout_subject",
                p["normalization_mode"] == "outer_train_scaler")

    return dict(
        is_v4=False,
        eligible=eligible_v3,
        cell_of=cell_of_v3,
        columns=("legacy_loso", "legacy_norm"),
        effects=("B1 (held-out model selection)",
                 "B2 (normalization regime)",
                 "Interaction"),
        protocol_of=lambda legacy, norm: protocol_id(
            legacy, "outer_train_scaler" if norm else "within_trial"),
        note=("B2 is a normalization-regime effect: the outer-train pooled "
              "scaler arm and the within-trial arm differ in more than one "
              "preprocessing choice.\n\n"),
        unavailable=("No EEGNet / BNCI2014_001 factorization runs with "
                     "within-trial or outer-train-scaler normalization were "
                     "present."),
    )


def _write_factorization(logs, out_dir, manifest_id):
    axes = _factorization_axes(logs)
    eligible = [
        log for log in _paper_logs(logs)
        if log.get("dataset") == "BNCI2014_001"
        and log.get("variant") == "eegnet"
        and log.get("weighting") == "equal"
        and axes["eligible"](log)
    ]
    out = Path(out_dir)
    cells = [(False, False), (False, True), (True, False), (True, True)]
    if not eligible:
        _write_factorization_unavailable(out, manifest_id, axes["unavailable"])
        return

    cell_of = axes["cell_of"]

    cell_logs = defaultdict(list)
    for log in eligible:
        cell_logs[cell_of(log)].append(log)
    missing = [cell for cell in cells if not cell_logs[cell]]
    if missing:
        _write_factorization_unavailable(
            out,
            manifest_id,
            "Incomplete 2x2 design; missing cells: " + ", ".join(map(str, missing)),
        )
        return

    seed_sets = {
        cell: {int(log["seed"]) for log in cell_logs[cell]}
        for cell in cells
    }
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        _write_factorization_unavailable(
            out,
            manifest_id,
            "The four factorization cells do not have the same seed set: "
            + "; ".join(f"{cell}={sorted(seeds)}" for cell, seeds in seed_sets.items()),
        )
        return

    subject_sets = {
        cell: {tuple(sorted(map(str, log["subjects"]))) for log in cell_logs[cell]}
        for cell in cells
    }
    if any(len(sets) != 1 for sets in subject_sets.values()) or len(
        {next(iter(sets)) for sets in subject_sets.values()}
    ) != 1:
        _write_factorization_unavailable(
            out,
            manifest_id,
            "The four factorization cells do not have an identical held-out subject set.",
        )
        return

    signatures = {
        cell: {_comparison_signature(log, include_protocol=False) for log in cell_logs[cell]}
        for cell in cells
    }
    if any(len(values) != 1 for values in signatures.values()) or len(
        {next(iter(values)) for values in signatures.values()}
    ) != 1:
        _write_factorization_unavailable(
            out,
            manifest_id,
            "The four factorization cells do not have an identical training/source configuration.",
        )
        return

    grouped = defaultdict(list)
    for log in eligible:
        grouped[cell_of(log)].append(_metric_value(log))
    rows = []
    for axis_a, axis_b in cells:
        values = grouped.get((axis_a, axis_b), [])
        if not values:
            continue
        m, sd, n = _summary(values)
        rows.append([
            axis_a, axis_b, f"{m:.6f}", f"{sd:.6f}", n,
            axes["protocol_of"](axis_a, axis_b),
            manifest_id, SCHEMA_VERSION, "accuracy",
        ])
    if rows:
        _write_csv(rows, [
            axes["columns"][0], axes["columns"][1], "mean_acc", "std_acc",
            "n_seeds", "protocol_id", "manifest_id", "schema_version", "metric",
        ], out / "audit_factorization.csv")

    # Paired subject-level effects are averaged over seeds before contrasts.
    by_cell_subject = defaultdict(lambda: defaultdict(list))
    for log in eligible:
        cell = cell_of(log)
        for subject, result in log.get("per_subject", {}).items():
            by_cell_subject[cell][str(subject)].append(float(result["main_accuracy"]))
    common_subjects = sorted(
        set(by_cell_subject[(False, False)])
        & set(by_cell_subject[(False, True)])
        & set(by_cell_subject[(True, False)])
        & set(by_cell_subject[(True, True)]),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    name_a, name_b, name_i = axes["effects"]
    effects = {name_a: [], name_b: [], name_i: []}
    for subject in common_subjects:
        ff = mean(by_cell_subject[(False, False)][subject])
        ft = mean(by_cell_subject[(False, True)][subject])
        tf = mean(by_cell_subject[(True, False)][subject])
        tt = mean(by_cell_subject[(True, True)][subject])
        effects[name_a].append(0.5 * ((tf - ff) + (tt - ft)))
        effects[name_b].append(0.5 * ((ft - ff) + (tt - tf)))
        effects[name_i].append(tt - tf - ft + ff)

    def t_summary(values):
        if len(values) < 2:
            return (mean(values) if values else float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        m = mean(values)
        sd = stdev(values)
        se = sd / math.sqrt(len(values))
        t = m / se if se else float("nan")
        try:
            from scipy import stats
            p = float(2 * stats.t.sf(abs(t), df=len(values) - 1))
        except Exception:
            p = float("nan")
        return m, sd, se, t, p

    with open(out / "paired_stats.md", "w", encoding="utf-8") as f:
        f.write("# Paired subject-wise protocol effects\n\n")
        f.write(f"Manifest: `{manifest_id}`; subjects paired: {len(common_subjects)}.\n\n")
        f.write(axes["note"])
        f.write("| Effect | Mean | SD | SE | t | p | n |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for label, values in effects.items():
            m, sd, se, t, p = t_summary(values)
            p_text = "n/a" if p != p else f"{p:.6f}"
            f.write(f"| {label} | {m:+.6f} | {sd:.6f} | {se:.6f} | {t:.3f} | {p_text} | {len(values)} |\n")


def _write_factorization_unavailable(out_dir, manifest_id, reason):
    """Record why a 2x2 artifact was deliberately not generated."""

    with open(Path(out_dir) / "factorization_unavailable.md", "w", encoding="utf-8") as f:
        f.write("# Factorization unavailable\n\n")
        f.write(f"Manifest: `{manifest_id}`.\n\n")
        f.write(f"{reason}\n")


def _write_mixed_protocol_note(out_dir, manifest_id, protocol_values):
    """Explain why generic ablation tables are omitted from a mixed manifest."""

    with open(Path(out_dir) / "standard_blocks_skipped.md", "w", encoding="utf-8") as f:
        f.write("# Standard block tables skipped\n\n")
        f.write(f"Manifest: `{manifest_id}`.\n\n")
        f.write(
            "This manifest intentionally combines protocol cells for the 2x2 audit. "
            "Generic block tables are not generated because grouping across those cells "
            "would mix incomparable evaluation protocols. Generate an exact-protocol "
            "manifest to create the standard ablation tables.\n\n"
        )
        f.write("Protocols present:\n\n")
        for value in protocol_values:
            f.write(f"- `{value}`\n")


def _metric_raw(log):
    return mean([float(r.get("main_accuracy", 0.0)) for r in log.get("per_subject", {}).values()])


def _metric_balanced(log):
    return mean([float(r.get("balanced_accuracy", 0.0)) for r in log.get("per_subject", {}).values()])


def _majority_baseline(log):
    values = []
    for result in log.get("per_subject", {}).values():
        matrix = result.get("confusion_matrix") or []
        row_sums = [sum(row) for row in matrix]
        total = sum(row_sums)
        values.append(max(row_sums) / total if total else 0.0)
    return mean(values) if values else 0.0


def _write_p300_summary(logs, out_dir, manifest_id):
    rows = []
    for variant, weighting in [("eegnet", "equal"), ("minimal", "equal"), ("difficulty-mtl", "kendall")]:
        selected = [
            log for log in _paper_logs(logs)
            if log.get("dataset") == "BNCI2014_009"
            and log.get("variant") == variant
            and log.get("weighting") == weighting
            and log.get("protocol", {}).get("protocol_id") == protocol_id(False, "within_trial")
        ]
        if not selected:
            continue
        rows.append((
            variant, weighting, mean([_metric_raw(log) for log in selected]),
            mean([_metric_balanced(log) for log in selected]),
            mean([_majority_baseline(log) for log in selected]), len(selected),
        ))
    if not rows:
        return
    with open(Path(out_dir) / "p300_balanced_accuracy.md", "w", encoding="utf-8") as f:
        f.write("# P300 metric-sensitivity boundary case\n\n")
        f.write(f"Manifest: `{manifest_id}`. Values are means over schema-v2 corrected runs.\n\n")
        f.write("| Variant | Weighting | Raw accuracy | Balanced accuracy | Majority baseline | n seeds |\n|---|---|---:|---:|---:|---:|\n")
        for variant, weighting, raw, balanced, majority, n in rows:
            f.write(f"| {variant} | {weighting} | {raw:.6f} | {balanced:.6f} | {majority:.6f} | {n} |\n")


def _write_summary(logs, out_dir, manifest_id, skipped):
    paper_logs = _paper_logs(logs)
    with open(Path(out_dir) / "paper_summary.md", "w", encoding="utf-8") as f:
        f.write("# Schema-v2 paper results summary\n\n")
        f.write(f"Manifest: `{manifest_id}`\n\n")
        f.write(f"Accepted schema-v2 runs: {len(paper_logs)}; skipped files: {len(skipped)}.\n\n")
        f.write("Every table generated from this directory carries the manifest ID above; legacy/unversioned logs are not used in paper tables.\n\n")
        corrected = [
            log for log in paper_logs
            if log.get("protocol", {}).get("protocol_id") == protocol_id(False, "within_trial")
            and log.get("dataset") == "BNCI2014_001"
        ]
        if corrected:
            f.write("## Corrected BNCI2014_001 comparison\n\n")
            for (variant, weighting), (m, s, n) in sorted(_group(corrected, ["variant", "weighting"]).items()):
                f.write(f"- `{variant}` / `{weighting}`: {m:.4f} +/- {s:.4f} over {n} runs\n")
            f.write("\n")
        f.write("## Protocol and task boundary\n\n")
        f.write("The runner records the selection split, normalization fit scope, exact task heads and label mappings, subject list, source revision, command, and package versions in every log.\n")


def aggregate(log_dir="logs", out_dir="results", protocol_filter=None,
              allow_partial=False, include_legacy=False):
    logs, skipped, _ = load_logs(
        log_dir,
        protocol_filter=protocol_filter,
        allow_partial=allow_partial,
        include_legacy=include_legacy,
    )
    if not logs:
        raise RuntimeError(
            f"no valid schema-v2 logs found under {log_dir}; old or incomplete logs were rejected"
        )
    protocol_values = sorted({
        log.get("protocol", {}).get("protocol_id")
        for log in _paper_logs(logs)
    })
    if protocol_filter is None and len(protocol_values) > 1:
        raise ValueError(
            "multiple protocols found; pass --protocol-id with one exact ID or --protocol-id all. Found: "
            + ", ".join(protocol_values)
        )
    manifest = _manifest(logs, skipped, log_dir)
    _write_manifest(manifest, out_dir)
    _write_blind_manifest(manifest, out_dir)
    if len(protocol_values) == 1:
        _write_standard_blocks(logs, out_dir, manifest["manifest_id"])
    else:
        _write_mixed_protocol_note(out_dir, manifest["manifest_id"], protocol_values)
    _write_corrected_table(logs, out_dir, manifest["manifest_id"])
    _write_factorization(logs, out_dir, manifest["manifest_id"])
    _write_p300_summary(logs, out_dir, manifest["manifest_id"])
    _write_summary(logs, out_dir, manifest["manifest_id"], skipped)
    print(f"Accepted {len(logs)} logs; skipped {len(skipped)}; manifest={manifest['manifest_id']}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "--log-dir", default="logs")
    parser.add_argument("--out_dir", "--out-dir", default="results")
    parser.add_argument("--protocol_id", "--protocol-id", default=None,
                        help="Exact protocol ID, or 'all' for a factorization manifest.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow pilot logs with fewer than the expected subject count.")
    parser.add_argument("--include-legacy", action="store_true",
                        help="Record old unversioned logs in the manifest, never silently mix them.")
    args = parser.parse_args()
    aggregate(
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        protocol_filter=args.protocol_id,
        allow_partial=args.allow_partial,
        include_legacy=args.include_legacy,
    )


if __name__ == "__main__":
    main()
