"""Paper experiment runner — single-entry configurable training script.

Runs one (seed, variant, dataset) combination. The orchestrator
`experiments/run_all.py` calls this with different arg combos to produce the
full experiment table.

Variants covered:
    --variant difficulty-mtl     flagship (MultiDecoderHybrid + class-hierarchy aux)
    --variant single             single-task head only (C1 baseline)
    --variant recon              main + reconstruction aux (C1 competitor)
    --variant metric             main + supervised-contrastive aux (C1 competitor)
    --variant no-transformer     flagship without transformer refinement
    --variant no-channel-attn    flagship with LightDecoderHead (no SE)
    --variant shared-channel-attn flagship with one shared SE before all heads (C3 baseline)
    --variant eegnet             stock EEGNet-8,2 (Lawhern 2018) — sanity-check baseline
    --variant minimal            SingleTaskModel without transformer and without channel attention

Weighting schemes (for --variant difficulty-mtl):
    --weighting kendall | fixed | equal | gradnorm | dwa

Datasets: BNCI2014_001 (4-class MI), BNCI2015_004 (5-class MI), BNCI2014_009 (P300).
Protocol: LOSO via MOABB.

Logs to logs/<run_id>.json with per-subject metrics and schema-v2 provenance.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
# Decoded MOABB trials live outside the source tree so the provenance collector
# never hashes them and the paper's snapshot stays stable.
MOABB_CACHE_ROOT = ROOT / ".cache" / "moabb_trials"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'EEGNets'))
sys.path.insert(0, str(ROOT / 'configs'))

from train_multidecoder import (  # noqa: E402
    remap_labels_for_tasks, MultiTaskLoss, build_model, load_dataset,
)
from multidecoder_config import (  # noqa: E402
    ExperimentConfig, EncoderConfig, TransformerConfig, DecoderConfig,
    TaskConfig, TrainingConfig, DataConfig,
)
from experiments.mtl_weighting import build_criterion, GradNormLoss, DWALoss  # noqa: E402
from experiments.aux_baselines import build_aux_model, supcon_loss  # noqa: E402
from experiments.official_bciciv_2a import OfficialBCICIV2aSource  # noqa: E402
from experiments.schema import (  # noqa: E402
    DATASET_EXPECTED_SUBJECT_IDS,
    NORMALIZATION_MODES,
    RUNNER_VERSION,
    SCHEMA_VERSION,
    build_run_id,
    canonical_sha256,
    metric_name,
    normalization_fit_scope,
    evaluation_unit,
    SPLIT_LEVELS,
    test_subject_used_for_normalization,
    test_subject_used_for_training,
    protocol_id,
    protocol_id_v4,
    runtime_provenance,
    selection_mode,
    selection_split,
    selection_scope_v4,
    target_access_mode,
    target_reference_scope,
    task_definition,
)


# Expected share of every training batch drawn from the held-out subject's
# available trials, when a condition is allowed to use their labels.  Fixing it
# is what makes the target-supervised contrast mean the same thing on a 9-
# subject dataset and a 105-subject one: without it the realized share is
# whatever the cohort size happens to imply, and the contrast is a dose-response
# to that share rather than to target supervision.
TARGET_SAMPLE_WEIGHT = 0.10


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Dataset Config Builders ──────────────────────────────────────────────────

DATASET_META = {
    'BNCI2014_001': {'paradigm': 'MotorImagery', 'n_cls_hint': 4},
    'BNCI2015_004': {'paradigm': 'MotorImagery', 'n_cls_hint': 5},
    'BNCI2014_009': {'paradigm': 'P300', 'n_cls_hint': 2},
    # 105 usable subjects, an order of magnitude more than 2a, which is what
    # tightens the subject-paired intervals the ladder reports.
    'PhysionetMI': {'paradigm': 'MotorImagery', 'n_cls_hint': 4},
    # 14 subjects, sitting between 2a's nine and PhysionetMI's 105, which is
    # what makes it useful for the cohort-size reading.
    'BNCI2014_002': {'paradigm': 'MotorImagery', 'n_cls_hint': 2},
}


class MoabbDataSource:
    """Adapter that makes MOABB and the official local source interchangeable."""

    def __init__(self, config):
        self.dataset, self.paradigm = load_dataset(config)
        self.dataset_id = config.data.dataset_name
        self._subject_disk_cache = {}
        self._cache_key_value = None
        declared = DATASET_EXPECTED_SUBJECT_IDS.get(self.dataset_id)
        available = list(self.dataset.subject_list)
        if declared is None:
            self.subject_list = available
        else:
            # PhysionetMI advertises 109 subjects but four recordings are
            # unusable.  Intersecting here means the evaluated subject set and
            # the set the schema declares are the same object, rather than the
            # runner silently evaluating subjects the gate will later reject.
            self.subject_list = [s for s in available if s in set(declared)]
            missing = sorted(set(declared) - set(available))
            if missing:
                raise ValueError(
                    f"{self.dataset_id} is missing declared subjects {missing}"
                )

    def _cache_key(self):
        """Key covering the dataset, paradigm and this module's own source.

        The descriptor alone would not notice a change to how epochs are cut,
        so the runner's source hash is folded in: any edit to the loading path
        misses into a different directory rather than being served stale
        arrays.
        """

        if self._cache_key_value is None:
            digest = hashlib.sha256()
            digest.update(str(self.provenance()["source_sha256"]).encode("utf-8"))
            digest.update(_sha256_file(Path(__file__)).encode("utf-8"))
            self._cache_key_value = digest.hexdigest()
        return self._cache_key_value

    def get_data(self, subjects):
        """Fetch trials, caching each subject's decoded arrays on disk.

        MOABB re-reads and re-filters the raw recordings on every call. With
        105 subjects that costs minutes per run, and running several runs at
        once multiplies it: eight concurrent loaders drove this machine to a
        load average of 206 on 256 cores and starved the jobs already
        training. Caching turns the cost into a one-off.
        """

        subject_ids = list(subjects)
        arrays, labels = [], []
        for subject in subject_ids:
            cached = self._subject_disk_cache.get(subject)
            if cached is None:
                cached = self._load_subject_cached(subject)
                self._subject_disk_cache[subject] = cached
            arrays.append(cached[0])
            labels.append(cached[1])
        if not arrays:
            raise ValueError("at least one subject is required")
        return np.concatenate(arrays, axis=0), np.concatenate(labels, axis=0), {}

    def _load_subject_cached(self, subject):
        cache_dir = MOABB_CACHE_ROOT / self.dataset_id / self._cache_key()
        cache_path = cache_dir / f"S{int(subject):03d}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=True) as bundle:
                    return bundle["X"], bundle["y"]
            except (OSError, ValueError, KeyError):
                cache_path.unlink(missing_ok=True)

        X, y, _ = self.paradigm.get_data(dataset=self.dataset, subjects=[subject])
        # MOABB hands back float64. The model consumes float32 anyway, and the
        # per-fold normalization allocates two more arrays the size of the
        # training pool, so carrying float64 that far costs real time: measured
        # on PhysionetMI, one fold's z-score took 168 s at float64 against a
        # 2.3 GB pool. Casting at the cache boundary halves every downstream
        # copy without changing what the model sees.
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Unique temporary name plus fsync: several runs share this directory,
        # and a fixed name would let two writers interleave into one file.
        handle, tmp_name = tempfile.mkstemp(
            dir=str(cache_dir), prefix=f"S{int(subject):03d}.", suffix=".npz")
        os.close(handle)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("wb") as stream:
                np.savez(stream, X=X, y=y)
                stream.flush()
                os.fsync(stream.fileno())
            tmp_path.replace(cache_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
        return X, y

    def provenance(self):
        descriptor = {
            "kind": "moabb",
            "dataset_id": self.dataset_id,
            # Derived from the dataset, never hard-coded: a literal here silently
            # recorded the BNCI2014_001 documentation page as the provenance of
            # every MOABB dataset, including PhysionetMI.
            "source_urls": [
                "https://moabb.neurotechx.com/docs/generated/"
                f"moabb.datasets.{self.dataset_id}.html"
            ],
            "dataset_code": getattr(self.dataset, "code", self.dataset_id),
            "dataset_doi": getattr(self.dataset, "doi", None),
            "paradigm": type(self.paradigm).__name__,
            "filters_hz": getattr(self.paradigm, "filters", None),
            "interval": getattr(self.dataset, "interval", None),
        }
        descriptor["source_sha256"] = canonical_sha256(descriptor)
        return descriptor


def build_data_source(config, args):
    """Select a declared data source; never replace an unavailable mirror silently."""

    if args.data_source == "official-gdf":
        if args.dataset != "BNCI2014_001":
            raise ValueError(
                "--data-source official-gdf currently supports only BNCI2014_001 "
                "(BCI Competition IV 2a)"
            )
        return OfficialBCICIV2aSource(args.official_gdf_root)
    return MoabbDataSource(config)


def make_task_list(dataset_name, variant):
    """Return the TaskConfig list for a (dataset, variant) combination.

    difficulty-mtl: dataset-specific auxiliary task definitions.
    others: main task only.
    """
    n_main = DATASET_META[dataset_name]['n_cls_hint']
    if variant != 'difficulty-mtl':
        return [TaskConfig(name='main', n_classes=n_main, loss_weight=1.0)]

    if n_main == 4:
        # The implemented 4-class task has exactly two heads.  Keep this
        # explicit: it is not a binary/coarse/main hierarchy.
        return [
            TaskConfig(name='binary', n_classes=2, loss_weight=0.3),
            TaskConfig(name='main', n_classes=4, loss_weight=1.0),
        ]
    elif n_main == 5:
        return [
            TaskConfig(name='binary', n_classes=2, loss_weight=0.3),
            TaskConfig(name='coarse', n_classes=3, loss_weight=0.5),
            TaskConfig(name='main', n_classes=5, loss_weight=1.0),
        ]
    elif n_main == 2:
        # P300 is binary — auxiliary "binary" = main; use a parity-style aux
        return [
            TaskConfig(name='main', n_classes=2, loss_weight=1.0),
        ]
    raise ValueError(f"No hierarchy defined for n_main={n_main}")


def make_config(args):
    task_list = make_task_list(args.dataset, args.variant)
    transformer_cfg = TransformerConfig(
        use_transformer=(args.variant not in ('no-transformer', 'minimal',
                                              'eegnet', 'braindecode-eegnet')),
    )
    return ExperimentConfig(
        encoder=EncoderConfig(),
        transformer=transformer_cfg,
        decoder=DecoderConfig(),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.wd,
            use_class_weights=True,
            use_dynamic_weighting=(args.weighting == 'kendall'),
            # The stock StepLR(15, 0.5) halves the rate every 15 epochs, so a
            # larger --epochs alone cannot buy more training: by epoch 150 the
            # rate is 1e-3 * 0.5**10. Exposing the schedule lets the reference
            # control vary the budget and the schedule together.
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            early_stopping_patience=args.patience,
        ),
        data=DataConfig(
            dataset_name=args.dataset,
            paradigm=DATASET_META[args.dataset]['paradigm'],
        ),
        tasks=task_list,
        model_type='hybrid',
        save_dir=args.save_dir,
    )


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_experiment_model(config, args, n_chan, T_input):
    """Build the right model for the chosen variant."""
    F_T = config.encoder.F_T
    n_main = config.tasks[-1].n_classes
    use_tf = config.transformer.use_transformer
    model_kwargs = dict(
        F_T=F_T, K_T=config.encoder.K_T, L=config.encoder.L,
        use_transformer=use_tf,
        n_heads=config.transformer.n_heads,
        n_transformer_layers=config.transformer.n_layers,
        hidden_dim=config.decoder.hidden_dim,
        dropout=config.decoder.dropout,
    )
    if args.variant == 'difficulty-mtl':
        return build_model(config, n_chan=n_chan)
    if args.variant == 'no-transformer':
        return build_model(config, n_chan=n_chan)  # config already reflects flag
    if args.variant == 'single':
        return build_aux_model('single', n_chan=n_chan, n_cls=n_main, **model_kwargs)
    if args.variant == 'recon':
        return build_aux_model('recon', n_chan=n_chan, n_cls=n_main,
                               T_input=T_input, **model_kwargs)
    if args.variant == 'metric':
        return build_aux_model('metric', n_chan=n_chan, n_cls=n_main, **model_kwargs)
    if args.variant == 'braindecode-eegnet':
        # Reference check: swap only the model, leaving the split, alignment,
        # standardization, sampler, optimizer and stopping rule untouched, so
        # any change in the reported effects is attributable to the decoder
        # implementation.  Every hyper-parameter is passed explicitly rather
        # than inherited from the library defaults, even where the two agree.
        import moabb.datasets as _md
        for _old, _new in (('BNCI2014001', 'BNCI2014_001'),
                           ('BNCI2014004', 'BNCI2014_004'),
                           ('BNCI2015001', 'BNCI2015_001')):
            # braindecode 0.8 imports these pre-1.0 moabb spellings at package
            # import time from dataset helpers we never call.  Aliasing them
            # keeps moabb itself untouched; the model code is upstream's.
            if not hasattr(_md, _old) and hasattr(_md, _new):
                setattr(_md, _old, getattr(_md, _new))
        from braindecode.models import EEGNetv4

        net = EEGNetv4(
            n_chans=n_chan, n_outputs=n_main, n_times=T_input,
            final_conv_length='auto', pool_mode='mean',
            F1=8, D=2, F2=16, kernel_length=64,
            third_kernel_size=(8, 4), drop_prob=0.25,
        )

        class BraindecodeWrapper(nn.Module):
            """Adapt (B, 1, C, T) tensors and dict-valued heads to the library."""

            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                if x.dim() == 4 and x.shape[1] == 1:
                    x = x.squeeze(1)
                out = self.net(x)
                while out.dim() > 2:
                    out = out.squeeze(-1)
                return {'main': out}

        return BraindecodeWrapper(net)
    if args.variant == 'eegnet':
        from EEGNet_orig import EEGNet
        net = EEGNet(C=n_chan, T=T_input, D=2, N=n_main, F_1=8, F_2=16)

        class EEGNetWrapper(nn.Module):
            def __init__(self, eegnet):
                super().__init__()
                self.net = eegnet

            def forward(self, x):
                return {'main': self.net(x)}

        return EEGNetWrapper(net)
    if args.variant == 'minimal':
        m = build_aux_model('single', n_chan=n_chan, n_cls=n_main,
                            **{**model_kwargs, 'use_transformer': False})
        if hasattr(m, 'head') and hasattr(m.head, 'attention'):
            m.head.attention = nn.Identity()
        return m
    if args.variant in ('no-channel-attn', 'shared-channel-attn'):
        # Build flagship but with custom decoder head configuration
        from MultiDecoder_imply import MultiDecoderHybrid
        from MultiDecoderEEG import LightDecoderHead, ChannelAttention
        m = MultiDecoderHybrid(
            n_chan=n_chan, n_cls_list=[t.n_classes for t in config.tasks],
            task_names=[t.name for t in config.tasks],
            **model_kwargs,
        )
        if args.variant == 'no-channel-attn':
            # Swap DecoderHead attention for identity
            for name, dec in m.decoders.items():
                dec.attention = nn.Identity()
        elif args.variant == 'shared-channel-attn':
            shared = ChannelAttention(F_T)
            m.shared_attention = shared
            orig_forward = m.forward

            def _fwd(x, _m=m, _shared=shared, _orig=orig_forward):
                f = _m.encoder(x)
                if _m.use_transformer:
                    f = f + _m.transformer(f)
                f = _shared(f)
                outputs = {}
                for n, dec in _m.decoders.items():
                    dec.attention = nn.Identity()
                    outputs[n] = dec(f)
                return outputs
            m.forward = _fwd
        return m
    raise ValueError(f"Unknown variant: {args.variant}")


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def compute_aux_loss(variant, outputs, targets_main, base_loss):
    """Add variant-specific auxiliary loss to the base classification loss."""
    if variant == 'recon' and 'recon' in outputs and 'recon_target' in outputs:
        recon_loss = F.mse_loss(outputs['recon'], outputs['recon_target'])
        return base_loss + 0.1 * recon_loss, {'recon': recon_loss.item()}
    if variant == 'metric' and 'metric_z' in outputs:
        metric_loss = supcon_loss(outputs['metric_z'], targets_main)
        return base_loss + 0.3 * metric_loss, {'metric': metric_loss.item()}
    return base_loss, {}


def get_shared_params(model):
    """Return a tensor whose gradients represent the shared encoder trunk.
    Used by GradNorm as the reference tensor."""
    # Use the last TCN block's last conv weight as a stable reference
    for blk in reversed(model.encoder.tcn_blocks):
        return blk.conv2[0].weight
    return next(model.encoder.parameters())


def train_one_epoch(model, criterion, optimizer, train_loader, task_configs,
                    n_original_classes, device, variant, gradnorm=False):
    model.train()
    total_loss = 0.0
    per_task_correct = {t.name: 0 for t in task_configs}
    per_task_total = {t.name: 0 for t in task_configs}
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y_np = batch_y.numpy()
        task_labels = remap_labels_for_tasks(batch_y_np, n_original_classes, task_configs)
        targets = {n: torch.LongTensor(l).to(device) for n, l in task_labels.items()}

        optimizer.zero_grad()
        outputs = model(batch_x)

        if gradnorm:
            # GradNorm needs per-task losses for a secondary backward
            per_task_losses = []
            for n in criterion.task_names:
                if n in outputs and n in targets:
                    ce = criterion.criteria[n](outputs[n], targets[n])
                    per_task_losses.append(ce)
                else:
                    per_task_losses.append(torch.zeros(1, device=device).squeeze())
            weights = torch.softmax(criterion.log_weights, dim=0) * criterion.n_tasks
            loss = sum(w * l for w, l in zip(weights, per_task_losses))
        else:
            loss, _ = criterion({n: outputs[n] for n in outputs if n in targets}, targets)
            per_task_losses = None

        # Aux-task add-ons
        main_t = targets.get('main')
        if main_t is not None:
            loss, _aux = compute_aux_loss(variant, outputs, main_t, loss)

        loss.backward(retain_graph=gradnorm)

        if gradnorm:
            shared = get_shared_params(model)
            criterion.update_weights(per_task_losses, shared)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            for n in outputs:
                if n in targets:
                    preds = outputs[n].argmax(dim=1)
                    per_task_correct[n] += (preds == targets[n]).sum().item()
                    per_task_total[n] += targets[n].size(0)

    avg_loss = total_loss / max(n_batches, 1)
    task_acc = {n: per_task_correct[n] / max(per_task_total[n], 1) for n in per_task_correct}
    return avg_loss, task_acc


def evaluate(model, val_loader, task_configs, n_original_classes, device, variant):
    model.eval()
    preds_store = {t.name: [] for t in task_configs}
    true_store = {t.name: [] for t in task_configs}
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y_np = batch_y.numpy()
            task_labels = remap_labels_for_tasks(batch_y_np, n_original_classes, task_configs)
            outputs = model(batch_x)
            for t in task_configs:
                if t.name in outputs and t.name in task_labels:
                    p = outputs[t.name].argmax(dim=1).cpu().numpy()
                    preds_store[t.name].extend(p)
                    true_store[t.name].extend(task_labels[t.name])
    acc = {n: accuracy_score(true_store[n], preds_store[n]) if preds_store[n] else 0.0
           for n in preds_store}
    return acc, preds_store, true_store


# ─── Main LOSO Loop ───────────────────────────────────────────────────────────

def within_trial_zscore(X):
    """Per-trial, per-channel z-score along the time axis. No cross-subject pooling.
    X shape: (n_trials, n_chan, n_time).

    Note that forcing every channel to unit variance inside a trial removes the
    inter-channel amplitude ratio.  For motor imagery that ratio carries the
    ERD/ERS lateralization, so this mode is an audit condition rather than a
    neutral default; ``within_trial_global`` is the matched control that keeps
    the ratio intact."""
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


def within_trial_global_zscore(X):
    """Per-trial z-score over the channel and time axes jointly.

    Differs from :func:`within_trial_zscore` only in the reduction axes, so the
    contrast between the two isolates one question: does preserving the
    inter-channel amplitude ratio change the result?  Both fit their statistics
    inside a single trial and are therefore equally leakage-free.
    X shape: (n_trials, n_chan, n_time)."""
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def _ea_whitener(X):
    """Symmetric inverse square root of the mean spatial covariance of ``X``.

    Separated from the transform so a reference estimated on one set of trials
    can be frozen and applied to another.  Clipping guards the rank-deficient
    case that a heavily band-passed 22-channel epoch can reach.
    """

    covariances = np.einsum('nct,ndt->ncd', X, X) / X.shape[2]
    reference = covariances.mean(axis=0)
    values, vectors = np.linalg.eigh(reference)
    values = np.clip(values, 1e-10, None)
    return (vectors * values ** -0.5) @ vectors.T


def _apply_whitener(whitener, X):
    return np.einsum('cd,ndt->nct', whitener, X).astype(X.dtype, copy=False)


def euclidean_align(X):
    """Whiten one subject's trials by their own mean spatial covariance.

    Euclidean Alignment (He and Wu, 2020) is the standard cross-subject motor
    imagery transform.  It is computed from a subject's own trials and uses no
    labels, which makes the access it needs label-free: for the held-out subject
    it is transductive but unsupervised, which is one of the two factors the
    experiments separate.

    X shape: (n_trials, n_chan, n_time).
    """

    return _apply_whitener(_ea_whitener(X), X)


def _resolve_normalization(normalization='within_trial', legacy_norm=None):
    """Resolve the old boolean flag without allowing ambiguous protocols."""
    if legacy_norm is not None:
        legacy_mode = 'outer_train_scaler' if legacy_norm else 'within_trial'
        if normalization not in (None, 'within_trial', legacy_mode):
            raise ValueError('legacy_norm conflicts with --normalization')
        normalization = legacy_mode
    normalization = normalization or 'within_trial'
    if normalization not in NORMALIZATION_MODES:
        raise ValueError(
            f'unknown normalization={normalization!r}; '
            f'expected one of {NORMALIZATION_MODES}'
        )
    return normalization


def _fit_flat_scaler(X):
    scaler = StandardScaler()
    scaler.fit(np.asarray(X).reshape(len(X), -1))
    return scaler


def _normalize_array(X, normalization, scaler=None):
    """Apply a declared normalization using an optionally fitted scaler."""
    X = np.asarray(X)
    if normalization == 'within_trial':
        return within_trial_zscore(X), None
    if normalization == 'within_trial_global':
        return within_trial_global_zscore(X), None
    if scaler is None:
        scaler = _fit_flat_scaler(X)
    X_normed = scaler.transform(X.reshape(len(X), -1)).reshape(X.shape)
    return X_normed, scaler


def prepare_data(X, y, label_encoder=None, scaler=None,
                 normalization='within_trial', legacy_norm=None):
    """Encode labels and apply one explicit normalization mode.

    For ``outer_train_scaler`` and ``inner_train_scaler`` the caller controls
    the fitting scope by deciding which samples are passed when ``scaler`` is
    ``None``.  The LOSO runner fits the latter only after the inner split.

    Returns: X_4d (torch), y (torch), label_encoder, scaler_or_None, C, T,
    n_classes.
    """
    normalization = _resolve_normalization(normalization, legacy_norm)
    if label_encoder is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        le = label_encoder
        y_encoded = le.transform(y)
    n_classes = len(le.classes_)
    X_normed, sc_out = _normalize_array(X, normalization, scaler=scaler)
    C, T = X.shape[1], X.shape[2]
    X4 = X_normed[:, np.newaxis, :, :]
    return torch.FloatTensor(X4), torch.LongTensor(y_encoded), le, sc_out, C, T, n_classes


def _tensorize(X):
    return torch.FloatTensor(np.asarray(X)[:, np.newaxis, :, :])


def _training_sample_weights(n_source, n_target, target_weight):
    """Per-sample draw weights that realize a declared target share.

    With no target trials this is the uniform distribution, so the source-only
    conditions still draw with replacement and differ from the supervised ones
    in the presence of target trials alone rather than in the sampling regime.
    """

    if n_target <= 0:
        return np.full(n_source, 1.0 / n_source, dtype=np.float64)
    per_source = (1.0 - target_weight) / n_source
    per_target = target_weight / n_target
    return np.concatenate((
        np.full(n_source, per_source, dtype=np.float64),
        np.full(n_target, per_target, dtype=np.float64),
    ))


def _measure_target_fraction(sample_weights, n_source, n_target, seed):
    """Estimate the realized target share of one epoch's draws.

    Drawn from an independent generator so reading the number cannot perturb
    the training stream.  The expectation is the declared weight by
    construction; this records what a single epoch actually receives.
    """

    if n_target <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    probs = sample_weights / sample_weights.sum()
    drawn = rng.choice(len(probs), size=n_source, replace=True, p=probs)
    return float((drawn >= n_source).mean())


def _split_outer_train(y_encoded, seed):
    """Create a stratified trial-level split pooled across outer-train subjects.

    The held-out subject is never included in this split.  Keeping the sample
    level explicit prevents the provenance record from being misread as a
    subject-level validation protocol.
    """
    return train_test_split(
        np.arange(len(y_encoded)),
        test_size=0.2,
        stratify=y_encoded,
        random_state=seed,
    )


def _subject_blocks(data_source, subjects):
    """Load each subject's raw trials once, before any per-subject transform.

    Alignment is applied later, after the held-out subject has been divided,
    because the reference covariance of the held-out subject must be estimated
    from that subject's available half alone.
    """

    blocks = {}
    for subject in subjects:
        X, y, _ = data_source.get_data([subject])
        blocks[subject] = (np.asarray(X), np.asarray(y))
    return blocks


TARGET_TEST_FRACTION = 0.5


def _nested_available_subset(idx_avail, labels, n_keep, rng_seed):
    """Take a class-stratified subset of the available half, nested by size.

    Nested means a smaller budget's trials are a subset of a larger one's: each
    class is permuted once under a per-subject seed and the first trials of that
    fixed order are kept, so a 45-trial budget draws from the same ordering a
    90-trial budget does.  Without nesting, two budgets would differ both in
    size and in which trials they happened to contain.
    """

    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.RandomState(rng_seed)
    order = {}
    for c in classes:
        pos = np.where(labels == c)[0]
        order[c] = pos[rng.permutation(len(pos))]

    # Largest-remainder allocation, so the kept subset stays balanced and sums
    # to exactly n_keep rather than drifting with per-class rounding.
    share = {c: n_keep * len(order[c]) / len(labels) for c in classes}
    take = {c: int(np.floor(share[c])) for c in classes}
    for c in sorted(classes, key=lambda c: share[c] - take[c], reverse=True):
        if sum(take.values()) >= n_keep:
            break
        take[c] += 1

    keep = np.concatenate([order[c][:take[c]] for c in classes])
    keep.sort()
    return idx_avail[keep]


SESSION_SPLIT_DATASETS = {
    # Datasets whose loader concatenates whole recording sessions in a known,
    # chronological order, so the first half of a subject's trials is an
    # earlier session than the second.  Only these can be split by session.
    "BNCI2014_001": "official 2a: session T then session E, 288 trials each",
}


def _session_split(n_trials, dataset):
    """Split a subject chronologically at the session boundary.

    The random split answers a question about an offline calibration block; it
    says nothing about whether a decoder holds up on a later recording.  Here
    the earlier session becomes the available half and the later one is
    reserved, so the calibration budget is unchanged and only the partition
    moves from arbitrary to chronological.
    """

    if dataset not in SESSION_SPLIT_DATASETS:
        raise ValueError(
            f"--target_split session is not defined for {dataset!r}; the "
            "loader must concatenate sessions in a known order "
            f"(supported: {sorted(SESSION_SPLIT_DATASETS)})"
        )
    if n_trials % 2:
        raise ValueError(
            f"{dataset} subject has {n_trials} trials, which does not divide "
            "into two equal sessions"
        )
    half = n_trials // 2
    return np.arange(half), np.arange(half, n_trials)


def _iter_evaluation_folds(data_source, subjects, seed, align=False,
                           available_trials=None, target_split="random",
                           dataset=None):
    """Yield one fold per held-out subject, with that subject's trials divided.

    Every condition scores the same object: one held-out subject's reserved test
    trials.  Each held-out subject is divided once, stratified by class, using a
    per-subject seed, so no condition can change which trials are reserved --
    only what it is permitted to do with the other half:

      ``test``       reserved for the final score, touched by no condition
      ``available``  the only material a condition may draw on

    Under alignment the source subjects are whitened by their own full-trial
    reference, which is legitimate because every one of their trials is
    available.  The held-out subject's whitener is estimated from the available
    half alone and then frozen; the reserved test trials are transformed by it
    but never contribute to it.

    Returns ``(subject, X_outer, y_outer, X_avail, y_avail, X_test, y_test)``.
    """

    subjects = list(subjects)
    raw = _subject_blocks(data_source, subjects)

    # Seeded exactly as in schema-v3 so the reserved test trials are the same
    # objects across the two runner generations and the clean arms stay
    # numerically comparable.
    if target_split == "session":
        splits = {s: _session_split(len(raw[s][1]), dataset) for s in subjects}
    else:
        splits = {
            s: train_test_split(
                np.arange(len(raw[s][1])),
                test_size=TARGET_TEST_FRACTION,
                stratify=raw[s][1],
                random_state=seed * 1000 + int(s),
            )
            for s in subjects
        }

    source_view = {}
    for s in subjects:
        X = raw[s][0]
        source_view[s] = _apply_whitener(_ea_whitener(X), X) if align else X

    for test_subj in subjects:
        train_subs = [s for s in subjects if s != test_subj]
        X_outer = np.concatenate([source_view[s] for s in train_subs], axis=0)
        y_outer = np.concatenate([raw[s][1] for s in train_subs], axis=0)

        X_s, y_s = raw[test_subj]
        idx_avail, idx_test = splits[test_subj]
        if available_trials is not None and available_trials < len(idx_avail):
            # The reserved test trials are untouched: only the calibration
            # budget shrinks, which is the one lever that separates a
            # procedure's effect from the exposure its set size implies.
            idx_avail = _nested_available_subset(
                idx_avail, y_s[idx_avail], int(available_trials),
                seed * 1000 + int(test_subj))
        X_avail, X_te = X_s[idx_avail], X_s[idx_test]
        if align:
            whitener = _ea_whitener(X_avail)
            X_avail = _apply_whitener(whitener, X_avail)
            X_te = _apply_whitener(whitener, X_te)

        yield (test_subj, X_outer, y_outer,
               X_avail, y_s[idx_avail], X_te, y_s[idx_test])



def run(args):
    set_seed(args.seed)
    legacy_loso = bool(getattr(args, 'legacy_loso', False))
    legacy_norm = getattr(args, 'legacy_norm', None)
    normalization = _resolve_normalization(
        getattr(args, 'normalization', 'within_trial'), legacy_norm
    )
    if legacy_loso and normalization == 'inner_train_scaler':
        raise ValueError(
            '--normalization inner_train_scaler is only defined for the '
            'corrected inner-train-subject selection protocol'
        )
    lr_schedule = getattr(args, 'lr_schedule', 'step') or 'step'
    align = bool(getattr(args, 'euclidean_alignment', False))
    target_supervised = bool(getattr(args, 'target_supervised', False))
    split_level = 'trial' if target_supervised else 'subject'
    target_split = getattr(args, 'target_split', 'random') or 'random'
    available_trials = getattr(args, 'available_trials', None)
    if available_trials is not None and int(available_trials) < 1:
        raise ValueError('--available_trials must be a positive trial count')
    target_weight = float(getattr(args, 'target_sample_weight', TARGET_SAMPLE_WEIGHT))
    if not 0.0 <= target_weight < 1.0:
        raise ValueError(
            f"--target-sample-weight must lie in [0, 1), got {target_weight!r}"
        )
    if target_supervised and target_weight <= 0.0:
        raise ValueError(
            'a target-supervised condition with zero target sampling weight '
            'would train on source data alone; declare a positive weight'
        )
    protocol = protocol_id_v4(align, target_supervised, normalization, legacy_loso)
    config = make_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    print(f"[variant] {args.variant} [weighting] {args.weighting} "
          f"[dataset] {args.dataset} [seed] {args.seed}")

    data_source = build_data_source(config, args)
    subjects = list(data_source.subject_list)
    if getattr(args, 'max_subjects', None):
        subjects = subjects[:args.max_subjects]
    print(f"[data-source] {args.data_source}")
    print(f"[subjects] {subjects}")

    requested_run_id = getattr(args, 'run_id', None)
    run_id = requested_run_id or build_run_id(
        args.variant, args.weighting, args.dataset, args.seed, protocol
    )
    if requested_run_id and protocol not in requested_run_id:
        raise ValueError(
            f"explicit run_id must contain protocol id {protocol!r} to prevent collisions"
        )
    log_path = Path(args.log_dir) / f"{run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and not getattr(args, 'overwrite', False):
        raise FileExistsError(
            f"refusing to overwrite existing run log {log_path}; use --overwrite"
        )
    print(
        f"[protocol] id={protocol} selection={selection_mode(legacy_loso)} "
        f"normalization={normalization} run_id={run_id}"
    )

    metric = metric_name(args.dataset)
    protocol_meta = {
        'protocol_id': protocol,
        'protocol_family': 'cross-subject-loso',
        'schema_version': SCHEMA_VERSION,
        'selection_mode': selection_mode(legacy_loso),
        'selection_split': selection_split(legacy_loso),
        'normalization_mode': normalization,
        'normalization_fit_scope': normalization_fit_scope(normalization, legacy_loso),
        'test_subject_used_for_model_selection': legacy_loso,
        'test_subject_used_for_normalization': test_subject_used_for_normalization(
            normalization
        ),
        'split_level': split_level,
        'evaluation_unit': evaluation_unit(split_level),
        'test_subject_used_for_training': test_subject_used_for_training(split_level),
        'euclidean_alignment': align,
        # Alignment reads the held-out subject's own trials but never their
        # labels, so it is recorded separately from the label-consuming flags.
        'test_subject_signals_used_for_alignment': align,
        # None means the full available half; an integer is a declared
        # calibration budget with the reserved test half unchanged.
        'target_split': target_split,
        'available_trials_cap': (
            int(available_trials) if available_trials is not None else None),
        # --- schema-v4 factorial identity -----------------------------------
        'target_supervised': target_supervised,
        'target_access': target_access_mode(align, target_supervised),
        'target_reference_scope': target_reference_scope(align),
        'selection_scope': selection_scope_v4(legacy_loso),
        'target_sample_weight': target_weight if target_supervised else 0.0,
        'source_partition':
            'fixed_stratified_80_20_of_outer_pool_shared_by_all_conditions',
        'metric': metric,
    }
    provenance = runtime_provenance(ROOT, [sys.executable, *sys.argv])
    provenance['data_source'] = data_source.provenance()
    all_results = {}
    start_time = time.time()

    for (test_subj, X_outer, y_outer, X_avail, y_avail, X_te, y_te
         ) in _iter_evaluation_folds(data_source, subjects, args.seed, align,
                                     available_trials=available_trials,
                                     target_split=target_split,
                                     dataset=args.dataset):
        t0 = time.time()

        # The label encoder sees the outer pool, which spans every class.
        le = LabelEncoder()
        y_outer_enc = le.fit_transform(y_outer)
        y_avail_enc = le.transform(y_avail)
        y_te_encoded = le.transform(y_te)
        n_classes = len(le.classes_)
        C, T = X_outer.shape[1], X_outer.shape[2]

        # The source partition is derived from the outer pool alone, so it is
        # the same index set in all four conditions.  In schema-v3 the
        # target-supervised arm re-split after mixing target trials into the
        # pool, which changed the source train/validation membership and the
        # validation set's composition at the same time as it added target
        # labels; nothing could then be attributed to target supervision alone.
        idx_src_tr, idx_src_val = _split_outer_train(y_outer_enc, args.seed)
        X_src_tr, y_src_tr = X_outer[idx_src_tr], y_outer_enc[idx_src_tr]
        n_source_train = len(y_src_tr)

        # Validation is the same source-only set in every condition, unless the
        # secondary held-out-selection arm is requested, which changes the
        # selection set and nothing else.
        if legacy_loso:
            X_val_raw, y_val_np = X_avail, y_avail_enc
        else:
            X_val_raw, y_val_np = X_outer[idx_src_val], y_outer_enc[idx_src_val]

        if target_supervised:
            X_train_raw = np.concatenate((X_src_tr, X_avail), axis=0)
            y_train_np = np.concatenate((y_src_tr, y_avail_enc), axis=0)
            n_target_train = len(y_avail_enc)
        else:
            X_train_raw, y_train_np = X_src_tr, y_src_tr
            n_target_train = 0

        scaler = None
        if normalization == 'outer_train_scaler':
            scaler = _fit_flat_scaler(X_outer)
        elif normalization == 'pooled_all_subject_scaler':
            # The secondary pooled-scaler arm differs in fit scope alone: the
            # same flat scaler, fitted with the held-out subject's available
            # trials added.
            scaler = _fit_flat_scaler(np.concatenate((X_outer, X_avail), axis=0))
        elif normalization == 'inner_train_scaler':
            scaler = _fit_flat_scaler(X_src_tr)

        X_train_np, _ = _normalize_array(X_train_raw, normalization, scaler)
        X_val_np, _ = _normalize_array(X_val_raw, normalization, scaler)
        X_test_np, _ = _normalize_array(X_te, normalization, scaler)
        X_train, y_train = _tensorize(X_train_np), torch.LongTensor(y_train_np)
        X_val, y_val = _tensorize(X_val_np), torch.LongTensor(y_val_np)
        X_test, y_test = _tensorize(X_test_np), torch.LongTensor(y_te_encoded)

        model = build_experiment_model(config, args, n_chan=C, T_input=T).to(device)

        task_names = [t.name for t in config.tasks]
        n_cls_list = [t.n_classes for t in config.tasks]
        initial_weights = [t.loss_weight for t in config.tasks]

        # Build criterion - for non-difficulty-mtl variants with a single head,
        # `kendall/gradnorm/dwa` degenerate to equivalents since n_tasks=1.
        criterion, needs_shared = build_criterion(
            scheme=args.weighting,
            task_names=task_names,
            n_cls_list=n_cls_list,
            initial_weights=initial_weights,
        )
        criterion = criterion.to(device)

        # Class-balanced weights are computed from inner training labels only.
        if config.training.use_class_weights:
            for task in config.tasks:
                task_y = remap_labels_for_tasks(y_train_np, n_classes, [task])[task.name]
                cnt = Counter(task_y)
                total = len(task_y)
                w = torch.tensor(
                    [total / (len(cnt) * cnt.get(i, 1)) for i in range(task.n_classes)],
                    dtype=torch.float32).to(device)
                criterion.set_class_weights(task.name, w)

        all_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(all_params, lr=config.training.learning_rate,
                                     weight_decay=config.training.weight_decay)
        if lr_schedule == 'cosine':
            # Anneals across the whole declared budget, so raising --epochs
            # actually buys training instead of decaying the rate to nothing.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.training.lr_step_size,
                gamma=config.training.lr_gamma)

        # Every condition draws exactly ``n_source_train`` samples per epoch,
        # with replacement, so the optimizer budget is identical whether or not
        # target trials were added.  When they were, their expected share of a
        # batch is the declared weight rather than whatever the cohort size
        # happens to imply.
        sample_weights = _training_sample_weights(
            n_source_train, n_target_train, target_weight)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=n_source_train, replacement=True)
        realized_target_fraction = _measure_target_fraction(
            sample_weights, n_source_train, n_target_train, args.seed)

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=config.training.batch_size,
                                  sampler=sampler)
        steps_per_epoch = math.ceil(n_source_train / config.training.batch_size)
        val_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=config.training.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=config.training.batch_size, shuffle=False)

        best_acc = 0.0
        best_state = None
        best_epoch = 0
        patience = 0
        inner_val_trace = []

        for epoch in range(config.training.num_epochs):
            tr_loss, _tr_acc = train_one_epoch(
                model, criterion, optimizer, train_loader, config.tasks, n_classes,
                device, args.variant, gradnorm=needs_shared)
            scheduler.step()
            if isinstance(criterion, DWALoss):
                criterion.epoch_end()

            val_acc, _, _ = evaluate(
                model, val_loader, config.tasks, n_classes, device, args.variant)
            main_acc = val_acc.get('main', 0.0)
            inner_val_trace.append({
                'epoch': epoch + 1,
                'train_loss': float(tr_loss),
                'main_accuracy': float(main_acc),
            })
            if main_acc > best_acc:
                best_acc = main_acc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  subj={test_subj} ep={epoch+1:3d} "
                      f"loss={tr_loss:.4f} val_main={main_acc:.3f} best={best_acc:.3f}")

            if patience >= config.training.early_stopping_patience:
                print(f"  subj={test_subj} early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        # Final evaluation on the held-out test subject (the only test metric).
        final_acc, preds, true = evaluate(
            model, test_loader, config.tasks, n_classes, device, args.variant)
        main_preds = preds.get('main', [])
        main_true = true.get('main', [])
        cm = confusion_matrix(main_true, main_preds).tolist() if main_preds else None
        balanced_acc = (
            float(balanced_accuracy_score(main_true, main_preds))
            if main_preds else 0.0
        )
        all_results[str(test_subj)] = {
            'main_accuracy': float(final_acc.get('main', 0.0)),
            'balanced_accuracy': balanced_acc,
            'task_accuracy': {k: float(v) for k, v in final_acc.items()},
            'confusion_matrix': cm,
            'n_params': int(sum(p.numel() for p in model.parameters())),
            'selected_epoch': best_epoch,
            'best_inner_val_accuracy': float(best_acc),
            'inner_val_trace': inner_val_trace,
            'validation_source': protocol_meta['selection_mode'],
            # Stored so that the aggregation unit can be varied after the fact:
            # the same predictions can be grouped by true subject or by random
            # groups of equal size, without retraining anything.
            'test_predictions': [int(v) for v in main_preds],
            'test_labels': [int(v) for v in main_true],
            # What the fixed controls actually delivered on this fold.
            'controls': {
                'n_source_train': int(n_source_train),
                'n_source_val': int(len(y_val_np)),
                'n_target_train': int(n_target_train),
                'n_target_available': int(len(y_avail_enc)),
                'n_test': int(len(y_te_encoded)),
                'samples_drawn_per_epoch': int(n_source_train),
                'steps_per_epoch': int(steps_per_epoch),
                'epochs_run': int(len(inner_val_trace)),
                'total_update_steps': int(steps_per_epoch * len(inner_val_trace)),
                'declared_target_sample_weight': (
                    float(target_weight) if target_supervised else 0.0),
                'realized_target_fraction': float(realized_target_fraction),
                # Expected draws of each available trial per epoch: the
                # quantity a fixed batch share leaves free across datasets.
                'target_repeats_per_epoch': (
                    float(steps_per_epoch * config.training.batch_size
                          * realized_target_fraction / max(n_target_train, 1))
                    if target_supervised else 0.0),
            },
            'time_sec': float(time.time() - t0),
        }
        print(f"  subj={test_subj} final main={all_results[str(test_subj)]['main_accuracy']:.4f} "
              f"balanced={balanced_acc:.4f} ({all_results[str(test_subj)]['time_sec']:.1f}s)")

    main_accs = [r['main_accuracy'] for r in all_results.values()]
    metric_values = [
        r['balanced_accuracy'] if metric == 'balanced_accuracy' else r['main_accuracy']
        for r in all_results.values()
    ]
    mean_acc = float(np.mean(main_accs)) if main_accs else 0.0
    std_acc = float(np.std(main_accs)) if main_accs else 0.0
    mean_metric = float(np.mean(metric_values)) if metric_values else 0.0
    std_metric = float(np.std(metric_values)) if metric_values else 0.0
    subjects_serialized = [
        int(s) if isinstance(s, (int, np.integer)) else str(s) for s in subjects
    ]

    summary = {
        'schema_version': SCHEMA_VERSION,
        'runner_version': RUNNER_VERSION,
        'status': 'completed',
        'run_id': run_id,
        'variant': args.variant,
        'weighting': args.weighting,
        'dataset': args.dataset,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'wd': args.wd,
        'max_subjects': getattr(args, 'max_subjects', None),
        'subjects': subjects_serialized,
        'protocol': protocol_meta,
        'metric': metric,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_metric': mean_metric,
        'std_metric': std_metric,
        'task_definition': task_definition(args.dataset, config.tasks),
        'config': {
            'task_names': task_names,
            'task_n_classes': n_cls_list,
            'task_loss_weights': initial_weights,
            'lr_schedule': lr_schedule,
            'early_stopping_patience': config.training.early_stopping_patience,
            'lr_step_size': config.training.lr_step_size,
            'lr_gamma': config.training.lr_gamma,
            'class_weights': bool(config.training.use_class_weights),
        },
        # Compatibility fields are retained, but the nested protocol object is
        # authoritative and is what the aggregator validates.
        'legacy_loso': legacy_loso,
        'legacy_norm': normalization == 'outer_train_scaler',
        'normalization': normalization,
        'split_level': split_level,
        'euclidean_alignment': align,
        'target_supervised': target_supervised,
        'target_sample_weight': target_weight if target_supervised else 0.0,
        'per_subject': all_results,
        'provenance': provenance,
        'wall_sec': float(time.time() - start_time),
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("\n" + "=" * 60)
    print(f"[DONE] {run_id}  metric={mean_metric:.4f} ± {std_metric:.4f}  "
          f"wall={summary['wall_sec']:.1f}s")
    print(f"[LOG]  {log_path}")
    return summary


def build_parser():
    p = argparse.ArgumentParser(description='Paper experiment runner')
    p.add_argument('--variant', type=str, default='difficulty-mtl',
                   choices=['difficulty-mtl', 'single', 'recon', 'metric',
                            'no-transformer', 'no-channel-attn', 'shared-channel-attn',
                            'eegnet', 'braindecode-eegnet', 'minimal'])
    p.add_argument('--legacy_loso', action='store_true',
                   help='Audit-only: use held-out subject as val (test-leak protocol)')
    p.add_argument('--euclidean_alignment', action='store_true',
                   help=('Apply per-subject Euclidean Alignment (He and Wu, 2020). '
                         'Uses the held-out subject\'s own trials but none of '
                         'their labels, so it is a label-free transductive arm.'))
    p.add_argument('--lr_schedule', choices=('step', 'cosine'), default='step',
                   help=('LR schedule. step reproduces the historical '
                         'StepLR(lr_step_size, lr_gamma); cosine anneals over '
                         'the full --epochs budget.'))
    p.add_argument('--lr_step_size', type=int, default=15,
                   help='StepLR period in epochs (ignored by --lr_schedule cosine).')
    p.add_argument('--lr_gamma', type=float, default=0.5,
                   help='StepLR decay factor (ignored by --lr_schedule cosine).')
    p.add_argument('--patience', type=int, default=10,
                   help='Early-stopping patience in epochs on the selection split.')
    p.add_argument('--target_supervised', action='store_true',
                   help=('Add the held-out subject\'s available labelled trials '
                         'to the training set.  The source partition, the number '
                         'of samples drawn per epoch and the expected target '
                         'share of a batch are held fixed, so this differs from '
                         'the source-only condition in target supervision alone.'))
    p.add_argument('--target_split', type=str, default='random',
                   choices=['random', 'session'],
                   help=('How the held-out subject is divided. "random" is a '
                         'stratified 50/50 draw; "session" reserves a later '
                         'recording session, which tests whether the effects '
                         'survive a chronological split rather than an '
                         'arbitrary one.'))
    p.add_argument('--available_trials', type=int, default=None,
                   help=('Cap the held-out subject\'s available half at this '
                         'many class-stratified trials, nested across budgets. '
                         'The reserved test half is unchanged, so this is the '
                         'one lever that separates a procedure from the '
                         'exposure its calibration-set size implies.'))
    p.add_argument('--target_sample_weight', type=float,
                   default=TARGET_SAMPLE_WEIGHT,
                   help=('Expected share of every training batch drawn from the '
                         'held-out subject under --target_supervised (default '
                         f'{TARGET_SAMPLE_WEIGHT}).  Fixing it makes the '
                         'supervised contrast mean the same thing across cohort '
                         'sizes.'))
    p.add_argument('--normalization', choices=NORMALIZATION_MODES,
                   default='within_trial',
                   help=('Normalization mode: within_trial (default), '
                         'within_trial_global (matched control that keeps the '
                         'inter-channel amplitude ratio), outer_train_scaler, '
                         'inner_train_scaler (strict train-only scaler arm), or '
                         'pooled_all_subject_scaler (audit-only: fitted with '
                         'held-out-subject trials).'))
    p.add_argument('--legacy_norm', action='store_true', default=None,
                   help=('Deprecated audit alias for '
                         '--normalization outer_train_scaler.'))
    p.add_argument('--weighting', type=str, default='kendall',
                   choices=['kendall', 'fixed', 'equal', 'gradnorm', 'dwa'])
    p.add_argument('--dataset', type=str, default='BNCI2014_001',
                   choices=['BNCI2014_001', 'BNCI2015_004', 'BNCI2014_009', 'PhysionetMI',
                            'BNCI2014_002'])
    p.add_argument('--data-source', choices=['moabb', 'official-gdf'], default='moabb',
                   help=('Declared dataset loader. official-gdf reads only the local '
                         'official BCI Competition IV 2a archives and supports '
                         'BNCI2014_001.'))
    p.add_argument('--official-gdf-root', default='.cache/bciciv_2a',
                   help='Directory containing BCICIV_2a_gdf.zip and true_labels.zip.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--run_id', type=str, default=None,
                   help='Optional explicit ID; it must include the protocol ID.')
    p.add_argument('--overwrite', action='store_true',
                   help='Allow replacement of an existing log with the same run ID.')
    p.add_argument('--max_subjects', type=int, default=None,
                   help='If set, only use the first N subjects (for pilots)')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    run(args)
