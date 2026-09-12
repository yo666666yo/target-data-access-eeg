"""Multi-decoder EEG training pipeline with multi-task learning.

Supports three model types:
- 'train': MultiDecoder4Train — shared encoder + multiple difficulty-level decoders
- 'control': MultiDecoder4Control — shared encoder + intensity/direction decoders
- 'hybrid': MultiDecoderHybrid — CNN + Transformer + multi-decoder (default)

Uses MOABB datasets with Leave-One-Subject-Out cross-validation.
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EEGNets'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))

from MultiDecoder_imply import MultiDecoder4Train, MultiDecoder4Control, MultiDecoderHybrid
from multidecoder_config import (
    ExperimentConfig, get_4class_mi_config, get_5class_mi_config
)


# ─── Label Remapping for Multi-Task Learning ─────────────────────────────────

def remap_labels_for_tasks(y, n_original_classes, task_configs):
    """Generate sub-task labels from the original classification labels.

    Strategy:
    - 2-class (binary): class 0 vs rest → useful as auxiliary regularizer
    - 3-class (coarse): group classes into 3 clusters
    - N-class (original): keep original labels

    Args:
        y: Original label array (int, 0-indexed)
        n_original_classes: Number of unique classes in original task
        task_configs: List of TaskConfig defining each sub-task

    Returns:
        Dict[str, np.ndarray] mapping task_name → label array
    """
    task_labels = {}
    for task in task_configs:
        n_cls = task.n_classes
        if n_cls == n_original_classes:
            task_labels[task.name] = y.copy()
        elif n_cls == 2:
            # Binary: first class vs. rest
            task_labels[task.name] = (y > 0).astype(np.int64)
        elif n_cls == 3:
            if n_original_classes == 4:
                # 4→3: merge classes {0,1}→0, {2}→1, {3}→2
                mapping = {0: 0, 1: 0, 2: 1, 3: 2}
            elif n_original_classes == 5:
                # 5→3: merge {0,1}→0, {2}→1, {3,4}→2
                mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
            else:
                # Generic: evenly partition
                mapping = {}
                per_group = max(1, n_original_classes // n_cls)
                for c in range(n_original_classes):
                    mapping[c] = min(c // per_group, n_cls - 1)
            task_labels[task.name] = np.array([mapping[int(c)] for c in y], dtype=np.int64)
        else:
            # For other intermediate class counts, use modular mapping
            task_labels[task.name] = (y % n_cls).astype(np.int64)
    return task_labels


# ─── Dynamic Task Weighting (Uncertainty-Based) ──────────────────────────────

class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty-based task weighting.

    Based on Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics" (CVPR 2018).

    Each task has a learnable log-variance parameter. The loss for task i is:
        L_i = (1 / (2 * sigma_i^2)) * CE_i + log(sigma_i)

    This automatically balances tasks — harder tasks get lower weight.
    """
    def __init__(self, task_names, n_cls_list, initial_weights=None,
                 use_dynamic=True, use_class_weights=True):
        super(MultiTaskLoss, self).__init__()
        self.task_names = task_names
        self.use_dynamic = use_dynamic
        n_tasks = len(task_names)

        if use_dynamic:
            # Learnable log-variance for each task
            self.log_vars = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(n_tasks)
            ])

        # Static weights as fallback
        if initial_weights is not None:
            self.static_weights = {name: w for name, w in zip(task_names, initial_weights)}
        else:
            self.static_weights = {name: 1.0 for name in task_names}

        self.criteria = nn.ModuleDict()
        for name, n_cls in zip(task_names, n_cls_list):
            self.criteria[name] = nn.CrossEntropyLoss()

    def set_class_weights(self, task_name, weights):
        """Set class-balanced weights for a specific task's CE loss."""
        self.criteria[task_name] = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets):
        """Compute multi-task loss.

        Args:
            outputs: Dict[str, Tensor] — task_name → (B, n_cls) logits
            targets: Dict[str, Tensor] — task_name → (B,) labels

        Returns:
            total_loss, loss_dict (per-task losses for logging)
        """
        total_loss = 0
        loss_dict = {}

        for i, name in enumerate(self.task_names):
            if name not in outputs or name not in targets:
                continue
            ce_loss = self.criteria[name](outputs[name], targets[name])

            if self.use_dynamic:
                precision = torch.exp(-self.log_vars[i])
                task_loss = precision * ce_loss + self.log_vars[i]
            else:
                task_loss = self.static_weights[name] * ce_loss

            total_loss = total_loss + task_loss
            loss_dict[name] = ce_loss.item()

        return total_loss, loss_dict

    def get_task_weights(self):
        """Get current effective task weights for logging."""
        if self.use_dynamic:
            return {name: torch.exp(-self.log_vars[i]).item()
                    for i, name in enumerate(self.task_names)}
        return self.static_weights.copy()


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(config, n_chan):
    """Build model from config."""
    task_names = [t.name for t in config.tasks]
    n_cls_list = [t.n_classes for t in config.tasks]

    if config.model_type == 'train':
        model = MultiDecoder4Train(
            n_chan=n_chan,
            n_cls_list=n_cls_list,
            task_names=task_names,
            F_T=config.encoder.F_T,
            K_T=config.encoder.K_T,
            L=config.encoder.L,
            hidden_dim=config.decoder.hidden_dim,
            dropout=config.decoder.dropout,
        )
    elif config.model_type == 'control':
        model = MultiDecoder4Control(
            n_chan=n_chan,
            n_intensity=n_cls_list[0],
            n_direction=n_cls_list[1],
            F_T=config.encoder.F_T,
            K_T=config.encoder.K_T,
            L=config.encoder.L,
            hidden_dim=config.decoder.hidden_dim,
            dropout=config.decoder.dropout,
        )
    elif config.model_type == 'hybrid':
        model = MultiDecoderHybrid(
            n_chan=n_chan,
            n_cls_list=n_cls_list,
            task_names=task_names,
            F_T=config.encoder.F_T,
            K_T=config.encoder.K_T,
            L=config.encoder.L,
            n_heads=config.transformer.n_heads,
            n_transformer_layers=config.transformer.n_layers,
            hidden_dim=config.decoder.hidden_dim,
            dropout=config.decoder.dropout,
            use_transformer=config.transformer.use_transformer,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    return model


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(config):
    """Load dataset using MOABB."""
    from moabb import datasets
    from moabb.paradigms import MotorImagery, P300

    dataset_map = {
        'BNCI2014_001': datasets.BNCI2014_001,
        'BNCI2014_009': datasets.BNCI2014_009,
        'BNCI2015_004': datasets.BNCI2015_004,
        'PhysionetMI': datasets.PhysionetMI,
        'BNCI2014_002': datasets.BNCI2014_002,
    }
    paradigm_map = {
        'MotorImagery': MotorImagery,
        'P300': P300,
    }

    dataset = dataset_map[config.data.dataset_name]()
    if config.data.dataset_name == 'BNCI2014_002':
        paradigm = MotorImagery(events=['right_hand', 'feet'], n_classes=2)
    elif config.data.dataset_name == 'PhysionetMI':
        # Drop the 'rest' class so the task is the same four-class motor
        # imagery decision as BNCI2014_001, which is what makes the two
        # datasets comparable in the ladder.
        paradigm = MotorImagery(
            events=['left_hand', 'right_hand', 'hands', 'feet'], n_classes=4)
    else:
        paradigm = paradigm_map[config.data.paradigm]()
    return dataset, paradigm


def prepare_data(X, y, device):
    """Preprocess: normalize, reshape, convert to tensors."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    C, T = X.shape[1], X.shape[2]
    X_4d = X_scaled[:, np.newaxis, :, :]  # (N, 1, C, T)

    X_tensor = torch.FloatTensor(X_4d)
    y_tensor = torch.LongTensor(y_encoded)

    return X_tensor, y_tensor, le, scaler, C, T, n_classes


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, optimizer, train_loader, task_configs,
                    n_original_classes, device):
    """Train for one epoch with multi-task labels."""
    model.train()
    total_loss = 0
    task_correct = {t.name: 0 for t in task_configs}
    task_total = {t.name: 0 for t in task_configs}
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y_np = batch_y.numpy()

        # Generate multi-task labels
        task_labels = remap_labels_for_tasks(batch_y_np, n_original_classes, task_configs)
        targets = {name: torch.LongTensor(labels).to(device)
                   for name, labels in task_labels.items()}

        optimizer.zero_grad()
        outputs = model(batch_x)

        # For control model, map outputs to task names
        if isinstance(outputs, dict) and 'intensity' in outputs:
            mapped_outputs = {}
            mapped_targets = {}
            for t in task_configs:
                if t.name == 'intensity' and 'intensity' in outputs:
                    mapped_outputs[t.name] = outputs['intensity']
                    mapped_targets[t.name] = targets[t.name]
                elif t.name == 'direction' and 'direction' in outputs:
                    mapped_outputs[t.name] = outputs['direction']
                    mapped_targets[t.name] = targets[t.name]
            outputs = mapped_outputs
            targets = mapped_targets

        loss, _ = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Track per-task accuracy
        with torch.no_grad():
            for name in outputs:
                if name in targets:
                    preds = outputs[name].argmax(dim=1)
                    task_correct[name] += (preds == targets[name]).sum().item()
                    task_total[name] += targets[name].size(0)

    avg_loss = total_loss / max(n_batches, 1)
    task_acc = {name: task_correct[name] / max(task_total[name], 1)
                for name in task_correct}
    return avg_loss, task_acc


def evaluate(model, val_loader, task_configs, n_original_classes, device):
    """Evaluate model on validation set."""
    model.eval()
    task_preds = {t.name: [] for t in task_configs}
    task_true = {t.name: [] for t in task_configs}

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y_np = batch_y.numpy()

            task_labels = remap_labels_for_tasks(batch_y_np, n_original_classes, task_configs)
            outputs = model(batch_x)

            if isinstance(outputs, dict) and 'intensity' in outputs:
                pass  # control model outputs already named correctly

            for t in task_configs:
                if t.name in outputs and t.name in task_labels:
                    preds = outputs[t.name].argmax(dim=1).cpu().numpy()
                    task_preds[t.name].extend(preds)
                    task_true[t.name].extend(task_labels[t.name])

    task_acc = {}
    for name in task_preds:
        if task_preds[name]:
            task_acc[name] = accuracy_score(task_true[name], task_preds[name])
    return task_acc, task_preds, task_true


# ─── Main Training Function ──────────────────────────────────────────────────

def train_multidecoder(config=None):
    """Full multi-decoder training pipeline with LOSO cross-validation.

    Args:
        config: ExperimentConfig. If None, uses 4-class MI default.

    Returns:
        Dict with per-subject results and aggregate metrics.
    """
    if config is None:
        config = get_4class_mi_config()

    print("=" * 70)
    print(f"  Multi-Decoder EEG Training Pipeline")
    print(f"  Model: {config.model_type} | Dataset: {config.data.dataset_name}")
    print(f"  Tasks: {[f'{t.name}({t.n_classes}cls)' for t in config.tasks]}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset, paradigm = load_dataset(config)
    subjects = dataset.subject_list
    print(f"Subjects: {subjects}")

    all_results = {}
    main_task_name = config.tasks[-1].name

    # LOSO Cross-Validation
    for test_subj in subjects:
        print(f"\n{'─' * 60}")
        print(f"  LOSO: Test subject = {test_subj}")
        print(f"{'─' * 60}")

        train_subs = [s for s in subjects if s != test_subj]
        X_train_raw, y_train_raw, _ = paradigm.get_data(dataset=dataset, subjects=train_subs)
        X_val_raw, y_val_raw, _ = paradigm.get_data(dataset=dataset, subjects=[test_subj])

        # Prepare data
        X_train, y_train, le, scaler, C, T, n_classes = prepare_data(
            X_train_raw, y_train_raw, device)

        # Apply same transform to validation
        y_val_encoded = le.transform(y_val_raw)
        X_val_flat = X_val_raw.reshape(X_val_raw.shape[0], -1)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val_raw.shape)
        X_val = torch.FloatTensor(X_val_scaled[:, np.newaxis, :, :])
        y_val = torch.LongTensor(y_val_encoded)

        print(f"  Channels: {C}, Time points: {T}, Classes: {n_classes}")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"  Label distribution: {Counter(y_train.numpy())}")

        # Build model
        model = build_model(config, n_chan=C).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        # Build multi-task loss
        task_names = [t.name for t in config.tasks]
        n_cls_list = [t.n_classes for t in config.tasks]
        initial_weights = [t.loss_weight for t in config.tasks]

        criterion = MultiTaskLoss(
            task_names=task_names,
            n_cls_list=n_cls_list,
            initial_weights=initial_weights,
            use_dynamic=config.training.use_dynamic_weighting,
        ).to(device)

        # Set class-balanced weights for each task
        if config.training.use_class_weights:
            y_np = y_train.numpy()
            for task in config.tasks:
                task_y = remap_labels_for_tasks(y_np, n_classes, [task])[task.name]
                counts = Counter(task_y)
                total = len(task_y)
                w = torch.tensor([total / (len(counts) * counts[i])
                                  for i in range(task.n_classes)],
                                 dtype=torch.float32).to(device)
                criterion.set_class_weights(task.name, w)

        # Optimizer includes both model and loss parameters (for dynamic weighting)
        all_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(all_params, lr=config.training.learning_rate,
                                      weight_decay=config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.training.lr_step_size,
            gamma=config.training.lr_gamma)

        # Data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=config.training.batch_size, shuffle=False)

        # Training loop
        best_main_acc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(config.training.num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, criterion, optimizer, train_loader,
                config.tasks, n_classes, device)
            scheduler.step()

            val_acc, _, _ = evaluate(
                model, val_loader, config.tasks, n_classes, device)

            main_val_acc = val_acc.get(main_task_name, 0)

            if main_val_acc > best_main_acc:
                best_main_acc = main_val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                task_weights = criterion.get_task_weights()
                acc_str = " | ".join([f"{n}: {val_acc.get(n, 0):.3f}" for n in task_names])
                weight_str = " | ".join([f"{n}: {task_weights.get(n, 0):.2f}" for n in task_names])
                print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                      f"Val [{acc_str}] | Weights [{weight_str}]")

            if patience_counter >= config.training.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Final evaluation with best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)

        final_acc, final_preds, final_true = evaluate(
            model, val_loader, config.tasks, n_classes, device)

        print(f"\n  Subject {test_subj} Results:")
        for name in task_names:
            acc = final_acc.get(name, 0)
            print(f"    {name}: {acc:.4f}")
        if main_task_name in final_preds and final_preds[main_task_name]:
            print(classification_report(
                final_true[main_task_name], final_preds[main_task_name],
                target_names=[str(c) for c in le.classes_], zero_division=0))

        all_results[test_subj] = {
            'task_accuracy': final_acc,
            'main_accuracy': final_acc.get(main_task_name, 0),
            'confusion_matrix': confusion_matrix(
                final_true.get(main_task_name, []),
                final_preds.get(main_task_name, [])
            ).tolist() if main_task_name in final_preds else None,
        }

    # Aggregate results
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    main_accs = [r['main_accuracy'] for r in all_results.values()]
    mean_acc = np.mean(main_accs) if main_accs else 0
    std_acc = np.std(main_accs) if main_accs else 0

    print(f"\n  Per-subject main task ({main_task_name}) accuracy:")
    for subj, result in all_results.items():
        print(f"    Subject {subj}: {result['main_accuracy']:.4f}")

    print(f"\n  Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Per-task aggregate
    for task in config.tasks:
        task_accs = [r['task_accuracy'].get(task.name, 0) for r in all_results.values()]
        print(f"  {task.name}: {np.mean(task_accs):.4f} +/- {np.std(task_accs):.4f}")

    # Save results
    os.makedirs(config.save_dir, exist_ok=True)
    results_path = os.path.join(config.save_dir,
        f'results_{config.model_type}_{config.data.dataset_name}.json')
    save_data = {
        'model_type': config.model_type,
        'dataset': config.data.dataset_name,
        'tasks': [{'name': t.name, 'n_classes': t.n_classes} for t in config.tasks],
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'per_subject': {str(k): {
            'main_accuracy': float(v['main_accuracy']),
            'task_accuracy': {tn: float(ta) for tn, ta in v['task_accuracy'].items()}
        } for k, v in all_results.items()},
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_results, mean_acc, std_acc


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Decoder EEG Training')
    parser.add_argument('--config', type=str, default='4class',
                        choices=['4class', '5class', 'control',
                                 'ablation_no_transformer', 'ablation_single_decoder'],
                        help='Experiment configuration preset')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--no_dynamic_weighting', action='store_true',
                        help='Disable dynamic task weighting')
    args = parser.parse_args()

    config_map = {
        '4class': get_4class_mi_config,
        '5class': get_5class_mi_config,
    }

    # Import ablation configs
    from multidecoder_config import (
        get_ablation_no_transformer_config,
        get_ablation_single_decoder_config,
        get_control_config,
    )
    config_map['control'] = get_control_config
    config_map['ablation_no_transformer'] = get_ablation_no_transformer_config
    config_map['ablation_single_decoder'] = get_ablation_single_decoder_config

    config = config_map[args.config]()

    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.no_dynamic_weighting:
        config.training.use_dynamic_weighting = False

    results, mean_acc, std_acc = train_multidecoder(config)
    print(f"\nDone! Final: {mean_acc:.4f} +/- {std_acc:.4f}")
