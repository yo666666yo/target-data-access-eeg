"""Alternative multi-task weighting schemes for the paper's ablation.

All schemes expose the same interface as MultiTaskLoss (from train_multidecoder):
    loss, loss_dict = criterion(outputs, targets)

Supported schemes (paper claim C2):
    - "fixed"         fixed loss weights from config
    - "equal"         all weights 1.0 (degenerate fixed)
    - "kendall"       Kendall et al. 2018 learnable log-variance (already in MultiTaskLoss)
    - "gradnorm"      GradNorm (Chen et al. 2018)
    - "dwa"           Dynamic Weight Averaging (Liu et al. 2019)

Kendall is provided by `MultiTaskLoss` in train_multidecoder.py with use_dynamic=True.
The schemes here share parameters with that class so the experiment runner can drop
them in without a second inheritance branch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedWeightLoss(nn.Module):
    """Fixed-weight multi-task CE loss. Baseline for ablation C2."""

    def __init__(self, task_names, weights=None, class_weights=None):
        super().__init__()
        self.task_names = task_names
        if weights is None:
            weights = {n: 1.0 for n in task_names}
        self.weights = dict(weights)
        self.criteria = nn.ModuleDict()
        for n in task_names:
            w = class_weights.get(n) if class_weights else None
            self.criteria[n] = nn.CrossEntropyLoss(weight=w)

    def set_class_weights(self, task_name, weights):
        self.criteria[task_name] = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets):
        total = 0.0
        loss_dict = {}
        for n in self.task_names:
            if n not in outputs or n not in targets:
                continue
            ce = self.criteria[n](outputs[n], targets[n])
            total = total + self.weights[n] * ce
            loss_dict[n] = ce.item()
        return total, loss_dict

    def get_task_weights(self):
        return dict(self.weights)


class GradNormLoss(nn.Module):
    """GradNorm (Chen et al. 2018). Dynamically balances task gradient magnitudes.

    On each step, measures the gradient norm of each task loss w.r.t. a chosen
    "shared" parameter tensor and nudges per-task weights so that (relative loss)^α
    is approximately equal across tasks.
    """

    def __init__(self, task_names, alpha=1.5, lr_w=0.025):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        self.lr_w = lr_w
        self.n_tasks = len(task_names)
        self.log_weights = nn.Parameter(torch.zeros(self.n_tasks))
        self.register_buffer('init_loss', torch.zeros(self.n_tasks))
        self.init_loss_set = False
        self.criteria = nn.ModuleDict({n: nn.CrossEntropyLoss() for n in task_names})

    def set_class_weights(self, task_name, weights):
        self.criteria[task_name] = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets, shared_params=None):
        task_losses = []
        loss_dict = {}
        for i, n in enumerate(self.task_names):
            if n in outputs and n in targets:
                ce = self.criteria[n](outputs[n], targets[n])
                task_losses.append(ce)
                loss_dict[n] = ce.item()
            else:
                task_losses.append(torch.zeros(1, device=self.log_weights.device).squeeze())

        if not self.init_loss_set:
            with torch.no_grad():
                for i, l in enumerate(task_losses):
                    self.init_loss[i] = l.detach()
            self.init_loss_set = True

        weights = torch.softmax(self.log_weights, dim=0) * self.n_tasks
        total = sum(w * l for w, l in zip(weights, task_losses))
        return total, loss_dict

    def update_weights(self, task_losses, shared_params):
        """Compute GradNorm update. Call after main loss.backward() but before
        optimizer step. `shared_params` is a tensor that all tasks' losses flow through."""
        if not self.init_loss_set:
            return
        grad_norms = []
        for l in task_losses:
            g = torch.autograd.grad(l, shared_params, retain_graph=True,
                                    create_graph=False, allow_unused=True)[0]
            if g is None:
                grad_norms.append(torch.tensor(0.0, device=self.log_weights.device))
            else:
                grad_norms.append(g.norm(2))
        grad_norms = torch.stack(grad_norms)

        losses_ratio = torch.stack([l / (self.init_loss[i] + 1e-8)
                                     for i, l in enumerate(task_losses)])
        inverse_train_rate = losses_ratio / (losses_ratio.mean() + 1e-8)
        target = grad_norms.mean().detach() * (inverse_train_rate ** self.alpha).detach()
        gn_loss = F.l1_loss(grad_norms, target)

        g_w = torch.autograd.grad(gn_loss, self.log_weights, retain_graph=True)[0]
        with torch.no_grad():
            self.log_weights.sub_(self.lr_w * g_w)

    def get_task_weights(self):
        with torch.no_grad():
            w = torch.softmax(self.log_weights, dim=0) * self.n_tasks
        return {n: float(w[i].item()) for i, n in enumerate(self.task_names)}


class DWALoss(nn.Module):
    """Dynamic Weight Averaging (Liu et al. 2019 End-to-End Multi-Task Learning).

    Weights = softmax(loss_ratio_t / T) where loss_ratio_t = L_{t-1} / L_{t-2}.
    No gradient computation overhead (unlike GradNorm).
    """

    def __init__(self, task_names, temp=2.0):
        super().__init__()
        self.task_names = task_names
        self.temp = temp
        self.n_tasks = len(task_names)
        self.register_buffer('prev_loss', torch.ones(self.n_tasks))
        self.register_buffer('curr_loss', torch.ones(self.n_tasks))
        self.register_buffer('loss_accumulator', torch.zeros(self.n_tasks))
        self.register_buffer('n_accumulated', torch.tensor(0))
        self.register_buffer('weights', torch.ones(self.n_tasks))
        self.criteria = nn.ModuleDict({n: nn.CrossEntropyLoss() for n in task_names})

    def set_class_weights(self, task_name, weights):
        self.criteria[task_name] = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets):
        total = 0.0
        loss_dict = {}
        per_task_ce = []
        for i, n in enumerate(self.task_names):
            if n in outputs and n in targets:
                ce = self.criteria[n](outputs[n], targets[n])
                per_task_ce.append(ce.detach())
                total = total + self.weights[i] * ce
                loss_dict[n] = ce.item()
            else:
                per_task_ce.append(torch.tensor(0.0, device=self.weights.device))
        with torch.no_grad():
            self.loss_accumulator += torch.stack([l.detach() for l in per_task_ce])
            self.n_accumulated += 1
        return total, loss_dict

    def epoch_end(self):
        """Call at the end of each training epoch."""
        if self.n_accumulated.item() == 0:
            return
        avg = self.loss_accumulator / self.n_accumulated.float()
        self.prev_loss = self.curr_loss.clone()
        self.curr_loss = avg.clone()
        ratio = self.curr_loss / (self.prev_loss + 1e-8)
        self.weights = torch.softmax(ratio / self.temp, dim=0) * self.n_tasks
        self.loss_accumulator.zero_()
        self.n_accumulated.zero_()

    def get_task_weights(self):
        return {n: float(self.weights[i].item()) for i, n in enumerate(self.task_names)}


def build_criterion(scheme, task_names, n_cls_list, initial_weights=None,
                    class_weights_map=None):
    """Factory. Returns (criterion, needs_shared_params).

    For 'kendall', import and return the existing MultiTaskLoss.
    """
    scheme = scheme.lower()
    if scheme == 'kendall':
        from train_multidecoder import MultiTaskLoss
        c = MultiTaskLoss(task_names=task_names, n_cls_list=n_cls_list,
                          initial_weights=initial_weights, use_dynamic=True)
        return c, False
    elif scheme == 'fixed':
        weights = {n: w for n, w in zip(task_names, initial_weights or [1.0] * len(task_names))}
        c = FixedWeightLoss(task_names=task_names, weights=weights,
                            class_weights=class_weights_map)
        return c, False
    elif scheme == 'equal':
        c = FixedWeightLoss(task_names=task_names,
                            weights={n: 1.0 for n in task_names},
                            class_weights=class_weights_map)
        return c, False
    elif scheme == 'gradnorm':
        c = GradNormLoss(task_names=task_names)
        return c, True
    elif scheme == 'dwa':
        c = DWALoss(task_names=task_names)
        return c, False
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
