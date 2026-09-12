"""Configuration system for multi-decoder EEG experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EncoderConfig:
    F_T: int = 64          # TCN output channels
    K_T: int = 3           # TCN kernel size
    L: int = 2             # Number of TCN layers


@dataclass
class TransformerConfig:
    use_transformer: bool = True
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    dropout: float = 0.5


@dataclass
class TaskConfig:
    """Defines a single classification sub-task."""
    name: str
    n_classes: int
    loss_weight: float = 1.0  # weight in multi-task loss


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_step_size: int = 15
    lr_gamma: float = 0.5
    early_stopping_patience: int = 10
    use_class_weights: bool = True
    use_dynamic_weighting: bool = True  # uncertainty-based task weighting


@dataclass
class DataConfig:
    dataset_name: str = 'BNCI2014_001'  # 4-class motor imagery
    paradigm: str = 'MotorImagery'


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tasks: List[TaskConfig] = field(default_factory=list)
    model_type: str = 'hybrid'  # 'train', 'control', 'hybrid'
    save_dir: str = 'checkpoints'


# ─── Preset configurations ───────────────────────────────────────────────────

def get_4class_mi_config():
    """4-class motor imagery with BCI Competition IV 2a dataset."""
    config = ExperimentConfig(
        data=DataConfig(dataset_name='BNCI2014_001', paradigm='MotorImagery'),
        tasks=[
            TaskConfig(name='binary', n_classes=2, loss_weight=0.3),
            TaskConfig(name='main_4cls', n_classes=4, loss_weight=1.0),
        ],
        model_type='hybrid',
    )
    return config


def get_5class_mi_config():
    """5-class motor imagery with BNCI2015_004 dataset."""
    config = ExperimentConfig(
        data=DataConfig(dataset_name='BNCI2015_004', paradigm='MotorImagery'),
        tasks=[
            TaskConfig(name='binary', n_classes=2, loss_weight=0.3),
            TaskConfig(name='coarse_3cls', n_classes=3, loss_weight=0.5),
            TaskConfig(name='main_5cls', n_classes=5, loss_weight=1.0),
        ],
        model_type='hybrid',
    )
    return config


def get_control_config():
    """Control mode: intensity (3-class) + direction (5-class)."""
    config = ExperimentConfig(
        data=DataConfig(dataset_name='BNCI2015_004', paradigm='MotorImagery'),
        tasks=[
            TaskConfig(name='intensity', n_classes=3, loss_weight=0.8),
            TaskConfig(name='direction', n_classes=5, loss_weight=1.0),
        ],
        model_type='control',
    )
    return config


def get_ablation_no_transformer_config():
    """Ablation: hybrid model without transformer refinement."""
    config = get_4class_mi_config()
    config.transformer.use_transformer = False
    return config


def get_ablation_single_decoder_config():
    """Ablation: single decoder (standard classification, no multi-task)."""
    config = ExperimentConfig(
        data=DataConfig(dataset_name='BNCI2014_001', paradigm='MotorImagery'),
        tasks=[
            TaskConfig(name='main_4cls', n_classes=4, loss_weight=1.0),
        ],
        model_type='hybrid',
    )
    return config
