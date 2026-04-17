"""
训练配置模块
============
定义所有训练方法共用的配置参数和数据标签映射。
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path


# ======================== 路径配置 ========================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ======================== 数据集配置 ========================

MODEL_NAME = "roberta-base"
DATASET_NAME = "tweet_eval"
DATASET_SUBSET = "emotion"

LABEL_MAP = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness"
}
LABEL_NAMES = ["anger", "joy", "optimism", "sadness"]
NUM_LABELS = len(LABEL_NAMES)

RANDOM_SEED = 42


# ======================== 训练超参数 ========================

@dataclass
class TrainingConfig:
    """通用训练配置"""
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3


@dataclass
class LoRAConfig:
    """LoRA 特定配置"""
    rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "value"]


@dataclass
class ExperimentConfig:
    """实验配置"""
    method: str  # full / lora / bitfit / zeroshot
    sample_size: int  # 200 / 500 / 1000 / 2000
    training: TrainingConfig = None
    lora: LoRAConfig = None

    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.lora is None and self.method == "lora":
            self.lora = LoRAConfig()


# ======================== 数据子集规模 ========================

SAMPLE_SIZES = [200, 500, 1000, 2000]


# ======================== 可复现性检查 ========================

def validate_config(config: ExperimentConfig) -> bool:
    """验证配置参数合法性"""
    valid_methods = ["full", "lora", "bitfit", "zeroshot"]

    if config.method not in valid_methods:
        raise ValueError(f"无效的训练方法: {config.method}，可选: {valid_methods}")

    if config.sample_size not in SAMPLE_SIZES:
        raise ValueError(
            f"无效的样本规模: {config.sample_size}，可选: {SAMPLE_SIZES}"
        )

    return True
