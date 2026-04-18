"""
基础工具模块
============
提供训练脚本共用的工具函数，包括：
- 设备检测与自动回退（CUDA / MPS / CPU）
- VRAM 监控
- 日志配置
- 指标追踪
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
from sklearn.metrics import accuracy_score


# ======================== MPS 显存峰值追踪 ========================
# MPS 没有 torch.mps.max_memory_allocated() API，只能手动记录峰值
_mps_peak_memory_gb = 0.0


def record_mps_peak():
    """在每个 batch forward 后调用，更新 MPS 峰值显存追踪"""
    global _mps_peak_memory_gb
    current = torch.mps.current_allocated_memory() / 1e9
    if current > _mps_peak_memory_gb:
        _mps_peak_memory_gb = current


# ======================== 设备管理 ========================

def get_device() -> torch.device:
    """
    自动检测可用设备并返回最佳设备。

    检测优先级：CUDA > MPS > CPU
    对于 CUDA，确保计算能力兼容；对于 MPS，捕获异常并回退到 CPU。

    Returns:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"检测到 CUDA 设备: {gpu_name}, 显存: {total_mem:.1f} GB")
        return device

    if torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1).to("mps")
            device = torch.device("mps")
            logging.info("检测到 Apple MPS 设备")
            return device
        except Exception as e:
            logging.warning(f"MPS 设备测试失败 ({e})，回退到 CPU")

    device = torch.device("cpu")
    logging.info("未检测到 GPU，使用 CPU 设备")
    return device


def reset_memory_stats(device: torch.device):
    """重置 GPU/MPS 内存统计"""
    global _mps_peak_memory_gb
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
        _mps_peak_memory_gb = 0.0


def get_peak_memory_gb(device: torch.device) -> float:
    """
    获取峰值显存占用（GB）

    Args:
        device: 计算设备

    Returns:
        float: 峰值显存占用（GB），CPU 返回 0
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    elif device.type == "mps":
        return _mps_peak_memory_gb
    return 0.0


# ======================== 随机种子 ========================

def set_seed(seed: int = 42):
    """
    设置所有随机种子，确保实验完全可复现

    包括：Python random、NumPy、PyTorch（CPU/CUDA）
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"全局随机种子已设置为: {seed}")


# ======================== 指标计算 ========================

def compute_accuracy(predictions: list, references: list) -> float:
    """计算分类准确率"""
    return accuracy_score(references, predictions)


def compute_per_class_accuracy(
    predictions: list,
    references: list,
    label_names: list
) -> Dict[str, float]:
    """计算每个类别的准确率"""
    from collections import defaultdict
    correct = defaultdict(int)
    total = defaultdict(int)

    for pred, ref in zip(predictions, references):
        total[ref] += 1
        if pred == ref:
            correct[ref] += 1

    return {
        label_names[i]: (correct[i] / total[i] if total[i] > 0 else 0.0)
        for i in range(len(label_names))
    }


# ======================== 时间工具 ========================

class Timer:
    """简易计时器，用于测量训练/推理耗时"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ======================== 日志配置 ========================

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    配置日志输出格式

    同时输出到 stdout 和指定文件（可选）
    格式示例: 2024-04-01 10:30:45 - INFO - 训练开始
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


# ======================== 路径工具 ========================

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def ensure_dir(path: Path):
    """确保目录存在，不存在则创建"""
    path.mkdir(parents=True, exist_ok=True)


# ======================== 实验结果保存 ========================

class ResultsTracker:
    """实验结果追踪器，保存关键指标到 JSON"""

    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        self.results[key] = value

    def update_dict(self, data: Dict[str, Any]):
        self.results.update(data)

    def save(self):
        import json
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logging.info(f"实验结果已保存至: {self.save_path}")

    def load(self) -> Dict[str, Any]:
        import json
        if self.save_path.exists():
            with open(self.save_path, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            return self.results
        return {}
