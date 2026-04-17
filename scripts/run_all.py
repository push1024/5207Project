#!/usr/bin/env python3
"""
统一实验运行脚本
===================
运行所有配置组合的实验，或按指定方法运行。

使用方法:
    # 运行所有方法的全部实验
    python scripts/run_all.py

    # 仅运行特定方法
    python scripts/run_all.py --methods full lora

    # 仅运行特定样本规模
    python scripts/run_all.py --sample-sizes 200 1000

    # 组合运行
    python scripts/run_all.py --methods bitfit --sample-sizes 200 500
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.config import SAMPLE_SIZES, RESULTS_DIR


METHODS = ["full", "lora", "bitfit", "zeroshot"]
SCRIPTS = {
    "full": "scripts/train_full.py",
    "lora": "scripts/train_lora.py",
    "bitfit": "scripts/train_bitfit.py",
    "zeroshot": "scripts/train_zeroshot.py"
}


def run_experiment(
    method: str,
    sample_size: int,
    batch_size: int = 32,
    epochs: int = 5,
    log_dir: Optional[Path] = None
) -> dict:
    """
    运行单个实验并返回结果

    Args:
        method: 训练方法 (full/lora/bitfit/zeroshot)
        sample_size: 样本规模
        batch_size: 批次大小
        epochs: 训练轮数
        log_dir: 日志目录

    Returns:
        dict: 实验结果
    """
    script_path = SCRIPTS[method]
    cmd = [
        sys.executable,
        script_path,
        "--sample-size", str(sample_size),
        "--batch-size", str(batch_size),
        "--seed", "42"
    ]
    # Zero-shot 无训练，train_zeroshot.py 不接受 --epochs
    if method != "zeroshot":
        cmd.extend(["--epochs", str(epochs)])

    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{method}_{sample_size}_{timestamp}.log"
        cmd.extend(["--log-file", str(log_file)])

    results_file = RESULTS_DIR / f"{method}_{sample_size}_results.json"

    if results_file.exists():
        logging.info(f"结果已存在，跳过: {method} @ {sample_size}")
        with open(results_file, "r") as f:
            return json.load(f)

    logging.info(f"\n{'='*60}")
    logging.info(f"开始实验: {method} @ {sample_size}")
    logging.info(f"{'='*60}")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"实验完成，耗时: {elapsed:.2f} 秒")

        if results_file.exists():
            with open(results_file, "r") as f:
                return json.load(f)

    except subprocess.CalledProcessError as e:
        logging.error(f"实验失败: {method} @ {sample_size}")
        logging.error(f"错误码: {e.returncode}")
        return {
            "method": method,
            "sample_size": sample_size,
            "status": "failed",
            "error": str(e)
        }

    return {"method": method, "sample_size": sample_size, "status": "unknown"}


def main():
    parser = argparse.ArgumentParser(description="统一实验运行脚本")
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help=f"要运行的方法 (默认: {METHODS})"
    )
    parser.add_argument(
        "--sample-sizes", "-n",
        type=int,
        nargs="+",
        default=SAMPLE_SIZES,
        help=f"样本规模列表 (默认: {SAMPLE_SIZES})"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="批次大小 (默认: 32)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="训练轮数 (默认: 5)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在结果的实验"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="日志保存目录"
    )

    args = parser.parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parent.parent / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(log_dir / "run_all.log"), logging.INFO)

    logging.info("=" * 60)
    logging.info("实验批次开始")
    logging.info(f"方法: {args.methods}")
    logging.info(f"样本规模: {args.sample_sizes}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"训练轮数: {args.epochs}")
    logging.info(f"日志目录: {log_dir}")
    logging.info("=" * 60)

    results = []

    for method in args.methods:
        for sample_size in args.sample_sizes:
            if method == "zeroshot" and sample_size != 200:
                continue

            result = run_experiment(
                method=method,
                sample_size=sample_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                log_dir=log_dir
            )
            results.append(result)

    logging.info("\n" + "=" * 60)
    logging.info("实验批次完成!")
    logging.info(f"总实验数: {len(results)}")
    logging.info("=" * 60)

    summary_file = log_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果摘要已保存至: {summary_file}")


if __name__ == "__main__":
    main()
