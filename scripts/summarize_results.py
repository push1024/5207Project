#!/usr/bin/env python3
"""
实验结果汇总与对比分析脚本
============================
汇总所有实验结果，生成对比表格和统计报告。

使用方法:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --output results/comparison.csv
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.config import RESULTS_DIR, SAMPLE_SIZES, LABEL_NAMES


METHODS = ["full", "lora", "bitfit", "zeroshot"]
METHOD_NAMES = {
    "full": "Full Fine-Tuning",
    "lora": "LoRA",
    "bitfit": "BitFit",
    "zeroshot": "Zero-shot"
}


def load_results(method: str, sample_size: int) -> Optional[Dict]:
    """加载单个实验结果"""
    results_file = RESULTS_DIR / f"{method}_{sample_size}_results.json"

    if not results_file.exists():
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def collect_all_results() -> pd.DataFrame:
    """收集所有实验结果并转换为 DataFrame"""
    records = []

    for method in METHODS:
        for sample_size in SAMPLE_SIZES:
            if method == "zeroshot" and sample_size != 200:
                result = load_results("zeroshot", 200)
            else:
                result = load_results(method, sample_size)

            if result:
                record = {
                    "Method": METHOD_NAMES.get(method, method),
                    "Sample Size": sample_size,
                    "Method_Code": method,
                    "Status": result.get("status", "completed")
                }

                if "best_val_accuracy" in result:
                    record["Val Accuracy"] = result["best_val_accuracy"]
                elif "val_accuracy" in result:
                    record["Val Accuracy"] = result["val_accuracy"]
                else:
                    record["Val Accuracy"] = None

                if "test_accuracy" in result:
                    record["Test Accuracy"] = result["test_accuracy"]
                else:
                    record["Test Accuracy"] = None

                if "training_time_seconds" in result:
                    record["Training Time (s)"] = result["training_time_seconds"]
                elif "inference_time_seconds" in result:
                    record["Training Time (s)"] = result["inference_time_seconds"]
                else:
                    record["Training Time (s)"] = None

                if "peak_vram_gb" in result:
                    record["Peak VRAM (GB)"] = result["peak_vram_gb"]
                else:
                    record["Peak VRAM (GB)"] = 0.0

                if "learning_rate" in result:
                    record["Learning Rate"] = result["learning_rate"]

                records.append(record)

    df = pd.DataFrame(records)
    return df


def compute_comparisons(df: pd.DataFrame) -> Dict:
    """计算方法间的对比指标"""
    comparisons = {}

    for sample_size in SAMPLE_SIZES:
        size_df = df[df["Sample Size"] == sample_size]

        if size_df.empty:
            continue

        lora_row = size_df[size_df["Method_Code"] == "lora"]
        bitfit_row = size_df[size_df["Method_Code"] == "bitfit"]

        if not lora_row.empty and not bitfit_row.empty:
            lora_acc = lora_row["Val Accuracy"].values[0]
            bitfit_acc = bitfit_row["Val Accuracy"].values[0]

            lora_vram = lora_row["Peak VRAM (GB)"].values[0]
            bitfit_vram = bitfit_row["Peak VRAM (GB)"].values[0]

            lora_time = lora_row["Training Time (s)"].values[0]
            bitfit_time = bitfit_row["Training Time (s)"].values[0]

            comparisons[sample_size] = {
                "bitfit_to_lora_accuracy_ratio": bitfit_acc / lora_acc if lora_acc else None,
                "bitfit_to_lora_vram_ratio": bitfit_vram / lora_vram if lora_vram else None,
                "bitfit_to_lora_time_ratio": bitfit_time / lora_time if lora_time else None,
                "hypothesis_supported": (
                    (bitfit_acc / lora_acc >= 0.97) and
                    (bitfit_vram / lora_vram <= 0.6) and
                    (bitfit_time / lora_time <= 0.4)
                ) if all([lora_acc, bitfit_acc, lora_vram, bitfit_vram, lora_time, bitfit_time]) else None
            }

    return comparisons


def generate_summary_report(df: pd.DataFrame, comparisons: Dict) -> str:
    """生成文本格式的汇总报告"""
    report = []
    report.append("=" * 80)
    report.append("PEFT 方法对比实验汇总报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("1. 完整实验结果表")
    report.append("-" * 80)

    display_df = df.copy()
    display_df = display_df.sort_values(["Sample Size", "Method"])
    display_df["Val Accuracy"] = display_df["Val Accuracy"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
    )
    display_df["Test Accuracy"] = display_df["Test Accuracy"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
    )
    display_df["Training Time (s)"] = display_df["Training Time (s)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    display_df["Peak VRAM (GB)"] = display_df["Peak VRAM (GB)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    print(display_df[["Method", "Sample Size", "Val Accuracy", "Test Accuracy",
                       "Training Time (s)", "Peak VRAM (GB)"]].to_string(index=False))

    report.append("")
    report.append("2. BitFit vs LoRA 对比分析")
    report.append("-" * 80)

    for sample_size, comp in sorted(comparisons.items()):
        report.append(f"\n样本规模: {sample_size}")
        report.append(f"  准确率比率 (BitFit/LoRA): {comp['bitfit_to_lora_accuracy_ratio']*100:.2f}%")
        report.append(f"  VRAM 比率 (BitFit/LoRA): {comp['bitfit_to_lora_vram_ratio']*100:.2f}%")
        report.append(f"  训练时间比率 (BitFit/LoRA): {comp['bitfit_to_lora_time_ratio']*100:.2f}%")

        if comp["hypothesis_supported"]:
            report.append(f"  ✓ 假设验证: 支持 (满足准确率≥97% 且 VRAM≤60% 且时间≤40%)")
        else:
            report.append(f"  ✗ 假设验证: 不满足")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="实验结果汇总")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 CSV 文件路径")
    parser.add_argument("--report", "-r", action="store_true",
                        help="输出详细报告")

    args = parser.parse_args()

    setup_logging(level=logging.INFO)

    df = collect_all_results()

    if df.empty:
        logging.warning("未找到任何实验结果！请先运行实验。")
        logging.info(f"结果目录: {RESULTS_DIR}")
        return

    logging.info(f"成功加载 {len(df)} 条实验记录")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
            logging.info(f"结果已保存至: {output_path}")
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
            logging.info(f"结果已保存至: {output_path}")

    comparisons = compute_comparisons(df)

    if args.report or not args.output:
        report = generate_summary_report(df, comparisons)
        print("\n" + report)

    summary_file = RESULTS_DIR / "summary_report.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        report = generate_summary_report(df, comparisons)
        f.write(report)

    logging.info(f"\n汇总报告已保存至: {summary_file}")


if __name__ == "__main__":
    main()
