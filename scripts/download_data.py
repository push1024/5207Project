#!/usr/bin/env python3
"""
数据下载与子集抽取脚本
======================
从 HuggingFace 下载 TweetEval-emotion 数据集，并按指定规模抽取训练子集。
所有子集使用固定随机种子 seed=42 确保可复现性。
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset


# ======================== 配置常量 ========================
DATASET_NAME = "tweet_eval"
DATASET_SUBSET = "emotion"
LABEL_MAP = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness"
}
RANDOM_SEED = 42

# 数据集根目录（相对于脚本位置）
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


def set_seed(seed: int = 42):
    """设置全局随机种子，确保实验可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_and_save_dataset():
    """从 HuggingFace 下载数据集并保存到本地"""
    print("=" * 60)
    print("开始下载 TweetEval-emotion 数据集...")
    print("=" * 60)

    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)

    print(f"\n原始数据集划分:")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data)} 条样本")

    local_dir = DATA_DIR / "raw"
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(local_dir))
    print(f"\n原始数据已保存至: {local_dir}")

    metadata = {
        "dataset_name": DATASET_NAME,
        "subset": DATASET_SUBSET,
        "label_map": LABEL_MAP,
        "splits": {split: len(data) for split, data in dataset.items()},
        "random_seed": RANDOM_SEED
    }

    with open(local_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return dataset


def extract_subsets(dataset, sample_sizes: list):
    """
    从训练集中抽取不同规模的子集

    Args:
        dataset: HuggingFace DatasetDict
        sample_sizes: 要抽取的样本数量列表，如 [200, 500, 1000, 2000]
    """
    train_data = dataset["train"]
    total_available = len(train_data)

    print("\n" + "=" * 60)
    print("开始抽取训练子集...")
    print("=" * 60)

    subsets_dir = DATA_DIR / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    for size in sample_sizes:
        if size > total_available:
            print(f"  [警告] 请求的样本数 {size} 超过可用数据 {total_available}，跳过")
            continue

        print(f"\n抽取 {size} 条训练样本...")

        indices = np.random.RandomState(RANDOM_SEED).permutation(total_available)[:size]
        subset = train_data.select(indices)

        subset_dir = subsets_dir / str(size)
        subset_dir.mkdir(parents=True, exist_ok=True)

        subset.save_to_disk(str(subset_dir))
        print(f"  已保存至: {subset_dir}")
        print(f"  实际样本数: {len(subset)}")

        class_dist = {}
        for label_id in range(4):
            count = sum(1 for ex in subset["label"] if ex == label_id)
            class_dist[LABEL_MAP[label_id]] = count

        print(f"  类别分布: {class_dist}")

        meta_path = subset_dir / "subset_info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "sample_size": size,
                "actual_size": len(subset),
                "class_distribution": class_dist,
                "random_seed": RANDOM_SEED,
                "parent_split": "train"
            }, f, indent=2, ensure_ascii=False)


def verify_subsets(sample_sizes: list):
    """验证已抽取的子集是否正确"""
    print("\n" + "=" * 60)
    print("验证子集完整性...")
    print("=" * 60)

    subsets_dir = DATA_DIR / "subsets"
    all_valid = True

    for size in sample_sizes:
        subset_path = subsets_dir / str(size)
        if not subset_path.exists():
            print(f"  [错误] 子集 {size} 不存在: {subset_path}")
            all_valid = False
            continue

        from datasets import load_from_disk
        subset = load_from_disk(str(subset_path))
        if len(subset) != size:
            print(f"  [错误] 子集 {size} 样本数不匹配: 期望 {size}, 实际 {len(subset)}")
            all_valid = False
        else:
            print(f"  [OK] 子集 {size}: {len(subset)} 条样本")

    if all_valid:
        print("\n所有子集验证通过！")
    else:
        print("\n存在验证失败的子集，请重新生成。")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="TweetEval-emotion 数据集下载与子集抽取工具"
    )
    parser.add_argument(
        "--sample-sizes", "-n",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 2000],
        help="要抽取的训练子集规模（默认: 200 500 1000 2000）"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，直接使用本地数据抽取子集"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证现有子集，不重新生成"
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    if args.verify_only:
        verify_subsets(args.sample_sizes)
        return

    if not args.skip_download:
        download_and_save_dataset()
    else:
        print("[提示] 跳过下载步骤，直接使用本地数据")

    from datasets import load_from_disk
    raw_path = DATA_DIR / "raw"
    if raw_path.exists():
        dataset = load_from_disk(str(raw_path))
    else:
        print("[错误] 找不到本地数据，请先运行不带 --skip-download 的命令")
        return

    extract_subsets(dataset, args.sample_sizes)
    verify_subsets(args.sample_sizes)

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print(f"数据存放位置: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
