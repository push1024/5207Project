#!/usr/bin/env python3
"""
Zero-Shot 推理脚本
====================
直接使用预训练的 RoBERTa-base 模型进行情感分类，不进行任何梯度更新。
作为成本下界基线。

使用方法:
    python scripts/train_zeroshot.py --sample-size 200
    python scripts/train_zeroshot.py --sample-size 500 --batch-size 16
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    get_device,
    set_seed,
    setup_logging,
    Timer,
    compute_accuracy,
    ResultsTracker
)
from src.config import (
    MODEL_NAME,
    LABEL_NAMES,
    NUM_LABELS,
    RANDOM_SEED,
    RESULTS_DIR
)
from src.data_loader import load_data, get_data_loaders


class ZeroShotClassifier:
    """Zero-Shot 分类器（无训练）"""

    def __init__(
        self,
        sample_size: int = None,
        batch_size: int = 32,
        device: torch.device = None
    ):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.device = device or get_device()

        self.model = None
        self.val_loader = None
        self.test_loader = None

        self.results = {
            "method": "zeroshot",
            "sample_size": sample_size,
            "batch_size": batch_size
        }

    def setup(self):
        """初始化模型和数据加载器"""
        logging.info(f"{'='*60}")
        logging.info(f"Zero-Shot 推理配置")
        logging.info(f"{'='*60}")
        logging.info(f"样本规模: {self.sample_size or 'full'}")
        logging.info(f"批次大小: {self.batch_size}")
        logging.info(f"设备: {self.device}")
        logging.info(f"{'='*60}")

        logging.info("加载数据集...")
        train_data, val_data, test_data = load_data(sample_size=self.sample_size)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _, self.val_loader, self.test_loader, _ = get_data_loaders(
            train_data, val_data, test_data,
            tokenizer, batch_size=self.batch_size
        )

        logging.info("加载预训练模型（不进行任何训练）...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label={i: name for i, name in enumerate(LABEL_NAMES)},
            label2id={name: i for i, name in enumerate(LABEL_NAMES)}
        )
        self.model.to(self.device)
        self.model.eval()

        logging.info("模型已加载，准备推理...")

    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "val") -> dict:
        """评估数据集，返回详细指标"""
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"{split_name} 推理中", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        accuracy = compute_accuracy(all_preds, all_labels)

        logging.info(f"{split_name} 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels
        }

    def run(self) -> dict:
        """执行 Zero-Shot 推理"""
        timer = Timer()
        timer.start()

        val_metrics = self.evaluate(self.val_loader, "validation")
        test_metrics = self.evaluate(self.test_loader, "test")

        elapsed = timer.stop()

        self.results.update({
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "inference_time_seconds": elapsed
        })

        logging.info(f"\n{'='*60}")
        logging.info(f"Zero-Shot 推理完成!")
        logging.info(f"验证准确率: {val_metrics['accuracy']:.4f} "
                     f"({val_metrics['accuracy']*100:.2f}%)")
        logging.info(f"测试准确率: {test_metrics['accuracy']:.4f} "
                     f"({test_metrics['accuracy']*100:.2f}%)")
        logging.info(f"推理时间: {elapsed:.2f} 秒")
        logging.info(f"{'='*60}")

        return self.results

    def save_results(self):
        """保存实验结果"""
        results_file = RESULTS_DIR / f"zeroshot_{self.sample_size or 'full'}_results.json"
        tracker = ResultsTracker(results_file)
        tracker.update_dict(self.results)
        tracker.save()


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot 推理")
    parser.add_argument("--sample-size", "-n", type=int, default=None,
                        help="使用的训练样本数（仅用于参考，不影响推理）")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")

    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    set_seed(args.seed)
    device = get_device()

    classifier = ZeroShotClassifier(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=device
    )

    classifier.setup()
    classifier.run()
    classifier.save_results()

    logging.info("Zero-Shot 实验成功完成！")


if __name__ == "__main__":
    main()
