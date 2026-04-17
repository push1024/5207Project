#!/usr/bin/env python3
"""
BitFit Fine-Tuning 训练脚本
============================
仅更新模型中的 bias 参数（<0.1% 可训练参数），冻结所有其他权重。

使用方法:
    python scripts/train_bitfit.py --sample-size 200 --epochs 5
    python scripts/train_bitfit.py --sample-size 500 --batch-size 16
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    get_device,
    reset_memory_stats,
    get_peak_memory_gb,
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
    RESULTS_DIR,
    LOGS_DIR
)
from src.data_loader import load_data, get_data_loaders


def freeze_except_bias(model: torch.nn.Module):
    """
    冻结模型所有参数，仅保留 name 包含 'bias' 的参数可训练

    这是 BitFit 的核心实现：遍历所有参数，按名称筛选 bias 项
    但保留 classifier（分类头）始终可训练，因为它不是 bias 参数
    """
    for name, param in model.named_parameters():
        if "bias" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logging.info(f"BitFit 参数配置:")
    logging.info(f"  可训练参数: {trainable_params:,} / {total_params:,} "
                 f"({100*trainable_params/total_params:.4f}%)")
    logging.info(f"  训练参数占比: ~0.1% (classifier 约 6000 参数除外)")


class BitFitFineTuner:
    """BitFit Fine-Tuning 训练器"""

    def __init__(
        self,
        sample_size: int,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        device: torch.device = None
    ):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or get_device()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.results = {
            "method": "bitfit",
            "sample_size": sample_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        }

    def setup(self):
        """初始化 BitFit 模型、优化器、数据加载器"""
        logging.info(f"{'='*60}")
        logging.info(f"BitFit Fine-Tuning 实验配置")
        logging.info(f"{'='*60}")
        logging.info(f"样本规模: {self.sample_size}")
        logging.info(f"批次大小: {self.batch_size}")
        logging.info(f"学习率: {self.learning_rate}")
        logging.info(f"权重衰减: {self.weight_decay}")
        logging.info(f"设备: {self.device}")
        logging.info(f"{'='*60}")

        logging.info("加载数据集...")
        train_data, val_data, test_data = load_data(sample_size=self.sample_size)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.train_loader, self.val_loader, self.test_loader, _ = get_data_loaders(
            train_data, val_data, test_data,
            tokenizer, batch_size=self.batch_size
        )

        logging.info("加载预训练模型（将应用 BitFit）...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label={i: name for i, name in enumerate(LABEL_NAMES)},
            label2id={name: i for i, name in enumerate(LABEL_NAMES)}
        )

        freeze_except_bias(self.model)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        reset_memory_stats(self.device)

    def train_epoch(self) -> float:
        """训练一个 epoch（默认使用 self.optimizer）"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="训练中", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def train_epoch_custom(self, optimizer, scheduler) -> float:
        """训练一个 epoch（使用外部传入的 optimizer/scheduler）"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="训练中", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, dataloader) -> tuple:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in tqdm(dataloader, desc="评估中", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        accuracy = compute_accuracy(all_preds, all_labels)
        return accuracy, all_preds, all_labels

    def train(self) -> dict:
        """执行完整训练流程：两阶段训练"""
        timer = Timer()
        timer.start()

        # ===== 阶段1: 只训练 classifier（高学习率，让它脱离随机初始化）=====
        logging.info(f"\n{'='*60}")
        logging.info(f"阶段1: 训练 Classifier（高学习率）")
        logging.info(f"{'='*60}")

        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_1 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"阶段1 可训练参数: {trainable_1:,}")

        optimizer_p1 = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=self.weight_decay
        )
        total_steps_p1 = len(self.train_loader) * 2
        scheduler_p1 = get_linear_schedule_with_warmup(
            optimizer_p1, num_warmup_steps=0, num_training_steps=total_steps_p1
        )

        for epoch in range(2):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc="阶段1", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                optimizer_p1.zero_grad()
                loss.backward()
                optimizer_p1.step()
                scheduler_p1.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            val_acc, _, _ = self.evaluate(self.val_loader)
            logging.info(f"阶段1 Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        # ===== 阶段2: 训练 classifier + bias =====
        logging.info(f"\n{'='*60}")
        logging.info(f"阶段2: 训练 Classifier + Bias（BitFit）")
        logging.info(f"{'='*60}")

        for name, param in self.model.named_parameters():
            if "bias" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_2 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"阶段2 可训练参数: {trainable_2:,}")

        # 差异化学习率: classifier 用较高，bias 用较低
        bias_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "classifier" in name:
                    classifier_params.append(param)
                else:
                    bias_params.append(param)

        optimizer_p2 = AdamW([
            {"params": classifier_params, "lr": 5e-4},
            {"params": bias_params, "lr": 5e-4},
        ], weight_decay=self.weight_decay)

        total_steps_p2 = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps_p2 * 0.1)
        scheduler_p2 = get_linear_schedule_with_warmup(
            optimizer_p2, num_warmup_steps=warmup_steps, num_training_steps=total_steps_p2
        )

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(self.epochs):
            logging.info(f"\n{'='*40}")
            logging.info(f"Epoch {epoch + 1}/{self.epochs} (阶段2)")
            logging.info(f"{'='*40}")

            train_loss = self.train_epoch_custom(optimizer_p2, scheduler_p2)
            val_acc, _, _ = self.evaluate(self.val_loader)

            logging.info(f"训练 Loss: {train_loss:.4f}")
            logging.info(f"验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_path = RESULTS_DIR / f"bitfit_{self.sample_size}_best.pt"
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"✓ 保存最佳 BitFit 模型至: {best_model_path}")

        elapsed = timer.stop()
        peak_memory = get_peak_memory_gb(self.device)

        self.results.update({
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "training_time_seconds": elapsed,
            "peak_vram_gb": peak_memory
        })

        logging.info(f"\n{'='*60}")
        logging.info(f"BitFit 训练完成!")
        logging.info(f"最佳验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        logging.info(f"对应 Epoch: {best_epoch}")
        logging.info(f"训练时间: {elapsed:.2f} 秒")
        logging.info(f"峰值 VRAM: {peak_memory:.2f} GB")
        logging.info(f"{'='*60}")

        return self.results

    @torch.no_grad()
    def test(self) -> float:
        """在测试集上评估"""
        test_acc, _, _ = self.evaluate(self.test_loader)
        logging.info(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
        self.results["test_accuracy"] = test_acc
        return test_acc

    def save_results(self):
        """保存实验结果"""
        results_file = RESULTS_DIR / f"bitfit_{self.sample_size}_results.json"
        tracker = ResultsTracker(results_file)
        tracker.update_dict(self.results)
        tracker.save()


def main():
    parser = argparse.ArgumentParser(description="BitFit Fine-Tuning 训练")
    parser.add_argument("--sample-size", "-n", type=int, default=200,
                        help="训练子集样本数 (默认: 200)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--epochs", "-e", type=int, default=5,
                        help="训练轮数 (默认: 5)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率 (默认: 2e-5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")

    args = parser.parse_args()

    setup_logging(log_file=args.log_file, level=logging.INFO)
    set_seed(args.seed)
    device = get_device()

    try:
        trainer = BitFitFineTuner(
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device
        )

        trainer.setup()
        trainer.train()
        trainer.test()
        trainer.save_results()

        logging.info("BitFit 实验成功完成！")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error("GPU 显存不足！尝试减小 batch_size")
            if args.batch_size > 16:
                logging.info("建议: 使用 --batch-size 16 重新运行")
        raise


if __name__ == "__main__":
    main()
