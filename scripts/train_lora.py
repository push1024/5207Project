#!/usr/bin/env python3
"""
LoRA Fine-Tuning 训练脚本
==========================
使用低秩适配（Low-Rank Adaptation）方法，仅更新 attention 中 Q/V 矩阵的降维分解参数。

使用方法:
    python scripts/train_lora.py --sample-size 200 --epochs 5
    python scripts/train_lora.py --sample-size 500 --rank 8 --alpha 32
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
from peft import LoraConfig, get_peft_model, TaskType
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
    LOGS_DIR,
    LoRAConfig as ConfigLoRA
)
from src.data_loader import load_data, get_data_loaders


class LoRAFineTuner:
    """LoRA Fine-Tuning 训练器"""

    def __init__(
        self,
        sample_size: int,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None,
        device: torch.device = None
    ):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query", "value"]
        self.device = device or get_device()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.results = {
            "method": "lora",
            "sample_size": sample_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": self.target_modules
        }

    def setup(self):
        """初始化 LoRA 模型、优化器、数据加载器"""
        logging.info(f"{'='*60}")
        logging.info(f"LoRA Fine-Tuning 实验配置")
        logging.info(f"{'='*60}")
        logging.info(f"样本规模: {self.sample_size}")
        logging.info(f"批次大小: {self.batch_size}")
        logging.info(f"LoRA Rank (r): {self.lora_rank}")
        logging.info(f"LoRA Alpha: {self.lora_alpha}")
        logging.info(f"LoRA Dropout: {self.lora_dropout}")
        logging.info(f"目标模块: {self.target_modules}")
        logging.info(f"设备: {self.device}")
        logging.info(f"{'='*60}")

        logging.info("加载数据集...")
        train_data, val_data, test_data = load_data(sample_size=self.sample_size)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.train_loader, self.val_loader, self.test_loader, _ = get_data_loaders(
            train_data, val_data, test_data,
            tokenizer, batch_size=self.batch_size
        )

        logging.info("加载预训练模型并应用 LoRA...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label={i: name for i, name in enumerate(LABEL_NAMES)},
            label2id={name: i for i, name in enumerate(LABEL_NAMES)}
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            modules_to_save=["classifier"]
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
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
        """训练一个 epoch"""
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

        # 先冻结 LoRA 参数，只开放 classifier
        for name, param in self.model.named_parameters():
            if "lora_" not in name.lower() and "classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_1 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"阶段1 可训练参数: {trainable_1:,}")

        optimizer_p1 = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=self.weight_decay
        )
        total_steps_p1 = len(self.train_loader) * 2  # 2 epoch
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

        # ===== 阶段2: 训练 classifier + LoRA =====
        logging.info(f"\n{'='*60}")
        logging.info(f"阶段2: 训练 Classifier + LoRA")
        logging.info(f"{'='*60}")

        # 恢复所有 LoRA 参数为可训练
        for name, param in self.model.named_parameters():
            if "lora_" in name.lower() or "classifier" in name:
                param.requires_grad = True

        trainable_2 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"阶段2 可训练参数: {trainable_2:,}")

        # 差异化学习率: classifier 用较高 LR，LoRA 用较低 LR
        lora_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "lora_" in name.lower():
                    lora_params.append(param)
                else:
                    classifier_params.append(param)

        optimizer_p2 = AdamW([
            {"params": classifier_params, "lr": 5e-4},
            {"params": lora_params, "lr": self.learning_rate},
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

            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc="训练中", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                optimizer_p2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer_p2.step()
                scheduler_p2.step()
                total_loss += loss.item()

            train_loss = total_loss / len(self.train_loader)
            val_acc, _, _ = self.evaluate(self.val_loader)
            logging.info(f"训练 Loss: {train_loss:.4f}")
            logging.info(f"验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_path = RESULTS_DIR / f"lora_{self.sample_size}_best.pt"
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"✓ 保存最佳 LoRA 适配器至: {best_model_path}")

        elapsed = timer.stop()
        peak_memory = get_peak_memory_gb(self.device)

        self.results.update({
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "training_time_seconds": elapsed,
            "peak_vram_gb": peak_memory
        })

        logging.info(f"\n{'='*60}")
        logging.info(f"LoRA 训练完成!")
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
        results_file = RESULTS_DIR / f"lora_{self.sample_size}_results.json"
        tracker = ResultsTracker(results_file)
        tracker.update_dict(self.results)
        tracker.save()


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning 训练")
    parser.add_argument("--sample-size", "-n", type=int, default=200,
                        help="训练子集样本数 (默认: 200)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--epochs", "-e", type=int, default=5,
                        help="训练轮数 (默认: 5)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率 (默认: 2e-5)")
    parser.add_argument("--rank", "-r", type=int, default=8,
                        help="LoRA rank (默认: 8)")
    parser.add_argument("--alpha", "-a", type=int, default=32,
                        help="LoRA alpha (默认: 32)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="LoRA dropout (默认: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")

    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    set_seed(args.seed)
    device = get_device()

    try:
        trainer = LoRAFineTuner(
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            device=device
        )

        trainer.setup()
        trainer.train()
        trainer.test()
        trainer.save_results()

        logging.info("LoRA 实验成功完成！")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error("GPU 显存不足！尝试减小 batch_size")
            if args.batch_size > 16:
                logging.info("建议: 使用 --batch-size 16 重新运行")
        raise


if __name__ == "__main__":
    main()
