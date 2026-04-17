"""
数据加载模块
============
提供 TweetEval-emotion 数据集的加载和预处理功能。
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer
)


def load_data(
    sample_size: Optional[int] = None,
    data_dir: Optional[Path] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    加载数据集（支持本地缓存）

    Args:
        sample_size: 如果指定，则从本地子集加载；否则使用完整训练集
        data_dir: 数据根目录，默认使用配置中的 DATA_DIR

    Returns:
        Tuple[Dataset, Dataset, Dataset]: (train_dataset, val_dataset, test_dataset)
    """
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR

    if sample_size is not None:
        subset_path = data_dir / "subsets" / str(sample_size)
        if subset_path.exists():
            logging.info(f"从本地加载子集 (样本数: {sample_size})")
            from datasets import load_from_disk
            dataset = load_from_disk(str(subset_path))
            from src.config import DATA_DIR as config_data_dir
            from datasets import load_from_disk
            raw_dataset = load_from_disk(str(config_data_dir / "raw"))
            val_dataset = raw_dataset["validation"]
            test_dataset = raw_dataset["test"]
            return dataset, val_dataset, test_dataset
        else:
            logging.warning(f"本地子集不存在: {subset_path}，尝试从远程加载")
            dataset = load_dataset("tweet_eval", "emotion", split="train")
            if len(dataset) > sample_size:
                dataset = dataset.select(range(sample_size))

    dataset = load_dataset("tweet_eval", "emotion")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    logging.info(f"数据集加载完成 - 训练: {len(train_dataset)}, "
                 f"验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def load_tokenizer(model_name: str = "roberta-base") -> PreTrainedTokenizer:
    """加载预训练模型对应的分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info(f"分词器加载完成: {model_name}")
    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128
) -> Dataset:
    """
    对数据集进行 tokenize 处理

    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大序列长度

    Returns:
        Dataset: tokenize 后的数据集
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized


def get_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 128
) -> Tuple:
    """
    构建训练和评估用的 DataLoader

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        tokenizer: 分词器
        batch_size: 批大小
        max_length: 最大序列长度

    Returns:
        Tuple: (train_loader, val_loader, test_loader, data_collator)
    """
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, max_length)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer, max_length)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, max_length)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        tokenized_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        tokenized_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, data_collator
