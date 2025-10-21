import argparse
import os
from pathlib import Path
import torch
import transformers
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


def load_text_data(author_name):
    """加载指定作者的文本数据"""
    data_path = Path(f"data/{author_name}/clean.txt")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {data_path}\n"
            f"请确保文件存在于 data/{author_name}/clean.txt"
        )
    
    print(f"正在加载数据: {data_path}")
    text = data_path.read_text(encoding='utf-8')
    return text


def prepare_dataset(text, tokenizer, block_size=512, stride=256):
    """
    改进版：使用token级别切分，支持滑动窗口
    
    Args:
        text: 原始文本
        tokenizer: GPT2分词器
        block_size: 每个训练样本的token数量
        stride: 滑动窗口步长（可以有重叠，提高数据利用率）
    
    Returns:
        Dataset: 处理好的数据集
    """
    print("正在tokenize文本...")
    
    # 预处理文本：移除多余空格和换行
    text = " ".join(text.split())
    
    # 先将整个文本tokenize（这样不会破坏词的完整性）
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"总token数: {len(tokenized_text):,}")
    
    # 使用滑动窗口切分token序列
    input_ids_list = []
    attention_mask_list = []
    
    # 从头到尾滑动窗口
    for i in range(0, len(tokenized_text) - block_size + 1, stride):
        # 提取一个block
        chunk_ids = tokenized_text[i:i + block_size]
        
        # 如果是最后一个不完整的块，进行padding
        if len(chunk_ids) < block_size:
            padding_length = block_size - len(chunk_ids)
            chunk_ids = chunk_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(chunk_ids) + [0] * padding_length
        else:
            attention_mask = [1] * block_size
        
        input_ids_list.append(chunk_ids)
        attention_mask_list.append(attention_mask)
    
    print(f"共创建了 {len(input_ids_list)} 个训练样本")
    print(f"滑动窗口参数: block_size={block_size}, stride={stride}")
    
    # 创建Dataset
    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list
    })
    
    return dataset


def train_model(author_name, epochs=3, batch_size=2, learning_rate=5e-5, 
                gradient_accumulation_steps=4, eval_split=0.1):
    """
    训练GPT-2模型（改进版）
    
    Args:
        author_name: 作者名称
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        gradient_accumulation_steps: 梯度累积步数（模拟更大的batch_size）
        eval_split: 验证集比例
    """
    print(f"\n{'='*50}")
    print(f"开始训练模型: {author_name}")
    print(f"{'='*50}\n")
    
    # 1. 从本地加载GPT-2模型和tokenizer
    print("正在从本地加载GPT-2模型...")
    local_model_path = Path("./gpt2")
    
    if not local_model_path.exists():
        raise FileNotFoundError(
            f"找不到本地GPT-2模型: {local_model_path}\n"
            f"请确保模型文件在 ./gpt2/ 目录下"
        )
    
    print(f"从本地路径加载: {local_model_path}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(str(local_model_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(str(local_model_path))
    config = GPT2Config.from_pretrained(str(local_model_path))
    
    print(f"模型参数量: {model.num_parameters():,}")
    
    # 2. 加载和准备数据
    text_data = load_text_data(author_name)
    print(f"文本长度: {len(text_data):,} 字符")
    
    # 使用改进的数据准备函数
    dataset = prepare_dataset(
        text_data, 
        tokenizer, 
        block_size=512,  # 可以根据显存调整
        stride=256  # 使用50%重叠，增加数据多样性
    )
    
    # 划分训练集和验证集
    if eval_split > 0:
        split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"训练集样本数: {len(train_dataset)}")
    
    # 3. 设置输出目录
    output_dir = Path(f"models/{author_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 配置训练参数（改进版）
    fp16_enabled = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # 评估可以用更大的batch
        
        # 梯度累积 - 模拟更大的batch size
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # 评估策略
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        
        # 保存策略
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # 保留最近3个checkpoint
        
        # 日志
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        logging_first_step=True,
        
        # 优化器参数
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # 梯度裁剪
        
        # 学习率调度
        lr_scheduler_type="cosine",  # 使用余弦退火
        
        # 性能优化
        fp16=fp16_enabled,
        dataloader_num_workers=0,  # CPU上设为0，GPU可以设为2-4
        dataloader_pin_memory=True if fp16_enabled else False,
        
        # 其他
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )
    
    # 5. 改进的数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2使用CLM而非MLM
        pad_to_multiple_of=8 if fp16_enabled else None,  # FP16时对齐到8的倍数
    )
    
    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # 传入tokenizer以便保存
    )
    
    # 7. 开始训练
    print("\n开始训练...")
    print(f"有效batch size: {batch_size * gradient_accumulation_steps}")
    print(f"总训练步数: {len(train_dataset) // (batch_size * gradient_accumulation_steps) * epochs}")
    print()
    
    # 检查是否有checkpoint可以恢复
    checkpoint_dir = None
    if (output_dir / "checkpoint-*").exists():
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoint_dir = str(sorted(checkpoints)[-1])
            print(f"发现checkpoint，从 {checkpoint_dir} 恢复训练")
    
    trainer.train(resume_from_checkpoint=checkpoint_dir)
    
    # 8. 保存最终模型
    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)
    
    print(f"\n保存最终模型到: {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(final_model_path)
    
    # 保存训练统计
    if eval_dataset:
        print("\n最终评估结果:")
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\n{'='*50}")
    print(f"训练完成！模型已保存到: {final_model_path}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="微调GPT-2模型")
    parser.add_argument(
        "author",
        type=str,
        help="作者名称（对应data/<作者名>/clean.txt文件）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数（默认: 3）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="批次大小（默认: 2）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="学习率（默认: 5e-5）"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="梯度累积步数（默认: 4）"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="验证集比例（默认: 0.1，设为0则不划分）"
    )
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print(f"已创建数据目录: {data_dir}")
        print(f"请将训练数据放在 data/{args.author}/clean.txt")
        return
    
    try:
        train_model(
            author_name=args.author,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_split=args.eval_split
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()