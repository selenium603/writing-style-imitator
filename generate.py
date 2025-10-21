import argparse
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(author_name):
    """
    加载指定作者的微调模型
    
    Args:
        author_name: 作者名称
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    model_path = Path(f"models/{author_name}/final_model")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型: {model_path}\n"
            f"请先使用 train.py 训练模型"
        )
    
    print(f"正在加载模型: {model_path}")
    
    # 加载tokenizer和model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # 设置为评估模式
    model.eval()
    
    # 如果有GPU，使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"模型已加载到: {device}")
    
    return model, tokenizer, device


def generate_text(
    model,
    tokenizer,
    device,
    prompt,
    max_length=200,
    min_length=10,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,  # 新增：重复惩罚
    num_return_sequences=1,
    no_repeat_ngram_size=3,  # 从2改为3
    num_beams=1,  # 新增：beam search
):
    """
    改进的文本生成函数
    
    Args:
        model: GPT-2模型
        tokenizer: 分词器
        device: 设备
        prompt: 起始句子
        max_length: 生成的最大长度
        min_length: 生成的最小长度
        temperature: 温度参数
        top_k: Top-K采样
        top_p: Top-P采样
        repetition_penalty: 重复惩罚系数（>1减少重复）
        num_return_sequences: 生成的序列数量
        no_repeat_ngram_size: 防止重复的n-gram大小
        num_beams: beam search的beam数量（1表示greedy）
    
    Returns:
        list: 生成的文本列表
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    print(f"\n起始句子: {prompt}")
    print(f"输入长度: {input_length} tokens")
    print(f"生成参数:")
    print(f"  max_length={max_length}, min_length={min_length}")
    print(f"  temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print(f"  repetition_penalty={repetition_penalty}")
    print(f"  num_beams={num_beams}, no_repeat_ngram_size={no_repeat_ngram_size}")
    print(f"\n正在生成...\n")
    
    # 生成文本
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            do_sample=True if num_beams == 1 else False,  # beam search时不采样
            early_stopping=True if num_beams > 1 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=None,  # 可以添加禁用词
        )
    
    # 解码生成的文本
    generated_texts = []
    for i, sequence in enumerate(output):
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="使用微调后的GPT-2模型生成文本")
    parser.add_argument(
        "author",
        type=str,
        help="作者名称"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="起始句子"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="最大长度（默认: 200）"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="最小长度（默认: 10）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="温度（默认: 0.8）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K（默认: 50）"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-P（默认: 0.95）"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="重复惩罚（默认: 1.2）"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1,
        help="生成数量（默认: 1）"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search数量（默认: 1，即greedy）"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载模型
        model, tokenizer, device = load_model(args.author)
        
        # 生成文本
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_sequences,
            num_beams=args.num_beams
        )
        
        # 输出结果
        print("=" * 80)
        print("生成的文本:")
        print("=" * 80)
        
        for i, text in enumerate(generated_texts, 1):
            if args.num_sequences > 1:
                print(f"\n[版本 {i}]")
            print(text)
            print()
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()