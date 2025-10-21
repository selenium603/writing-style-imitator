import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading
import queue
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 全局变量存储已加载的模型
loaded_models = {}
model_loading_status = {}
generation_queue = queue.Queue()

def load_model(author_name):
    """加载指定作者的微调模型"""
    try:
        model_loading_status[author_name] = {"status": "loading", "progress": 0}
        
        model_path = Path(f"models/{author_name}/final_model")
        
        if not model_path.exists():
            model_loading_status[author_name] = {
                "status": "error", 
                "message": f"模型不存在，请先训练模型"
            }
            return None, None, None
        
        model_loading_status[author_name]["progress"] = 30
        
        # 加载tokenizer和model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model_loading_status[author_name]["progress"] = 60
        
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()
        
        model_loading_status[author_name]["progress"] = 80
        
        # 如果有GPU，使用GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        model_loading_status[author_name]["progress"] = 100
        model_loading_status[author_name]["status"] = "loaded"
        
        return model, tokenizer, device
    
    except Exception as e:
        model_loading_status[author_name] = {
            "status": "error",
            "message": str(e)
        }
        return None, None, None

def generate_text_async(model, tokenizer, device, prompt, params):
    """异步生成文本"""
    try:
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # 生成文本
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=params.get("max_length", 200),
                min_length=params.get("min_length", 10),
                temperature=params.get("temperature", 0.8),
                top_k=params.get("top_k", 50),
                top_p=params.get("top_p", 0.95),
                repetition_penalty=params.get("repetition_penalty", 1.2),
                num_return_sequences=params.get("num_sequences", 1),
                no_repeat_ngram_size=params.get("no_repeat_ngram_size", 3),
                num_beams=params.get("num_beams", 1),
                do_sample=True if params.get("num_beams", 1) == 1 else False,
                early_stopping=True if params.get("num_beams", 1) > 1 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_texts = []
        for sequence in output:
            text = tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text)
        
        return {"status": "success", "texts": generated_texts}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    models_dir = Path("models")
    available_models = []
    
    if models_dir.exists():
        for author_dir in models_dir.iterdir():
            if author_dir.is_dir() and (author_dir / "final_model").exists():
                model_info = {
                    "name": author_dir.name,
                    "loaded": author_dir.name in loaded_models,
                    "status": model_loading_status.get(author_dir.name, {"status": "not_loaded"})
                }
                available_models.append(model_info)
    
    # 检查训练数据
    data_dir = Path("data")
    training_data = []
    if data_dir.exists():
        for data_folder in data_dir.iterdir():
            if data_folder.is_dir() and (data_folder / "clean.txt").exists():
                training_data.append(data_folder.name)
    
    return jsonify({
        "models": available_models,
        "training_data": training_data
    })

@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """加载模型的API端点"""
    data = request.json
    author_name = data.get('author')
    
    if not author_name:
        return jsonify({"status": "error", "message": "请提供作者名称"}), 400
    
    if author_name in loaded_models:
        return jsonify({"status": "success", "message": "模型已加载"})
    
    # 在后台线程中加载模型
    def load_in_background():
        model, tokenizer, device = load_model(author_name)
        if model is not None:
            loaded_models[author_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device
            }
    
    thread = threading.Thread(target=load_in_background)
    thread.start()
    
    return jsonify({"status": "loading", "message": "模型正在加载中..."})

@app.route('/api/generate', methods=['POST'])
def generate():
    """生成文本的API端点"""
    data = request.json
    author = data.get('author')
    prompt = data.get('prompt')
    params = data.get('params', {})
    
    if not author or not prompt:
        return jsonify({"status": "error", "message": "请提供作者名称和起始文本"}), 400
    
    if author not in loaded_models:
        return jsonify({"status": "error", "message": "请先加载模型"}), 400
    
    model_info = loaded_models[author]
    result = generate_text_async(
        model_info["model"],
        model_info["tokenizer"],
        model_info["device"],
        prompt,
        params
    )
    
    return jsonify(result)

@app.route('/api/model_status/<author>', methods=['GET'])
def get_model_status(author):
    """获取模型加载状态"""
    status = model_loading_status.get(author, {"status": "not_loaded"})
    return jsonify(status)

@app.route('/api/train', methods=['POST'])
def train_model():
    """启动模型训练（简化版本，实际应该在后台进程中运行）"""
    data = request.json
    author = data.get('author')
    params = data.get('params', {})
    
    if not author:
        return jsonify({"status": "error", "message": "请提供作者名称"}), 400
    
    # 检查训练数据是否存在
    data_path = Path(f"data/{author}/clean.txt")
    if not data_path.exists():
        return jsonify({
            "status": "error", 
            "message": f"训练数据不存在: {data_path}"
        }), 400
    
    # 这里应该启动一个后台进程来训练模型
    # 为了简化，返回一个提示信息
    return jsonify({
        "status": "info",
        "message": "请在终端中运行训练命令",
        "command": f"python train.py {author} --epochs {params.get('epochs', 3)} --batch-size {params.get('batch_size', 2)}"
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # 确保必要的目录存在
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(parents=True, exist_ok=True)
    Path("static/js").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("文本风格迁移与模仿器 Web 服务")
    print("="*50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\n访问 http://localhost:5000 开始使用")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
