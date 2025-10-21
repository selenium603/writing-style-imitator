# 文本风格迁移与模仿器 (Writing Style Imitator)

基于GPT-2的文本风格迁移与模仿系统，可以学习并模仿不同作家的写作风格。

## 功能特点

- 🎨 支持多个作者风格的模型训练和切换
- 🚀 基于GPT-2的深度学习模型
- 💻 美观的Web界面，实时交互
- ⚙️ 可调节的生成参数（温度、Top-K、Top-P等）
- 📊 支持训练进度监控

## 已训练模型

项目包含以下作者风格的模型：
- Harry Potter (哈利·波特)
- Hemingway (海明威)
- Isaac Asimov (艾萨克·阿西莫夫)
- J.R.R. Tolkien (托尔金)
- The Tao-te Ching (道德经)
- Twilight (暮光之城)
- Анна Каренина (安娜·卡列尼娜)

## 系统要求

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- Flask 2.0+

## 安装说明

1. 克隆项目
```bash
git clone https://github.com/selenium603/writing-style-imitator.git
cd writing-style-imitator
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载GPT-2基础模型（如果没有）
```bash
# 将GPT-2模型文件放在 ./gpt2/ 目录下
```

## 使用方法

### 1. 启动Web服务

#### 方法一：使用启动脚本（Windows）
```bash
run_web.bat
```

#### 方法二：使用Python脚本
```bash
python run_web.py
```

#### 方法三：直接运行Flask应用
```bash
python app.py
```

启动后访问：http://localhost:5000

### 2. 训练新模型

准备训练数据（放在 `data/<作者名>/clean.txt`），然后运行：

```bash
python train.py <作者名> --epochs 3 --batch-size 2
```

参数说明：
- `--epochs`: 训练轮数（默认：3）
- `--batch-size`: 批次大小（默认：2）
- `--learning-rate`: 学习率（默认：5e-5）
- `--gradient-accumulation-steps`: 梯度累积步数（默认：4）
- `--eval-split`: 验证集比例（默认：0.1）

### 3. 生成文本（命令行）

```bash
python generate.py <作者名> "起始文本" --max-length 200 --temperature 0.8
```

参数说明：
- `--max-length`: 最大生成长度（默认：200）
- `--min-length`: 最小生成长度（默认：10）
- `--temperature`: 温度参数（默认：0.8）
- `--top-k`: Top-K采样（默认：50）
- `--top-p`: Top-P采样（默认：0.95）
- `--repetition-penalty`: 重复惩罚（默认：1.2）
- `--num-sequences`: 生成数量（默认：1）

## 项目结构

```
writing-style-imitator/
├── app.py                  # Flask Web应用
├── generate.py             # 文本生成脚本
├── train.py               # 模型训练脚本
├── run_web.py             # Web服务启动脚本
├── requirements.txt       # Python依赖
├── data/                  # 训练数据
│   └── <作者名>/
│       └── clean.txt
├── models/                # 训练好的模型
│   └── <作者名>/
│       └── final_model/
├── gpt2/                  # GPT-2基础模型
├── templates/             # HTML模板
│   └── index.html
└── static/                # 静态资源
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## 技术栈

- **后端**: Flask, PyTorch, Transformers
- **前端**: HTML5, CSS3, JavaScript (原生)
- **模型**: GPT-2 (fine-tuned)
- **训练**: Hugging Face Trainer API

## Web界面功能

1. **模型选择**: 在侧边栏选择不同作者的模型
2. **参数调节**: 实时调整生成参数
3. **文本生成**: 输入起始文本，获取续写结果
4. **多版本生成**: 支持同时生成多个版本

## 注意事项

- 模型文件较大（每个约500MB），训练完成的模型存储在 `models/` 目录
- 训练数据文件不包含在仓库中，需自行准备
- GPU训练速度远快于CPU，建议使用GPU训练
- 首次加载模型需要一定时间，请耐心等待

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

GitHub: [@selenium603](https://github.com/selenium603)

