// 全局变量
let currentModel = null;
let isGenerating = false;

// DOM 元素
const elements = {
    modelList: document.getElementById('model-list'),
    refreshModels: document.getElementById('refresh-models'),
    chatMessages: document.getElementById('chat-messages'),
    userInput: document.getElementById('user-input'),
    sendButton: document.getElementById('send-button'),
    clearChat: document.getElementById('clear-chat'),
    charCount: document.getElementById('char-count'),
    modelStatus: document.getElementById('model-status'),
    trainModal: document.getElementById('train-modal'),
    trainModelBtn: document.getElementById('train-model'),
    startTraining: document.getElementById('start-training'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    resetParams: document.getElementById('reset-params'),
    // 参数控件
    maxLength: document.getElementById('max-length'),
    temperature: document.getElementById('temperature'),
    topK: document.getElementById('top-k'),
    topP: document.getElementById('top-p'),
    repetitionPenalty: document.getElementById('repetition-penalty'),
    numSequences: document.getElementById('num-sequences')
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    setupEventListeners();
    updateCharCount();
});

// 设置事件监听器
function setupEventListeners() {
    // 模型相关
    elements.refreshModels.addEventListener('click', loadModels);
    
    // 聊天相关
    elements.sendButton.addEventListener('click', sendMessage);
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    elements.userInput.addEventListener('input', updateCharCount);
    elements.clearChat.addEventListener('click', clearChat);
    
    // 参数滑块
    const sliders = ['temperature', 'topP', 'repetitionPenalty'];
    sliders.forEach(id => {
        const slider = elements[id.replace(/-([a-z])/g, (g) => g[1].toUpperCase())];
        slider.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
    });
    
    // 数字输入
    const numberInputs = ['maxLength', 'topK', 'numSequences'];
    numberInputs.forEach(id => {
        const input = elements[id.replace(/-([a-z])/g, (g) => g[1].toUpperCase())];
        input.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
        });
    });
    
    // 重置参数
    elements.resetParams.addEventListener('click', resetParameters);
    
    // 模态框
    const closeButtons = document.querySelectorAll('.close, .close-modal');
    closeButtons.forEach(btn => {
        btn.addEventListener('click', closeModal);
    });
    
    window.addEventListener('click', (e) => {
        if (e.target === elements.trainModal) {
            closeModal();
        }
    });
    
    elements.startTraining.addEventListener('click', startTraining);
}

// 加载模型列表
async function loadModels() {
    try {
        showLoading('加载模型列表...');
        const response = await fetch('/api/models');
        const data = await response.json();
        
        elements.modelList.innerHTML = '';
        
        if (data.models.length === 0) {
            elements.modelList.innerHTML = '<p style="color: rgba(255,255,255,0.5); text-align: center;">暂无可用模型</p>';
        } else {
            data.models.forEach(model => {
                const modelItem = createModelItem(model);
                elements.modelList.appendChild(modelItem);
            });
        }
        
        hideLoading();
    } catch (error) {
        console.error('加载模型失败:', error);
        showError('加载模型列表失败');
        hideLoading();
    }
}

// 创建模型项
function createModelItem(model) {
    const div = document.createElement('div');
    div.className = `model-item ${model.loaded ? 'loaded' : ''} ${model.name === currentModel ? 'selected' : ''}`;
    div.innerHTML = `
        <span class="model-name">${model.name}</span>
        <span class="model-status">
            <span class="status-dot"></span>
            ${model.loaded ? '已加载' : '未加载'}
        </span>
    `;
    
    div.addEventListener('click', () => selectModel(model.name));
    return div;
}

// 选择模型
async function selectModel(modelName) {
    if (currentModel === modelName) return;
    
    currentModel = modelName;
    updateModelStatus('loading', `正在加载模型 ${modelName}...`);
    
    // 更新UI
    document.querySelectorAll('.model-item').forEach(item => {
        item.classList.remove('selected');
        if (item.querySelector('.model-name').textContent === modelName) {
            item.classList.add('selected');
        }
    });
    
    try {
        showLoading(`加载模型 ${modelName} 中...`);
        
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ author: modelName })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' || data.status === 'loading') {
            // 轮询检查加载状态
            checkModelLoadingStatus(modelName);
        } else {
            throw new Error(data.message || '模型加载失败');
        }
        
    } catch (error) {
        console.error('加载模型失败:', error);
        showError(`加载模型失败: ${error.message}`);
        updateModelStatus('error', '模型加载失败');
        currentModel = null;
        hideLoading();
    }
}

// 检查模型加载状态
async function checkModelLoadingStatus(modelName) {
    const checkInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/model_status/${modelName}`);
            const status = await response.json();
            
            if (status.status === 'loaded') {
                clearInterval(checkInterval);
                updateModelStatus('connected', `模型 ${modelName} 已就绪`);
                elements.sendButton.disabled = false;
                hideLoading();
                
                // 更新模型列表中的状态
                loadModels();
                
                // 清除欢迎消息
                if (elements.chatMessages.querySelector('.welcome-message')) {
                    elements.chatMessages.innerHTML = '';
                }
                
                addMessage('assistant', `模型 ${modelName} 已加载成功！现在可以开始生成文本了。`);
            } else if (status.status === 'error') {
                clearInterval(checkInterval);
                throw new Error(status.message || '模型加载失败');
            } else if (status.progress) {
                elements.loadingText.textContent = `加载进度: ${status.progress}%`;
            }
        } catch (error) {
            clearInterval(checkInterval);
            console.error('检查模型状态失败:', error);
            updateModelStatus('error', '模型加载失败');
            hideLoading();
        }
    }, 1000);
}

// 发送消息
async function sendMessage() {
    const prompt = elements.userInput.value.trim();
    if (!prompt || !currentModel || isGenerating) return;
    
    isGenerating = true;
    elements.sendButton.disabled = true;
    
    // 添加用户消息
    addMessage('user', prompt);
    elements.userInput.value = '';
    updateCharCount();
    
    // 获取参数
    const params = {
        max_length: parseInt(elements.maxLength.value),
        temperature: parseFloat(elements.temperature.value),
        top_k: parseInt(elements.topK.value),
        top_p: parseFloat(elements.topP.value),
        repetition_penalty: parseFloat(elements.repetitionPenalty.value),
        num_sequences: parseInt(elements.numSequences.value)
    };
    
    try {
        updateModelStatus('loading', '正在生成...');
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                author: currentModel,
                prompt: prompt,
                params: params
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            if (data.texts.length === 1) {
                addMessage('assistant', data.texts[0]);
            } else {
                // 多个版本
                const versionsHtml = data.texts.map((text, index) => `
                    <div class="version-item">
                        <div class="version-header">版本 ${index + 1}</div>
                        <div>${text}</div>
                    </div>
                `).join('');
                
                addMessage('assistant', `<div class="generated-versions">${versionsHtml}</div>`, true);
            }
        } else {
            throw new Error(data.message || '生成失败');
        }
        
    } catch (error) {
        console.error('生成文本失败:', error);
        addMessage('assistant', `生成失败: ${error.message}`, false, 'error');
    } finally {
        isGenerating = false;
        elements.sendButton.disabled = false;
        updateModelStatus('connected', `模型 ${currentModel} 已就绪`);
    }
}

// 添加消息到聊天界面
function addMessage(role, content, isHtml = false, type = 'normal') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isHtml) {
        contentDiv.innerHTML = content;
    } else {
        contentDiv.textContent = content;
    }
    
    if (type === 'error') {
        contentDiv.style.backgroundColor = '#fee';
        contentDiv.style.color = '#c00';
    }
    
    messageDiv.appendChild(contentDiv);
    
    // 添加时间戳
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(metaDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// 清空聊天记录
function clearChat() {
    if (confirm('确定要清空所有对话记录吗？')) {
        elements.chatMessages.innerHTML = '';
        if (!currentModel) {
            // 恢复欢迎消息
            elements.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <h2>欢迎使用文本风格迁移与模仿器！</h2>
                    <p>请在左侧选择一个已训练的模型，然后输入起始文本，系统将以该作者的风格续写。</p>
                    <div class="tips">
                        <h3>使用提示：</h3>
                        <ul>
                            <li>选择不同的作者模型，体验不同的写作风格</li>
                            <li>调整生成参数来控制文本的创造性和多样性</li>
                            <li>Temperature 越高，生成的文本越有创意但可能不太连贯</li>
                            <li>重复惩罚可以减少生成文本中的重复内容</li>
                        </ul>
                    </div>
                </div>
            `;
        }
    }
}

// 更新字符计数
function updateCharCount() {
    elements.charCount.textContent = elements.userInput.value.length;
}

// 更新模型状态
function updateModelStatus(status, text) {
    const statusElement = elements.modelStatus;
    statusElement.className = `status-indicator ${status}`;
    statusElement.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
}

// 重置参数
function resetParameters() {
    elements.maxLength.value = 200;
    elements.temperature.value = 0.8;
    elements.topK.value = 50;
    elements.topP.value = 0.95;
    elements.repetitionPenalty.value = 1.2;
    elements.numSequences.value = 1;
    
    // 更新显示值
    document.querySelectorAll('.param-value').forEach(span => {
        const input = span.previousElementSibling;
        span.textContent = input.value;
    });
}

// 加载训练数据列表
async function loadTrainingData() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const dataList = document.getElementById('training-data-list');
        dataList.innerHTML = '';
        
        if (data.training_data.length === 0) {
            dataList.innerHTML = '<p style="color: #999;">暂无训练数据</p>';
        } else {
            data.training_data.forEach(name => {
                const span = document.createElement('span');
                span.className = 'data-item';
                span.textContent = name;
                span.style.cursor = 'pointer';
                span.addEventListener('click', () => {
                    document.getElementById('train-author').value = name;
                });
                dataList.appendChild(span);
            });
        }
    } catch (error) {
        console.error('加载训练数据失败:', error);
    }
}

// 开始训练
async function startTraining() {
    const author = document.getElementById('train-author').value.trim();
    const epochs = parseInt(document.getElementById('train-epochs').value);
    const batchSize = parseInt(document.getElementById('train-batch-size').value);
    
    if (!author) {
        alert('请输入作者名称');
        return;
    }
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                author: author,
                params: {
                    epochs: epochs,
                    batch_size: batchSize
                }
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'error') {
            alert(`错误: ${data.message}`);
        } else {
            alert(`请在终端运行以下命令来训练模型:\n\n${data.command}\n\n训练完成后刷新模型列表即可使用。`);
            closeModal();
        }
    } catch (error) {
        console.error('启动训练失败:', error);
        alert('启动训练失败');
    }
}

// 显示加载动画
function showLoading(text = '加载中...') {
    elements.loadingOverlay.style.display = 'flex';
    elements.loadingText.textContent = text;
}

// 隐藏加载动画
function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

// 显示错误消息
function showError(message) {
    alert(message);
}

// 关闭模态框
function closeModal() {
    elements.trainModal.style.display = 'none';
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
