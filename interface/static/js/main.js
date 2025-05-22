/**
 * GHOST AGI Web界面主JavaScript文件
 */

// 建立Socket.IO连接
const socket = io();

// DOM元素引用
const chatContainer = document.getElementById('chat-container');
const userInputForm = document.getElementById('user-input-form');
const userInputField = document.getElementById('user-input');
const eventsContainer = document.getElementById('events-container');
const clearChatBtn = document.getElementById('clear-chat');
const clearEventsBtn = document.getElementById('clear-events');
const refreshStatusBtn = document.getElementById('refresh-status');
const statusBadge = document.getElementById('status-badge');
const memoryCountEl = document.getElementById('memory-count');
const uptimeEl = document.getElementById('uptime');

// 消息历史
let messageHistory = [];

// 连接成功
socket.on('connect', () => {
    console.log('已连接到服务器');
    addEventToContainer({
        type: 'connection.established',
        data: { timestamp: new Date().getTime() / 1000 }
    });
    
    // 更新系统状态
    updateSystemStatus();
});

// 断开连接
socket.on('disconnect', () => {
    console.log('已断开连接');
    statusBadge.className = 'badge bg-danger';
    statusBadge.textContent = '已断开';
    
    addEventToContainer({
        type: 'connection.lost',
        data: { timestamp: new Date().getTime() / 1000 }
    });
});

// 接收系统事件
socket.on('system_event', (event) => {
    addEventToContainer(event);
    
    // 处理特定事件
    if (event.type === 'system.started') {
        statusBadge.className = 'badge bg-success';
        statusBadge.textContent = '运行中';
    } else if (event.type === 'system.stopped') {
        statusBadge.className = 'badge bg-danger';
        statusBadge.textContent = '已停止';
    } else if (event.type === 'memory.new' || event.type === 'memory.deleted') {
        updateSystemStatus();
    }
});

// 接收响应
socket.on('response', (data) => {
    if (data.status === 'success') {
        addSystemMessage(data.response);
    }
});

// 接收错误
socket.on('error', (data) => {
    addSystemMessage({
        status: 'error',
        message: data.message
    });
});

// 用户输入处理
userInputForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const message = userInputField.value.trim();
    
    if (!message) return;
    
    // 添加用户消息到聊天界面
    addUserMessage(message);
    
    // 发送到服务器
    socket.emit('user_input', { message });
    
    // 清空输入框
    userInputField.value = '';
});

// 添加用户消息到聊天界面
function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'user-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = formatTime(new Date());
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // 保存到历史
    messageHistory.push({
        type: 'user',
        content: message,
        time: new Date()
    });
}

// 添加系统消息到聊天界面
function addSystemMessage(messageData) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'system-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 处理不同类型的响应
    if (typeof messageData === 'string') {
        contentDiv.textContent = messageData;
    } else if (messageData.status === 'error') {
        contentDiv.innerHTML = `<span class="text-danger">错误: ${messageData.message}</span>`;
    } else if (messageData.response) {
        // 格式化响应对象
        if (typeof messageData.response === 'string') {
            contentDiv.textContent = messageData.response;
        } else {
            try {
                const formattedResponse = formatResponse(messageData.response);
                contentDiv.innerHTML = formattedResponse;
            } catch (e) {
                contentDiv.textContent = JSON.stringify(messageData.response);
            }
        }
    } else {
        contentDiv.textContent = JSON.stringify(messageData);
    }
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = formatTime(new Date());
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // 保存到历史
    messageHistory.push({
        type: 'system',
        content: messageData,
        time: new Date()
    });
}

// 格式化响应
function formatResponse(response) {
    // 如果是字符串，直接返回
    if (typeof response === 'string') {
        return response;
    }
    
    // 如果有特定属性，尝试智能格式化
    if (response.message) {
        return response.message;
    }
    
    // 如果是带状态的响应
    if (response.status === 'success') {
        let result = '';
        
        if (response.message) {
            result = response.message;
        } else if (response.content) {
            if (typeof response.content === 'string') {
                result = response.content;
            } else {
                result = `<pre>${JSON.stringify(response.content, null, 2)}</pre>`;
            }
        } else {
            result = '操作成功';
        }
        
        return result;
    } else if (response.status === 'error') {
        return `<span class="text-danger">错误: ${response.message}</span>`;
    }
    
    // 默认格式化为JSON
    return `<pre>${JSON.stringify(response, null, 2)}</pre>`;
}

// 添加事件到事件容器
function addEventToContainer(event) {
    const eventItem = document.createElement('div');
    eventItem.className = 'event-item';
    
    const eventTime = document.createElement('span');
    eventTime.className = 'event-time';
    
    const timestamp = event.data && event.data.timestamp 
        ? new Date(event.data.timestamp * 1000) 
        : new Date();
        
    eventTime.textContent = formatTime(timestamp);
    
    const eventType = document.createElement('span');
    eventType.className = 'event-type';
    eventType.textContent = event.type;
    
    const eventData = document.createElement('span');
    eventData.className = 'event-data';
    
    // 简化事件数据显示
    let dataText = '';
    if (event.data) {
        const { timestamp, ...otherData } = event.data;
        if (Object.keys(otherData).length > 0) {
            dataText = JSON.stringify(otherData);
        }
    }
    
    eventData.textContent = dataText;
    
    eventItem.appendChild(eventTime);
    eventItem.appendChild(eventType);
    
    if (dataText) {
        eventItem.appendChild(document.createTextNode(' - '));
        eventItem.appendChild(eventData);
    }
    
    eventsContainer.appendChild(eventItem);
    
    // 滚动到底部
    eventsContainer.scrollTop = eventsContainer.scrollHeight;
}

// 更新系统状态
function updateSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // 更新状态徽章
            if (data.status === 'running') {
                statusBadge.className = 'badge bg-success';
                statusBadge.textContent = '运行中';
            } else {
                statusBadge.className = 'badge bg-danger';
                statusBadge.textContent = '已停止';
            }
            
            // 更新记忆数量
            if (data.memory && data.memory.short_term_count !== undefined) {
                const shortTerm = data.memory.short_term_count || 0;
                const longTerm = data.memory.long_term_count || 0;
                memoryCountEl.textContent = `${shortTerm + longTerm}`;
            }
            
            // 更新运行时间
            if (data.uptime !== undefined) {
                uptimeEl.textContent = formatUptime(data.uptime);
            }
        })
        .catch(error => {
            console.error('获取状态出错:', error);
            statusBadge.className = 'badge bg-warning';
            statusBadge.textContent = '状态未知';
        });
}

// 清空聊天
clearChatBtn.addEventListener('click', () => {
    chatContainer.innerHTML = '';
    messageHistory = [];
    
    // 添加欢迎消息
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'system-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<p>欢迎使用GHOST AGI智能系统！我可以帮助您完成各种任务，请告诉我您需要什么帮助。</p>';
    
    welcomeDiv.appendChild(contentDiv);
    chatContainer.appendChild(welcomeDiv);
});

// 清空事件
clearEventsBtn.addEventListener('click', () => {
    eventsContainer.innerHTML = '';
});

// 刷新状态
refreshStatusBtn.addEventListener('click', () => {
    updateSystemStatus();
});

// 格式化时间
function formatTime(date) {
    return date.toLocaleTimeString();
}

// 格式化运行时间
function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    seconds %= 86400;
    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;
    const minutes = Math.floor(seconds / 60);
    seconds = Math.floor(seconds % 60);
    
    if (days > 0) {
        return `${days}天 ${hours}小时`;
    } else if (hours > 0) {
        return `${hours}小时 ${minutes}分`;
    } else if (minutes > 0) {
        return `${minutes}分 ${seconds}秒`;
    } else {
        return `${seconds}秒`;
    }
}

// 定期更新状态
setInterval(updateSystemStatus, 30000);

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化状态
    updateSystemStatus();
}); 