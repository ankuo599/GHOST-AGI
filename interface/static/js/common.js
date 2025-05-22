/**
 * GHOST AGI Web界面公共JavaScript文件
 * 提供所有页面共用的功能
 */

// 全局变量
const socket = io();
const statusBadge = document.getElementById('status-badge');
const memoryCountEl = document.getElementById('memory-count');
const uptimeEl = document.getElementById('uptime');

// 初始化Socket.IO连接
function initSocketConnection() {
    // 连接成功
    socket.on('connect', () => {
        console.log('已连接到服务器');
        updateStatusUI('运行中', 'success');
    });

    // 断开连接
    socket.on('disconnect', () => {
        console.log('已断开连接');
        updateStatusUI('已断开', 'danger');
    });

    // 接收系统事件
    socket.on('system_event', (event) => {
        // 处理特定事件
        if (event.type === 'system.started') {
            updateStatusUI('运行中', 'success');
        } else if (event.type === 'system.stopped') {
            updateStatusUI('已停止', 'danger');
        } else if (event.type === 'memory.new' || event.type === 'memory.deleted') {
            updateSystemStatus();
        }
    });
}

// 更新状态UI
function updateStatusUI(status, type) {
    if (statusBadge) {
        statusBadge.className = `badge bg-${type}`;
        statusBadge.textContent = status;
    }
}

// 更新系统状态
function updateSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // 更新状态徽章
            if (data.status === 'running') {
                updateStatusUI('运行中', 'success');
            } else {
                updateStatusUI('已停止', 'danger');
            }
            
            // 更新记忆数量
            if (memoryCountEl && data.memory) {
                const shortTerm = data.memory.short_term?.count || 0;
                const longTerm = data.memory.long_term?.count || 0;
                memoryCountEl.textContent = `${shortTerm + longTerm}`;
            }
            
            // 更新运行时间
            if (uptimeEl && data.uptime !== undefined) {
                uptimeEl.textContent = formatUptime(data.uptime);
            }
        })
        .catch(error => {
            console.error('获取状态出错:', error);
            updateStatusUI('状态未知', 'warning');
        });
}

// 格式化时间
function formatTime(date) {
    return date.toLocaleTimeString();
}

// 格式化日期时间
function formatDateTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
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

// 格式化JSON显示
function formatJson(json) {
    if (typeof json === 'string') {
        try {
            json = JSON.parse(json);
        } catch (e) {
            return json;
        }
    }
    return JSON.stringify(json, null, 2);
}

// 处理错误
function handleError(error, container) {
    console.error('错误:', error);
    
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <h5>发生错误</h5>
                <p>${error.message || error}</p>
            </div>
        `;
    }
    
    updateStatusUI('错误', 'danger');
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

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化Socket.IO连接
    initSocketConnection();
    
    // 初始化状态
    updateSystemStatus();
    
    // 定期更新状态
    setInterval(updateSystemStatus, 30000);
}); 