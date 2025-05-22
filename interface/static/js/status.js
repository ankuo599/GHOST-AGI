/**
 * GHOST AGI 状态页面JavaScript文件
 */

// DOM元素引用
const refreshStatusBtn = document.getElementById('refresh-status-btn');
const mainStatusBadge = document.getElementById('main-status-badge');
const mainUptime = document.getElementById('main-uptime');
const agentsCount = document.getElementById('agents-count');
const memoryTotalCount = document.getElementById('memory-total-count');
const toolsCount = document.getElementById('tools-count');
const shortTermProgress = document.getElementById('short-term-progress');
const shortTermInfo = document.getElementById('short-term-info');
const longTermProgress = document.getElementById('long-term-progress');
const longTermInfo = document.getElementById('long-term-info');
const recentRetrievals = document.getElementById('recent-retrievals');
const agentsTableBody = document.getElementById('agents-table-body');
const pendingTasks = document.getElementById('pending-tasks');
const platformInfo = document.getElementById('platform-info');
const processorInfo = document.getElementById('processor-info');
const pythonInfo = document.getElementById('python-info');
const startTimeInfo = document.getElementById('start-time-info');
const subscribersCount = document.getElementById('subscribers-count');
const recentEventsCount = document.getElementById('recent-events-count');
const toolExecutions = document.getElementById('tool-executions');
const lastUpdateTime = document.getElementById('last-update-time');

// 图表
let memoryChart;
let eventsChart;

// 初始化图表
function initCharts() {
    // 记忆图表
    const memoryCtx = document.getElementById('memory-chart').getContext('2d');
    memoryChart = new Chart(memoryCtx, {
        type: 'bar',
        data: {
            labels: ['短期记忆', '长期记忆'],
            datasets: [{
                label: '记忆数量',
                data: [0, 0],
                backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(75, 192, 192, 0.5)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(75, 192, 192, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '记忆系统'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // 事件图表
    const eventsCtx = document.getElementById('events-chart').getContext('2d');
    eventsChart = new Chart(eventsCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 10}, (_, i) => i + 1),
            datasets: [{
                label: '事件数量',
                data: Array(10).fill(0),
                fill: false,
                borderColor: 'rgba(153, 102, 255, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '系统事件'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// 更新图表数据
function updateCharts(data) {
    if (memoryChart && data.memory) {
        memoryChart.data.datasets[0].data = [
            data.memory.short_term?.count || 0,
            data.memory.long_term?.count || 0
        ];
        memoryChart.update();
    }

    if (eventsChart && data.events) {
        // 更新事件图表
        if (data.events.recent_events > 0) {
            // 移除最旧的数据点并添加新的数据点
            eventsChart.data.datasets[0].data.shift();
            eventsChart.data.datasets[0].data.push(data.events.recent_events);
            eventsChart.update();
        }
    }
}

// 更新详细状态
function updateDetailedStatus(data) {
    // 更新主状态
    if (mainStatusBadge) {
        if (data.status === 'running') {
            mainStatusBadge.innerHTML = '<span class="badge bg-success">运行中</span>';
        } else {
            mainStatusBadge.innerHTML = '<span class="badge bg-danger">已停止</span>';
        }
    }

    // 更新主运行时间
    if (mainUptime && data.uptime !== undefined) {
        mainUptime.textContent = `运行时间: ${formatUptime(data.uptime)}`;
    }

    // 更新统计数字
    if (agentsCount && data.agents) {
        agentsCount.textContent = data.agents.registered_agents || 0;
    }

    if (memoryTotalCount && data.memory) {
        const shortTerm = data.memory.short_term?.count || 0;
        const longTerm = data.memory.long_term?.count || 0;
        memoryTotalCount.textContent = shortTerm + longTerm;
    }

    if (toolsCount && data.tools) {
        toolsCount.textContent = data.tools.tool_count || 0;
    }

    // 更新记忆进度条
    if (data.memory) {
        const shortTermCount = data.memory.short_term?.count || 0;
        const longTermCount = data.memory.long_term?.count || 0;
        const total = shortTermCount + longTermCount;

        if (shortTermProgress && shortTermInfo) {
            const shortTermPercent = total > 0 ? (shortTermCount / total * 100) : 0;
            shortTermProgress.style.width = `${shortTermPercent}%`;
            shortTermInfo.textContent = `${shortTermCount} 条短期记忆`;
        }

        if (longTermProgress && longTermInfo) {
            const longTermPercent = total > 0 ? (longTermCount / total * 100) : 0;
            longTermProgress.style.width = `${longTermPercent}%`;
            longTermInfo.textContent = `${longTermCount} 条长期记忆`;
        }
    }

    // 更新智能体表格
    if (agentsTableBody && data.agents) {
        if (data.agents.registered_agents > 0) {
            let agentsHtml = '';
            
            // 模拟一些智能体数据
            const agentTypes = ['core', 'meta'];
            const statuses = ['活跃', '空闲'];
            
            agentTypes.forEach((type, index) => {
                const lastActivity = new Date(data.timestamp * 1000 - index * 60000);
                
                agentsHtml += `
                    <tr>
                        <td>${type}</td>
                        <td><span class="badge ${index === 0 ? 'bg-success' : 'bg-secondary'}">${statuses[index]}</span></td>
                        <td>${index === 0 ? '1' : '0'}</td>
                        <td>${formatTime(lastActivity)}</td>
                    </tr>
                `;
            });
            
            agentsTableBody.innerHTML = agentsHtml;
        } else {
            agentsTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">无注册智能体</td></tr>';
        }
    }

    // 更新待处理任务
    if (pendingTasks && data.agents) {
        if (data.agents.pending_tasks > 0) {
            pendingTasks.innerHTML = `<p class="text-warning">${data.agents.pending_tasks} 个任务等待处理</p>`;
        } else {
            pendingTasks.innerHTML = '<p class="text-muted">无待处理任务</p>';
        }
    }

    // 更新系统信息
    if (platformInfo) {
        platformInfo.textContent = navigator.platform;
    }

    if (processorInfo) {
        processorInfo.textContent = navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} 核` : '未知';
    }

    if (pythonInfo) {
        pythonInfo.textContent = '3.9.x'; // 模拟数据
    }

    if (startTimeInfo && data.timestamp && data.uptime) {
        const startTime = new Date((data.timestamp - data.uptime) * 1000);
        startTimeInfo.textContent = formatDateTime(data.timestamp - data.uptime);
    }

    // 更新事件和工具信息
    if (subscribersCount && data.events) {
        subscribersCount.textContent = data.events.subscribers || 0;
    }

    if (recentEventsCount && data.events) {
        recentEventsCount.textContent = data.events.recent_events || 0;
    }

    if (toolExecutions && data.tools) {
        toolExecutions.textContent = data.tools.recent_executions || 0;
    }

    // 更新最后更新时间
    if (lastUpdateTime && data.timestamp) {
        lastUpdateTime.textContent = formatDateTime(data.timestamp);
    }
}

// 获取并更新状态
function fetchAndUpdateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // 更新详细状态
            updateDetailedStatus(data);
            
            // 更新图表
            updateCharts(data);
        })
        .catch(error => {
            console.error('获取状态出错:', error);
            handleError(error);
        });
}

// 刷新按钮点击事件
if (refreshStatusBtn) {
    refreshStatusBtn.addEventListener('click', fetchAndUpdateStatus);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化图表
    initCharts();
    
    // 获取并更新状态
    fetchAndUpdateStatus();
    
    // 定期更新状态
    setInterval(fetchAndUpdateStatus, 10000);
    
    // 订阅系统事件
    socket.on('system_event', (event) => {
        // 更新状态
        setTimeout(fetchAndUpdateStatus, 500);
    });
}); 