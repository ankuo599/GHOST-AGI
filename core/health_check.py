"""
健康检查系统
实现系统状态监控、异常检测和自动恢复
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import asyncio
import psutil
import time
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from sklearn.linear_model import LinearRegression
from collections import deque
import threading

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    timestamp: str
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    response_time: Optional[float] = None
    error_rate: Optional[float] = None
    queue_length: Optional[int] = None
    active_connections: Optional[int] = None

@dataclass
class HealthStatus:
    """健康状态"""
    is_healthy: bool
    metrics: SystemMetrics
    issues: List[Dict[str, Any]]
    recovery_actions: List[Dict[str, Any]]
    timestamp: str
    predictions: Optional[Dict[str, Any]] = None
    optimization_suggestions: Optional[List[Dict[str, Any]]] = None

class MetricsPredictor:
    """指标预测器"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.models = {}
        self.feature_names = [
            "cpu_usage", "memory_usage", "disk_usage",
            "process_count", "response_time", "error_rate"
        ]
        
    def update(self, metrics: SystemMetrics):
        """更新历史数据"""
        self.metrics_history.append(metrics)
        
    def predict(self) -> Dict[str, Any]:
        """预测未来指标"""
        if len(self.metrics_history) < self.window_size:
            return {}
            
        predictions = {}
        for feature in self.feature_names:
            if not hasattr(self.metrics_history[0], feature):
                continue
                
            # 准备训练数据
            X = np.array(range(len(self.metrics_history))).reshape(-1, 1)
            y = np.array([getattr(m, feature) for m in self.metrics_history])
            
            # 训练模型
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测下一个值
            next_value = model.predict([[len(self.metrics_history)]])[0]
            predictions[feature] = {
                "next_value": float(next_value),
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
                "confidence": model.score(X, y)
            }
            
        return predictions

class SystemOptimizer:
    """系统优化器"""
    def __init__(self):
        self.optimization_rules = []
        self.performance_history = []
        
    def add_rule(self, condition: Callable[[SystemMetrics], bool],
                 action: Callable[[SystemMetrics], Dict[str, Any]]):
        """添加优化规则"""
        self.optimization_rules.append((condition, action))
        
    def analyze(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """分析系统并提供优化建议"""
        suggestions = []
        
        # 记录性能历史
        self.performance_history.append(metrics)
        
        # 应用优化规则
        for condition, action in self.optimization_rules:
            if condition(metrics):
                suggestion = action(metrics)
                suggestions.append(suggestion)
                
        return suggestions

class HealthChecker:
    """健康检查器"""
    def __init__(self):
        self.logger = logging.getLogger("HealthChecker")
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "process_count": 1000,
            "gpu_usage": 90.0,
            "gpu_memory": 90.0,
            "response_time": 1000.0,  # 毫秒
            "error_rate": 0.01,  # 1%
            "queue_length": 1000,
            "active_connections": 10000
        }
        
        # 添加动态阈值调整
        self.threshold_history = {k: [] for k in self.thresholds}
        self.threshold_adjustment_factor = 0.1
        
    def adjust_thresholds(self, metrics: SystemMetrics):
        """动态调整阈值"""
        for key, value in self.thresholds.items():
            if hasattr(metrics, key):
                current_value = getattr(metrics, key)
                self.threshold_history[key].append(current_value)
                
                # 计算移动平均
                if len(self.threshold_history[key]) > 10:
                    avg = sum(self.threshold_history[key][-10:]) / 10
                    # 根据历史数据调整阈值
                    self.thresholds[key] = (1 - self.threshold_adjustment_factor) * self.thresholds[key] + \
                                         self.threshold_adjustment_factor * avg

    def check_metrics(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """检查所有指标"""
        issues = []
        
        # 调整阈值
        self.adjust_thresholds(metrics)
        
        # 检查各项指标
        for key, threshold in self.thresholds.items():
            if hasattr(metrics, key):
                value = getattr(metrics, key)
                if value is not None and value > threshold:
                    issues.append({
                        "type": f"{key}_high",
                        "severity": "warning",
                        "message": f"{key}过高: {value}%",
                        "threshold": threshold,
                        "current_value": value
                    })
                    
        return issues

class RecoveryHandler:
    """恢复处理器"""
    def __init__(self):
        self.logger = logging.getLogger("RecoveryHandler")
        self.recovery_actions: Dict[str, Callable] = {}
        
    def register_action(self, issue_type: str, action: Callable):
        """注册恢复动作"""
        self.recovery_actions[issue_type] = action
        
    async def handle_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """处理问题"""
        try:
            if issue["type"] in self.recovery_actions:
                action = self.recovery_actions[issue["type"]]
                result = await action(issue)
                return {
                    "issue": issue,
                    "action": action.__name__,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "issue": issue,
                    "action": "none",
                    "result": "No recovery action registered",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"问题处理失败: {str(e)}")
            return {
                "issue": issue,
                "action": "error",
                "result": str(e),
                "timestamp": datetime.now().isoformat()
            }

class HealthMonitor:
    """健康监控器"""
    def __init__(self, check_interval: int = 60):
        self.logger = logging.getLogger("HealthMonitor")
        self.check_interval = check_interval
        self.checker = HealthChecker()
        self.recovery = RecoveryHandler()
        self.predictor = MetricsPredictor()
        self.optimizer = SystemOptimizer()
        self.metrics_history: List[SystemMetrics] = []
        self.health_history: List[HealthStatus] = []
        self.is_running = False
        self._setup_recovery_actions()
        self._setup_optimization_rules()
        
    def _setup_recovery_actions(self):
        """设置恢复动作"""
        # 注册CPU相关恢复动作
        self.recovery.register_action(
            "cpu_high_usage",
            self._handle_cpu_high_usage
        )
        
        # 注册内存相关恢复动作
        self.recovery.register_action(
            "memory_high_usage",
            self._handle_memory_high_usage
        )
        
        # 注册磁盘相关恢复动作
        self.recovery.register_action(
            "disk_high_usage",
            self._handle_disk_high_usage
        )
        
        # 注册进程相关恢复动作
        self.recovery.register_action(
            "too_many_processes",
            self._handle_too_many_processes
        )
        
    def _setup_optimization_rules(self):
        """设置优化规则"""
        # CPU优化规则
        self.optimizer.add_rule(
            lambda m: m.cpu_usage > 70,
            lambda m: {
                "type": "cpu_optimization",
                "suggestion": "考虑增加CPU核心或优化计算密集型任务",
                "priority": "high"
            }
        )
        
        # 内存优化规则
        self.optimizer.add_rule(
            lambda m: m.memory_usage > 75,
            lambda m: {
                "type": "memory_optimization",
                "suggestion": "建议增加内存或优化内存使用",
                "priority": "high"
            }
        )
        
        # 响应时间优化规则
        self.optimizer.add_rule(
            lambda m: m.response_time and m.response_time > 500,
            lambda m: {
                "type": "response_time_optimization",
                "suggestion": "优化请求处理流程或增加处理节点",
                "priority": "medium"
            }
        )
        
    async def start(self):
        """启动监控"""
        self.is_running = True
        self.logger.info("健康监控启动")
        
        while self.is_running:
            try:
                # 收集系统指标
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查健康状态
                health_status = await self._check_health(metrics)
                self.health_history.append(health_status)
                
                # 处理发现的问题
                if not health_status.is_healthy:
                    await self._handle_issues(health_status.issues)
                    
                # 等待下一次检查
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {str(e)}")
                await asyncio.sleep(5)  # 出错后短暂等待
                
    async def stop(self):
        """停止监控"""
        self.is_running = False
        self.logger.info("健康监控停止")
        
    async def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # 基本系统指标
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # 获取进程信息
            process_count = len(psutil.pids())
            
            # 尝试获取GPU信息
            gpu_usage = None
            gpu_memory = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_memory = (gpu_info.used / gpu_info.total) * 100
            except:
                pass
                
            # 创建指标对象
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                process_count=process_count,
                timestamp=datetime.now().isoformat(),
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {str(e)}")
            raise
            
    async def _check_health(self, metrics: SystemMetrics) -> HealthStatus:
        """检查健康状态"""
        # 更新预测器
        self.predictor.update(metrics)
        
        # 检查问题
        issues = self.checker.check_metrics(metrics)
        
        # 获取预测
        predictions = self.predictor.predict()
        
        # 获取优化建议
        optimization_suggestions = self.optimizer.analyze(metrics)
        
        # 创建健康状态
        health_status = HealthStatus(
            is_healthy=len(issues) == 0,
            metrics=metrics,
            issues=issues,
            recovery_actions=[],
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            optimization_suggestions=optimization_suggestions
        )
        
        return health_status

    async def _handle_issues(self, issues: List[Dict[str, Any]]):
        """处理问题"""
        try:
            for issue in issues:
                # 处理问题
                result = await self.recovery.handle_issue(issue)
                
                # 记录恢复动作
                self.health_history[-1].recovery_actions.append(result)
                
        except Exception as e:
            self.logger.error(f"问题处理失败: {str(e)}")
            
    async def _handle_cpu_high_usage(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """处理CPU使用率过高"""
        try:
            # 获取CPU使用率最高的进程
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # 按CPU使用率排序
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            # 终止最耗CPU的进程
            if processes:
                top_process = processes[0]
                if top_process['cpu_percent'] > 50:  # 只终止CPU使用率超过50%的进程
                    proc = psutil.Process(top_process['pid'])
                    proc.terminate()
                    return {
                        "action": "terminate_process",
                        "process": top_process,
                        "result": "success"
                    }
                    
            return {
                "action": "no_action",
                "result": "No high CPU process found"
            }
            
        except Exception as e:
            self.logger.error(f"CPU问题处理失败: {str(e)}")
            return {
                "action": "error",
                "result": str(e)
            }
            
    async def _handle_memory_high_usage(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """处理内存使用率过高"""
        try:
            # 获取内存使用率最高的进程
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # 按内存使用率排序
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            # 终止最耗内存的进程
            if processes:
                top_process = processes[0]
                if top_process['memory_percent'] > 10:  # 只终止内存使用率超过10%的进程
                    proc = psutil.Process(top_process['pid'])
                    proc.terminate()
                    return {
                        "action": "terminate_process",
                        "process": top_process,
                        "result": "success"
                    }
                    
            return {
                "action": "no_action",
                "result": "No high memory process found"
            }
            
        except Exception as e:
            self.logger.error(f"内存问题处理失败: {str(e)}")
            return {
                "action": "error",
                "result": str(e)
            }
            
    async def _handle_disk_high_usage(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """处理磁盘使用率过高"""
        try:
            # 获取最大的文件
            largest_files = []
            for path in Path('/').rglob('*'):
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        largest_files.append((path, size))
                except (PermissionError, OSError):
                    continue
                    
            # 按文件大小排序
            largest_files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除最大的文件
            if largest_files:
                largest_file = largest_files[0]
                if largest_file[1] > 1e9:  # 只删除大于1GB的文件
                    largest_file[0].unlink()
                    return {
                        "action": "delete_file",
                        "file": str(largest_file[0]),
                        "size": largest_file[1],
                        "result": "success"
                    }
                    
            return {
                "action": "no_action",
                "result": "No large file found"
            }
            
        except Exception as e:
            self.logger.error(f"磁盘问题处理失败: {str(e)}")
            return {
                "action": "error",
                "result": str(e)
            }
            
    async def _handle_too_many_processes(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """处理进程数量过多"""
        try:
            # 获取所有进程
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # 按资源使用率排序
            processes.sort(key=lambda x: x['cpu_percent'] + x['memory_percent'], reverse=True)
            
            # 终止资源使用率最高的进程
            if processes:
                top_process = processes[0]
                if top_process['cpu_percent'] + top_process['memory_percent'] > 30:
                    proc = psutil.Process(top_process['pid'])
                    proc.terminate()
                    return {
                        "action": "terminate_process",
                        "process": top_process,
                        "result": "success"
                    }
                    
            return {
                "action": "no_action",
                "result": "No high resource process found"
            }
            
        except Exception as e:
            self.logger.error(f"进程问题处理失败: {str(e)}")
            return {
                "action": "error",
                "result": str(e)
            }
            
    def get_health_status(self) -> HealthStatus:
        """获取当前健康状态"""
        if self.health_history:
            return self.health_history[-1]
        return None
        
    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """获取指标历史"""
        return self.metrics_history[-limit:]
        
    def get_health_history(self, limit: int = 100) -> List[HealthStatus]:
        """获取健康状态历史"""
        return self.health_history[-limit:]
        
    def save_state(self, path: str):
        """保存状态"""
        data = {
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "health_history": [asdict(h) for h in self.health_history]
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_state(self, path: str):
        """加载状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.metrics_history = [
                SystemMetrics(**m) for m in data["metrics_history"]
            ]
            self.health_history = [
                HealthStatus(**h) for h in data["health_history"]
            ]
            
        except FileNotFoundError:
            self.logger.warning("状态文件不存在，使用默认状态") 