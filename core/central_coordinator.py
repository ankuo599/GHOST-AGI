"""
中央协调引擎 (Central Coordination Engine)

负责协调系统各核心组件间的交互和信息流，确保系统组件协同工作。
管理系统的运行状态，调度任务执行，协调资源分配。
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Union
from collections import defaultdict
import threading
import importlib
import psutil
import asyncio
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    id: str
    type: str
    data: Dict[str, Any]
    priority: TaskPriority
    created_at: float
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None
    timeout: int = 300

class ResourceManager:
    """资源管理器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_limits = {
            "cpu_percent": 80.0,
            "memory_percent": 80.0,
            "disk_percent": 90.0
        }
        self.resource_usage = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0
        }
        self.resource_history = []
        self.max_history_size = 1000
        
    def update_usage(self):
        """更新资源使用情况"""
        self.resource_usage = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        self.resource_history.append({
            "timestamp": time.time(),
            "usage": self.resource_usage.copy()
        })
        if len(self.resource_history) > self.max_history_size:
            self.resource_history.pop(0)
            
    def check_resources(self) -> bool:
        """检查资源是否充足"""
        return all(
            self.resource_usage[key] < self.resource_limits[key]
            for key in self.resource_limits
        )
        
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        return {
            "current": self.resource_usage,
            "limits": self.resource_limits,
            "history": self.resource_history[-100:] if self.resource_history else []
        }

class FaultRecovery:
    """故障恢复管理器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recovery_strategies = {}
        self.fault_history = []
        self.max_history_size = 1000
        
    def register_strategy(self, fault_type: str, strategy: Callable):
        """注册恢复策略"""
        self.recovery_strategies[fault_type] = strategy
        
    def handle_fault(self, fault_type: str, fault_data: Dict[str, Any]) -> bool:
        """处理故障"""
        if fault_type in self.recovery_strategies:
            try:
                success = self.recovery_strategies[fault_type](fault_data)
                self.fault_history.append({
                    "timestamp": time.time(),
                    "type": fault_type,
                    "data": fault_data,
                    "success": success
                })
                if len(self.fault_history) > self.max_history_size:
                    self.fault_history.pop(0)
                return success
            except Exception as e:
                logging.error(f"故障恢复失败: {str(e)}")
                return False
        return False
        
    def get_fault_stats(self) -> Dict[str, Any]:
        """获取故障统计信息"""
        return {
            "total_faults": len(self.fault_history),
            "successful_recoveries": sum(1 for f in self.fault_history if f["success"]),
            "recent_faults": self.fault_history[-100:] if self.fault_history else []
        }

class CentralCoordinator:
    """中央协调引擎，负责协调系统各组件间的交互和合作"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化中央协调引擎
        
        Args:
            config: 配置参数
        """
        # 初始化配置
        self.config = config or {}
        self.default_config = {
            "module_load_timeout": 30,
            "task_execution_timeout": 300,
            "enable_parallel_execution": True,
            "max_parallel_tasks": 5,
            "log_level": "info",
            "auto_recovery": True,
            "performance_monitoring": True,
            "resource_monitoring": True,
            "fault_tolerance": True
        }
        
        # 合并默认配置
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # 设置日志记录器
        self.logger = self._setup_logger()
        self.logger.info("中央协调引擎初始化中...")
        
        # 系统组件
        self.modules = {}
        self.required_modules = [
            "knowledge.self_organizing_knowledge.SelfOrganizingKnowledge",
            "reasoning.creative_thinking_engine.CreativeThinkingEngine",
            "metacognition.metacognitive_monitor.MetacognitiveMonitor"
        ]
        
        # 任务管理
        self.tasks = {}
        self.task_queues = {
            priority: [] for priority in TaskPriority
        }
        self.running_tasks = set()
        self.completed_tasks = []
        self.task_lock = threading.RLock()
        
        # 资源管理
        self.resource_manager = ResourceManager(self.config)
        
        # 故障恢复
        self.fault_recovery = FaultRecovery(self.config)
        
        # 系统状态
        self.system_state = {
            "status": "initializing",
            "start_time": time.time(),
            "active_modules": set(),
            "resource_usage": self.resource_manager.resource_usage,
            "health": {
                "overall": "good",
                "issues": []
            }
        }
        
        # 性能指标
        self.performance_metrics = {
            "task_completion_times": [],
            "module_response_times": defaultdict(list),
            "error_rates": defaultdict(int),
            "resource_usage_history": []
        }
        
        # 加载核心模块
        self._load_core_modules()
        
        self.logger.info("中央协调引擎初始化完成")
        self.system_state["status"] = "ready"
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("CentralCoordinator")
        
        # 根据配置设置日志级别
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        log_level = level_map.get(self.config["log_level"].lower(), logging.INFO)
        logger.setLevel(log_level)
        
        # 添加处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 文件处理器
            file_handler = logging.FileHandler("coordinator.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_core_modules(self):
        """加载核心模块"""
        self.logger.info("开始加载核心模块...")
        
        for module_path in self.required_modules:
            try:
                self._load_module(module_path)
            except Exception as e:
                self.logger.error(f"加载模块 {module_path} 失败: {str(e)}")
                self.system_state["health"]["issues"].append(f"模块加载失败: {module_path}")
                
        if len(self.modules) < len(self.required_modules):
            self.system_state["health"]["overall"] = "degraded"
            self.logger.warning("部分核心模块加载失败，系统处于降级状态")
        else:
            self.logger.info("所有核心模块加载完成")
            
    def _load_module(self, module_path: str) -> bool:
        """
        加载单个模块
        
        Args:
            module_path: 模块路径，格式为 "package.module.ClassName"
            
        Returns:
            bool: 是否成功加载
        """
        module_parts = module_path.split('.')
        class_name = module_parts[-1]
        package_path = '.'.join(module_parts[:-1])
        
        self.logger.info(f"加载模块: {module_path}")
        
        try:
            # 加载模块
            module = importlib.import_module(package_path)
            
            # 获取类
            module_class = getattr(module, class_name)
            
            # 实例化模块
            module_instance = module_class()
            
            # 保存模块实例
            module_id = class_name.lower()
            self.modules[module_id] = {
                "instance": module_instance,
                "path": module_path,
                "loaded_at": time.time(),
                "status": "active"
            }
            
            self.system_state["active_modules"].add(module_id)
            
            self.logger.info(f"模块 {module_id} 加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模块 {module_path} 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def start_system(self) -> Dict[str, Any]:
        """
        启动系统
        
        Returns:
            Dict: 启动结果
        """
        if self.system_state["status"] == "running":
            return {
                "status": "error",
                "message": "系统已在运行中"
            }
            
        self.logger.info("启动系统...")
        
        # 初始化所有模块
        for module_id, module_info in self.modules.items():
            instance = module_info["instance"]
            
            # 启动模块
            try:
                if hasattr(instance, "start") and callable(getattr(instance, "start")):
                    instance.start()
            except Exception as e:
                self.logger.error(f"启动模块 {module_id} 失败: {str(e)}")
                self.system_state["health"]["issues"].append(f"模块启动失败: {module_id}")
                
        # 启动任务处理线程
        if self.config["enable_parallel_execution"]:
            self._start_task_processor()
            
        # 启动性能监控
        if self.config["performance_monitoring"]:
            self._start_performance_monitoring()
            
        self.system_state["status"] = "running"
        
        return {
            "status": "success",
            "message": "系统已启动",
            "active_modules": list(self.system_state["active_modules"])
        }
        
    def stop_system(self) -> Dict[str, Any]:
        """
        停止系统
        
        Returns:
            Dict: 停止结果
        """
        if self.system_state["status"] != "running":
            return {
                "status": "error",
                "message": "系统未在运行"
            }
            
        self.logger.info("停止系统...")
        
        # 停止所有模块
        for module_id, module_info in self.modules.items():
            instance = module_info["instance"]
            
            # 停止模块
            try:
                if hasattr(instance, "stop") and callable(getattr(instance, "stop")):
                    instance.stop()
            except Exception as e:
                self.logger.error(f"停止模块 {module_id} 失败: {str(e)}")
                
        # 停止任务处理
        self._stop_task_processor()
        
        # 停止性能监控
        self._stop_performance_monitoring()
        
        self.system_state["status"] = "stopped"
        
        return {
            "status": "success",
            "message": "系统已停止"
        }
        
    async def execute_task(self, task_type: str, task_data: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL) -> Dict[str, Any]:
        """执行任务"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            type=task_type,
            data=task_data,
            priority=priority,
            created_at=time.time()
        )
        
        with self.task_lock:
            self.tasks[task_id] = task
            self.task_queues[priority].append(task)
            
        try:
            # 检查资源
            if not self.resource_manager.check_resources():
                raise Exception("系统资源不足")
                
            # 执行任务
            result = await self._process_task(task)
            
            # 更新任务状态
            with self.task_lock:
                task.status = "completed"
                task.result = result
                
            return {
                "status": "success",
                "task_id": task_id,
                "result": result
            }
            
        except Exception as e:
            # 处理故障
            if self.config["fault_tolerance"]:
                self.fault_recovery.handle_fault("task_execution_failed", {
                    "task_id": task_id,
                    "error": str(e)
                })
                
            # 更新任务状态
            with self.task_lock:
                task.status = "failed"
                task.error = str(e)
                
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
            
    async def _process_task(self, task: Task) -> Any:
        """处理任务"""
        start_time = time.time()
        
        try:
            # 检查任务依赖
            if task.dependencies:
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if not dep_task or dep_task.status != "completed":
                        raise Exception(f"任务依赖 {dep_id} 未完成")
                        
            # 执行任务
            result = await self._dispatch_task(task.type, task.data)
            
            # 记录性能指标
            execution_time = time.time() - start_time
            self.performance_metrics["task_completion_times"].append(execution_time)
            
            return result
            
        except Exception as e:
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(1)  # 等待一秒后重试
                return await self._process_task(task)
            raise
            
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": self.system_state["status"],
            "uptime": time.time() - self.system_state["start_time"],
            "active_modules": list(self.system_state["active_modules"]),
            "resource_usage": self.resource_manager.get_resource_stats(),
            "health": self.system_state["health"],
            "performance": {
                "task_metrics": {
                    "total": len(self.tasks),
                    "running": len(self.running_tasks),
                    "completed": len(self.completed_tasks)
                },
                "error_rates": dict(self.performance_metrics["error_rates"]),
                "avg_completion_time": sum(self.performance_metrics["task_completion_times"]) / 
                    len(self.performance_metrics["task_completion_times"]) if self.performance_metrics["task_completion_times"] else 0
            },
            "fault_stats": self.fault_recovery.get_fault_stats()
        }
        
    def subscribe_event(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            
        Returns:
            Dict: 订阅结果
        """
        self.event_listeners[event_type].append(callback)
        
        return {
            "status": "success",
            "message": f"已订阅事件: {event_type}",
            "listeners_count": len(self.event_listeners[event_type])
        }
        
    def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
            
        Returns:
            Dict: 发布结果
        """
        # 创建事件记录
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        }
        
        # 记录事件
        self.event_history.append(event)
        
        # 通知订阅者
        listeners = self.event_listeners.get(event_type, [])
        notify_count = 0
        
        for callback in listeners:
            try:
                callback(event_data)
                notify_count += 1
            except Exception as e:
                self.logger.error(f"事件处理器异常: {str(e)}")
                
        return {
            "status": "success",
            "message": f"事件已发布: {event_type}",
            "notified": notify_count
        }
        
    def _start_task_processor(self):
        """启动任务处理线程"""
        self.task_processor_active = True
        self.task_processor_thread = threading.Thread(target=self._task_processor_loop)
        self.task_processor_thread.daemon = True
        self.task_processor_thread.start()
        
    def _stop_task_processor(self):
        """停止任务处理线程"""
        self.task_processor_active = False
        if hasattr(self, 'task_processor_thread') and self.task_processor_thread.is_alive():
            self.task_processor_thread.join(timeout=5.0)
            
    def _task_processor_loop(self):
        """任务处理循环"""
        while self.task_processor_active:
            # 检查是否有任务需要处理
            with self.task_lock:
                available_slots = self.config["max_parallel_tasks"] - len(self.running_tasks)
                
                tasks_to_process = []
                for _ in range(min(available_slots, len(self.task_queue))):
                    if self.task_queue:
                        task_id = self.task_queue.pop(0)
                        tasks_to_process.append(task_id)
                        self.running_tasks.add(task_id)
                    
            # 处理任务
            for task_id in tasks_to_process:
                # 启动单独的线程处理每个任务
                task_thread = threading.Thread(target=self._process_task, args=(task_id,))
                task_thread.daemon = True
                task_thread.start()
                
            # 休眠一段时间
            time.sleep(0.1)
            
    def _process_task(self, task_id: str):
        """
        处理任务
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.tasks:
            self.logger.error(f"找不到任务: {task_id}")
            return
            
        task = self.tasks[task_id]
        task["status"] = "running"
        task["started_at"] = time.time()
        
        self.logger.info(f"开始处理任务: {task_id} ({task['type']})")
        
        try:
            # 根据任务类型分发到相应模块
            task_type = task["type"]
            task_data = task["data"]
            
            result = self._dispatch_task(task_type, task_data)
            
            # 更新任务状态
            task["status"] = "completed"
            task["completed_at"] = time.time()
            task["result"] = result
            
            # 记录任务完成时间
            completion_time = task["completed_at"] - task["started_at"]
            self.performance_metrics["task_completion_times"].append(completion_time)
            
            self.logger.info(f"任务完成: {task_id} ({task['type']}), 耗时: {completion_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"任务执行异常: {task_id} - {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 更新任务状态
            task["status"] = "failed"
            task["completed_at"] = time.time()
            task["error"] = str(e)
            
        finally:
            # 更新任务列表
            with self.task_lock:
                if task_id in self.running_tasks:
                    self.running_tasks.remove(task_id)
                self.completed_tasks.append(task_id)
                
    def _dispatch_task(self, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据任务类型分发到相应模块
        
        Args:
            task_type: 任务类型
            task_data: 任务数据
            
        Returns:
            Dict: 任务结果
        """
        # 任务类型到模块的映射
        task_module_map = {
            "knowledge_add": ("selforganizingknowledge", "add_concept"),
            "knowledge_query": ("selforganizingknowledge", "get_concept"),
            "knowledge_search": ("selforganizingknowledge", "search_concepts"),
            "creative_idea": ("creativethinkingengine", "generate_creative_idea"),
            "creative_validate": ("creativethinkingengine", "validate_creative_idea"),
            "metacognition_analyze": ("metacognitivemonitor", "analyze_thinking_process"),
            "metacognition_state": ("metacognitivemonitor", "get_cognitive_state")
        }
        
        # 检查任务类型是否支持
        if task_type not in task_module_map:
            raise ValueError(f"不支持的任务类型: {task_type}")
            
        # 获取处理模块和方法
        module_id, method_name = task_module_map[task_type]
        
        # 检查模块是否存在
        if module_id not in self.modules:
            raise ValueError(f"模块不存在: {module_id}")
            
        module_info = self.modules[module_id]
        instance = module_info["instance"]
        
        # 检查方法是否存在
        if not hasattr(instance, method_name) or not callable(getattr(instance, method_name)):
            raise ValueError(f"模块 {module_id} 不支持方法: {method_name}")
            
        # 执行方法
        method = getattr(instance, method_name)
        start_time = time.time()
        result = method(**task_data)
        end_time = time.time()
        
        # 记录模块响应时间
        response_time = end_time - start_time
        self.performance_metrics["module_response_times"][module_id].append(response_time)
        
        return result
        
    def _start_performance_monitoring(self):
        """启动性能监控"""
        self.performance_monitor_active = True
        self.performance_monitor_thread = threading.Thread(target=self._performance_monitor_loop)
        self.performance_monitor_thread.daemon = True
        self.performance_monitor_thread.start()
        
    def _stop_performance_monitoring(self):
        """停止性能监控"""
        self.performance_monitor_active = False
        if hasattr(self, 'performance_monitor_thread') and self.performance_monitor_thread.is_alive():
            self.performance_monitor_thread.join(timeout=5.0)
            
    def _performance_monitor_loop(self):
        """性能监控循环"""
        while self.performance_monitor_active:
            self._update_resource_usage()
            
            # 检测性能异常
            self._detect_performance_anomalies()
            
            # 清理旧的性能数据
            self._cleanup_performance_data()
            
            # 每30秒更新一次
            time.sleep(30)
            
    def _update_resource_usage(self):
        """更新资源使用情况"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            
            # 内存使用率
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 更新系统状态
            self.system_state["resource_usage"] = {
                "cpu": cpu_percent,
                "memory": memory_percent,
                "disk": disk_percent
            }
            
            # 记录历史数据
            self.performance_metrics["resource_usage_history"].append({
                "timestamp": time.time(),
                "cpu": cpu_percent,
                "memory": memory_percent,
                "disk": disk_percent
            })
            
        except ImportError:
            # psutil库不可用
            self.logger.warning("无法获取系统资源使用情况，psutil库不可用")
            
        except Exception as e:
            self.logger.error(f"更新资源使用情况失败: {str(e)}")
            
    def _detect_performance_anomalies(self):
        """检测性能异常"""
        resource_usage = self.system_state["resource_usage"]
        
        # 检查资源使用是否超过阈值
        if resource_usage["cpu"] > 90:
            self.logger.warning(f"CPU使用率过高: {resource_usage['cpu']}%")
            self.system_state["health"]["issues"].append(f"CPU使用率过高: {resource_usage['cpu']}%")
            
        if resource_usage["memory"] > 90:
            self.logger.warning(f"内存使用率过高: {resource_usage['memory']}%")
            self.system_state["health"]["issues"].append(f"内存使用率过高: {resource_usage['memory']}%")
            
        if resource_usage["disk"] > 95:
            self.logger.warning(f"磁盘使用率过高: {resource_usage['disk']}%")
            self.system_state["health"]["issues"].append(f"磁盘使用率过高: {resource_usage['disk']}%")
            
        # 限制健康问题列表大小
        if len(self.system_state["health"]["issues"]) > 20:
            self.system_state["health"]["issues"] = self.system_state["health"]["issues"][-20:]
            
        # 更新整体健康状态
        if len(self.system_state["health"]["issues"]) > 5:
            self.system_state["health"]["overall"] = "poor"
        elif len(self.system_state["health"]["issues"]) > 0:
            self.system_state["health"]["overall"] = "degraded"
        else:
            self.system_state["health"]["overall"] = "good"
            
    def _cleanup_performance_data(self):
        """清理旧的性能数据"""
        # 限制任务完成时间记录数量
        if len(self.performance_metrics["task_completion_times"]) > 1000:
            self.performance_metrics["task_completion_times"] = self.performance_metrics["task_completion_times"][-1000:]
            
        # 限制模块响应时间记录数量
        for module_id in self.performance_metrics["module_response_times"]:
            if len(self.performance_metrics["module_response_times"][module_id]) > 1000:
                self.performance_metrics["module_response_times"][module_id] = \
                    self.performance_metrics["module_response_times"][module_id][-1000:]
                    
        # 限制资源使用历史记录数量
        if len(self.performance_metrics["resource_usage_history"]) > 1000:
            self.performance_metrics["resource_usage_history"] = \
                self.performance_metrics["resource_usage_history"][-1000:]
            
        # 限制事件历史记录数量
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:] 