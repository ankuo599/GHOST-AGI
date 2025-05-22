"""
任务调度系统
实现任务的规划、执行和监控
"""

from typing import Dict, List, Any, Optional, Set, Callable, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import uuid
from enum import Enum
import networkx as nx
from collections import defaultdict
import psutil
import traceback
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    BLOCKED = "blocked"
    SCHEDULED = "scheduled"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"

@dataclass
class ResourceRequirement:
    """资源需求"""
    type: ResourceType
    amount: float
    unit: str
    priority: int = 0
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class Task:
    """任务"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    dependencies: List[str]
    resources: List[ResourceRequirement]
    metadata: Dict[str, Any]
    result: Optional[Any]
    error: Optional[str]
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    last_error: Optional[str] = None
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    predicted_duration: Optional[float] = None
    resource_usage_history: Optional[List[Dict[str, float]]] = None

class TaskPredictor:
    """任务预测器"""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.feature_names = [
            "priority", "resource_count", "dependency_count",
            "retry_count", "time_of_day", "day_of_week"
        ]
        self.is_trained = False
        
    def prepare_features(self, task: Task) -> np.ndarray:
        """准备特征"""
        features = [
            task.priority.value,
            len(task.resources),
            len(task.dependencies),
            task.retry_count,
            datetime.fromisoformat(task.created_at).hour,
            datetime.fromisoformat(task.created_at).weekday()
        ]
        return np.array(features).reshape(1, -1)
        
    def train(self, tasks: List[Task]):
        """训练模型"""
        if not tasks:
            return
            
        X = []
        y = []
        
        for task in tasks:
            if task.actual_duration is not None:
                X.append(self.prepare_features(task)[0])
                y.append(task.actual_duration)
                
        if X and y:
            X = np.array(X)
            y = np.array(y)
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X, y)
            self.is_trained = True
            
    def predict(self, task: Task) -> Optional[float]:
        """预测任务执行时间"""
        if not self.is_trained:
            return None
            
        features = self.prepare_features(task)
        features = self.scaler.transform(features)
        return float(self.model.predict(features)[0])

class ResourceOptimizer:
    """资源优化器"""
    def __init__(self):
        self.resource_usage_history = defaultdict(list)
        self.optimization_strategies = {}
        self.performance_metrics = defaultdict(list)
        
    def record_usage(self, resource_type: ResourceType, usage: float):
        """记录资源使用情况"""
        self.resource_usage_history[resource_type].append({
            "timestamp": datetime.now().isoformat(),
            "usage": usage
        })
        
    def analyze_usage(self, resource_type: ResourceType) -> Dict[str, Any]:
        """分析资源使用情况"""
        if not self.resource_usage_history[resource_type]:
            return {}
            
        usages = [h["usage"] for h in self.resource_usage_history[resource_type]]
        return {
            "mean": np.mean(usages),
            "std": np.std(usages),
            "max": np.max(usages),
            "min": np.min(usages),
            "trend": self._calculate_trend(usages)
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "stable"
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        return "stable"
        
    def optimize_allocation(self, task: Task, available_resources: Dict[ResourceType, float]) -> Dict[ResourceType, float]:
        """优化资源分配"""
        allocation = {}
        
        for req in task.resources:
            if req.type in available_resources:
                # 获取历史使用情况
                usage_stats = self.analyze_usage(req.type)
                
                # 根据历史数据调整分配
                if usage_stats:
                    # 如果资源使用趋势上升，增加分配
                    if usage_stats["trend"] == "increasing":
                        allocation[req.type] = min(
                            req.amount * 1.2,
                            available_resources[req.type]
                        )
                    # 如果资源使用趋势下降，减少分配
                    elif usage_stats["trend"] == "decreasing":
                        allocation[req.type] = max(
                            req.amount * 0.8,
                            req.amount
                        )
                    else:
                        allocation[req.type] = min(
                            req.amount,
                            available_resources[req.type]
                        )
                else:
                    allocation[req.type] = min(
                        req.amount,
                        available_resources[req.type]
                    )
                    
        return allocation

class AdaptiveScheduler:
    """自适应调度器"""
    def __init__(self):
        self.scheduling_history = []
        self.performance_metrics = defaultdict(list)
        self.scheduling_strategies = {
            "fifo": self._fifo_schedule,
            "priority": self._priority_schedule,
            "resource_aware": self._resource_aware_schedule,
            "deadline_aware": self._deadline_aware_schedule
        }
        self.current_strategy = "priority"
        
    def _fifo_schedule(self, tasks: List[Task]) -> List[Task]:
        """先进先出调度"""
        return sorted(tasks, key=lambda t: t.created_at)
        
    def _priority_schedule(self, tasks: List[Task]) -> List[Task]:
        """基于优先级调度"""
        return sorted(tasks, key=lambda t: (-t.priority.value, t.created_at))
        
    def _resource_aware_schedule(self, tasks: List[Task]) -> List[Task]:
        """基于资源感知调度"""
        return sorted(tasks, key=lambda t: (
            -t.priority.value,
            len(t.resources),
            t.created_at
        ))
        
    def _deadline_aware_schedule(self, tasks: List[Task]) -> List[Task]:
        """基于截止时间调度"""
        return sorted(tasks, key=lambda t: (
            -t.priority.value,
            t.timeout if t.timeout else float('inf'),
            t.created_at
        ))
        
    def record_performance(self, strategy: str, metrics: Dict[str, float]):
        """记录调度性能"""
        self.performance_metrics[strategy].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
    def analyze_performance(self) -> Dict[str, Any]:
        """分析调度性能"""
        analysis = {}
        
        for strategy, metrics in self.performance_metrics.items():
            if not metrics:
                continue
                
            # 计算平均性能指标
            avg_metrics = defaultdict(list)
            for m in metrics:
                for k, v in m["metrics"].items():
                    avg_metrics[k].append(v)
                    
            analysis[strategy] = {
                k: np.mean(v) for k, v in avg_metrics.items()
            }
            
        return analysis
        
    def select_strategy(self) -> str:
        """选择最佳调度策略"""
        analysis = self.analyze_performance()
        if not analysis:
            return self.current_strategy
            
        # 根据性能指标选择最佳策略
        best_strategy = max(
            analysis.items(),
            key=lambda x: sum(x[1].values())
        )[0]
        
        self.current_strategy = best_strategy
        return best_strategy
        
    def schedule(self, tasks: List[Task]) -> List[Task]:
        """调度任务"""
        strategy = self.select_strategy()
        return self.scheduling_strategies[strategy](tasks)

class TaskScheduler:
    """任务调度器"""
    def __init__(self, max_workers: int = 10):
        self.logger = logging.getLogger("TaskScheduler")
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.resource_manager = ResourceManager()
        self.event_bus = EventBus()
        self.predictor = TaskPredictor()
        self.optimizer = ResourceOptimizer()
        self.scheduler = AdaptiveScheduler()
        self.workers = []
        self.is_running = False
        
    async def submit_task(self, name: str, description: str,
                         priority: TaskPriority,
                         dependencies: List[str] = None,
                         resources: List[ResourceRequirement] = None,
                         metadata: Dict[str, Any] = None,
                         timeout: Optional[int] = None,
                         max_retries: Optional[int] = None) -> str:
        """提交任务"""
            task_id = str(uuid.uuid4())
        
        # 创建任务
            task = Task(
                id=task_id,
                name=name,
                description=description,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                dependencies=dependencies or [],
            resources=resources or [],
                metadata=metadata or {},
                result=None,
                error=None,
            max_retries=max_retries or 3,
            timeout=timeout
        )
        
        # 预测任务执行时间
        task.predicted_duration = self.predictor.predict(task)
            
            # 保存任务
            self.tasks[task_id] = task
            
        # 发布事件
        await self.event_bus.publish(Event(
            type=EventType.TASK_CREATED,
            timestamp=datetime.now().isoformat(),
            data={"task_id": task_id},
            source="scheduler"
            ))
            
            return task_id
            
    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            
            # 优化资源分配
            available_resources = self.resource_manager.get_available_resources()
            optimized_allocation = self.optimizer.optimize_allocation(task, available_resources)
                    
                # 分配资源
            for resource_type, amount in optimized_allocation.items():
                self.resource_manager.allocate(task.id, resource_type, amount)
                    
                # 执行任务
            start_time = time.time()
            result = await self._run_task(task)
            end_time = time.time()
            
            # 记录实际执行时间
            task.actual_duration = end_time - start_time
            
            # 更新预测器
            self.predictor.train([task])
            
            # 记录资源使用情况
            for resource_type, amount in optimized_allocation.items():
                self.optimizer.record_usage(resource_type, amount)
                        
                # 更新任务状态
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
            task.result = result
            
            # 发布事件
            await self.event_bus.publish(Event(
                type=EventType.TASK_COMPLETED,
                timestamp=datetime.now().isoformat(),
                data={"task_id": task.id},
                source="scheduler"
            ))
                
            except Exception as e:
            # 处理错误
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.last_error = str(e)
                
            # 发布事件
            await self.event_bus.publish(Event(
                type=EventType.TASK_FAILED,
                timestamp=datetime.now().isoformat(),
                data={
                    "task_id": task.id,
                    "error": str(e)
                },
                source="scheduler"
            ))
            
            # 重试逻辑
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
                await self._handle_retry(task)
                
        finally:
            # 释放资源
            await self._release_resources(task)
            
    async def _worker(self):
        """工作线程"""
        while self.is_running:
            try:
                # 获取待执行任务
                pending_tasks = [
                    t for t in self.tasks.values()
                    if t.status == TaskStatus.PENDING
                ]
                
                if pending_tasks:
                    # 使用自适应调度器选择任务
                    scheduled_tasks = self.scheduler.schedule(pending_tasks)
                    
                    for task in scheduled_tasks:
                        # 检查依赖
                        if await self._check_dependencies(task):
                            # 执行任务
                            await self._execute_task(task)
                            
                            # 记录调度性能
                            self.scheduler.record_performance(
                                self.scheduler.current_strategy,
                                {
                                    "completion_time": task.actual_duration or 0,
                                    "resource_efficiency": self._calculate_resource_efficiency(task),
                                    "success_rate": 1 if task.status == TaskStatus.COMPLETED else 0
                                }
                            )
                            
                await asyncio.sleep(0.1)
            
        except Exception as e:
                self.logger.error(f"工作线程错误: {str(e)}")
                await asyncio.sleep(1)
                
    def _calculate_resource_efficiency(self, task: Task) -> float:
        """计算资源使用效率"""
        if not task.resource_usage_history:
            return 0.0
            
        # 计算资源使用率的标准差
        usage_std = np.std([u["usage"] for u in task.resource_usage_history])
        # 计算资源使用率的平均值
        usage_mean = np.mean([u["usage"] for u in task.resource_usage_history])
        
        # 效率 = 1 - (标准差/平均值)
        return 1 - (usage_std / usage_mean if usage_mean > 0 else 0)

# ... existing code ...