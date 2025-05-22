"""
任务规划引擎
实现多步任务分解与执行
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class Task:
    """任务"""
    id: str
    name: str
    description: str
    dependencies: List[str]
    resources: Dict[str, Any]
    priority: int
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    estimated_duration: float = 0.0  # 预计执行时间
    actual_duration: float = 0.0     # 实际执行时间
    retry_count: int = 0             # 重试次数
    max_retries: int = 3             # 最大重试次数
    timeout: float = 300.0           # 超时时间（秒）

@dataclass
class Plan:
    """执行计划"""
    id: str
    tasks: List[Task]
    dependencies: Dict[str, List[str]]
    resources: Dict[str, Any]
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    optimization_history: List[Dict[str, Any]] = None  # 优化历史记录

class TaskPlanner:
    """任务规划器"""
    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.task_templates: Dict[str, Dict[str, Any]] = {}
        self.resource_pool: Dict[str, Any] = {}
        self.logger = logging.getLogger("TaskPlanner")
        self.executor = ThreadPoolExecutor(max_workers=4)  # 任务执行器
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)  # 性能指标
        
    def register_task_template(self, template_id: str, template: Dict[str, Any]) -> bool:
        """注册任务模板"""
        try:
            self.task_templates[template_id] = template
            return True
        except Exception as e:
            self.logger.error(f"注册任务模板失败: {str(e)}")
            return False
            
    def create_plan(self, goal: str, context: Dict[str, Any]) -> Optional[Plan]:
        """创建执行计划"""
        try:
            # 分解目标为子任务
            tasks = self._decompose_goal(goal, context)
            
            # 分析任务依赖
            dependencies = self._analyze_dependencies(tasks)
            
            # 分配资源
            resources = self._allocate_resources(tasks)
            
            # 创建计划
            plan = Plan(
                id=str(uuid.uuid4()),
                tasks=tasks,
                dependencies=dependencies,
                resources=resources,
                optimization_history=[]
            )
            
            # 优化计划
            self._optimize_plan(plan)
            
            self.plans[plan.id] = plan
            return plan
            
        except Exception as e:
            self.logger.error(f"创建计划失败: {str(e)}")
            return None
            
    def _decompose_goal(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """分解目标为子任务"""
        tasks = []
        
        # 使用任务模板分解目标
        for template_id, template in self.task_templates.items():
            if self._is_template_applicable(template, goal, context):
                # 创建子任务
                subtasks = self._create_subtasks(template, goal, context)
                tasks.extend(subtasks)
                
        return tasks
        
    def _is_template_applicable(self, template: Dict[str, Any], 
                              goal: str, context: Dict[str, Any]) -> bool:
        """检查模板是否适用于目标"""
        # 检查目标类型
        if template.get("goal_type") and goal not in template["goal_type"]:
            return False
            
        # 检查上下文条件
        conditions = template.get("conditions", {})
        for key, value in conditions.items():
            if key not in context or context[key] != value:
                return False
                
        return True
        
    def _create_subtasks(self, template: Dict[str, Any], 
                        goal: str, context: Dict[str, Any]) -> List[Task]:
        """根据模板创建子任务"""
        subtasks = []
        
        # 获取任务定义
        task_defs = template.get("tasks", [])
        
        for task_def in task_defs:
            # 创建任务
            task = Task(
                id=str(uuid.uuid4()),
                name=task_def["name"],
                description=task_def["description"],
                dependencies=task_def.get("dependencies", []),
                resources=task_def.get("resources", {}),
                priority=task_def.get("priority", 0)
            )
            
            subtasks.append(task)
            
        return subtasks
        
    def _analyze_dependencies(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """分析任务依赖关系"""
        dependencies = defaultdict(list)
        
        # 构建依赖图
        dep_graph = nx.DiGraph()
        
        for task in tasks:
            dep_graph.add_node(task.id)
            for dep in task.dependencies:
                dep_graph.add_edge(dep, task.id)
                
        # 检查循环依赖
        if not nx.is_directed_acyclic_graph(dep_graph):
            raise ValueError("检测到循环依赖")
            
        # 获取依赖关系
        for task in tasks:
            dependencies[task.id] = list(dep_graph.predecessors(task.id))
            
        return dict(dependencies)
        
    def _allocate_resources(self, tasks: List[Task]) -> Dict[str, Any]:
        """分配资源"""
        allocated = {}
        
        for task in tasks:
            for resource_type, amount in task.resources.items():
                if resource_type not in allocated:
                    allocated[resource_type] = 0
                allocated[resource_type] += amount
                
                # 检查资源是否足够
                if resource_type in self.resource_pool:
                    if allocated[resource_type] > self.resource_pool[resource_type]:
                        raise ValueError(f"资源 {resource_type} 不足")
                        
        return allocated
        
    def _optimize_plan(self, plan: Plan):
        """优化执行计划"""
        # 记录优化前的状态
        initial_metrics = self._calculate_plan_metrics(plan)
        
        # 1. 任务优先级优化
        self._optimize_task_priorities(plan)
        
        # 2. 资源分配优化
        self._optimize_resource_allocation(plan)
        
        # 3. 并行度优化
        self._optimize_parallelism(plan)
        
        # 记录优化后的状态
        final_metrics = self._calculate_plan_metrics(plan)
        
        # 记录优化历史
        plan.optimization_history.append({
            "timestamp": time.time(),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "improvements": {
                k: final_metrics[k] - initial_metrics[k]
                for k in initial_metrics
            }
        })
        
    def _calculate_plan_metrics(self, plan: Plan) -> Dict[str, float]:
        """计算计划性能指标"""
        return {
            "estimated_duration": sum(task.estimated_duration for task in plan.tasks),
            "resource_efficiency": self._calculate_resource_efficiency(plan),
            "parallelism_score": self._calculate_parallelism_score(plan),
            "priority_score": sum(task.priority for task in plan.tasks)
        }
        
    def _calculate_resource_efficiency(self, plan: Plan) -> float:
        """计算资源使用效率"""
        total_resources = sum(self.resource_pool.values())
        allocated_resources = sum(plan.resources.values())
        return allocated_resources / total_resources if total_resources > 0 else 0
        
    def _calculate_parallelism_score(self, plan: Plan) -> float:
        """计算并行度得分"""
        # 构建依赖图
        dep_graph = nx.DiGraph()
        for task in plan.tasks:
            dep_graph.add_node(task.id)
            for dep in plan.dependencies[task.id]:
                dep_graph.add_edge(dep, task.id)
                
        # 计算关键路径长度
        critical_path_length = max(
            len(path)
            for path in nx.all_simple_paths(dep_graph, 
                                          source=min(dep_graph.nodes()),
                                          target=max(dep_graph.nodes()))
        )
        
        # 计算并行度得分
        return len(plan.tasks) / critical_path_length if critical_path_length > 0 else 0
        
    def _optimize_task_priorities(self, plan: Plan):
        """优化任务优先级"""
        # 根据依赖关系和资源需求调整优先级
        for task in plan.tasks:
            # 增加关键路径任务的优先级
            if self._is_critical_path_task(task.id, plan):
                task.priority += 2
                
            # 增加资源密集型任务的优先级
            resource_intensity = sum(task.resources.values())
            task.priority += int(resource_intensity * 0.5)
            
    def _is_critical_path_task(self, task_id: str, plan: Plan) -> bool:
        """判断任务是否在关键路径上"""
        # 构建依赖图
        dep_graph = nx.DiGraph()
        for task in plan.tasks:
            dep_graph.add_node(task.id)
            for dep in plan.dependencies[task.id]:
                dep_graph.add_edge(dep, task.id)
                
        # 获取所有最长路径
        longest_paths = []
        max_length = 0
        for path in nx.all_simple_paths(dep_graph, 
                                      source=min(dep_graph.nodes()),
                                      target=max(dep_graph.nodes())):
            if len(path) > max_length:
                max_length = len(path)
                longest_paths = [path]
            elif len(path) == max_length:
                longest_paths.append(path)
                
        # 检查任务是否在任何最长路径上
        return any(task_id in path for path in longest_paths)
        
    def _optimize_resource_allocation(self, plan: Plan):
        """优化资源分配"""
        # 计算每个任务的资源效率
        task_efficiencies = {}
        for task in plan.tasks:
            resource_usage = sum(task.resources.values())
            task_efficiencies[task.id] = task.priority / resource_usage if resource_usage > 0 else 0
            
        # 根据效率调整资源分配
        for task in plan.tasks:
            efficiency = task_efficiencies[task.id]
            # 增加高效任务的资源
            if efficiency > np.mean(list(task_efficiencies.values())):
                for resource_type in task.resources:
                    task.resources[resource_type] *= 1.2
                    
    def _optimize_parallelism(self, plan: Plan):
        """优化并行度"""
        # 识别可以并行的任务
        parallel_groups = self._identify_parallel_tasks(plan)
        
        # 调整任务依赖以增加并行度
        for group in parallel_groups:
            if len(group) > 1:
                # 移除组内任务间的依赖
                for task_id in group:
                    task = next(t for t in plan.tasks if t.id == task_id)
                    task.dependencies = [
                        dep for dep in task.dependencies
                        if dep not in group
                    ]
                    
    def _identify_parallel_tasks(self, plan: Plan) -> List[Set[str]]:
        """识别可以并行的任务组"""
        # 构建依赖图
        dep_graph = nx.DiGraph()
        for task in plan.tasks:
            dep_graph.add_node(task.id)
            for dep in plan.dependencies[task.id]:
                dep_graph.add_edge(dep, task.id)
                
        # 获取所有无依赖的任务
        independent_tasks = set()
        for task in plan.tasks:
            if not plan.dependencies[task.id]:
                independent_tasks.add(task.id)
                
        # 识别可以并行的任务组
        parallel_groups = []
        visited = set()
        
        for task_id in independent_tasks:
            if task_id in visited:
                continue
                
            # 获取可以并行的任务组
            group = {task_id}
            visited.add(task_id)
            
            # 查找其他可以并行的任务
            for other_id in independent_tasks - {task_id}:
                if other_id not in visited and not self._has_common_dependencies(
                    task_id, other_id, plan):
                    group.add(other_id)
                    visited.add(other_id)
                    
            parallel_groups.append(group)
            
        return parallel_groups
        
    def _has_common_dependencies(self, task1_id: str, task2_id: str, plan: Plan) -> bool:
        """检查两个任务是否有共同依赖"""
        deps1 = set(plan.dependencies[task1_id])
        deps2 = set(plan.dependencies[task2_id])
        return bool(deps1 & deps2)
        
    def execute_plan(self, plan_id: str) -> bool:
        """执行计划"""
        if plan_id not in self.plans:
            return False
            
        plan = self.plans[plan_id]
        plan.start_time = time.time()
        plan.status = "running"
        
        try:
            # 获取任务执行顺序
            execution_order = self._get_execution_order(plan)
            
            # 创建任务执行队列
            task_queue = []
            for task_id in execution_order:
                task = next(t for t in plan.tasks if t.id == task_id)
                task_queue.append(task)
                
            # 并行执行任务
            futures = []
            while task_queue:
                # 获取可执行的任务
                executable_tasks = [
                    task for task in task_queue
                    if all(
                        next(t for t in plan.tasks if t.id == dep).status == "completed"
                        for dep in task.dependencies
                    )
                ]
                
                if not executable_tasks:
                    # 等待某个任务完成
                    if futures:
                        done, _ = as_completed(futures, timeout=1.0)
                        for future in done:
                            task = future.result()
                            task_queue.remove(task)
                    continue
                    
                # 提交任务到执行器
                for task in executable_tasks:
                    future = self.executor.submit(self._execute_task, task)
                    futures.append(future)
                    task_queue.remove(task)
                    
            # 等待所有任务完成
            for future in as_completed(futures):
                task = future.result()
                if task.status == "failed":
                    plan.status = "failed"
                    return False
                    
            plan.status = "completed"
            plan.end_time = time.time()
            
            # 更新性能指标
            self._update_performance_metrics(plan)
            
            return True
            
        except Exception as e:
            self.logger.error(f"执行计划失败: {str(e)}")
            plan.status = "failed"
            return False
            
    def _update_performance_metrics(self, plan: Plan):
        """更新性能指标"""
        duration = plan.end_time - plan.start_time
        self.performance_metrics["execution_time"].append(duration)
        self.performance_metrics["success_rate"].append(
            sum(1 for task in plan.tasks if task.status == "completed") / len(plan.tasks)
        )
        
    def _execute_task(self, task: Task) -> Task:
        """执行单个任务"""
        start_time = time.time()
        task.status = "running"
        
        try:
            # 执行任务逻辑
            # TODO: 实现具体的任务执行逻辑
            
            # 更新任务状态
            task.status = "completed"
            task.actual_duration = time.time() - start_time
            
            # 更新任务模板中的执行时间估计
            self._update_task_estimates(task)
            
            return task
            
        except Exception as e:
            self.logger.error(f"执行任务失败: {str(e)}")
            task.status = "failed"
            task.error = str(e)
            
            # 重试任务
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                return self._execute_task(task)
                
            return task
            
    def _update_task_estimates(self, task: Task):
        """更新任务执行时间估计"""
        # 更新任务模板中的时间估计
        for template in self.task_templates.values():
            if template.get("name") == task.name:
                if "estimated_duration" not in template:
                    template["estimated_duration"] = []
                template["estimated_duration"].append(task.actual_duration)
                template["estimated_duration"] = template["estimated_duration"][-10:]  # 保留最近10次
                
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取计划状态"""
        if plan_id not in self.plans:
            return None
            
        plan = self.plans[plan_id]
        
        return {
            "id": plan.id,
            "status": plan.status,
            "progress": self._calculate_progress(plan),
            "start_time": plan.start_time,
            "end_time": plan.end_time,
            "duration": plan.end_time - plan.start_time if plan.end_time else None,
            "task_status": {
                task.id: {
                    "status": task.status,
                    "progress": self._calculate_task_progress(task),
                    "duration": task.actual_duration,
                    "retries": task.retry_count
                }
                for task in plan.tasks
            },
            "optimization_history": plan.optimization_history
        }
        
    def _calculate_task_progress(self, task: Task) -> float:
        """计算任务进度"""
        if task.status == "completed":
            return 1.0
        elif task.status == "failed":
            return 0.0
        elif task.status == "running":
            return min(1.0, task.actual_duration / task.estimated_duration) if task.estimated_duration > 0 else 0.5
        return 0.0
        
    def _calculate_progress(self, plan: Plan) -> float:
        """计算计划进度"""
        if not plan.tasks:
            return 0.0
            
        total_progress = sum(
            self._calculate_task_progress(task)
            for task in plan.tasks
        )
        return total_progress / len(plan.tasks)
        
    def update_resource_pool(self, resources: Dict[str, Any]):
        """更新资源池"""
        self.resource_pool.update(resources)
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            "total": self.resource_pool,
            "allocated": {
                plan_id: plan.resources
                for plan_id, plan in self.plans.items()
                if plan.status == "running"
            },
            "efficiency": {
                plan_id: self._calculate_resource_efficiency(plan)
                for plan_id, plan in self.plans.items()
                if plan.status == "running"
            }
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "execution_time": {
                "mean": np.mean(self.performance_metrics["execution_time"]),
                "std": np.std(self.performance_metrics["execution_time"]),
                "history": self.performance_metrics["execution_time"]
            },
            "success_rate": {
                "mean": np.mean(self.performance_metrics["success_rate"]),
                "std": np.std(self.performance_metrics["success_rate"]),
                "history": self.performance_metrics["success_rate"]
            }
        } 