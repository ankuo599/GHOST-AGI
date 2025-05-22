# -*- coding: utf-8 -*-
"""
智能体调度器 (Agent Scheduler)

负责管理和调度系统中的各个智能体，实现任务分配和优先级管理
支持基于能力和性能的任务分配和负载均衡
"""

import time
import uuid
import threading
from queue import PriorityQueue, Empty
from typing import Dict, List, Any, Callable, Optional, Set, Tuple, Union

class AgentScheduler:
    def __init__(self, event_system=None):
        """
        初始化智能体调度器
        
        Args:
            event_system: 事件系统实例，用于发布调度相关事件
        """
        self.agents = {}  # 注册的智能体 {agent_id: agent_data}
        self.task_queue = PriorityQueue()  # 任务优先级队列
        self.running = False  # 调度器是否运行
        self.scheduler_thread = None  # 调度线程
        self.lock = threading.RLock()  # 线程锁
        self.event_system = event_system  # 事件系统
        self.task_history = []  # 任务历史
        self.max_history = 100  # 最大历史记录数
        self.task_callbacks = {}  # 任务完成回调函数
        self.default_timeout = 60  # 默认任务超时（秒）
        
    def register_agent(self, agent_id: str, agent_instance: Any, 
                        capabilities: Optional[List[str]] = None,
                        max_tasks: int = 1) -> bool:
        """
        注册智能体
        
        Args:
            agent_id (str): 智能体ID
            agent_instance (Any): 智能体实例
            capabilities (List[str], optional): 智能体能力列表
            max_tasks (int): 最大并行任务数
            
        Returns:
            bool: 是否成功注册
        """
        with self.lock:
            if agent_id in self.agents:
                return False  # 智能体已存在
                
            self.agents[agent_id] = {
                "instance": agent_instance,
                "capabilities": capabilities or [],
                "status": "idle",
                "performance": {
                    "success_rate": 0.5,  # 初始成功率
                    "avg_response_time": 1.0,  # 初始平均响应时间
                    "task_count": 0  # 已完成任务数
                },
                "current_tasks": [],  # 当前正在处理的任务
                "max_tasks": max_tasks,  # 最大并行任务数
                "last_task_time": 0,  # 上次分配任务的时间
                "registered_at": time.time()
            }
            
            # 发布智能体注册事件
            if self.event_system:
                self.event_system.publish("agent.registered", {
                    "agent_id": agent_id,
                    "capabilities": capabilities
                })
                
            return True
            
    def unregister_agent(self, agent_id: str) -> bool:
        """
        取消注册智能体
        
        Args:
            agent_id (str): 智能体ID
            
        Returns:
            bool: 是否成功取消注册
        """
        with self.lock:
            if agent_id not in self.agents:
                return False
                
            # 检查是否有正在处理的任务
            current_tasks = self.agents[agent_id]["current_tasks"]
            if current_tasks:
                # 重新分配未完成的任务
                for task_id in current_tasks:
                    self._reassign_task(task_id)
                    
            # 移除智能体
            del self.agents[agent_id]
            
            # 发布智能体注销事件
            if self.event_system:
                self.event_system.publish("agent.unregistered", {
                    "agent_id": agent_id
                })
                
            return True
            
    def assign_task(self, task_description: Dict[str, Any], 
                     required_capabilities: Optional[List[str]] = None,
                     priority: int = 0,
                     timeout: Optional[int] = None,
                     callback: Optional[Callable] = None) -> str:
        """
        分配任务
        
        Args:
            task_description (Dict[str, Any]): 任务描述
            required_capabilities (List[str], optional): 所需能力列表
            priority (int): 任务优先级
            timeout (int, optional): 任务超时时间（秒）
            callback (Callable, optional): 任务完成后的回调函数
            
        Returns:
            str: 任务ID，如果无法分配则返回None
        """
        task_id = str(uuid.uuid4())
        
        # 创建任务对象
        task = {
            "id": task_id,
            "description": task_description,
            "status": "pending",
            "required_capabilities": required_capabilities or [],
            "priority": priority,
            "created_at": time.time(),
            "assigned_to": None,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "timeout": timeout or self.default_timeout
        }
        
        # 尝试立即分配
        assigned = self._assign_task_to_agent(task)
        
        if not assigned:
            # 加入队列，等待调度
            self.task_queue.put((priority, task))
            
        # 记录任务
        with self.lock:
            # 保存回调函数
            if callback:
                self.task_callbacks[task_id] = callback
                
            # 添加到历史记录
            self.task_history.append(task)
            if len(self.task_history) > self.max_history:
                self.task_history = self.task_history[-self.max_history:]
                
        # 确保调度器正在运行
        if not self.running:
            self.start()
            
        return task_id
        
    def _assign_task_to_agent(self, task: Dict[str, Any]) -> bool:
        """
        将任务分配给合适的智能体
        
        Args:
            task (Dict[str, Any]): 任务对象
            
        Returns:
            bool: 是否成功分配
        """
        with self.lock:
            suitable_agents = []
            required_capabilities = task["required_capabilities"]
            
            # 找到具有所需能力的可用智能体
            for agent_id, agent_data in self.agents.items():
                # 检查智能体是否可以接受更多任务
                if len(agent_data["current_tasks"]) >= agent_data["max_tasks"]:
                    continue
                    
                # 检查是否具备所需能力
                if required_capabilities:
                    if not all(cap in agent_data["capabilities"] for cap in required_capabilities):
                        continue
                        
                suitable_agents.append((agent_id, agent_data))
                
            if not suitable_agents:
                return False  # 没有合适的智能体
                
            # 根据性能指标排序：先考虑成功率，再考虑响应时间
            suitable_agents.sort(key=lambda x: (
                x[1]["performance"]["success_rate"], 
                -x[1]["performance"]["avg_response_time"],
                -x[1]["last_task_time"]  # 负值，使得最久未分配任务的优先
            ), reverse=True)
            
            # 分配给最合适的智能体
            selected_agent_id, selected_agent = suitable_agents[0]
            task["assigned_to"] = selected_agent_id
            task["status"] = "assigned"
            task["started_at"] = time.time()
            
            # 更新智能体状态
            selected_agent["current_tasks"].append(task["id"])
            selected_agent["last_task_time"] = time.time()
            if len(selected_agent["current_tasks"]) >= selected_agent["max_tasks"]:
                selected_agent["status"] = "busy"
            else:
                selected_agent["status"] = "active"
                
            # 发布任务分配事件
            if self.event_system:
                self.event_system.publish("task.assigned", {
                    "task_id": task["id"],
                    "agent_id": selected_agent_id,
                    "description": task["description"]
                })
                
            # 将任务发送给智能体
            try:
                if hasattr(selected_agent["instance"], "execute_task"):
                    # 异步执行任务
                    threading.Thread(
                        target=self._execute_task,
                        args=(selected_agent_id, task),
                        daemon=True
                    ).start()
                    
                return True
            except Exception as e:
                # 任务分配失败，标记为错误
                task["status"] = "error"
                task["result"] = {"error": str(e)}
                
                # 从智能体当前任务中移除
                selected_agent["current_tasks"].remove(task["id"])
                
                # 发布任务错误事件
                if self.event_system:
                    self.event_system.publish("task.error", {
                        "task_id": task["id"],
                        "agent_id": selected_agent_id,
                        "error": str(e)
                    })
                    
                return False
                
    def _execute_task(self, agent_id: str, task: Dict[str, Any]) -> None:
        """
        执行任务
        
        Args:
            agent_id (str): 智能体ID
            task (Dict[str, Any]): 任务对象
        """
        with self.lock:
            agent_data = self.agents.get(agent_id)
            if not agent_data:
                return
                
        # 获取智能体实例
        agent_instance = agent_data["instance"]
        
        try:
            # 设置任务开始时间
            start_time = time.time()
            
            # 执行任务
            result = agent_instance.execute_task(task["description"])
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            with self.lock:
                # 更新任务状态
                task["status"] = "completed"
                task["completed_at"] = time.time()
                task["result"] = result
                
                # 更新智能体状态
                agent_data = self.agents.get(agent_id)
                if agent_data:
                    # 从当前任务列表中移除
                    if task["id"] in agent_data["current_tasks"]:
                        agent_data["current_tasks"].remove(task["id"])
                        
                    # 更新性能指标
                    perf = agent_data["performance"]
                    perf["task_count"] += 1
                    
                    # 更新平均响应时间（使用加权平均）
                    weight = 0.2  # 新值的权重
                    perf["avg_response_time"] = (
                        (1 - weight) * perf["avg_response_time"] + 
                        weight * execution_time
                    )
                    
                    # 更新成功率（假设任务完成就是成功）
                    success = True
                    if success:
                        new_success_rate = (
                            (perf["success_rate"] * (perf["task_count"] - 1) + 1) / 
                            perf["task_count"]
                        )
                    else:
                        new_success_rate = (
                            (perf["success_rate"] * (perf["task_count"] - 1)) / 
                            perf["task_count"]
                        )
                    perf["success_rate"] = new_success_rate
                    
                    # 更新智能体状态
                    if not agent_data["current_tasks"]:
                        agent_data["status"] = "idle"
                    
                # 调用任务完成回调
                if task["id"] in self.task_callbacks:
                    callback = self.task_callbacks[task["id"]]
                    del self.task_callbacks[task["id"]]
                    
                    try:
                        callback(task)
                    except Exception as e:
                        print(f"任务回调错误: {str(e)}")
                
            # 发布任务完成事件
            if self.event_system:
                self.event_system.publish("task.completed", {
                    "task_id": task["id"],
                    "agent_id": agent_id,
                    "execution_time": execution_time,
                    "result": result
                })
                
        except Exception as e:
            with self.lock:
                # 更新任务状态为错误
                task["status"] = "error"
                task["completed_at"] = time.time()
                task["result"] = {"error": str(e)}
                
                # 更新智能体状态
                agent_data = self.agents.get(agent_id)
                if agent_data:
                    # 从当前任务列表中移除
                    if task["id"] in agent_data["current_tasks"]:
                        agent_data["current_tasks"].remove(task["id"])
                        
                    # 更新性能指标
                    perf = agent_data["performance"]
                    perf["task_count"] += 1
                    
                    # 更新成功率（失败）
                    new_success_rate = (
                        (perf["success_rate"] * (perf["task_count"] - 1)) / 
                        perf["task_count"]
                    )
                    perf["success_rate"] = new_success_rate
                    
                    # 更新智能体状态
                    if not agent_data["current_tasks"]:
                        agent_data["status"] = "idle"
                    
                # 调用任务完成回调
                if task["id"] in self.task_callbacks:
                    callback = self.task_callbacks[task["id"]]
                    del self.task_callbacks[task["id"]]
                    
                    try:
                        callback(task)
                    except Exception as e:
                        print(f"任务回调错误: {str(e)}")
                
            # 发布任务错误事件
            if self.event_system:
                self.event_system.publish("task.error", {
                    "task_id": task["id"],
                    "agent_id": agent_id,
                    "error": str(e)
                })
                
    def _reassign_task(self, task_id: str) -> bool:
        """
        重新分配任务
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            bool: 是否成功重新分配
        """
        with self.lock:
            # 查找任务
            task = None
            for t in self.task_history:
                if t["id"] == task_id:
                    task = t
                    break
                    
            if not task:
                return False
                
            # 重置任务状态
            task["status"] = "pending"
            task["assigned_to"] = None
            task["started_at"] = None
            
            # 加入队列，等待重新分配
            self.task_queue.put((task["priority"], task))
            
            # 发布任务重新分配事件
            if self.event_system:
                self.event_system.publish("task.reassigned", {
                    "task_id": task_id
                })
                
            return True
            
    def start(self) -> bool:
        """
        启动任务调度器
        
        Returns:
            bool: 是否成功启动
        """
        if self.running:
            return False
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        return True
        
    def stop(self) -> bool:
        """
        停止任务调度器
        
        Returns:
            bool: 是否成功停止
        """
        if not self.running:
            return False
            
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
            
        return True
        
    def _scheduler_loop(self) -> None:
        """
        调度器主循环
        """
        while self.running:
            try:
                # 检查任务超时
                self._check_task_timeouts()
                
                # 尝试获取任务，设置超时以便能响应停止请求
                try:
                    priority, task = self.task_queue.get(timeout=0.5)
                    
                    # 再次尝试分配任务
                    success = self._assign_task_to_agent(task)
                    
                    if not success:
                        # 无法立即分配，重新加入队列（延迟一点以避免CPU过载）
                        time.sleep(0.1)
                        self.task_queue.put((priority, task))
                        
                    self.task_queue.task_done()
                except Empty:
                    # 队列为空，等待下一次循环
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"调度器错误: {str(e)}")
                time.sleep(0.5)  # 发生错误时暂停一下
                
    def _check_task_timeouts(self) -> None:
        """
        检查任务是否超时
        """
        current_time = time.time()
        
        with self.lock:
            # 检查所有已分配但尚未完成的任务
            for task in self.task_history:
                if task["status"] == "assigned" and task["started_at"]:
                    # 计算任务已经运行的时间
                    task_time = current_time - task["started_at"]
                    
                    # 检查是否超时
                    if task_time > task["timeout"]:
                        # 标记任务为超时
                        task["status"] = "timeout"
                        
                        # 从智能体当前任务中移除
                        agent_id = task["assigned_to"]
                        if agent_id in self.agents:
                            agent_data = self.agents[agent_id]
                            if task["id"] in agent_data["current_tasks"]:
                                agent_data["current_tasks"].remove(task["id"])
                                
                                # 更新智能体状态
                                if not agent_data["current_tasks"]:
                                    agent_data["status"] = "idle"
                                    
                        # 发布任务超时事件
                        if self.event_system:
                            self.event_system.publish("task.timeout", {
                                "task_id": task["id"],
                                "agent_id": agent_id,
                                "timeout": task["timeout"],
                                "actual_time": task_time
                            })
                            
                        # 尝试重新分配任务
                        self._reassign_task(task["id"])
                        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            Dict[str, Any]: 任务状态，如果任务不存在则返回None
        """
        with self.lock:
            for task in self.task_history:
                if task["id"] == task_id:
                    return {
                        "id": task["id"],
                        "status": task["status"],
                        "assigned_to": task["assigned_to"],
                        "created_at": task["created_at"],
                        "started_at": task["started_at"],
                        "completed_at": task["completed_at"],
                        "execution_time": task["completed_at"] - task["started_at"] if task["completed_at"] and task["started_at"] else None,
                        "result": task["result"]
                    }
                    
        return None
        
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取智能体状态
        
        Args:
            agent_id (str): 智能体ID
            
        Returns:
            Dict[str, Any]: 智能体状态，如果智能体不存在则返回None
        """
        with self.lock:
            if agent_id not in self.agents:
                return None
                
            agent_data = self.agents[agent_id]
            return {
                "id": agent_id,
                "status": agent_data["status"],
                "capabilities": agent_data["capabilities"],
                "current_tasks": agent_data["current_tasks"],
                "max_tasks": agent_data["max_tasks"],
                "performance": agent_data["performance"],
                "registered_at": agent_data["registered_at"]
            }
            
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有智能体状态
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有智能体的状态
        """
        result = {}
        
        with self.lock:
            for agent_id, agent_data in self.agents.items():
                result[agent_id] = {
                    "status": agent_data["status"],
                    "capabilities": agent_data["capabilities"],
                    "current_tasks": len(agent_data["current_tasks"]),
                    "performance": agent_data["performance"]
                }
                
        return result
        
    def get_task_queue_stats(self) -> Dict[str, Any]:
        """
        获取任务队列统计信息
        
        Returns:
            Dict[str, Any]: 队列统计信息
        """
        with self.lock:
            return {
                "queue_size": self.task_queue.qsize(),
                "completed_tasks": sum(1 for task in self.task_history if task["status"] == "completed"),
                "pending_tasks": sum(1 for task in self.task_history if task["status"] == "pending"),
                "assigned_tasks": sum(1 for task in self.task_history if task["status"] == "assigned"),
                "error_tasks": sum(1 for task in self.task_history if task["status"] == "error"),
                "timeout_tasks": sum(1 for task in self.task_history if task["status"] == "timeout")
            }