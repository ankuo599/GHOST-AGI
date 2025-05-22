# -*- coding: utf-8 -*-
"""
分布式协作系统 (Distributed Collaboration System)

负责多实例协同工作，实现任务分发和结果聚合
支持分布式环境下的系统协作和资源共享
"""

import time
import uuid
import json
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable
import requests

class CollaborationSystem:
    def __init__(self, instance_id=None, host="localhost", port=5000):
        """
        初始化分布式协作系统
        
        Args:
            instance_id: 实例ID，如果为None则自动生成
            host: 主机地址
            port: 端口号
        """
        self.logger = logging.getLogger("CollaborationSystem")
        self.instance_id = instance_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        
        # 节点信息
        self.is_coordinator = False  # 是否为协调节点
        self.coordinator_info = None  # 协调节点信息
        self.known_instances = {}  # 已知实例 {instance_id: instance_info}
        
        # 任务管理
        self.task_queue = queue.PriorityQueue()  # 任务队列
        self.task_results = {}  # 任务结果 {task_id: result}
        self.pending_tasks = {}  # 待处理任务 {task_id: task_info}
        self.assigned_tasks = {}  # 已分配任务 {task_id: instance_id}
        
        # 能力注册
        self.capabilities = set()  # 本实例能力集合
        self.instance_capabilities = {}  # 实例能力映射 {instance_id: capabilities}
        
        # 通信状态
        self.is_running = False
        self.communication_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat = {}  # 最后心跳时间 {instance_id: timestamp}
        
        # 结果聚合器
        self.result_aggregators = {}  # {task_type: aggregator_func}
        
        self.logger.info(f"分布式协作系统初始化完成，实例ID: {self.instance_id}")
    
    def start(self):
        """
        启动协作系统
        """
        if self.is_running:
            return
            
        self.is_running = True
        
        # 启动通信线程
        self.communication_thread = threading.Thread(target=self._communication_loop)
        self.communication_thread.daemon = True
        self.communication_thread.start()
        
        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        self.logger.info("分布式协作系统已启动")
    
    def stop(self):
        """
        停止协作系统
        """
        self.is_running = False
        
        if self.communication_thread:
            self.communication_thread.join(timeout=2.0)
            
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
            
        self.logger.info("分布式协作系统已停止")
    
    def register_capability(self, capability):
        """
        注册能力
        
        Args:
            capability: 能力标识
        """
        self.capabilities.add(capability)
        self.logger.info(f"已注册能力: {capability}")
    
    def unregister_capability(self, capability):
        """
        注销能力
        
        Args:
            capability: 能力标识
        """
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self.logger.info(f"已注销能力: {capability}")
    
    def register_result_aggregator(self, task_type, aggregator_func):
        """
        注册结果聚合器
        
        Args:
            task_type: 任务类型
            aggregator_func: 聚合函数，接收任务结果列表，返回聚合结果
        """
        self.result_aggregators[task_type] = aggregator_func
        self.logger.info(f"已注册结果聚合器: {task_type}")
    
    def discover_instances(self, discovery_endpoints=None):
        """
        发现其他实例
        
        Args:
            discovery_endpoints: 发现端点列表
            
        Returns:
            int: 发现的实例数量
        """
        if not discovery_endpoints:
            return 0
            
        discovered = 0
        
        for endpoint in discovery_endpoints:
            try:
                # 模拟发现请求
                # 实际实现中应该发送HTTP请求到发现端点
                # response = requests.get(f"http://{endpoint}/discover", timeout=5)
                # if response.status_code == 200:
                #     instances = response.json().get("instances", [])
                
                # 模拟发现结果
                instances = []
                
                for instance_info in instances:
                    instance_id = instance_info.get("id")
                    if instance_id and instance_id != self.instance_id:
                        self.known_instances[instance_id] = instance_info
                        discovered += 1
                        
                        # 记录实例能力
                        if "capabilities" in instance_info:
                            self.instance_capabilities[instance_id] = set(instance_info["capabilities"])
            except Exception as e:
                self.logger.error(f"发现实例时出错: {e}")
        
        self.logger.info(f"发现了 {discovered} 个新实例")
        return discovered
    
    def submit_task(self, task_type, task_data, priority=0, required_capabilities=None):
        """
        提交任务
        
        Args:
            task_type: 任务类型
            task_data: 任务数据
            priority: 优先级，数字越小优先级越高
            required_capabilities: 所需能力列表
            
        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "type": task_type,
            "data": task_data,
            "priority": priority,
            "required_capabilities": required_capabilities or [],
            "submitted_at": time.time(),
            "status": "pending",
            "submitter": self.instance_id
        }
        
        # 添加到任务队列
        self.task_queue.put((priority, task_id, task))
        self.pending_tasks[task_id] = task
        
        self.logger.info(f"已提交任务: {task_id}, 类型: {task_type}")
        return task_id
    
    def get_task_status(self, task_id):
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务状态
        """
        # 检查是否在结果中
        if task_id in self.task_results:
            return {
                "id": task_id,
                "status": "completed",
                "result": self.task_results[task_id]
            }
            
        # 检查是否在待处理任务中
        if task_id in self.pending_tasks:
            return {
                "id": task_id,
                "status": "pending",
                "info": self.pending_tasks[task_id]
            }
            
        # 检查是否在已分配任务中
        if task_id in self.assigned_tasks:
            return {
                "id": task_id,
                "status": "processing",
                "assigned_to": self.assigned_tasks[task_id]
            }
            
        return {
            "id": task_id,
            "status": "unknown"
        }
    
    def process_local_task(self, task):
        """
        处理本地任务
        
        Args:
            task: 任务信息
            
        Returns:
            Any: 处理结果
        """
        # 这里应该实现实际的任务处理逻辑
        # 根据任务类型调用相应的处理函数
        
        task_id = task["id"]
        task_type = task["type"]
        task_data = task["data"]
        
        # 模拟任务处理
        result = {
            "status": "success",
            "processed_by": self.instance_id,
            "processed_at": time.time(),
            "result_data": f"Processed {task_type} task"
        }
        
        # 存储结果
        self.task_results[task_id] = result
        
        # 如果是已分配任务，移除分配记录
        if task_id in self.assigned_tasks:
            del self.assigned_tasks[task_id]
            
        # 如果是待处理任务，移除待处理记录
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]
            
        self.logger.info(f"已完成本地任务: {task_id}")
        return result
    
    def distribute_task(self, task):
        """
        分发任务到合适的实例
        
        Args:
            task: 任务信息
            
        Returns:
            bool: 是否成功分发
        """
        task_id = task["id"]
        required_capabilities = task.get("required_capabilities", [])
        
        # 查找具备所需能力的实例
        capable_instances = []
        
        # 检查本实例是否具备能力
        if not required_capabilities or all(cap in self.capabilities for cap in required_capabilities):
            capable_instances.append(self.instance_id)
            
        # 检查其他已知实例
        for instance_id, capabilities in self.instance_capabilities.items():
            if not required_capabilities or all(cap in capabilities for cap in required_capabilities):
                capable_instances.append(instance_id)
                
        if not capable_instances:
            self.logger.warning(f"没有找到具备所需能力的实例处理任务: {task_id}")
            return False
            
        # 选择实例（简单实现：选择第一个）
        selected_instance = capable_instances[0]
        
        if selected_instance == self.instance_id:
            # 本地处理
            self.process_local_task(task)
        else:
            # 发送到远程实例
            # 实际实现中应该发送HTTP请求到远程实例
            # try:
            #     response = requests.post(
            #         f"http://{self.known_instances[selected_instance]['host']}:{self.known_instances[selected_instance]['port']}/tasks",
            #         json=task,
            #         timeout=5
            #     )
            #     success = response.status_code == 200
            # except Exception as e:
            #     self.logger.error(f"发送任务到远程实例时出错: {e}")
            #     success = False
            
            # 模拟发送结果
            success = True
            
            if success:
                # 记录任务分配
                self.assigned_tasks[task_id] = selected_instance
                
                # 从待处理任务中移除
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
                    
                self.logger.info(f"已将任务 {task_id} 分发到实例 {selected_instance}")
            
        return True
    
    def aggregate_results(self, task_type, results):
        """
        聚合任务结果
        
        Args:
            task_type: 任务类型
            results: 结果列表
            
        Returns:
            Any: 聚合结果
        """
        if task_type in self.result_aggregators:
            aggregator = self.result_aggregators[task_type]
            try:
                return aggregator(results)
            except Exception as e:
                self.logger.error(f"聚合结果时出错: {e}")
                return None
        else:
            # 默认聚合：返回结果列表
            return results
    
    def _communication_loop(self):
        """
        通信循环
        """
        while self.is_running:
            try:
                # 处理任务队列
                if not self.task_queue.empty():
                    _, task_id, task = self.task_queue.get(block=False)
                    self.distribute_task(task)
                    self.task_queue.task_done()
                    
                # 实际实现中应该处理接收到的消息
                # 例如：接收任务请求、结果返回等
                
                time.sleep(0.1)  # 避免CPU占用过高
            except Exception as e:
                self.logger.error(f"通信循环出错: {e}")
    
    def _heartbeat_loop(self):
        """
        心跳循环
        """
        while self.is_running:
            try:
                current_time = time.time()
                
                # 发送心跳到协调节点
                if self.coordinator_info and not self.is_coordinator:
                    # 实际实现中应该发送HTTP请求到协调节点
                    # try:
                    #     requests.post(
                    #         f"http://{self.coordinator_info['host']}:{self.coordinator_info['port']}/heartbeat",
                    #         json={
                    #             "instance_id": self.instance_id,
                    #             "timestamp": current_time,
                    #             "capabilities": list(self.capabilities)
                    #         },
                    #         timeout=2
                    #     )
                    # except Exception as e:
                    #     self.logger.error(f"发送心跳时出错: {e}")
                    pass
                
                # 检查其他实例心跳
                if self.is_coordinator:
                    for instance_id, last_time in list(self.last_heartbeat.items()):
                        # 如果超过30秒没有心跳，认为实例已离线
                        if current_time - last_time > 30:
                            if instance_id in self.known_instances:
                                del self.known_instances[instance_id]
                            if instance_id in self.instance_capabilities:
                                del self.instance_capabilities[instance_id]
                            del self.last_heartbeat[instance_id]
                            
                            self.logger.warning(f"实例离线: {instance_id}")
                
                time.sleep(10)  # 心跳间隔
            except Exception as e:
                self.logger.error(f"心跳循环出错: {e}")
    
    def become_coordinator(self):
        """
        成为协调节点
        """
        self.is_coordinator = True
        self.coordinator_info = {
            "id": self.instance_id,
            "host": self.host,
            "port": self.port
        }
        self.logger.info("已成为协调节点")
    
    def set_coordinator(self, coordinator_info):
        """
        设置协调节点
        
        Args:
            coordinator_info: 协调节点信息
        """
        self.is_coordinator = False
        self.coordinator_info = coordinator_info
        self.logger.info(f"已设置协调节点: {coordinator_info['id']}")
    
    def get_collaboration_status(self):
        """
        获取协作状态
        
        Returns:
            Dict[str, Any]: 协作状态
        """
        return {
            "instance_id": self.instance_id,
            "is_coordinator": self.is_coordinator,
            "coordinator_info": self.coordinator_info,
            "known_instances": len(self.known_instances),
            "capabilities": list(self.capabilities),
            "pending_tasks": len(self.pending_tasks),
            "assigned_tasks": len(self.assigned_tasks),
            "completed_tasks": len(self.task_results),
            "is_running": self.is_running
        }