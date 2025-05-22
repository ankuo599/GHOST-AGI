# -*- coding: utf-8 -*-
"""
增强版分布式协作系统 (Enhanced Collaboration System)

实现基于能力的任务分配算法、智能体间知识共享机制和冲突检测与解决机制
支持协作绩效评估和动态任务调度
"""

import time
import uuid
import json
import logging
import threading
import queue
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

# 导入基础协作系统
from collaboration_system import CollaborationSystem
# 导入目标冲突检测器
from goal_conflict_detector import GoalConflictDetector

class EnhancedCollaborationSystem(CollaborationSystem):
    def __init__(self, instance_id=None, host="localhost", port=5000):
        """
        初始化增强版分布式协作系统
        
        Args:
            instance_id: 实例ID，如果为None则自动生成
            host: 主机地址
            port: 端口号
        """
        super().__init__(instance_id, host, port)
        
        # 任务分配增强
        self.task_allocation = {
            "active": True,
            "capability_weights": {},  # 能力权重 {capability: weight}
            "agent_capability_scores": {},  # 智能体能力评分 {agent_id: {capability: score}}
            "task_requirements": {},  # 任务需求 {task_type: {capability: importance}}
            "allocation_history": [],  # 分配历史
            "allocation_success_rate": {}  # 分配成功率 {agent_id: rate}
        }
        
        # 知识共享机制
        self.knowledge_sharing = {
            "active": True,
            "shared_knowledge": {},  # 共享知识库
            "knowledge_access_control": {},  # 知识访问控制
            "sharing_history": [],  # 共享历史
            "knowledge_usage_stats": {}  # 知识使用统计
        }
        
        # 冲突检测与解决
        self.conflict_resolution = {
            "active": True,
            "conflict_patterns": {},  # 冲突模式
            "resolution_strategies": {},  # 解决策略
            "conflict_history": [],  # 冲突历史
            "resolution_success_rate": {}  # 解决成功率
        }
        
        # 协作绩效评估
        self.collaboration_metrics = {
            "active": True,
            "metrics": {},  # 评估指标
            "baseline_performance": {},  # 基准性能
            "evaluation_history": [],  # 评估历史
            "team_scores": {}  # 团队评分
        }
        
        # 智能体优先级
        self.agent_priorities = {}  # 智能体优先级 {agent_id: priority}
        
        # 初始化目标冲突检测器
        self.goal_conflict_detector = GoalConflictDetector()
        
        # 初始化冲突解决策略
        self._initialize_resolution_strategies()
        
    def _initialize_resolution_strategies(self):
        """
        初始化冲突解决策略
        """
        self.conflict_resolution["resolution_strategies"] = {
            "priority_based": self._resolve_by_priority,
            "voting": self._resolve_by_voting,
            "compromise": self._resolve_by_compromise,
            "capability_based": self._resolve_by_capability
        }
        
    def allocate_task(self, task_id: str, task_type: str, task_data: Dict[str, Any], 
                     required_capabilities: List[str] = None) -> str:
        """
        基于能力的任务分配
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            task_data: 任务数据
            required_capabilities: 所需能力列表
            
        Returns:
            str: 分配的智能体ID，如果无法分配则返回None
        """
        if not self.task_allocation["active"] or not self.known_instances:
            return None
            
        # 如果未指定所需能力，尝试从任务需求中获取
        if required_capabilities is None:
            if task_type in self.task_allocation["task_requirements"]:
                required_capabilities = list(self.task_allocation["task_requirements"][task_type].keys())
            else:
                required_capabilities = []  # 默认不需要特定能力
                
        # 计算每个智能体的适合度分数
        agent_scores = {}
        
        for instance_id, instance_info in self.known_instances.items():
            # 跳过不活跃的实例
            if instance_info.get("status") != "active":
                continue
                
            # 获取实例能力
            instance_capabilities = self.instance_capabilities.get(instance_id, set())
            
            # 计算能力匹配分数
            capability_score = 0
            capability_count = 0
            
            for capability in required_capabilities:
                if capability in instance_capabilities:
                    # 如果有能力权重，使用权重；否则使用默认权重1.0
                    weight = self.task_allocation["capability_weights"].get(capability, 1.0)
                    
                    # 如果有能力评分，使用评分；否则使用默认评分1.0
                    if instance_id in self.task_allocation["agent_capability_scores"] and \
                       capability in self.task_allocation["agent_capability_scores"][instance_id]:
                        score = self.task_allocation["agent_capability_scores"][instance_id][capability]
                    else:
                        score = 1.0
                        
                    capability_score += weight * score
                    capability_count += 1
                    
            # 如果没有匹配的能力，分数为0
            if capability_count == 0:
                agent_scores[instance_id] = 0
            else:
                # 归一化分数
                agent_scores[instance_id] = capability_score / capability_count
                
                # 考虑历史成功率
                if instance_id in self.task_allocation["allocation_success_rate"]:
                    success_rate = self.task_allocation["allocation_success_rate"][instance_id]
                    # 将成功率作为权重因子
                    agent_scores[instance_id] *= (0.5 + 0.5 * success_rate)  # 确保即使成功率为0也有50%的基础分
                    
                # 考虑负载均衡
                pending_tasks_count = sum(1 for t_id, a_id in self.assigned_tasks.items() if a_id == instance_id)
                # 任务越多，分数越低
                load_factor = max(0.5, 1.0 / (1.0 + 0.1 * pending_tasks_count))  # 确保至少有50%的基础分
                agent_scores[instance_id] *= load_factor
                
        # 如果没有合适的智能体，返回None
        if not agent_scores:
            return None
            
        # 选择得分最高的智能体
        best_agent_id = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        # 如果最高分数为0，表示没有合适的智能体
        if agent_scores[best_agent_id] == 0:
            return None
            
        # 分配任务
        self.assigned_tasks[task_id] = best_agent_id
        
        # 记录分配历史
        allocation_record = {
            "task_id": task_id,
            "task_type": task_type,
            "agent_id": best_agent_id,
            "score": agent_scores[best_agent_id],
            "timestamp": time.time()
        }
        self.task_allocation["allocation_history"].append(allocation_record)
        
        self.logger.info(f"任务 {task_id} 分配给智能体 {best_agent_id}，分数: {agent_scores[best_agent_id]:.4f}")
        return best_agent_id
    
    def update_capability_score(self, agent_id: str, capability: str, score: float):
        """
        更新智能体能力评分
        
        Args:
            agent_id: 智能体ID
            capability: 能力名称
            score: 评分 (0-1)
        """
        if agent_id not in self.task_allocation["agent_capability_scores"]:
            self.task_allocation["agent_capability_scores"][agent_id] = {}
            
        self.task_allocation["agent_capability_scores"][agent_id][capability] = score
    
    def update_allocation_success_rate(self, agent_id: str, success: bool):
        """
        更新任务分配成功率
        
        Args:
            agent_id: 智能体ID
            success: 是否成功
        """
        if agent_id not in self.task_allocation["allocation_success_rate"]:
            self.task_allocation["allocation_success_rate"][agent_id] = 1.0 if success else 0.0
        else:
            # 使用指数移动平均更新成功率
            current_rate = self.task_allocation["allocation_success_rate"][agent_id]
            alpha = 0.1  # 平滑因子
            new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
            self.task_allocation["allocation_success_rate"][agent_id] = new_rate
    
    def share_knowledge(self, agent_id: str, knowledge_id: str, knowledge_data: Dict[str, Any], 
                       access_control: List[str] = None) -> bool:
        """
        智能体间知识共享
        
        Args:
            agent_id: 共享知识的智能体ID
            knowledge_id: 知识ID
            knowledge_data: 知识数据
            access_control: 可访问此知识的智能体ID列表，None表示所有智能体可访问
            
        Returns:
            bool: 共享是否成功
        """
        if not self.knowledge_sharing["active"]:
            return False
            
        # 创建知识记录
        knowledge_record = {
            "id": knowledge_id,
            "data": knowledge_data,
            "source": agent_id,
            "created_at": time.time(),
            "access_count": 0,
            "last_accessed": None
        }
        
        # 存储知识
        self.knowledge_sharing["shared_knowledge"][knowledge_id] = knowledge_record
        
        # 设置访问控制
        if access_control is not None:
            self.knowledge_sharing["knowledge_access_control"][knowledge_id] = access_control
            
        # 记录共享历史
        sharing_record = {
            "knowledge_id": knowledge_id,
            "source": agent_id,
            "timestamp": time.time()
        }
        self.knowledge_sharing["sharing_history"].append(sharing_record)
        
        self.logger.info(f"智能体 {agent_id} 共享知识 {knowledge_id}")
        return True
    
    def access_shared_knowledge(self, agent_id: str, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        访问共享知识
        
        Args:
            agent_id: 访问知识的智能体ID
            knowledge_id: 知识ID
            
        Returns:
            Optional[Dict[str, Any]]: 知识数据，如果无法访问则返回None
        """
        if not self.knowledge_sharing["active"] or \
           knowledge_id not in self.knowledge_sharing["shared_knowledge"]:
            return None
            
        # 检查访问控制
        if knowledge_id in self.knowledge_sharing["knowledge_access_control"] and \
           agent_id not in self.knowledge_sharing["knowledge_access_control"][knowledge_id]:
            self.logger.warning(f"智能体 {agent_id} 无权访问知识 {knowledge_id}")
            return None
            
        # 获取知识记录
        knowledge_record = self.knowledge_sharing["shared_knowledge"][knowledge_id]
        
        # 更新访问统计
        knowledge_record["access_count"] += 1
        knowledge_record["last_accessed"] = time.time()
        
        # 更新知识使用统计
        if agent_id not in self.knowledge_sharing["knowledge_usage_stats"]:
            self.knowledge_sharing["knowledge_usage_stats"][agent_id] = {}
            
        if knowledge_id not in self.knowledge_sharing["knowledge_usage_stats"][agent_id]:
            self.knowledge_sharing["knowledge_usage_stats"][agent_id][knowledge_id] = 0
            
        self.knowledge_sharing["knowledge_usage_stats"][agent_id][knowledge_id] += 1
        
        return knowledge_record["data"]
    
    def detect_conflict(self, task_id: str, agent_actions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        检测智能体行动冲突
        
        Args:
            task_id: 任务ID
            agent_actions: 智能体行动 {agent_id: action_data}
            
        Returns:
            Optional[Dict[str, Any]]: 冲突信息，如果无冲突则返回None
        """
        if not self.conflict_resolution["active"] or len(agent_actions) < 2:
            return None
            
        # 冲突检测逻辑
        conflict_detected = False
        conflict_type = None
        conflicting_agents = []
        conflict_details = {}
        
        # 检查资源冲突（多个智能体尝试修改同一资源）
        resource_access = {}  # {resource_id: [agent_ids]}
        
        for agent_id, action in agent_actions.items():
            # 提取行动中涉及的资源
            resources = action.get("resources", [])
            for resource in resources:
                resource_id = resource.get("id")
                access_type = resource.get("access_type", "read")  # read或write
                
                if resource_id not in resource_access:
                    resource_access[resource_id] = {"read": [], "write": []}
                    
                resource_access[resource_id][access_type].append(agent_id)
                
                # 检测写冲突（多个智能体尝试写同一资源）
                if access_type == "write" and len(resource_access[resource_id]["write"]) > 1:
                    conflict_detected = True
                    conflict_type = "resource_write_conflict"
                    conflicting_agents = resource_access[resource_id]["write"]
                    conflict_details = {
                        "resource_id": resource_id,
                        "access_type": "write"
                    }
                    break
                    
                # 检测读写冲突（一个智能体尝试读，另一个尝试写）
                if access_type == "read" and resource_access[resource_id]["write"]:
                    conflict_detected = True
                    conflict_type = "resource_read_write_conflict"
                    conflicting_agents = resource_access[resource_id]["read"] + resource_access[resource_id]["write"]
                    conflict_details = {
                        "resource_id": resource_id,
                        "readers": resource_access[resource_id]["read"],
                        "writers": resource_access[resource_id]["write"]
                    }
                    break
                    
            if conflict_detected:
                break
                
        # 检查目标冲突（智能体目标不一致）
        if not conflict_detected:
            agent_goals = {}
            for agent_id, action in agent_actions.items():
                goal = action.get("goal")
                if goal:
                    if goal not in agent_goals:
                        agent_goals[goal] = []
                    agent_goals[goal].append(agent_id)
                    
            # 如果有多个不同的目标，可能存在冲突
            if len(agent_goals) > 1:
                # 检查目标是否互斥
                goals = list(agent_goals.keys())
                for i in range(len(goals)):
                    for j in range(i+1, len(goals)):
                        # 这里需要一个目标互斥性检查函数，简化为随机
                        if self._are_goals_conflicting(goals[i], goals[j]):
                            conflict_detected = True
                            conflict_type = "goal_conflict"
                            conflicting_agents = agent_goals[goals[i]] + agent_goals[goals[j]]
                            conflict_details = {
                                "goals": [goals[i], goals[j]]
                            }
                            break
                    if conflict_detected:
                        break
                        
        if not conflict_detected:
            return None
            
        # 创建冲突记录
        conflict = {
            "id": str(uuid.uuid4()),
            "task_id": task_id,
            "type": conflict_type,
            "agents": conflicting_agents,
            "details": conflict_details,
            "detected_at": time.time(),
            "resolved": False,
            "resolution": None
        }
        
        # 记录冲突历史
        self.conflict_resolution["conflict_history"].append(conflict)
        
        self.logger.warning(f"检测到冲突: {conflict_type}，涉及智能体: {conflicting_agents}")
        return conflict
    
    def resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        解决智能体冲突
        
        Args:
            conflict: 冲突信息
            
        Returns:
            Dict[str, Any]: 解决方案
        """
        if not self.conflict_resolution["active"]:
            return {"status": "inactive"}
            
        conflict_type = conflict["type"]
        resolution_strategy = None
        
        # 根据冲突类型选择解决策略
        if conflict_type == "resource_write_conflict":
            resolution_strategy = "priority_based"
        elif conflict_type == "resource_read_write_conflict":
            resolution_strategy = "compromise"
        elif conflict_type == "goal_conflict":
            resolution_strategy = "capability_based"
        else:
            resolution_strategy = "voting"  # 默认策略
            
        # 执行解决策略
        if resolution_strategy in self.conflict_resolution["resolution_strategies"]:
            resolution = self.conflict_resolution["resolution_strategies"][resolution_strategy](conflict)
        else:
            resolution = {"status": "failed", "reason": "未知的解决策略"}
            
        # 更新冲突记录
        for i, c in enumerate(self.conflict_resolution["conflict_history"]):
            if c["id"] == conflict["id"]:
                self.conflict_resolution["conflict_history"][i]["resolved"] = resolution["status"] == "success"
                self.conflict_resolution["conflict_history"][i]["resolution"] = resolution
                break
                
        self.logger.info(f"冲突 {conflict['id']} 解决方案: {resolution['status']}")
        return resolution
    
    def _resolve_by_priority(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于优先级解决冲突
        
        Args:
            conflict: 冲突信息
            
        Returns:
            Dict[str, Any]: 解决方案
        """
        agents = conflict["agents"]
        agent_priorities = {}
        
        # 获取智能体优先级
        for agent_id in agents:
            if agent_id in self.agent_priorities:
                agent_priorities[agent_id] = self.agent_priorities.get(agent_id, 0)
            else:
                # 如果没有设置优先级，使用默认值
                agent_priorities[agent_id] = 0
                
        # 选择优先级最高的智能体（数值越小优先级越高）
        selected_agent = min(agent_priorities.items(), key=lambda x: x[1])[0]
        
        return {
            "status": "success",
            "strategy": "priority_based",
            "selected_agent": selected_agent,
            "reason": "基于优先级选择"
        }
    
    def _resolve_by_voting(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于投票解决冲突
        
        Args:
            conflict: 冲突信息
            
        Returns:
            Dict[str, Any]: 解决方案
        """
        # 在实际应用中，这里应该实现真正的投票机制
        # 简化为随机选择一个解决方案
        options = ["option_a", "option_b", "option_c"]
        selected_option = random.choice(options)
        
        return {
            "status": "success",
            "strategy": "voting",
            "selected_option": selected_option,
            "reason": "基于投票选择"
        }
    
    def _resolve_by_compromise(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于妥协解决冲突
        
        Args:
            conflict: 冲突信息
            
        Returns:
            Dict[str, Any]: 解决方案
        """
        # 读写冲突的妥协策略：让写操作先完成，然后再执行读操作
        if conflict["type"] == "resource_read_write_conflict":
            details = conflict["details"]
            writers = details.get("writers", [])
            readers = details.get("readers", [])
            
            return {
                "status": "success",
                "strategy": "compromise",
                "sequence": {
                    "first": writers,  # 先执行写操作
                    "then": readers    # 再执行读操作
                },
                "reason": "先写后读策略"
            }
            
        return {
            "status": "failed",
            "strategy": "compromise",
            "reason": "无法为此类冲突找到妥协方案"
        }
    
    def _resolve_by_capability(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于能力解决冲突
        
        Args:
            conflict: 冲突信息
            
        Returns:
            Dict[str, Any]: 解决方案
        """
        agents = conflict["agents"]
        agent_scores = {}
        
        # 计算每个智能体的能力分数
        for agent_id in agents:
            if agent_id in self.task_allocation["agent_capability_scores"]:
                # 计算平均能力分数
                scores = self.task_allocation["agent_capability_scores"][agent_id].values()
                if scores:
                    agent_scores[agent_id] = sum(scores) / len(scores)
                else:
                    agent_scores[agent_id] = 0.5  # 默认中等能力
            else:
                agent_scores[agent_id] = 0.5  # 默认中等能力
                
        # 选择能力分数最高的智能体
        if agent_scores:
            selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
            
            return {
                "status": "success",
                "strategy": "capability_based",
                "selected_agent": selected_agent,
                "reason": "基于能力评分选择"
            }
            
        return {
            "status": "failed",
            "strategy": "capability_based",
            "reason": "无法评估智能体能力"
        }
    
    def _are_goals_conflicting(self, goal1, goal2) -> bool:
        """
        检查两个目标是否冲突
        
        Args:
            goal1: 目标1
            goal2: 目标2
            
        Returns:
            bool: 是否冲突
        """
        # 使用目标冲突检测器进行检测
        if isinstance(goal1, dict) and isinstance(goal2, dict):
            # 使用高级冲突检测器
            is_conflict, _, _ = self.goal_conflict_detector.detect_conflict(goal1, goal2)
            return is_conflict
        else:
            # 简单字符串目标，检查是否为已知的冲突对
            # 将简单字符串转换为字典格式
            g1 = {"id": str(goal1), "type": str(goal1)}
            g2 = {"id": str(goal2), "type": str(goal2)}
            
            # 使用高级冲突检测器
            is_conflict, _, _ = self.goal_conflict_detector.detect_conflict(g1, g2)
            
            # 如果检测器未检测到冲突，使用基本规则检查
            if not is_conflict:
                conflicting_pairs = [
                    ("maximize_profit", "minimize_cost"),
                    ("increase_speed", "reduce_errors"),
                    ("centralize", "distribute")
                ]
                
                for pair in conflicting_pairs:
                    if (goal1 == pair[0] and goal2 == pair[1]) or (goal1 == pair[1] and goal2 == pair[0]):
                        return True
                    
        # 默认情况下，假设目标不冲突
        return False
        
    def set_agent_priority(self, agent_id: str, priority: int) -> bool:
        """
        设置智能体优先级
        
        Args:
            agent_id: 智能体ID
            priority: 优先级值（数值越小优先级越高）
            
        Returns:
            bool: 设置是否成功
        """
        if not agent_id:
            return False
            
        self.agent_priorities[agent_id] = priority
        self.logger.info(f"设置智能体 {agent_id} 优先级为 {priority}")
        return True
        
    def get_agent_priority(self, agent_id: str) -> Optional[int]:
        """
        获取智能体优先级
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            Optional[int]: 优先级值，如果未设置则返回None
        """
        return self.agent_priorities.get(agent_id)
    
    def evaluate_collaboration(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        评估协作绩效
        
        Args:
            metrics: 性能指标
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.collaboration_metrics["active"]:
            return {"status": "inactive"}
            
        # 更新评估指标
        self.collaboration_metrics["metrics"].update(metrics)
        
        # 如果没有基准性能，将当前性能设为基准
        if not self.collaboration_metrics["baseline_performance"]:
            self.collaboration_metrics["baseline_performance"] = metrics.copy()
            return {"status": "baseline_set", "baseline": metrics}
            
        # 计算相对于基准的改进
        improvements = {}
        for metric, value in metrics.items():
            if metric in self.collaboration_metrics["baseline_performance"]:
                baseline = self.collaboration_metrics["baseline_performance"][metric]
                if baseline != 0:
                    relative_improvement = (value - baseline) / abs(baseline)
                else:
                    relative_improvement = value
                improvements[metric] = relative_improvement
                
        # 计算总体协作分数
        if improvements:
            collaboration_score = sum(improvements.values()) / len(improvements)
        else:
            collaboration_score = 0.0
            
        # 记录评估结果
        evaluation_result = {
            "timestamp": time.time(),
            "metrics": metrics.copy(),
            "improvements": improvements,
            "collaboration_score": collaboration_score
        }
        self.collaboration_metrics["evaluation_history"].append(evaluation_result)
        
        # 如果评估历史过长，移除最旧的记录
        if len(self.collaboration_metrics["evaluation_history"]) > 20:
            self.collaboration_metrics["evaluation_history"].pop(0)
            
        # 更新团队评分
        self.collaboration_metrics["team_scores"]["current"] = collaboration_score
        if "highest" not in self.collaboration_metrics["team_scores"] or \
           collaboration_score > self.collaboration_metrics["team_scores"]["highest"]:
            self.collaboration_metrics["team_scores"]["highest"] = collaboration_score
            
        return {
            "status": "evaluated",
            "improvements": improvements,
            "collaboration_score": collaboration_score,
            "team_scores": self.collaboration_metrics["team_scores"]
        }