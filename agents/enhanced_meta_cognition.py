# -*- coding: utf-8 -*-
"""
增强版元认知智能体 (Enhanced Meta-Cognition Agent)

实现第四阶段计划中的元认知智能体自我监督能力：
1. 实现智能体行为的自我监控机制
2. 提供决策过程的透明度和可解释性
3. 支持目标一致性检查
4. 实现自适应的资源分配
5. 提供性能评估和优化建议
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict

class EnhancedMetaCognitionAgent:
    def __init__(self):
        """
        初始化增强版元认知智能体
        """
        # 监控指标
        self.monitoring_metrics = {
            "decision_time": [],
            "goal_consistency": [],
            "resource_usage": [],
            "action_success_rate": [],
            "learning_progress": []
        }
        
        # 智能体状态跟踪
        self.agent_states = {}
        self.decision_history = []
        self.reflection_logs = []
        
        # 资源分配
        self.resource_allocation = {}
        self.resource_limits = {}
        
        # 性能评估
        self.performance_evaluations = {}
        self.optimization_suggestions = []
        
        # 目标管理
        self.current_goals = []
        self.goal_consistency_scores = {}
        
        # 日志系统
        self.logger = logging.getLogger("meta_cognition")
        
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """
        注册智能体到监控系统
        
        Args:
            agent_id (str): 智能体ID
            agent_type (str): 智能体类型
            capabilities (List[str]): 智能体能力列表
        """
        self.agent_states[agent_id] = {
            "type": agent_type,
            "capabilities": capabilities,
            "status": "idle",
            "current_task": None,
            "performance": {},
            "resource_usage": {},
            "last_active": time.time()
        }
        
        # 初始化资源分配
        self.resource_allocation[agent_id] = {
            "cpu": 0.1,  # 初始CPU分配比例
            "memory": 0.1,  # 初始内存分配比例
            "priority": 1.0  # 初始优先级
        }
        
    def set_resource_limits(self, limits: Dict[str, float]):
        """
        设置系统资源限制
        
        Args:
            limits (Dict[str, float]): 资源限制字典
        """
        self.resource_limits = limits.copy()
        
    def monitor_decision(self, agent_id: str, decision: Dict[str, Any], 
                         context: Dict[str, Any], execution_time: float):
        """
        监控智能体决策过程
        
        Args:
            agent_id (str): 智能体ID
            decision (Dict[str, Any]): 决策内容
            context (Dict[str, Any]): 决策上下文
            execution_time (float): 执行时间
        """
        if agent_id not in self.agent_states:
            self.logger.warning(f"未注册的智能体: {agent_id}")
            return
            
        # 记录决策
        decision_record = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "decision": decision,
            "context": context,
            "execution_time": execution_time
        }
        
        self.decision_history.append(decision_record)
        
        # 更新智能体状态
        self.agent_states[agent_id]["status"] = "active"
        self.agent_states[agent_id]["current_task"] = decision.get("task_id")
        self.agent_states[agent_id]["last_active"] = time.time()
        
        # 记录决策时间指标
        self.monitoring_metrics["decision_time"].append(execution_time)
        
        # 检查目标一致性
        if "goal" in decision:
            consistency_score = self._check_goal_consistency(decision["goal"], self.current_goals)
            self.goal_consistency_scores[decision.get("task_id")] = consistency_score
            self.monitoring_metrics["goal_consistency"].append(consistency_score)
            
    def _check_goal_consistency(self, goal: Any, current_goals: List[Any]) -> float:
        """
        检查目标与当前目标集的一致性
        
        Args:
            goal: 待检查目标
            current_goals: 当前目标集
            
        Returns:
            float: 一致性得分 (0-1)
        """
        if not current_goals:
            return 1.0  # 如果没有当前目标，则认为一致
            
        # 简单实现：检查目标是否在当前目标列表中
        if isinstance(goal, dict) and isinstance(current_goals[0], dict):
            # 如果目标是字典类型，计算键的重叠度
            goal_keys = set(goal.keys())
            consistency_scores = []
            
            for current_goal in current_goals:
                current_keys = set(current_goal.keys())
                overlap = len(goal_keys.intersection(current_keys))
                total = len(goal_keys.union(current_keys))
                consistency_scores.append(overlap / total if total > 0 else 0.0)
                
            return max(consistency_scores)  # 返回最高一致性分数
        else:
            # 其他类型目标，检查是否存在于当前目标中
            return 1.0 if goal in current_goals else 0.0
            
    def update_resource_usage(self, agent_id: str, usage: Dict[str, float]):
        """
        更新智能体资源使用情况
        
        Args:
            agent_id (str): 智能体ID
            usage (Dict[str, float]): 资源使用情况
        """
        if agent_id not in self.agent_states:
            return
            
        self.agent_states[agent_id]["resource_usage"] = usage.copy()
        
        # 记录资源使用指标
        total_usage = sum(usage.values())
        self.monitoring_metrics["resource_usage"].append(total_usage)
        
    def optimize_resource_allocation(self):
        """
        优化系统资源分配
        
        Returns:
            Dict[str, Dict[str, float]]: 优化后的资源分配
        """
        # 基于智能体性能和当前任务优先级调整资源分配
        active_agents = [agent_id for agent_id, state in self.agent_states.items() 
                        if state["status"] != "idle"]
        
        if not active_agents:
            return self.resource_allocation
            
        # 计算每个智能体的性能分数
        performance_scores = {}
        for agent_id in active_agents:
            # 基于历史性能计算分数
            if "performance" in self.agent_states[agent_id]:
                perf = self.agent_states[agent_id]["performance"]
                if perf:
                    avg_score = sum(perf.values()) / len(perf)
                    performance_scores[agent_id] = avg_score
                else:
                    performance_scores[agent_id] = 0.5  # 默认中等性能
            else:
                performance_scores[agent_id] = 0.5
        
        # 根据性能分数和优先级分配资源
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for agent_id in active_agents:
                score_ratio = performance_scores[agent_id] / total_score
                priority = self.resource_allocation[agent_id]["priority"]
                
                # 调整CPU分配
                self.resource_allocation[agent_id]["cpu"] = min(0.8, score_ratio * priority * 0.5 + 0.1)
                
                # 调整内存分配
                self.resource_allocation[agent_id]["memory"] = min(0.8, score_ratio * priority * 0.5 + 0.1)
        
        return self.resource_allocation
    
    def evaluate_performance(self, agent_id: str, metrics: Dict[str, float]):
        """
        评估智能体性能
        
        Args:
            agent_id (str): 智能体ID
            metrics (Dict[str, float]): 性能指标
            
        Returns:
            float: 综合性能得分 (0-1)
        """
        if agent_id not in self.agent_states:
            return 0.0
            
        # 更新性能记录
        self.agent_states[agent_id]["performance"] = metrics.copy()
        
        # 计算综合得分
        weights = {
            "accuracy": 0.4,
            "efficiency": 0.3,
            "resource_efficiency": 0.2,
            "adaptability": 0.1
        }
        
        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                score += value * weights[metric]
        
        # 记录到性能评估
        self.performance_evaluations[agent_id] = score
        
        # 记录成功率指标
        if "success_rate" in metrics:
            self.monitoring_metrics["action_success_rate"].append(metrics["success_rate"])
            
        return score
    
    def generate_optimization_suggestions(self, agent_id: str) -> List[str]:
        """
        生成智能体优化建议
        
        Args:
            agent_id (str): 智能体ID
            
        Returns:
            List[str]: 优化建议列表
        """
        if agent_id not in self.agent_states:
            return []
            
        suggestions = []
        agent_state = self.agent_states[agent_id]
        
        # 基于性能指标生成建议
        if "performance" in agent_state:
            perf = agent_state["performance"]
            
            # 检查准确率
            if "accuracy" in perf and perf["accuracy"] < 0.7:
                suggestions.append("提高决策准确率：考虑增加更多上下文信息或改进推理算法")
                
            # 检查效率
            if "efficiency" in perf and perf["efficiency"] < 0.6:
                suggestions.append("提高处理效率：优化算法复杂度或考虑缓存常用结果")
                
            # 检查资源效率
            if "resource_efficiency" in perf and perf["resource_efficiency"] < 0.5:
                suggestions.append("提高资源利用效率：减少不必要的计算或优化内存使用")
                
            # 检查适应性
            if "adaptability" in perf and perf["adaptability"] < 0.6:
                suggestions.append("提高环境适应性：增强学习能力或扩展知识库覆盖范围")
        
        # 基于资源使用情况生成建议
        if "resource_usage" in agent_state:
            usage = agent_state["resource_usage"]
            
            if "cpu" in usage and usage["cpu"] > 0.8:
                suggestions.append("CPU使用率过高：考虑任务分解或异步处理")
                
            if "memory" in usage and usage["memory"] > 0.8:
                suggestions.append("内存使用率过高：优化数据结构或实现增量处理")
        
        # 记录优化建议
        self.optimization_suggestions.append({
            "agent_id": agent_id,
            "timestamp": time.time(),
            "suggestions": suggestions
        })
        
        return suggestions
    
    def reflect_on_decisions(self, time_window: float = 3600) -> Dict[str, Any]:
        """
        对过去一段时间的决策进行反思
        
        Args:
            time_window (float): 时间窗口（秒）
            
        Returns:
            Dict[str, Any]: 反思结果
        """
        current_time = time.time()
        recent_decisions = [d for d in self.decision_history 
                          if current_time - d["timestamp"] <= time_window]
        
        if not recent_decisions:
            return {"insights": [], "patterns": [], "improvements": []}
            
        # 分析决策模式
        decision_types = defaultdict(int)
        decision_times = []
        goal_consistencies = []
        
        for decision in recent_decisions:
            # 统计决策类型
            d_type = decision["decision"].get("type", "unknown")
            decision_types[d_type] += 1
            
            # 收集决策时间
            decision_times.append(decision["execution_time"])
            
            # 收集目标一致性
            task_id = decision["decision"].get("task_id")
            if task_id in self.goal_consistency_scores:
                goal_consistencies.append(self.goal_consistency_scores[task_id])
        
        # 生成洞察
        insights = []
        
        # 决策类型分布
        most_common_type = max(decision_types.items(), key=lambda x: x[1])
        insights.append(f"最常见的决策类型是 {most_common_type[0]}，占比 {most_common_type[1]/len(recent_decisions):.1%}")
        
        # 决策时间分析
        avg_time = sum(decision_times) / len(decision_times)
        insights.append(f"平均决策时间: {avg_time:.4f}秒")
        
        # 目标一致性分析
        if goal_consistencies:
            avg_consistency = sum(goal_consistencies) / len(goal_consistencies)
            insights.append(f"平均目标一致性: {avg_consistency:.2f}")
        
        # 识别模式
        patterns = []
        
        # 检查决策时间趋势
        if len(decision_times) > 5:
            time_trend = np.polyfit(range(len(decision_times)), decision_times, 1)[0]
            if time_trend > 0:
                patterns.append("决策时间呈上升趋势，可能表明处理复杂度增加")
            elif time_trend < 0:
                patterns.append("决策时间呈下降趋势，表明效率有所提高")
        
        # 检查目标一致性趋势
        if len(goal_consistencies) > 5:
            consistency_trend = np.polyfit(range(len(goal_consistencies)), goal_consistencies, 1)[0]
            if consistency_trend < 0:
                patterns.append("目标一致性呈下降趋势，可能表明目标漂移或冲突")
        
        # 提出改进建议
        improvements = []
        
        # 基于决策时间
        if avg_time > 0.5:  # 假设0.5秒是一个合理的阈值
            improvements.append("考虑优化决策算法以减少处理时间")
            
        # 基于目标一致性
        if goal_consistencies and avg_consistency < 0.8:
            improvements.append("加强目标管理，减少目标冲突")
            
        # 记录反思日志
        reflection = {
            "timestamp": current_time,
            "time_window": time_window,
            "decision_count": len(recent_decisions),
            "insights": insights,
            "patterns": patterns,
            "improvements": improvements
        }
        
        self.reflection_logs.append(reflection)
        
        return reflection
    
    def set_goals(self, goals: List[Any]):
        """
        设置当前系统目标
        
        Args:
            goals (List[Any]): 目标列表
        """
        self.current_goals = goals.copy()
        
    def get_monitoring_metrics(self) -> Dict[str, List[float]]:
        """
        获取监控指标
        
        Returns:
            Dict[str, List[float]]: 监控指标字典
        """
        return self.monitoring_metrics.copy()
    
    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """
        获取智能体状态
        
        Args:
            agent_id (str, optional): 智能体ID，如果为None则返回所有智能体状态
            
        Returns:
            Dict[str, Any]: 智能体状态字典
        """
        if agent_id:
            return self.agent_states.get(agent_id, {}).copy()
        else:
            return {k: v.copy() for k, v in self.agent_states.items()}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        生成系统性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        report = {
            "timestamp": time.time(),
            "agent_performance": self.performance_evaluations.copy(),
            "resource_allocation": self.resource_allocation.copy(),
            "metrics": {
                k: (sum(v) / len(v) if v else 0.0) for k, v in self.monitoring_metrics.items()
            },
            "optimization_suggestions": [s for s in self.optimization_suggestions[-5:]]  # 最近5条建议
        }
        
        return report