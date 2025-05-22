# -*- coding: utf-8 -*-
"""
进化引擎 (Evolution Engine)

负责系统的自我评估与进化机制
实现基于奖励/成功率的智能体评估、有效行为保留和无效行为替换
提供知识注入与迁移接口
"""

import time
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Callable

class EvolutionEngine:
    def __init__(self, memory_system=None, learning_engine=None):
        """
        初始化进化引擎
        
        Args:
            memory_system: 记忆系统实例
            learning_engine: 学习引擎实例
        """
        self.memory_system = memory_system
        self.learning_engine = learning_engine
        self.logger = logging.getLogger("EvolutionEngine")
        
        # 进化参数
        self.evolution_interval = 3600  # 进化检查间隔（秒）
        self.last_evolution_check = time.time()
        self.evolution_threshold = 0.6  # 进化阈值
        self.mutation_rate = 0.1  # 变异率
        
        # 智能体评估数据
        self.agent_evaluations = {}  # {agent_id: evaluation_data}
        self.behavior_patterns = {}  # {pattern_id: pattern_data}
        self.knowledge_repository = {}  # {knowledge_id: knowledge_data}
        
        # 进化历史
        self.evolution_history = []
        self.max_history_size = 100
        
    def evaluate_agent(self, agent_id, performance_metrics):
        """
        评估智能体性能
        
        Args:
            agent_id: 智能体ID
            performance_metrics: 性能指标字典
            
        Returns:
            float: 评估分数 (0-1)
        """
        # 确保智能体评估数据存在
        if agent_id not in self.agent_evaluations:
            self.agent_evaluations[agent_id] = {
                "scores": [],
                "avg_score": 0.0,
                "evaluations_count": 0,
                "last_evaluation": time.time(),
                "improvement_rate": 0.0
            }
            
        eval_data = self.agent_evaluations[agent_id]
        
        # 计算评估分数
        score = self._calculate_score(performance_metrics)
        
        # 更新评估数据
        eval_data["scores"].append(score)
        if len(eval_data["scores"]) > 10:  # 只保留最近10次评估
            eval_data["scores"].pop(0)
            
        eval_data["evaluations_count"] += 1
        eval_data["last_evaluation"] = time.time()
        
        # 计算平均分数
        eval_data["avg_score"] = sum(eval_data["scores"]) / len(eval_data["scores"])
        
        # 计算改进率（如果有足够的评估数据）
        if len(eval_data["scores"]) >= 5:
            old_avg = sum(eval_data["scores"][:len(eval_data["scores"]) // 2]) / (len(eval_data["scores"]) // 2)
            new_avg = sum(eval_data["scores"][len(eval_data["scores"]) // 2:]) / (len(eval_data["scores"]) - len(eval_data["scores"]) // 2)
            eval_data["improvement_rate"] = (new_avg - old_avg) / max(old_avg, 0.001)  # 避免除零
            
        self.logger.info(f"智能体 {agent_id} 评估完成，分数: {score:.4f}, 平均分数: {eval_data['avg_score']:.4f}")
        return score
    
    def _calculate_score(self, metrics):
        """
        根据性能指标计算评估分数
        
        Args:
            metrics: 性能指标字典
            
        Returns:
            float: 评估分数 (0-1)
        """
        # 默认权重
        weights = {
            "success_rate": 0.4,
            "response_time": 0.2,
            "accuracy": 0.3,
            "resource_efficiency": 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        # 计算加权分数
        for metric, weight in weights.items():
            if metric in metrics:
                # 对于响应时间，转换为分数（越短越好）
                if metric == "response_time" and metrics[metric] > 0:
                    # 假设理想响应时间为0.1秒，最差为10秒
                    time_score = max(0, 1 - (metrics[metric] - 0.1) / 9.9)
                    score += weight * time_score
                else:
                    score += weight * metrics[metric]
                    
                total_weight += weight
                
        # 如果没有有效指标，返回默认分数
        if total_weight == 0:
            return 0.5
            
        # 归一化分数
        return score / total_weight
    
    def register_behavior_pattern(self, pattern_id, pattern_data):
        """
        注册行为模式
        
        Args:
            pattern_id: 模式ID
            pattern_data: 模式数据
            
        Returns:
            bool: 是否成功注册
        """
        if pattern_id in self.behavior_patterns:
            self.logger.warning(f"行为模式 {pattern_id} 已存在，将被覆盖")
            
        self.behavior_patterns[pattern_id] = {
            "data": pattern_data,
            "effectiveness": 0.5,  # 初始有效性为中等
            "usage_count": 0,
            "success_count": 0,
            "registered_at": time.time()
        }
        
        self.logger.info(f"行为模式 {pattern_id} 已注册")
        return True
    
    def update_behavior_effectiveness(self, pattern_id, success, context=None):
        """
        更新行为模式的有效性
        
        Args:
            pattern_id: 模式ID
            success: 是否成功
            context: 上下文信息
            
        Returns:
            bool: 是否成功更新
        """
        if pattern_id not in self.behavior_patterns:
            return False
            
        pattern = self.behavior_patterns[pattern_id]
        pattern["usage_count"] += 1
        
        if success:
            pattern["success_count"] += 1
            
        # 更新有效性
        if pattern["usage_count"] > 0:
            pattern["effectiveness"] = pattern["success_count"] / pattern["usage_count"]
            
        # 记录上下文（如果提供）
        if context:
            if "contexts" not in pattern:
                pattern["contexts"] = []
                
            pattern["contexts"].append({
                "data": context,
                "success": success,
                "timestamp": time.time()
            })
            
            # 限制上下文历史大小
            if len(pattern["contexts"]) > 10:
                pattern["contexts"].pop(0)
                
        self.logger.info(f"行为模式 {pattern_id} 有效性已更新: {pattern['effectiveness']:.4f}")
        return True
    
    def check_evolution(self, force=False):
        """
        检查是否需要进行进化
        
        Args:
            force: 是否强制进化
            
        Returns:
            bool: 是否执行了进化
        """
        current_time = time.time()
        
        # 检查是否达到进化间隔
        if not force and current_time - self.last_evolution_check < self.evolution_interval:
            return False
            
        self.last_evolution_check = current_time
        self.logger.info("开始进化检查...")
        
        # 执行进化
        evolution_result = self._evolve_system()
        
        # 记录进化历史
        evolution_record = {
            "timestamp": current_time,
            "changes": evolution_result["changes"],
            "metrics_before": evolution_result["metrics_before"],
            "metrics_after": evolution_result["metrics_after"]
        }
        
        self.evolution_history.append(evolution_record)
        
        # 限制历史记录大小
        if len(self.evolution_history) > self.max_history_size:
            self.evolution_history.pop(0)
            
        self.logger.info(f"进化检查完成，变更数量: {len(evolution_result['changes'])}")
        return True
    
    def _evolve_system(self):
        """
        执行系统进化
        
        Returns:
            dict: 进化结果
        """
        # 收集进化前的系统指标
        metrics_before = self._collect_system_metrics()
        
        changes = []
        
        # 1. 优化行为模式
        behavior_changes = self._optimize_behaviors()
        changes.extend(behavior_changes)
        
        # 2. 优化智能体
        agent_changes = self._optimize_agents()
        changes.extend(agent_changes)
        
        # 3. 知识优化
        knowledge_changes = self._optimize_knowledge()
        changes.extend(knowledge_changes)
        
        # 收集进化后的系统指标
        metrics_after = self._collect_system_metrics()
        
        return {
            "changes": changes,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after
        }
    
    def _collect_system_metrics(self):
        """
        收集系统指标
        
        Returns:
            dict: 系统指标
        """
        # 这里应该收集系统的各种指标
        # 简化实现，返回基本指标
        return {
            "timestamp": time.time(),
            "agent_count": len(self.agent_evaluations),
            "behavior_count": len(self.behavior_patterns),
            "knowledge_count": len(self.knowledge_repository),
            "avg_agent_score": self._get_average_agent_score()
        }
    
    def _get_average_agent_score(self):
        """
        获取平均智能体评分
        
        Returns:
            float: 平均评分
        """
        if not self.agent_evaluations:
            return 0.0
            
        total_score = sum(data["avg_score"] for data in self.agent_evaluations.values())
        return total_score / len(self.agent_evaluations)
    
    def _optimize_behaviors(self):
        """
        优化行为模式
        
        Returns:
            list: 变更列表
        """
        changes = []
        
        # 筛选低效行为模式
        low_effective_patterns = [
            (pattern_id, data) for pattern_id, data in self.behavior_patterns.items()
            if data["usage_count"] >= 5 and data["effectiveness"] < self.evolution_threshold
        ]
        
        # 筛选高效行为模式
        high_effective_patterns = [
            (pattern_id, data) for pattern_id, data in self.behavior_patterns.items()
            if data["usage_count"] >= 5 and data["effectiveness"] > 0.8
        ]
        
        # 替换或改进低效行为
        for pattern_id, data in low_effective_patterns:
            # 尝试从高效模式中学习
            if high_effective_patterns:
                # 选择一个高效模式作为参考
                ref_pattern_id, ref_data = random.choice(high_effective_patterns)
                
                # 创建改进版本
                improved_data = self._improve_behavior(data["data"], ref_data["data"])
                
                # 更新行为模式
                self.behavior_patterns[pattern_id]["data"] = improved_data
                self.behavior_patterns[pattern_id]["effectiveness"] = 0.5  # 重置有效性
                
                changes.append({
                    "type": "behavior_improved",
                    "pattern_id": pattern_id,
                    "reference_pattern": ref_pattern_id,
                    "old_effectiveness": data["effectiveness"],
                    "timestamp": time.time()
                })
                
                self.logger.info(f"行为模式 {pattern_id} 已改进，参考模式: {ref_pattern_id}")
            else:
                # 如果没有高效模式参考，尝试随机变异
                mutated_data = self._mutate_behavior(data["data"])
                
                # 更新行为模式
                self.behavior_patterns[pattern_id]["data"] = mutated_data
                self.behavior_patterns[pattern_id]["effectiveness"] = 0.5  # 重置有效性
                
                changes.append({
                    "type": "behavior_mutated",
                    "pattern_id": pattern_id,
                    "old_effectiveness": data["effectiveness"],
                    "timestamp": time.time()
                })
                
                self.logger.info(f"行为模式 {pattern_id} 已变异")
                
        return changes
    
    def _improve_behavior(self, behavior_data, reference_data):
        """
        改进行为模式
        
        Args:
            behavior_data: 原行为数据
            reference_data: 参考行为数据
            
        Returns:
            dict: 改进后的行为数据
        """
        # 这里应该实现行为模式的改进逻辑
        # 简化实现，合并部分参考行为的特性
        
        # 深拷贝原行为数据
        improved_data = behavior_data.copy() if isinstance(behavior_data, dict) else behavior_data
        
        # 如果是字典类型，尝试合并部分参考行为的特性
        if isinstance(improved_data, dict) and isinstance(reference_data, dict):
            # 随机选择一些参考行为的特性进行合并
            for key, value in reference_data.items():
                if random.random() < 0.3:  # 30%的概率合并该特性
                    improved_data[key] = value
                    
        return improved_data
    
    def _mutate_behavior(self, behavior_data):
        """
        变异行为模式
        
        Args:
            behavior_data: 原行为数据
            
        Returns:
            dict: 变异后的行为数据
        """
        # 这里应该实现行为模式的变异逻辑
        # 简化实现，随机修改部分数据
        
        # 深拷贝原行为数据
        mutated_data = behavior_data.copy() if isinstance(behavior_data, dict) else behavior_data
        
        # 如果是字典类型，尝试随机修改一些值
        if isinstance(mutated_data, dict):
            for key in mutated_data:
                if random.random() < self.mutation_rate:
                    # 根据值的类型进行变异
                    if isinstance(mutated_data[key], (int, float)):
                        # 数值类型，在原值基础上随机增减
                        mutated_data[key] *= random.uniform(0.8, 1.2)
                    elif isinstance(mutated_data[key], str):
                        # 字符串类型，暂不变异
                        pass
                    elif isinstance(mutated_data[key], bool):
                        # 布尔类型，随机翻转
                        mutated_data[key] = not mutated_data[key]
                    elif isinstance(mutated_data[key], list):
                        # 列表类型，随机调整顺序
                        random.shuffle(mutated_data[key])
                        
        return mutated_data
    
    def _optimize_agents(self):
        """
        优化智能体
        
        Returns:
            list: 变更列表
        """
        # 简化实现，仅记录需要优化的智能体
        changes = []
        
        # 筛选表现不佳的智能体
        low_performing_agents = [
            (agent_id, data) for agent_id, data in self.agent_evaluations.items()
            if data["evaluations_count"] >= 5 and data["avg_score"] < self.evolution_threshold
        ]
        
        for agent_id, data in low_performing_agents:
            changes.append({
                "type": "agent_needs_optimization",
                "agent_id": agent_id,
                "avg_score": data["avg_score"],
                "timestamp": time.time()
            })
            
            self.logger.info(f"智能体 {agent_id} 需要优化，平均分数: {data['avg_score']:.4f}")
            
        return changes
    
    def _optimize_knowledge(self):
        """
        优化知识库
        
        Returns:
            list: 变更列表
        """
        # 简化实现，暂不实现知识优化
        return []
    
    def inject_knowledge(self, knowledge_id, knowledge_data, source=None):
        """
        注入知识
        
        Args:
            knowledge_id: 知识ID
            knowledge_data: 知识数据
            source: 知识来源
            
        Returns:
            bool: 是否成功注入
        """
        if knowledge_id in self.knowledge_repository:
            self.logger.warning(f"知识 {knowledge_id} 已存在，将被覆盖")
            
        self.knowledge_repository[knowledge_id] = {
            "data": knowledge_data,
            "source": source,
            "injected_at": time.time(),
            "usage_count": 0,
            "last_used": None
        }
        
        # 如果有记忆系统，也存储到记忆系统中
        if self.memory_system:
            self.memory_system.add_to_long_term({
                "type": "knowledge",
                "id": knowledge_id,
                "content": knowledge_data,
                "source": source,
                "timestamp": time.time()
            })
            
        self.logger.info(f"知识 {knowledge_id} 已注入")
        return True
    
    def use_knowledge(self, knowledge_id):
        """
        使用知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            dict: 知识数据，如果不存在则返回None
        """
        if knowledge_id not in self.knowledge_repository:
            return None
            
        knowledge = self.knowledge_repository[knowledge_id]
        knowledge["usage_count"] += 1
        knowledge["last_used"] = time.time()
        
        return knowledge["data"]
    
    def transfer_knowledge(self, knowledge_id, target_system, transfer_format="json"):
        """
        迁移知识到其他系统
        
        Args:
            knowledge_id: 知识ID
            target_system: 目标系统
            transfer_format: 迁移格式
            
        Returns:
            bool: 是否成功迁移
        """
        if knowledge_id not in self.knowledge_repository:
            return False
            
        knowledge = self.knowledge_repository[knowledge_id]
        
        # 根据迁移格式准备数据
        if transfer_format == "json":
            transfer_data = json.dumps({
                "id": knowledge_id,
                "data": knowledge["data"],
                "source": knowledge["source"],
                "timestamp": time.time()
            })
        else:
            # 其他格式暂不支持
            self.logger.error(f"不支持的迁移格式: {transfer_format}")
            return False
            
        # 这里应该实现向目标系统传输数据的逻辑
        # 简化实现，仅记录日志
        self.logger.info(f"知识 {knowledge_id} 已准备迁移到 {target_system}")
        
        return True
    
    def get_evolution_history(self):
        """
        获取进化历史
        
        Returns:
            list: 进化历史记录
        """
        return self.evolution_history
    
    def get_system_evolution_metrics(self):
        """
        获取系统进化指标
        
        Returns:
            dict: 系统进化指标
        """
        # 计算整体进化指标
        metrics = {
            "timestamp": time.time(),
            "total_agents": len(self.agent_evaluations),
            "total_behaviors": len(self.behavior_patterns),
            "total_knowledge": len(self.knowledge_repository),
            "avg_agent_score": self._get_average_agent_score(),
            "evolution_count": len(self.evolution_history)
        }
        
        # 计算进化趋势（如果有足够的历史记录）
        if len(self.evolution_history) >= 2:
            first_metrics = self.evolution_history[0]["metrics_before"]
            last_metrics = self.evolution_history[-1]["metrics_after"]
            
            if "avg_agent_score" in first_metrics and "avg_agent_score" in last_metrics:
                metrics["score_improvement"] = last_metrics["avg_agent_score"] - first_metrics["avg_agent_score"]
                
        return metrics