# -*- coding: utf-8 -*-
"""
增强版学习引擎 (Enhanced Learning Engine)

实现强化学习算法、少样本学习机制和知识迁移功能
支持自适应学习率调整和学习效果评估
"""

import time
import random
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from collections import deque

# 导入基础学习引擎
from .learning_engine import LearningEngine
from .reinforcement_learning import ReinforcementLearning

class EnhancedLearningEngine(LearningEngine):
    def __init__(self, memory_system=None):
        """
        初始化增强版学习引擎
        
        Args:
            memory_system: 记忆系统实例，用于获取和存储学习数据
        """
        super().__init__(memory_system)
        
        # 强化学习增强组件
        self.rl_module = ReinforcementLearning()
        self.rl_enabled = True
        
        # 少样本学习增强
        self.few_shot_learning = {
            "active": True,
            "examples": {},  # 示例库 {task_type: [examples]}
            "similarity_threshold": 0.75,  # 相似度阈值
            "max_examples_per_task": 10,  # 每个任务的最大示例数
            "adaptation_speed": 0.3,  # 适应速度
            "task_embeddings": {},  # 任务嵌入 {task_id: embedding}
            "prototype_vectors": {},  # 原型向量 {task_type: vector}
            "similarity_cache": {}  # 相似度缓存 {(task_id1, task_id2): similarity}
        }
        
        # 知识迁移机制
        self.knowledge_transfer = {
            "active": True,
            "domain_knowledge": {},  # 领域知识 {domain: knowledge}
            "transfer_mappings": {},  # 迁移映射 {source_domain: {target_domain: mapping}}
            "transfer_history": [],  # 迁移历史
            "transfer_success_rate": {},  # 迁移成功率 {(source, target): rate}
            "domain_similarities": {}  # 领域相似度 {(domain1, domain2): similarity}
        }
        
        # 自适应学习率
        self.adaptive_learning = {
            "active": True,
            "base_learning_rate": 0.01,  # 基础学习率
            "min_learning_rate": 0.001,  # 最小学习率
            "max_learning_rate": 0.1,  # 最大学习率
            "adaptation_factor": 0.05,  # 适应因子
            "performance_window": deque(maxlen=10),  # 性能窗口
            "last_adjustment": time.time()  # 上次调整时间
        }
        
        # 学习效果评估
        self.learning_evaluation = {
            "active": True,
            "metrics": {},  # 评估指标 {metric_name: value}
            "baseline_performance": {},  # 基准性能
            "evaluation_history": [],  # 评估历史
            "improvement_threshold": 0.05  # 改进阈值
        }
        
    def learn_from_feedback(self, state: str, action: str, reward: float, next_state: str, 
                           available_actions: List[str]):
        """
        从反馈中学习（强化学习）
        
        Args:
            state: 当前状态
            action: 执行的行动
            reward: 获得的奖励
            next_state: 下一个状态
            available_actions: 可用行动列表
            
        Returns:
            bool: 学习是否成功
        """
        if not self.rl_enabled or not self.rl_module:
            return False
            
        # 更新Q值
        self.rl_module.update_q_value(state, action, reward, next_state, available_actions)
        
        # 记录学习历史
        self.learning_history.append({
            "type": "reinforcement",
            "state": state,
            "action": action,
            "reward": reward,
            "timestamp": time.time()
        })
        
        # 更新探索率（随时间衰减）
        self.rl_module.exploration_rate *= self.rl_module.exploration_decay
        self.rl_module.exploration_rate = max(self.rl_module.exploration_rate, 
                                            self.rl_module.min_exploration_rate)
        
        # 添加到经验回放缓冲区
        if self.rl_module.experience_replay:
            self.rl_module.replay_buffer.append((state, action, reward, next_state, available_actions))
            
            # 如果缓冲区足够大，进行批量学习
            if len(self.rl_module.replay_buffer) >= self.rl_module.batch_size:
                self._batch_learning()
                
        return True
    
    def _batch_learning(self, batch_size=None):
        """
        从经验回放缓冲区中批量学习
        
        Args:
            batch_size: 批处理大小，如果为None则使用默认值
        """
        if batch_size is None:
            batch_size = self.rl_module.batch_size
            
        if len(self.rl_module.replay_buffer) < batch_size:
            return
            
        # 随机采样批次
        batch = random.sample(self.rl_module.replay_buffer, batch_size)
        
        # 批量更新Q值
        for state, action, reward, next_state, available_actions in batch:
            self.rl_module.update_q_value(state, action, reward, next_state, available_actions)
            
        self.rl_module.training_count += 1
    
    def few_shot_learn(self, task_type: str, example: Dict[str, Any], result: Any):
        """
        少样本学习：从少量示例中学习新任务
        
        Args:
            task_type: 任务类型
            example: 示例数据
            result: 示例结果
            
        Returns:
            bool: 学习是否成功
        """
        if not self.few_shot_learning["active"]:
            return False
            
        # 确保任务类型在示例库中
        if task_type not in self.few_shot_learning["examples"]:
            self.few_shot_learning["examples"][task_type] = []
            
        # 创建示例记录
        example_record = {
            "input": example,
            "output": result,
            "embedding": self._generate_task_embedding(example),
            "added_at": time.time()
        }
        
        # 添加到示例库
        examples = self.few_shot_learning["examples"][task_type]
        examples.append(example_record)
        
        # 如果超过最大示例数，移除最旧的示例
        if len(examples) > self.few_shot_learning["max_examples_per_task"]:
            examples.pop(0)
            
        # 更新任务原型向量
        self._update_prototype_vector(task_type)
        
        # 记录学习历史
        self.learning_history.append({
            "type": "few_shot",
            "task_type": task_type,
            "timestamp": time.time()
        })
        
        return True
    
    def _generate_task_embedding(self, task_data: Dict[str, Any]) -> np.ndarray:
        """
        生成任务嵌入向量
        
        Args:
            task_data: 任务数据
            
        Returns:
            np.ndarray: 嵌入向量
        """
        # 实际应用中应使用适当的嵌入模型
        # 这里使用简化的随机向量表示
        task_id = hash(json.dumps(task_data, sort_keys=True))
        
        # 如果已有嵌入，直接返回
        if task_id in self.few_shot_learning["task_embeddings"]:
            return self.few_shot_learning["task_embeddings"][task_id]
            
        # 生成随机嵌入向量（实际应用中应替换为真实嵌入）
        embedding = np.random.randn(64)  # 64维嵌入向量
        embedding = embedding / np.linalg.norm(embedding)  # 归一化
        
        # 缓存嵌入向量
        self.few_shot_learning["task_embeddings"][task_id] = embedding
        
        return embedding
    
    def _update_prototype_vector(self, task_type: str):
        """
        更新任务类型的原型向量
        
        Args:
            task_type: 任务类型
        """
        if task_type not in self.few_shot_learning["examples"] or \
           not self.few_shot_learning["examples"][task_type]:
            return
            
        # 获取所有示例的嵌入向量
        embeddings = [example["embedding"] for example in self.few_shot_learning["examples"][task_type]]
        
        # 计算平均向量作为原型
        prototype = np.mean(embeddings, axis=0)
        prototype = prototype / np.linalg.norm(prototype)  # 归一化
        
        # 更新原型向量
        self.few_shot_learning["prototype_vectors"][task_type] = prototype
    
    def predict_from_few_shot(self, task_type: str, input_data: Dict[str, Any]) -> Tuple[Any, float]:
        """
        基于少样本学习进行预测
        
        Args:
            task_type: 任务类型
            input_data: 输入数据
            
        Returns:
            Tuple[Any, float]: (预测结果, 置信度)
        """
        if not self.few_shot_learning["active"] or \
           task_type not in self.few_shot_learning["examples"] or \
           not self.few_shot_learning["examples"][task_type]:
            return None, 0.0
            
        # 生成输入数据的嵌入向量
        input_embedding = self._generate_task_embedding(input_data)
        
        # 计算与所有示例的相似度
        examples = self.few_shot_learning["examples"][task_type]
        similarities = []
        
        for example in examples:
            example_embedding = example["embedding"]
            similarity = np.dot(input_embedding, example_embedding)  # 余弦相似度
            similarities.append((similarity, example))
            
        # 按相似度降序排序
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # 如果最高相似度低于阈值，返回None
        if similarities[0][0] < self.few_shot_learning["similarity_threshold"]:
            return None, 0.0
            
        # 获取最相似示例的结果
        best_match = similarities[0][1]
        confidence = similarities[0][0]
        
        return best_match["output"], confidence
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, knowledge: Dict[str, Any]) -> bool:
        """
        知识迁移：将一个领域的知识迁移到另一个领域
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            knowledge: 知识数据
            
        Returns:
            bool: 迁移是否成功
        """
        if not self.knowledge_transfer["active"]:
            return False
            
        # 确保领域知识存在
        if source_domain not in self.knowledge_transfer["domain_knowledge"]:
            self.knowledge_transfer["domain_knowledge"][source_domain] = {}
            
        if target_domain not in self.knowledge_transfer["domain_knowledge"]:
            self.knowledge_transfer["domain_knowledge"][target_domain] = {}
            
        # 确保迁移映射存在
        if source_domain not in self.knowledge_transfer["transfer_mappings"]:
            self.knowledge_transfer["transfer_mappings"][source_domain] = {}
            
        if target_domain not in self.knowledge_transfer["transfer_mappings"][source_domain]:
            # 创建默认映射（实际应用中应基于领域特性创建）
            self.knowledge_transfer["transfer_mappings"][source_domain][target_domain] = {
                "attribute_mapping": {},  # 属性映射
                "concept_mapping": {},   # 概念映射
                "rule_mapping": {}       # 规则映射
            }
            
        # 应用迁移映射
        mapping = self.knowledge_transfer["transfer_mappings"][source_domain][target_domain]
        transferred_knowledge = self._apply_knowledge_mapping(knowledge, mapping)
        
        # 更新目标领域知识
        self.knowledge_transfer["domain_knowledge"][target_domain].update(transferred_knowledge)
        
        # 记录迁移历史
        transfer_record = {
            "source": source_domain,
            "target": target_domain,
            "knowledge_size": len(knowledge),
            "timestamp": time.time()
        }
        self.knowledge_transfer["transfer_history"].append(transfer_record)
        
        return True
    
    def _apply_knowledge_mapping(self, knowledge: Dict[str, Any], mapping: Dict[str, Dict]) -> Dict[str, Any]:
        """
        应用知识映射
        
        Args:
            knowledge: 源知识
            mapping: 领域映射
            
        Returns:
            Dict[str, Any]: 映射后的知识
        """
        result = {}
        
        # 应用属性映射
        for key, value in knowledge.items():
            # 如果有明确的映射，使用映射后的键
            if key in mapping["attribute_mapping"]:
                mapped_key = mapping["attribute_mapping"][key]
            else:
                mapped_key = key
                
            # 递归处理嵌套字典
            if isinstance(value, dict):
                result[mapped_key] = self._apply_knowledge_mapping(value, mapping)
            else:
                result[mapped_key] = value
                
        return result
    
    def adjust_learning_rate(self):
        """
        自适应调整学习率
        
        Returns:
            float: 调整后的学习率
        """
        if not self.adaptive_learning["active"]:
            return self.learning_rate
            
        # 检查是否需要调整
        current_time = time.time()
        if current_time - self.adaptive_learning["last_adjustment"] < 300:  # 至少5分钟调整一次
            return self.learning_rate
            
        # 获取性能窗口数据
        performance_window = self.adaptive_learning["performance_window"]
        if len(performance_window) < 2:
            return self.learning_rate
            
        # 计算性能趋势
        recent_performance = list(performance_window)[-3:]  # 最近3次性能
        if len(recent_performance) < 2:
            return self.learning_rate
            
        # 计算性能变化率
        performance_change = (recent_performance[-1] - recent_performance[0]) / max(0.001, recent_performance[0])
        
        # 根据性能变化调整学习率
        if performance_change > 0.05:  # 性能显著提升
            # 增加学习率以加速学习
            new_learning_rate = self.learning_rate * (1 + self.adaptive_learning["adaptation_factor"])
        elif performance_change < -0.05:  # 性能显著下降
            # 减小学习率以稳定学习
            new_learning_rate = self.learning_rate * (1 - self.adaptive_learning["adaptation_factor"])
        else:  # 性能稳定
            # 保持学习率不变
            new_learning_rate = self.learning_rate
            
        # 确保学习率在合理范围内
        new_learning_rate = max(self.adaptive_learning["min_learning_rate"], 
                              min(self.adaptive_learning["max_learning_rate"], new_learning_rate))
        
        # 更新学习率
        self.learning_rate = new_learning_rate
        self.adaptive_learning["last_adjustment"] = current_time
        
        return new_learning_rate
    
    def evaluate_learning_effect(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        评估学习效果
        
        Args:
            metrics: 性能指标
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.learning_evaluation["active"]:
            return {"status": "inactive"}
            
        # 更新评估指标
        self.learning_evaluation["metrics"].update(metrics)
        
        # 如果没有基准性能，将当前性能设为基准
        if not self.learning_evaluation["baseline_performance"]:
            self.learning_evaluation["baseline_performance"] = metrics.copy()
            return {"status": "baseline_set", "baseline": metrics}
            
        # 计算相对于基准的改进
        improvements = {}
        for metric, value in metrics.items():
            if metric in self.learning_evaluation["baseline_performance"]:
                baseline = self.learning_evaluation["baseline_performance"][metric]
                if baseline != 0:
                    relative_improvement = (value - baseline) / abs(baseline)
                else:
                    relative_improvement = value
                improvements[metric] = relative_improvement
                
        # 计算总体改进
        if improvements:
            overall_improvement = sum(improvements.values()) / len(improvements)
        else:
            overall_improvement = 0.0
            
        # 记录评估结果
        evaluation_result = {
            "timestamp": time.time(),
            "metrics": metrics.copy(),
            "improvements": improvements,
            "overall_improvement": overall_improvement
        }
        self.learning_evaluation["evaluation_history"].append(evaluation_result)
        
        # 如果评估历史过长，移除最旧的记录
        if len(self.learning_evaluation["evaluation_history"]) > 20:
            self.learning_evaluation["evaluation_history"].pop(0)
            
        # 判断是否达到改进阈值
        significant_improvement = overall_improvement > self.learning_evaluation["improvement_threshold"]
        
        return {
            "status": "evaluated",
            "improvements": improvements,
            "overall_improvement": overall_improvement,
            "significant_improvement": significant_improvement
        }