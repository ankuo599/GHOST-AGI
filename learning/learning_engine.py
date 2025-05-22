# -*- coding: utf-8 -*-
"""
学习引擎 (Learning Engine)

负责系统的学习能力，包括记忆巩固和技能习得
支持多种学习机制，如监督学习、强化学习和自监督学习
"""

import time
import uuid
import random
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict
import math
import json
import os

class LearningEngine:
    def __init__(self, memory_system=None, event_system=None):
        """
        初始化学习引擎
        
        Args:
            memory_system: 记忆系统实例
            event_system: 事件系统实例
        """
        self.memory_system = memory_system
        self.event_system = event_system
        
        # 学习模型和参数
        self.models = {}
        self.model_configs = {}
        
        # 强化学习状态
        self.rl_states = {}  # 状态记录
        self.rl_actions = {}  # 动作空间
        self.rl_rewards = {}  # 奖励记录
        self.rl_q_values = {}  # Q值表
        
        # 学习统计
        self.learning_stats = {
            "interactions_processed": 0,
            "supervised_updates": 0,
            "reinforcement_updates": 0,
            "self_supervised_updates": 0,
            "knowledge_discoveries": 0,
            "pattern_recognitions": 0
        }
        
        # 学习配置
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        
        # 加载已有模型
        self._load_models()
        
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从交互中学习
        
        Args:
            interaction_data: 交互数据
            
        Returns:
            Dict: 学习结果
        """
        if not interaction_data:
            return {"status": "error", "message": "无交互数据"}
            
        # 更新统计
        self.learning_stats["interactions_processed"] += 1
        
        # 提取交互数据
        user_input = interaction_data.get("user_input", "")
        system_response = interaction_data.get("system_response", "")
        feedback = interaction_data.get("feedback", {})
        
        # 学习结果
        learning_results = {
            "supervised_learning": None,
            "reinforcement_learning": None,
            "self_supervised_learning": None,
            "discoveries": []
        }
        
        # 应用不同的学习方法
        
        # 1. 监督学习 - 如果有明确的反馈
        if feedback and "correctness" in feedback:
            supervised_result = self._apply_supervised_learning(
                input_data=user_input,
                expected_output=feedback.get("expected_output", ""),
                actual_output=system_response
            )
            learning_results["supervised_learning"] = supervised_result
            
        # 2. 强化学习 - 根据奖励信号
        if "reward" in feedback or "sentiment" in feedback:
            reward = feedback.get("reward", 0)
            if "sentiment" in feedback:
                # 从情感分析转换为奖励信号
                sentiment = feedback["sentiment"]
                if sentiment == "positive":
                    reward = 1.0
                elif sentiment == "negative":
                    reward = -1.0
                else:
                    reward = 0.0
                    
            rl_result = self._apply_reinforcement_learning(
                state=interaction_data.get("state", {}),
                action=interaction_data.get("action", ""),
                reward=reward,
                next_state=interaction_data.get("next_state", {})
            )
            learning_results["reinforcement_learning"] = rl_result
            
        # 3. 自监督学习 - 从数据本身学习模式
        self_supervised_result = self._apply_self_supervised_learning(
            text=user_input + " " + system_response,
            metadata=interaction_data.get("metadata", {})
        )
        learning_results["self_supervised_learning"] = self_supervised_result
        
        # 4. 模式发现 - 分析交互以发现新的知识或模式
        discoveries = self._discover_patterns(interaction_data)
        if discoveries:
            learning_results["discoveries"] = discoveries
            
        # 存储学习结果到记忆系统
        if self.memory_system:
            self.memory_system.add_to_short_term({
                "type": "learning_result",
                "interaction_id": interaction_data.get("id", str(uuid.uuid4())),
                "results": learning_results,
                "timestamp": time.time()
            })
            
        # 发布学习事件
        if self.event_system:
            self.event_system.publish("learning.interaction_processed", {
                "interaction_id": interaction_data.get("id", str(uuid.uuid4())),
                "results_summary": {
                    "supervised_applied": learning_results["supervised_learning"] is not None,
                    "reinforcement_applied": learning_results["reinforcement_learning"] is not None,
                    "self_supervised_applied": learning_results["self_supervised_learning"] is not None,
                    "discoveries_count": len(learning_results["discoveries"])
                },
                "timestamp": time.time()
            })
            
        return learning_results
        
    def _apply_supervised_learning(self, input_data: Any, expected_output: Any, 
                                 actual_output: Any) -> Dict[str, Any]:
        """
        应用监督学习
        
        Args:
            input_data: 输入数据
            expected_output: 期望输出
            actual_output: 实际输出
            
        Returns:
            Dict: 学习结果
        """
        # 更新统计
        self.learning_stats["supervised_updates"] += 1
        
        # 构建特征向量 (简化版)
        features = self._extract_features(input_data)
        
        # 计算损失
        loss = self._calculate_loss(expected_output, actual_output)
        
        # 更新模型参数 (简化版)
        model_id = "supervised_basic"
        if model_id not in self.models:
            self.models[model_id] = {"weights": {}, "bias": 0.0}
            
        model = self.models[model_id]
        
        # 简单的梯度下降更新
        for feature, value in features.items():
            if feature not in model["weights"]:
                model["weights"][feature] = 0.0
                
            # 计算梯度并更新权重
            gradient = value * loss
            model["weights"][feature] -= self.learning_rate * gradient
            
        model["bias"] -= self.learning_rate * loss
        
        # 保存更新后的模型
        self._save_models()
        
        return {
            "model_id": model_id,
            "loss": loss,
            "feature_count": len(features),
            "timestamp": time.time()
        }
        
    def _apply_reinforcement_learning(self, state: Dict[str, Any], action: str, 
                                   reward: float, next_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用强化学习
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            
        Returns:
            Dict: 学习结果
        """
        # 更新统计
        self.learning_stats["reinforcement_updates"] += 1
        
        # 状态和动作的字符串表示
        state_key = json.dumps(state, sort_keys=True)
        next_state_key = json.dumps(next_state, sort_keys=True)
        
        # 记录状态和动作
        if state_key not in self.rl_states:
            self.rl_states[state_key] = state
            
        if action not in self.rl_actions:
            self.rl_actions[action] = True
            
        # 更新Q值表
        if state_key not in self.rl_q_values:
            self.rl_q_values[state_key] = {}
            
        if action not in self.rl_q_values[state_key]:
            self.rl_q_values[state_key][action] = 0.0
            
        # 获取下一个状态的最大Q值
        max_next_q = 0
        if next_state_key in self.rl_q_values:
            next_q_values = self.rl_q_values[next_state_key].values()
            if next_q_values:
                max_next_q = max(next_q_values)
                
        # Q-learning更新公式
        current_q = self.rl_q_values[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.rl_q_values[state_key][action] = new_q
        
        # 更新探索率
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
        # 保存Q值表
        self._save_q_values()
        
        return {
            "state": state_key,
            "action": action,
            "reward": reward,
            "old_q": current_q,
            "new_q": new_q,
            "exploration_rate": self.exploration_rate,
            "timestamp": time.time()
        }
        
    def _apply_self_supervised_learning(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用自监督学习
        
        Args:
            text: 文本数据
            metadata: 元数据
            
        Returns:
            Dict: 学习结果
        """
        # 更新统计
        self.learning_stats["self_supervised_updates"] += 1
        
        # 简单的词频分析 (示例自监督任务)
        words = text.lower().split()
        word_counts = defaultdict(int)
        
        for word in words:
            if len(word) > 3:  # 忽略短词
                word_counts[word] += 1
                
        # 提取重要关键词
        important_words = [(word, count) for word, count in word_counts.items() if count > 1]
        important_words.sort(key=lambda x: x[1], reverse=True)
        
        # 更新词频模型
        model_id = "word_frequency"
        if model_id not in self.models:
            self.models[model_id] = {"word_counts": defaultdict(int), "total_words": 0}
            
        model = self.models[model_id]
        
        # 更新词频
        for word, count in word_counts.items():
            model["word_counts"][word] += count
            model["total_words"] += count
            
        # 保存更新后的模型
        self._save_models()
        
        return {
            "model_id": model_id,
            "important_words": important_words[:10],
            "total_words_processed": len(words),
            "timestamp": time.time()
        }
        
    def _discover_patterns(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从交互中发现模式和知识
        
        Args:
            interaction_data: 交互数据
            
        Returns:
            List[Dict]: 发现的模式和知识
        """
        discoveries = []
        
        # 提取用户输入和系统响应
        user_input = interaction_data.get("user_input", "")
        system_response = interaction_data.get("system_response", "")
        context = interaction_data.get("context", {})
        
        # 1. 检查是否有问答模式
        if "?" in user_input and system_response:
            discoveries.append({
                "type": "qa_pattern",
                "question": user_input,
                "answer": system_response,
                "confidence": 0.8
            })
            self.learning_stats["pattern_recognitions"] += 1
            
        # 2. 检查是否包含新的信息
        if self.memory_system:
            # 查询记忆系统，检查是否与现有记忆重复
            similar_memories = self.memory_system.search_by_content(user_input, limit=3)
            
            # 如果没有类似记忆，可能是新知识
            if not similar_memories:
                discoveries.append({
                    "type": "new_knowledge",
                    "content": user_input,
                    "context": context,
                    "confidence": 0.7
                })
                self.learning_stats["knowledge_discoveries"] += 1
                
        # 更多的模式发现逻辑可以在这里添加
        
        return discoveries
        
    def choose_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        使用强化学习策略选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            
        Returns:
            str: 选择的动作
        """
        state_key = json.dumps(state, sort_keys=True)
        
        # 探索 vs. 利用
        if random.random() < self.exploration_rate:
            # 探索: 随机选择
            return random.choice(available_actions)
        else:
            # 利用: 选择Q值最高的动作
            if state_key in self.rl_q_values:
                q_values = self.rl_q_values[state_key]
                
                # 过滤出可用动作的Q值
                available_q_values = {action: q_values.get(action, 0.0) for action in available_actions}
                
                if available_q_values:
                    # 选择Q值最高的动作
                    return max(available_q_values.items(), key=lambda x: x[1])[0]
                    
        # 如果没有Q值信息或没有可用动作，随机选择
        return random.choice(available_actions)
        
    def generate_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        基于学习内容生成洞察
        
        Args:
            limit: 返回洞察数量限制
            
        Returns:
            List[Dict]: 洞察列表
        """
        insights = []
        
        # 1. 基于词频模型生成洞察
        word_model = self.models.get("word_frequency")
        if word_model:
            word_counts = word_model["word_counts"]
            total_words = word_model["total_words"]
            
            # 找出最频繁的词
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            if top_words:
                insights.append({
                    "type": "frequent_topics",
                    "description": "用户经常提及的主题",
                    "data": [(word, count/total_words) for word, count in top_words[:10]],
                    "confidence": 0.8
                })
                
        # 2. 基于强化学习模型生成洞察
        if self.rl_q_values:
            # 找出高奖励的状态-动作对
            high_value_actions = []
            
            for state_key, actions in self.rl_q_values.items():
                for action, q_value in actions.items():
                    if q_value > 0.5:  # 阈值
                        high_value_actions.append({
                            "state": state_key,
                            "action": action,
                            "value": q_value
                        })
                        
            # 按Q值排序
            high_value_actions.sort(key=lambda x: x["value"], reverse=True)
            
            if high_value_actions:
                insights.append({
                    "type": "effective_strategies",
                    "description": "有效的交互策略",
                    "data": high_value_actions[:5],
                    "confidence": 0.7
                })
                
        # 3. 基于记忆系统内容生成洞察
        if self.memory_system:
            # 尝试查找交互模式
            recent_memories = self.memory_system.query_by_time_range(
                time.time() - 86400,  # 过去24小时
                time.time(),
                memory_type="user_input"
            )
            
            if recent_memories:
                user_activity = {
                    "count": len(recent_memories),
                    "avg_length": sum(len(str(m.get("content", ""))) for m in recent_memories) / len(recent_memories)
                }
                
                insights.append({
                    "type": "user_activity",
                    "description": "最近24小时的用户活动",
                    "data": user_activity,
                    "confidence": 0.9
                })
                
        return insights[:limit]
        
    def _extract_features(self, text: str) -> Dict[str, float]:
        """
        从文本提取特征
        
        Args:
            text: 输入文本
            
        Returns:
            Dict: 特征映射
        """
        features = {}
        
        # 简单的单词存在特征
        words = text.lower().split()
        for word in words:
            if len(word) > 3:  # 忽略短词
                features[f"word_{word}"] = 1.0
                
        # 文本长度特征
        features["text_length"] = min(1.0, len(text) / 500)  # 归一化
        
        # 问题标记
        features["is_question"] = 1.0 if "?" in text else 0.0
        
        return features
        
    def _calculate_loss(self, expected: Any, actual: Any) -> float:
        """
        计算损失
        
        Args:
            expected: 期望输出
            actual: 实际输出
            
        Returns:
            float: 损失值
        """
        # 对于文本，使用简单的相似度计算
        if isinstance(expected, str) and isinstance(actual, str):
            # 转换为小写并分词
            expected_words = set(expected.lower().split())
            actual_words = set(actual.lower().split())
            
            # 计算交集大小
            intersection = expected_words.intersection(actual_words)
            
            # 计算相似度
            similarity = len(intersection) / max(1, len(expected_words))
            
            # 损失 = 1 - 相似度
            return 1.0 - similarity
            
        # 对于数值，使用均方误差
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return (expected - actual) ** 2
            
        # 默认损失
        return 1.0
        
    def _save_models(self):
        """保存学习模型"""
        # 实际实现会将模型保存到文件或数据库
        pass
        
    def _load_models(self):
        """加载学习模型"""
        # 实际实现会从文件或数据库加载模型
        pass
        
    def _save_q_values(self):
        """保存Q值表"""
        # 实际实现会将Q值表保存到文件或数据库
        pass
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        获取学习统计信息
        
        Returns:
            Dict: 学习统计
        """
        # 复制基本统计
        stats = dict(self.learning_stats)
        
        # 添加模型信息
        stats["models_count"] = len(self.models)
        
        # 添加Q表信息
        stats["rl_states_count"] = len(self.rl_states)
        stats["rl_actions_count"] = len(self.rl_actions)
        
        # 添加配置信息
        stats["learning_rate"] = self.learning_rate
        stats["exploration_rate"] = self.exploration_rate
        
        # 添加时间戳
        stats["timestamp"] = time.time()
        
        return stats
        
    def update_learning_rate(self, new_rate: float) -> bool:
        """
        更新学习率
        
        Args:
            new_rate: 新的学习率
            
        Returns:
            bool: 是否成功更新
        """
        if new_rate < 0 or new_rate > 1:
            return False
            
        self.learning_rate = new_rate
        
        # 发布学习参数更新事件
        if self.event_system:
            self.event_system.publish("learning.param_updated", {
                "param": "learning_rate",
                "old_value": self.learning_rate,
                "new_value": new_rate,
                "timestamp": time.time()
            })
            
        return True
        
    def update_exploration_rate(self, new_rate: float) -> bool:
        """
        更新探索率
        
        Args:
            new_rate: 新的探索率
            
        Returns:
            bool: 是否成功更新
        """
        if new_rate < 0 or new_rate > 1:
            return False
            
        self.exploration_rate = new_rate
        
        # 发布学习参数更新事件
        if self.event_system:
            self.event_system.publish("learning.param_updated", {
                "param": "exploration_rate",
                "old_value": self.exploration_rate,
                "new_value": new_rate,
                "timestamp": time.time()
            })
            
        return True