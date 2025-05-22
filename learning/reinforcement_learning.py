# -*- coding: utf-8 -*-
"""
强化学习模块 (Reinforcement Learning Module)

实现基于反馈的强化学习算法，使系统能从经验中学习
支持Q-learning、经验回放和策略优化
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque

class ReinforcementLearning:
    def __init__(self, memory_system=None, event_system=None):
        """
        初始化强化学习模块
        
        Args:
            memory_system: 记忆系统实例
            event_system: 事件系统实例
        """
        self.memory_system = memory_system
        self.event_system = event_system
        
        # Q-learning参数
        self.learning_rate = 0.1  # 学习率
        self.discount_factor = 0.9  # 折扣因子
        self.exploration_rate = 0.2  # 探索率
        self.exploration_decay = 0.995  # 探索率衰减
        self.min_exploration_rate = 0.01  # 最小探索率
        
        # Q值表
        self.q_table = {}  # 状态-行动-价值映射
        
        # 经验回放
        self.experience_replay = True  # 是否启用经验回放
        self.replay_buffer = deque(maxlen=1000)  # 经验回放缓冲区
        self.batch_size = 32  # 批处理大小
        
        # 学习统计
        self.training_count = 0  # 训练次数
        self.rewards_history = []  # 奖励历史
        self.performance_history = []  # 性能历史
        
        # 策略优化
        self.policy_optimization = True  # 是否启用策略优化
        self.optimization_interval = 100  # 优化间隔
        
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """
        根据当前状态选择行动（探索或利用）
        
        Args:
            state: 当前状态的字符串表示
            available_actions: 可用行动列表
            
        Returns:
            str: 选择的行动
        """
        # 如果没有可用行动，返回None
        if not available_actions:
            return None
            
        # 探索：随机选择行动
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
            
        # 利用：选择Q值最高的行动
        if state in self.q_table:
            # 获取当前状态下所有可用行动的Q值
            q_values = {action: self.q_table[state].get(action, 0.0) 
                       for action in available_actions}
            
            # 选择Q值最高的行动（如果有多个最高值，随机选择一个）
            max_q = max(q_values.values())
            best_actions = [action for action, q_value in q_values.items() 
                          if q_value == max_q]
            return random.choice(best_actions)
        else:
            # 如果状态未知，随机选择
            return random.choice(available_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, 
                      next_available_actions: List[str]):
        """
        更新Q值表
        
        Args:
            state: 当前状态
            action: 执行的行动
            reward: 获得的奖励
            next_state: 下一个状态
            next_available_actions: 下一个状态的可用行动
        """
        # 确保状态在Q表中
        if state not in self.q_table:
            self.q_table[state] = {}
            
        # 获取当前Q值
        current_q = self.q_table[state].get(action, 0.0)
        
        # 计算下一个状态的最大Q值
        max_next_q = 0.0
        if next_state and next_available_actions:
            if next_state in self.q_table:
                next_q_values = [self.q_table[next_state].get(next_action, 0.0) 
                               for next_action in next_available_actions]
                if next_q_values:
                    max_next_q = max(next_q_values)
        
        # 计算新的Q值（Q-learning公式）
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # 更新Q表
        self.q_table[state][action] = new_q
        
        # 记录奖励
        self.rewards_history.append(reward)
        
        # 增加训练计数
        self.training_count += 1
        
        # 衰减探索率
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)
        
        # 如果启用了策略优化，定期优化策略
        if self.policy_optimization and self.training_count % self.optimization_interval == 0:
            self.optimize_policy()
    
    def add_experience(self, state: str, action: str, reward: float, next_state: str, 
                      next_available_actions: List[str]):
        """
        将经验添加到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的行动
            reward: 获得的奖励
            next_state: 下一个状态
            next_available_actions: 下一个状态的可用行动
        """
        if self.experience_replay:
            self.replay_buffer.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "next_available_actions": next_available_actions
            })
    
    def train_from_replay(self):
        """
        从经验回放缓冲区中随机抽样进行训练
        """
        if not self.experience_replay or len(self.replay_buffer) < self.batch_size:
            return
            
        # 随机抽取一批经验
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 对每个经验更新Q值
        for experience in batch:
            self.update_q_value(
                experience["state"],
                experience["action"],
                experience["reward"],
                experience["next_state"],
                experience["next_available_actions"]
            )
    
    def optimize_policy(self):
        """
        优化学习策略（学习率、折扣因子等）
        """
        # 计算最近的平均奖励
        recent_rewards = self.rewards_history[-100:] if len(self.rewards_history) > 100 else self.rewards_history
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        
        # 记录性能
        self.performance_history.append({
            "timestamp": time.time(),
            "avg_reward": avg_reward,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "q_table_size": len(self.q_table)
        })
        
        # 根据性能调整学习参数
        if len(self.performance_history) >= 2:
            prev_performance = self.performance_history[-2]["avg_reward"]
            current_performance = avg_reward
            
            # 如果性能提升，略微增加学习率
            if current_performance > prev_performance:
                self.learning_rate = min(0.5, self.learning_rate * 1.05)
            # 如果性能下降，略微减少学习率
            else:
                self.learning_rate = max(0.01, self.learning_rate * 0.95)
    
    def get_state_representation(self, state_dict: Dict[str, Any]) -> str:
        """
        将状态字典转换为字符串表示
        
        Args:
            state_dict: 状态字典
            
        Returns:
            str: 状态的字符串表示
        """
        # 对字典键进行排序，确保相同状态有相同的字符串表示
        sorted_items = sorted(state_dict.items())
        return str(sorted_items)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取强化学习统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "q_table_size": len(self.q_table),
            "states_count": len(self.q_table),
            "training_count": self.training_count,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "avg_reward": sum(self.rewards_history[-100:]) / max(1, len(self.rewards_history[-100:])) if self.rewards_history else 0,
            "experience_buffer_size": len(self.replay_buffer) if self.experience_replay else 0
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        保存Q值表到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            import json
            # 将Q表转换为可序列化的格式
            serializable_q_table = {}
            for state, actions in self.q_table.items():
                serializable_q_table[state] = actions
                
            # 保存到文件
            with open(filepath, 'w') as f:
                json.dump({
                    "q_table": serializable_q_table,
                    "params": {
                        "learning_rate": self.learning_rate,
                        "discount_factor": self.discount_factor,
                        "exploration_rate": self.exploration_rate
                    },
                    "stats": self.get_stats(),
                    "timestamp": time.time()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        从文件加载Q值表
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # 加载Q表
            self.q_table = data["q_table"]
            
            # 加载参数
            params = data.get("params", {})
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.discount_factor = params.get("discount_factor", self.discount_factor)
            self.exploration_rate = params.get("exploration_rate", self.exploration_rate)
            
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False