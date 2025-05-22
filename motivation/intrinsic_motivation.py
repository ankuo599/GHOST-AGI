"""
GHOST-AGI 内在动机系统

该模块实现系统的好奇心驱动探索和自主目标设定，使GHOST-AGI能够展现主动性与自发学习行为。
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict

class IntrinsicMotivation:
    """内在动机系统，使系统能够自主探索并设定目标"""
    
    def __init__(self, 
                 curiosity_weight: float = 0.7,
                 novelty_threshold: float = 0.6,
                 competence_weight: float = 0.5,
                 goal_horizon: int = 5,
                 logger: Optional[logging.Logger] = None):
        """
        初始化内在动机系统
        
        Args:
            curiosity_weight: 好奇心权重
            novelty_threshold: 新颖性阈值
            competence_weight: 能力进步权重
            goal_horizon: 目标规划视野
            logger: 日志记录器
        """
        # 动机参数
        self.curiosity_weight = curiosity_weight
        self.novelty_threshold = novelty_threshold
        self.competence_weight = competence_weight
        self.goal_horizon = goal_horizon
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 内在奖励系统
        self.prediction_errors = defaultdict(list)  # 预测错误历史
        self.novelty_scores = defaultdict(float)    # 新颖性分数
        self.competence_progress = defaultdict(list)  # 能力进步历史
        
        # 目标管理
        self.current_goals = []  # 当前目标列表
        self.goal_history = []   # 目标历史
        self.goal_achievements = defaultdict(float)  # 目标成就度
        
        # 探索策略
        self.exploration_strategies = {
            "random_exploration": self._random_exploration,
            "uncertainty_based": self._uncertainty_based_exploration,
            "competence_based": self._competence_based_exploration,
            "novelty_seeking": self._novelty_seeking_exploration
        }
        self.active_exploration_strategy = "novelty_seeking"
        
        # 统计信息
        self.motivation_stats = {
            "total_goals_generated": 0,
            "goals_achieved": 0,
            "exploration_episodes": 0,
            "curiosity_triggers": 0,
            "last_activity": time.time()
        }
        
        self.logger.info("内在动机系统初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("IntrinsicMotivation")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("intrinsic_motivation.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def update_prediction_error(self, context_id: str, error: float) -> None:
        """
        更新预测错误
        
        Args:
            context_id: 上下文ID
            error: 预测错误值
        """
        self.prediction_errors[context_id].append(error)
        # 保持历史记录不过长
        if len(self.prediction_errors[context_id]) > 100:
            self.prediction_errors[context_id] = self.prediction_errors[context_id][-100:]
    
    def update_competence(self, skill_id: str, competence_score: float) -> None:
        """
        更新技能能力值
        
        Args:
            skill_id: 技能ID
            competence_score: 能力分数
        """
        self.competence_progress[skill_id].append(competence_score)
        # 保持历史记录不过长
        if len(self.competence_progress[skill_id]) > 100:
            self.competence_progress[skill_id] = self.competence_progress[skill_id][-100:]
    
    def calculate_curiosity(self, context_id: str) -> float:
        """
        计算特定上下文的好奇心得分
        
        Args:
            context_id: 上下文ID
            
        Returns:
            好奇心得分
        """
        if context_id not in self.prediction_errors or not self.prediction_errors[context_id]:
            return 0.0
        
        # 获取最近的预测错误
        recent_errors = self.prediction_errors[context_id][-10:]
        
        # 计算错误的平均值和趋势
        avg_error = sum(recent_errors) / len(recent_errors)
        
        # 计算错误趋势
        if len(recent_errors) >= 2:
            error_trend = recent_errors[-1] - recent_errors[0]
        else:
            error_trend = 0
        
        # 好奇心分数结合当前错误和进步趋势
        # 高预测错误但有进步 = 高好奇心
        if error_trend < 0:  # 错误下降 = 正在学习
            curiosity_score = avg_error * (1.0 + abs(error_trend))
        else:  # 错误上升或不变
            curiosity_score = avg_error * 0.5
        
        return min(1.0, max(0.0, curiosity_score))
    
    def calculate_novelty(self, state_representation: Any) -> float:
        """
        计算状态的新颖性得分
        
        Args:
            state_representation: 状态表示
            
        Returns:
            新颖性得分
        """
        # 实际实现应该比较状态与已知状态的距离
        # 这里使用简化实现
        state_id = str(hash(str(state_representation)))
        
        if state_id in self.novelty_scores:
            # 随着时间推移，新颖性会自然衰减
            self.novelty_scores[state_id] *= 0.95
        else:
            # 新状态具有高新颖性
            self.novelty_scores[state_id] = 1.0
        
        return self.novelty_scores[state_id]
    
    def calculate_competence_progress(self, skill_id: str) -> float:
        """
        计算技能的能力进步
        
        Args:
            skill_id: 技能ID
            
        Returns:
            能力进步分数
        """
        if skill_id not in self.competence_progress or len(self.competence_progress[skill_id]) < 2:
            return 0.0
        
        # 获取最近的能力记录
        recent_scores = self.competence_progress[skill_id][-10:]
        
        if len(recent_scores) < 2:
            return 0.0
        
        # 计算进步趋势
        progress = recent_scores[-1] - recent_scores[0]
        
        # 正规化进步分数
        normalized_progress = np.tanh(progress * 3)  # tanh限制在-1到1之间
        
        # 只对正进步给予奖励
        if normalized_progress > 0:
            return normalized_progress
        return 0.0
    
    def generate_intrinsic_reward(self, 
                                context_id: str, 
                                state_representation: Any, 
                                skill_id: Optional[str] = None) -> Dict[str, float]:
        """
        生成内在奖励
        
        Args:
            context_id: 上下文ID
            state_representation: 状态表示
            skill_id: 技能ID
            
        Returns:
            内在奖励值
        """
        # 计算不同类型的内在奖励
        curiosity_reward = self.calculate_curiosity(context_id) * self.curiosity_weight
        novelty_reward = self.calculate_novelty(state_representation)
        
        competence_reward = 0.0
        if skill_id:
            competence_reward = self.calculate_competence_progress(skill_id) * self.competence_weight
        
        # 总内在奖励
        total_reward = curiosity_reward + novelty_reward + competence_reward
        
        # 记录统计
        if curiosity_reward > 0.3:
            self.motivation_stats["curiosity_triggers"] += 1
        
        rewards = {
            "curiosity": curiosity_reward,
            "novelty": novelty_reward,
            "competence": competence_reward,
            "total": total_reward
        }
        
        self.logger.debug(f"为上下文 {context_id} 生成内在奖励: {rewards}")
        return rewards
    
    def select_exploration_target(self, 
                                possible_targets: List[Dict[str, Any]], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        基于内在动机选择探索目标
        
        Args:
            possible_targets: 可能的探索目标列表
            context: 上下文信息
            
        Returns:
            选择的探索目标
        """
        if not possible_targets:
            self.logger.warning("没有可选的探索目标")
            return {}
        
        self.logger.info(f"从 {len(possible_targets)} 个候选目标中选择探索目标")
        
        # 使用当前活跃的探索策略
        strategy = self.exploration_strategies.get(
            self.active_exploration_strategy, 
            self.exploration_strategies["novelty_seeking"]
        )
        
        selected_target = strategy(possible_targets, context)
        
        # 更新统计
        self.motivation_stats["exploration_episodes"] += 1
        self.motivation_stats["last_activity"] = time.time()
        
        return selected_target
    
    def _random_exploration(self, 
                          possible_targets: List[Dict[str, Any]], 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """随机探索策略"""
        return np.random.choice(possible_targets)
    
    def _uncertainty_based_exploration(self, 
                                     possible_targets: List[Dict[str, Any]], 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """基于不确定性的探索"""
        if not context or "context_id" not in context:
            return self._random_exploration(possible_targets)
        
        # 计算每个目标的不确定性
        context_id = context["context_id"]
        targets_with_scores = []
        
        for target in possible_targets:
            target_id = target.get("id", str(hash(str(target))))
            uncertainty_score = self.calculate_curiosity(f"{context_id}_{target_id}")
            targets_with_scores.append((target, uncertainty_score))
        
        # 按不确定性排序
        sorted_targets = sorted(targets_with_scores, key=lambda x: x[1], reverse=True)
        
        # 选择不确定性最高的目标
        return sorted_targets[0][0]
    
    def _competence_based_exploration(self, 
                                    possible_targets: List[Dict[str, Any]], 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """基于能力进步的探索"""
        targets_with_scores = []
        
        for target in possible_targets:
            skill_id = target.get("skill_id", target.get("id", "unknown"))
            progress_score = self.calculate_competence_progress(skill_id)
            targets_with_scores.append((target, progress_score))
        
        # 按能力进步排序
        sorted_targets = sorted(targets_with_scores, key=lambda x: x[1], reverse=True)
        
        # 一定概率选择进步空间最大的目标
        if np.random.random() < 0.7 and sorted_targets[0][1] > 0:
            return sorted_targets[0][0]
        else:
            # 否则随机选择
            return np.random.choice(possible_targets)
    
    def _novelty_seeking_exploration(self, 
                                   possible_targets: List[Dict[str, Any]], 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """基于新颖性的探索"""
        targets_with_scores = []
        
        for target in possible_targets:
            novelty_score = self.calculate_novelty(target)
            targets_with_scores.append((target, novelty_score))
        
        # 按新颖性排序
        sorted_targets = sorted(targets_with_scores, key=lambda x: x[1], reverse=True)
        
        # 选择新颖性最高的目标
        if sorted_targets[0][1] > self.novelty_threshold:
            return sorted_targets[0][0]
        else:
            # 如果没有足够新颖的目标，随机选择
            return np.random.choice(possible_targets)
    
    def generate_goal(self, 
                     domain_knowledge: Dict[str, Any], 
                     current_state: Dict[str, Any], 
                     skill_inventory: Dict[str, float]) -> Dict[str, Any]:
        """
        生成自主目标
        
        Args:
            domain_knowledge: 领域知识
            current_state: 当前状态
            skill_inventory: 技能库存与能力水平
            
        Returns:
            生成的目标
        """
        self.logger.info("生成自主目标")
        
        # 目标候选项
        goal_candidates = []
        
        # 基于当前状态与领域知识生成目标候选
        goal_candidates.extend(self._generate_knowledge_based_goals(domain_knowledge, current_state))
        
        # 基于能力水平生成目标候选
        goal_candidates.extend(self._generate_competence_based_goals(skill_inventory))
        
        # 基于好奇心生成目标候选
        goal_candidates.extend(self._generate_curiosity_based_goals(current_state))
        
        if not goal_candidates:
            self.logger.warning("无法生成目标候选项")
            return {}
        
        # 评估目标候选项
        evaluated_goals = self._evaluate_goal_candidates(goal_candidates, current_state, skill_inventory)
        
        # 选择最佳目标
        if evaluated_goals:
            best_goal = max(evaluated_goals, key=lambda g: g["score"])
            
            # 更新目标列表
            goal_id = best_goal.get("id", f"goal_{int(time.time())}")
            goal_data = {
                "id": goal_id,
                "description": best_goal["description"],
                "type": best_goal["type"],
                "creation_time": time.time(),
                "estimated_difficulty": best_goal.get("difficulty", 0.5),
                "estimated_value": best_goal.get("value", 0.5),
                "score": best_goal["score"],
                "status": "active"
            }
            
            self.current_goals.append(goal_data)
            self.goal_history.append(goal_data.copy())
            
            # 更新统计
            self.motivation_stats["total_goals_generated"] += 1
            
            self.logger.info(f"生成新目标: {goal_data['description']}")
            return goal_data
        else:
            self.logger.warning("无法评估目标候选项")
            return {}
    
    def _generate_knowledge_based_goals(self, 
                                      domain_knowledge: Dict[str, Any], 
                                      current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于领域知识生成目标"""
        goals = []
        
        # 实际实现会基于领域知识图谱或本体推理生成目标
        # 简化版本
        
        return goals
    
    def _generate_competence_based_goals(self, skill_inventory: Dict[str, float]) -> List[Dict[str, Any]]:
        """基于能力水平生成目标"""
        goals = []
        
        # 实际实现会基于能力边界和最近发展区推理生成目标
        # 简化版本
        
        return goals
    
    def _generate_curiosity_based_goals(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于好奇心生成目标"""
        goals = []
        
        # 实际实现会基于预测模型的不确定区域生成目标
        # 简化版本
        
        return goals
    
    def _evaluate_goal_candidates(self, 
                                candidates: List[Dict[str, Any]], 
                                current_state: Dict[str, Any],
                                skill_inventory: Dict[str, float]) -> List[Dict[str, Any]]:
        """评估目标候选项"""
        evaluated_candidates = []
        
        for candidate in candidates:
            # 评估实现目标的难度
            difficulty = candidate.get("difficulty", 0.5)
            
            # 评估目标价值
            value = candidate.get("value", 0.5)
            
            # 评估新颖性
            novelty = self.calculate_novelty(candidate)
            
            # 如果目标需要特定技能，评估能力差距
            skill_match = 1.0
            if "required_skills" in candidate:
                skill_gaps = []
                for skill, required_level in candidate["required_skills"].items():
                    current_level = skill_inventory.get(skill, 0.0)
                    if current_level < required_level:
                        skill_gaps.append(required_level - current_level)
                
                if skill_gaps:
                    avg_gap = sum(skill_gaps) / len(skill_gaps)
                    # 较小的差距是好的，但过大的差距不是
                    if avg_gap > 0.5:
                        skill_match = 1.0 - avg_gap
                    else:
                        skill_match = 1.0 - avg_gap/2  # 小差距仍有较高匹配度
            
            # 计算最终分数
            # 公式：价值 * (1-难度权重*难度) * 新颖性 * 技能匹配度
            difficulty_weight = 0.7  # 难度的权重系数
            score = value * (1 - difficulty_weight * difficulty) * novelty * skill_match
            
            # 添加评估结果
            candidate_copy = candidate.copy()
            candidate_copy["score"] = score
            candidate_copy["evaluated_difficulty"] = difficulty
            candidate_copy["evaluated_value"] = value
            candidate_copy["evaluated_novelty"] = novelty
            candidate_copy["evaluated_skill_match"] = skill_match
            
            evaluated_candidates.append(candidate_copy)
        
        return evaluated_candidates
    
    def update_goal_achievement(self, goal_id: str, achievement_level: float) -> Dict[str, Any]:
        """
        更新目标成就度
        
        Args:
            goal_id: 目标ID
            achievement_level: 成就度 (0.0-1.0)
            
        Returns:
            更新结果
        """
        self.logger.info(f"更新目标 {goal_id} 的成就度: {achievement_level}")
        
        # 找到对应目标
        target_goal = None
        for goal in self.current_goals:
            if goal["id"] == goal_id:
                target_goal = goal
                break
        
        if not target_goal:
            self.logger.warning(f"目标 {goal_id} 不存在")
            return {"error": f"目标 {goal_id} 不存在"}
        
        # 更新成就度
        self.goal_achievements[goal_id] = achievement_level
        
        # 如果目标成就度满足条件，标记为已完成
        if achievement_level >= 0.95:
            target_goal["status"] = "completed"
            
            # 从当前目标中移除
            self.current_goals = [g for g in self.current_goals if g["id"] != goal_id]
            
            # 更新历史记录
            for goal in self.goal_history:
                if goal["id"] == goal_id:
                    goal["status"] = "completed"
                    goal["completion_time"] = time.time()
                    goal["achievement_level"] = achievement_level
                    break
            
            # 更新统计
            self.motivation_stats["goals_achieved"] += 1
            
            self.logger.info(f"目标 {goal_id} 已完成!")
            
            return {
                "status": "completed",
                "goal_id": goal_id,
                "achievement_level": achievement_level
            }
        else:
            # 更新历史记录
            for goal in self.goal_history:
                if goal["id"] == goal_id:
                    goal["achievement_level"] = achievement_level
                    break
            
            return {
                "status": "in_progress",
                "goal_id": goal_id,
                "achievement_level": achievement_level
            }
    
    def adjust_strategy(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        根据性能表现调整探索策略
        
        Args:
            performance_metrics: 性能指标
            
        Returns:
            调整结果
        """
        self.logger.info("调整内在动机策略")
        
        prev_strategy = self.active_exploration_strategy
        
        # 根据性能指标选择最佳策略
        # 低学习速率 -> 增加不确定性探索
        if "learning_rate" in performance_metrics and performance_metrics["learning_rate"] < 0.2:
            self.active_exploration_strategy = "uncertainty_based"
        
        # 高重复性 -> 增加新颖性探索
        elif "repetitiveness" in performance_metrics and performance_metrics["repetitiveness"] > 0.7:
            self.active_exploration_strategy = "novelty_seeking"
        
        # 能力停滞 -> 能力进步探索
        elif "skill_progress" in performance_metrics and performance_metrics["skill_progress"] < 0.1:
            self.active_exploration_strategy = "competence_based"
        
        # 默认维持当前策略
        
        # 更新参数
        if "exploration_vs_exploitation" in performance_metrics:
            # 根据探索与利用比例调整好奇心权重
            exploration_ratio = performance_metrics["exploration_vs_exploitation"]
            self.curiosity_weight = 0.4 + (exploration_ratio * 0.6)  # 0.4-1.0范围
        
        result = {
            "prev_strategy": prev_strategy,
            "new_strategy": self.active_exploration_strategy,
            "curiosity_weight": self.curiosity_weight,
            "novelty_threshold": self.novelty_threshold
        }
        
        return result
    
    def get_motivation_status(self) -> Dict[str, Any]:
        """
        获取动机系统状态
        
        Returns:
            系统状态
        """
        return {
            "active_goals": len(self.current_goals),
            "completed_goals": self.motivation_stats["goals_achieved"],
            "exploration_strategy": self.active_exploration_strategy,
            "curiosity_level": self.curiosity_weight,
            "current_goals": [g["description"] for g in self.current_goals],
            "stats": self.motivation_stats
        }
    
    def save_state(self, file_path: str) -> Dict[str, Any]:
        """
        保存系统状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存结果
        """
        try:
            state = {
                "motivation_params": {
                    "curiosity_weight": self.curiosity_weight,
                    "novelty_threshold": self.novelty_threshold,
                    "competence_weight": self.competence_weight,
                    "goal_horizon": self.goal_horizon
                },
                "current_goals": self.current_goals,
                "goal_history": self.goal_history,
                "goal_achievements": dict(self.goal_achievements),
                "active_exploration_strategy": self.active_exploration_strategy,
                "motivation_stats": self.motivation_stats,
                "saved_at": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"内在动机系统状态已保存到: {file_path}")
            
            return {"success": True, "file_path": file_path}
        
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return {"success": False, "error": str(e)} 