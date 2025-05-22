"""
自我意识系统
实现自我状态感知、目标管理和价值观系统
"""

from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from knowledge.knowledge_base import KnowledgeBase, KnowledgeNode

class Goal:
    """目标"""
    def __init__(self, description: str, priority: float = 1.0,
                 deadline: Optional[datetime] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.status = "pending"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.progress = 0.0
        self.subgoals: List[Goal] = []
        self.metadata: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "subgoals": [goal.to_dict() for goal in self.subgoals],
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        goal = cls(
            description=data["description"],
            priority=data["priority"],
            deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None
        )
        goal.id = data["id"]
        goal.status = data["status"]
        goal.created_at = datetime.fromisoformat(data["created_at"])
        goal.updated_at = datetime.fromisoformat(data["updated_at"])
        goal.progress = data["progress"]
        goal.subgoals = [cls.from_dict(subgoal) for subgoal in data["subgoals"]]
        goal.metadata = data["metadata"]
        return goal

class Value:
    """价值观"""
    def __init__(self, name: str, description: str,
                 importance: float = 1.0):
        self.name = name
        self.description = description
        self.importance = importance
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Value':
        value = cls(
            name=data["name"],
            description=data["description"],
            importance=data["importance"]
        )
        value.created_at = datetime.fromisoformat(data["created_at"])
        value.updated_at = datetime.fromisoformat(data["updated_at"])
        value.metadata = data["metadata"]
        return value

class SelfAwareness:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.logger = logging.getLogger("SelfAwareness")
        self.goals: Dict[str, Goal] = {}
        self.values: Dict[str, Value] = {}
        self.emotional_state: Dict[str, float] = {
            "happiness": 0.5,
            "confidence": 0.5,
            "curiosity": 0.5,
            "fear": 0.0,
            "anger": 0.0
        }
        self.self_concept: Dict[str, Any] = {
            "capabilities": {},
            "limitations": {},
            "preferences": {},
            "beliefs": {}
        }
        self.state_history: List[Dict[str, Any]] = []
        
    def add_goal(self, goal: Goal) -> str:
        """添加目标"""
        self.goals[goal.id] = goal
        return goal.id
        
    def update_goal(self, goal_id: str, 
                   updates: Dict[str, Any]) -> bool:
        """更新目标"""
        if goal_id not in self.goals:
            return False
            
        goal = self.goals[goal_id]
        for key, value in updates.items():
            if hasattr(goal, key):
                setattr(goal, key, value)
        goal.updated_at = datetime.now()
        return True
        
    def add_value(self, value: Value):
        """添加价值观"""
        self.values[value.name] = value
        
    def update_value(self, value_name: str, 
                    updates: Dict[str, Any]) -> bool:
        """更新价值观"""
        if value_name not in self.values:
            return False
            
        value = self.values[value_name]
        for key, val in updates.items():
            if hasattr(value, key):
                setattr(value, key, val)
        value.updated_at = datetime.now()
        return True
        
    def update_emotional_state(self, 
                             emotions: Dict[str, float]):
        """更新情感状态"""
        for emotion, intensity in emotions.items():
            if emotion in self.emotional_state:
                self.emotional_state[emotion] = max(0.0, min(1.0, intensity))
                
    def update_self_concept(self, 
                          concept_type: str,
                          updates: Dict[str, Any]):
        """更新自我概念"""
        if concept_type in self.self_concept:
            self.self_concept[concept_type].update(updates)
            
    async def evaluate_goal(self, goal_id: str) -> Dict[str, Any]:
        """评估目标"""
        try:
            if goal_id not in self.goals:
                return {
                    "error": "目标不存在",
                    "timestamp": datetime.now().isoformat()
                }
                
            goal = self.goals[goal_id]
            
            # 评估目标进度
            progress = self._calculate_goal_progress(goal)
            
            # 评估目标价值
            value = self._evaluate_goal_value(goal)
            
            # 评估目标可行性
            feasibility = self._evaluate_goal_feasibility(goal)
            
            result = {
                "goal_id": goal_id,
                "progress": progress,
                "value": value,
                "feasibility": feasibility,
                "timestamp": datetime.now().isoformat()
            }
            
            # 记录状态
            self._record_state("goal_evaluation", result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"目标评估出错: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def evaluate_decision(self, 
                              options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估决策"""
        try:
            # 评估每个选项
            evaluations = []
            for option in options:
                # 评估选项价值
                value_score = self._evaluate_option_value(option)
                
                # 评估选项可行性
                feasibility_score = self._evaluate_option_feasibility(option)
                
                # 评估选项风险
                risk_score = self._evaluate_option_risk(option)
                
                evaluations.append({
                    "option": option,
                    "value_score": value_score,
                    "feasibility_score": feasibility_score,
                    "risk_score": risk_score,
                    "total_score": value_score + feasibility_score - risk_score
                })
                
            # 选择最佳选项
            best_option = max(evaluations, key=lambda x: x["total_score"])
            
            result = {
                "evaluations": evaluations,
                "best_option": best_option,
                "timestamp": datetime.now().isoformat()
            }
            
            # 记录状态
            self._record_state("decision_evaluation", result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"决策评估出错: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def _calculate_goal_progress(self, goal: Goal) -> float:
        """计算目标进度"""
        if not goal.subgoals:
            return goal.progress
            
        total_progress = 0.0
        for subgoal in goal.subgoals:
            total_progress += self._calculate_goal_progress(subgoal)
        return total_progress / len(goal.subgoals)
        
    def _evaluate_goal_value(self, goal: Goal) -> float:
        """评估目标价值"""
        value_score = 0.0
        for value in self.values.values():
            # 检查目标是否符合价值观
            if self._is_goal_aligned_with_value(goal, value):
                value_score += value.importance
        return value_score
        
    def _evaluate_goal_feasibility(self, goal: Goal) -> float:
        """评估目标可行性"""
        # 检查能力
        capability_score = self._evaluate_capability(goal)
        
        # 检查资源
        resource_score = self._evaluate_resources(goal)
        
        # 检查时间
        time_score = self._evaluate_time(goal)
        
        return (capability_score + resource_score + time_score) / 3.0
        
    def _is_goal_aligned_with_value(self, goal: Goal, 
                                  value: Value) -> bool:
        """检查目标是否符合价值观"""
        # 实现价值观对齐检查逻辑
        return True
        
    def _evaluate_capability(self, goal: Goal) -> float:
        """评估能力"""
        # 实现能力评估逻辑
        return 0.5
        
    def _evaluate_resources(self, goal: Goal) -> float:
        """评估资源"""
        # 实现资源评估逻辑
        return 0.5
        
    def _evaluate_time(self, goal: Goal) -> float:
        """评估时间"""
        if not goal.deadline:
            return 1.0
            
        time_left = (goal.deadline - datetime.now()).total_seconds()
        if time_left <= 0:
            return 0.0
            
        return min(1.0, time_left / (24 * 3600))  # 假设一天为基准
        
    def _evaluate_option_value(self, option: Dict[str, Any]) -> float:
        """评估选项价值"""
        value_score = 0.0
        for value in self.values.values():
            # 检查选项是否符合价值观
            if self._is_option_aligned_with_value(option, value):
                value_score += value.importance
        return value_score
        
    def _evaluate_option_feasibility(self, 
                                   option: Dict[str, Any]) -> float:
        """评估选项可行性"""
        # 实现可行性评估逻辑
        return 0.5
        
    def _evaluate_option_risk(self, option: Dict[str, Any]) -> float:
        """评估选项风险"""
        # 实现风险评估逻辑
        return 0.5
        
    def _is_option_aligned_with_value(self, option: Dict[str, Any], 
                                    value: Value) -> bool:
        """检查选项是否符合价值观"""
        # 实现价值观对齐检查逻辑
        return True
        
    def _record_state(self, state_type: str, data: Dict[str, Any]):
        """记录状态"""
        self.state_history.append({
            "type": state_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "emotional_state": self.emotional_state.copy(),
            "self_concept": self.self_concept.copy()
        })
        
    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return self.state_history[-limit:]
        
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "goals": {goal_id: goal.to_dict() 
                     for goal_id, goal in self.goals.items()},
            "values": {value_name: value.to_dict() 
                      for value_name, value in self.values.items()},
            "emotional_state": self.emotional_state,
            "self_concept": self.self_concept,
            "timestamp": datetime.now().isoformat()
        }
        
    def clear_history(self):
        """清除历史"""
        self.state_history.clear()
        
    def save_state(self, path: str):
        """保存状态"""
        data = {
            "goals": {goal_id: goal.to_dict() 
                     for goal_id, goal in self.goals.items()},
            "values": {value_name: value.to_dict() 
                      for value_name, value in self.values.items()},
            "emotional_state": self.emotional_state,
            "self_concept": self.self_concept,
            "state_history": self.state_history
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_state(self, path: str):
        """加载状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.goals = {
                goal_id: Goal.from_dict(goal_data)
                for goal_id, goal_data in data["goals"].items()
            }
            self.values = {
                value_name: Value.from_dict(value_data)
                for value_name, value_data in data["values"].items()
            }
            self.emotional_state = data["emotional_state"]
            self.self_concept = data["self_concept"]
            self.state_history = data["state_history"]
            
        except FileNotFoundError:
            self.logger.warning("状态文件不存在，使用默认状态") 