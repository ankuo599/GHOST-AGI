"""
自我意识模块
实现内在状态建模和元认知监控
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
import time
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
import networkx as nx

@dataclass
class InternalState:
    """内在状态"""
    id: str
    type: str  # 状态类型：emotion, intention, belief, goal
    value: float  # 状态值
    confidence: float  # 置信度
    source: str  # 状态来源
    timestamp: float  # 时间戳
    metadata: Dict[str, Any]  # 元数据

@dataclass
class SelfModel:
    """自我模型"""
    id: str
    name: str
    type: str  # 模型类型：behavior, capability, preference
    parameters: Dict[str, Any]  # 模型参数
    performance: Dict[str, float]  # 性能指标
    updated_at: float  # 更新时间

class SelfAwareness:
    """自我意识系统"""
    def __init__(self, model_dir: str = "models/self"):
        self.states: Dict[str, InternalState] = {}
        self.models: Dict[str, SelfModel] = {}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("SelfAwareness")
        self.state_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._init_models()
        
    def _init_models(self):
        """初始化自我模型"""
        # 行为模型
        self.models["behavior"] = SelfModel(
            id="behavior",
            name="行为模型",
            type="behavior",
            parameters={
                "action_space": [],  # 动作空间
                "state_space": [],  # 状态空间
                "reward_function": None  # 奖励函数
            },
            performance={},
            updated_at=time.time()
        )
        
        # 能力模型
        self.models["capability"] = SelfModel(
            id="capability",
            name="能力模型",
            type="capability",
            parameters={
                "skills": [],  # 技能列表
                "resources": {},  # 资源列表
                "constraints": {}  # 约束条件
            },
            performance={},
            updated_at=time.time()
        )
        
        # 偏好模型
        self.models["preference"] = SelfModel(
            id="preference",
            name="偏好模型",
            type="preference",
            parameters={
                "goals": [],  # 目标列表
                "values": {},  # 价值判断
                "priorities": {}  # 优先级
            },
            performance={},
            updated_at=time.time()
        )
        
    def update_state(self, state: InternalState) -> bool:
        """更新内在状态"""
        try:
            self.states[state.id] = state
            
            # 记录状态历史
            self.state_history[state.id].append({
                "value": state.value,
                "confidence": state.confidence,
                "timestamp": state.timestamp
            })
            
            # 更新相关模型
            self._update_models(state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新状态失败: {str(e)}")
            return False
            
    def _update_models(self, state: InternalState):
        """更新自我模型"""
        if state.type == "emotion":
            self._update_behavior_model(state)
        elif state.type == "intention":
            self._update_capability_model(state)
        elif state.type == "belief":
            self._update_preference_model(state)
            
    def _update_behavior_model(self, state: InternalState):
        """更新行为模型"""
        model = self.models["behavior"]
        
        # 根据情绪状态调整行为参数
        if state.value > 0.7:  # 积极情绪
            model.parameters["exploration_rate"] = max(
                0.1,
                model.parameters.get("exploration_rate", 0.3) * 0.9
            )
        else:  # 消极情绪
            model.parameters["exploration_rate"] = min(
                0.9,
                model.parameters.get("exploration_rate", 0.3) * 1.1
            )
            
        model.updated_at = time.time()
        
    def _update_capability_model(self, state: InternalState):
        """更新能力模型"""
        model = self.models["capability"]
        
        # 根据意图调整能力评估
        if state.value > 0.7:  # 强烈意图
            model.parameters["confidence_threshold"] = min(
                0.9,
                model.parameters.get("confidence_threshold", 0.7) * 1.1
            )
        else:  # 弱意图
            model.parameters["confidence_threshold"] = max(
                0.5,
                model.parameters.get("confidence_threshold", 0.7) * 0.9
            )
            
        model.updated_at = time.time()
        
    def _update_preference_model(self, state: InternalState):
        """更新偏好模型"""
        model = self.models["preference"]
        
        # 根据信念调整偏好
        if state.value > 0.7:  # 强信念
            model.parameters["goal_priority"] = min(
                1.0,
                model.parameters.get("goal_priority", 0.5) * 1.2
            )
        else:  # 弱信念
            model.parameters["goal_priority"] = max(
                0.1,
                model.parameters.get("goal_priority", 0.5) * 0.8
            )
            
        model.updated_at = time.time()
        
    def get_state(self, state_id: str) -> Optional[InternalState]:
        """获取内在状态"""
        return self.states.get(state_id)
        
    def get_state_history(self, state_id: str) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return self.state_history.get(state_id, [])
        
    def get_model(self, model_id: str) -> Optional[SelfModel]:
        """获取自我模型"""
        return self.models.get(model_id)
        
    def update_model(self, model_id: str, 
                    parameters: Dict[str, Any]) -> bool:
        """更新自我模型"""
        if model_id not in self.models:
            return False
            
        try:
            model = self.models[model_id]
            model.parameters.update(parameters)
            model.updated_at = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
            return False
            
    def evaluate_self(self) -> Dict[str, Any]:
        """评估自我状态"""
        return {
            "emotional_state": self._evaluate_emotional_state(),
            "capability_state": self._evaluate_capability_state(),
            "preference_state": self._evaluate_preference_state(),
            "overall_state": self._evaluate_overall_state()
        }
        
    def _evaluate_emotional_state(self) -> Dict[str, Any]:
        """评估情绪状态"""
        emotional_states = [
            state for state in self.states.values()
            if state.type == "emotion"
        ]
        
        if not emotional_states:
            return {"status": "unknown"}
            
        # 计算情绪指标
        avg_value = np.mean([s.value for s in emotional_states])
        avg_confidence = np.mean([s.confidence for s in emotional_states])
        
        return {
            "status": "positive" if avg_value > 0.5 else "negative",
            "intensity": abs(avg_value - 0.5) * 2,
            "confidence": avg_confidence,
            "stability": self._calculate_stability(emotional_states)
        }
        
    def _evaluate_capability_state(self) -> Dict[str, Any]:
        """评估能力状态"""
        model = self.models["capability"]
        
        return {
            "available_skills": len(model.parameters["skills"]),
            "resource_utilization": self._calculate_resource_utilization(),
            "constraint_satisfaction": self._calculate_constraint_satisfaction(),
            "overall_capability": self._calculate_overall_capability()
        }
        
    def _evaluate_preference_state(self) -> Dict[str, Any]:
        """评估偏好状态"""
        model = self.models["preference"]
        
        return {
            "active_goals": len(model.parameters["goals"]),
            "value_alignment": self._calculate_value_alignment(),
            "priority_consistency": self._calculate_priority_consistency(),
            "overall_preference": self._calculate_overall_preference()
        }
        
    def _evaluate_overall_state(self) -> Dict[str, Any]:
        """评估整体状态"""
        emotional = self._evaluate_emotional_state()
        capability = self._evaluate_capability_state()
        preference = self._evaluate_preference_state()
        
        return {
            "emotional_health": emotional["confidence"] * emotional["stability"],
            "capability_health": capability["overall_capability"],
            "preference_health": preference["overall_preference"],
            "overall_health": (
                emotional["confidence"] * emotional["stability"] +
                capability["overall_capability"] +
                preference["overall_preference"]
            ) / 3
        }
        
    def _calculate_stability(self, states: List[InternalState]) -> float:
        """计算状态稳定性"""
        if len(states) < 2:
            return 1.0
            
        values = [s.value for s in states]
        return 1.0 - np.std(values)
        
    def _calculate_resource_utilization(self) -> float:
        """计算资源利用率"""
        model = self.models["capability"]
        resources = model.parameters["resources"]
        
        if not resources:
            return 0.0
            
        return sum(
            usage / total
            for usage, total in resources.values()
            if total > 0
        ) / len(resources)
        
    def _calculate_constraint_satisfaction(self) -> float:
        """计算约束满足度"""
        model = self.models["capability"]
        constraints = model.parameters["constraints"]
        
        if not constraints:
            return 1.0
            
        satisfied = sum(
            1 for constraint in constraints.values()
            if self._check_constraint(constraint)
        )
        
        return satisfied / len(constraints)
        
    def _check_constraint(self, constraint: Dict[str, Any]) -> bool:
        """检查约束条件"""
        # 实现约束检查逻辑
        return True
        
    def _calculate_overall_capability(self) -> float:
        """计算整体能力"""
        utilization = self._calculate_resource_utilization()
        satisfaction = self._calculate_constraint_satisfaction()
        
        return (utilization + satisfaction) / 2
        
    def _calculate_value_alignment(self) -> float:
        """计算价值一致性"""
        model = self.models["preference"]
        values = model.parameters["values"]
        
        if not values:
            return 1.0
            
        # 计算价值之间的冲突
        conflicts = 0
        total = 0
        
        for v1 in values:
            for v2 in values:
                if v1 != v2:
                    total += 1
                    if self._check_value_conflict(values[v1], values[v2]):
                        conflicts += 1
                        
        return 1.0 - (conflicts / total if total > 0 else 0)
        
    def _check_value_conflict(self, value1: Any, value2: Any) -> bool:
        """检查价值冲突"""
        # 实现价值冲突检查逻辑
        return False
        
    def _calculate_priority_consistency(self) -> float:
        """计算优先级一致性"""
        model = self.models["preference"]
        priorities = model.parameters["priorities"]
        
        if not priorities:
            return 1.0
            
        # 检查优先级是否形成有向无环图
        graph = nx.DiGraph()
        for goal, priority in priorities.items():
            graph.add_node(goal, priority=priority)
            
        for goal1 in priorities:
            for goal2 in priorities:
                if goal1 != goal2:
                    if priorities[goal1] > priorities[goal2]:
                        graph.add_edge(goal1, goal2)
                        
        return 1.0 if nx.is_directed_acyclic_graph(graph) else 0.0
        
    def _calculate_overall_preference(self) -> float:
        """计算整体偏好"""
        alignment = self._calculate_value_alignment()
        consistency = self._calculate_priority_consistency()
        
        return (alignment + consistency) / 2
        
    def save_state(self, file_path: str) -> bool:
        """保存状态"""
        try:
            data = {
                "states": {
                    state_id: {
                        "type": state.type,
                        "value": state.value,
                        "confidence": state.confidence,
                        "source": state.source,
                        "timestamp": state.timestamp,
                        "metadata": state.metadata
                    }
                    for state_id, state in self.states.items()
                },
                "models": {
                    model_id: {
                        "name": model.name,
                        "type": model.type,
                        "parameters": model.parameters,
                        "performance": model.performance,
                        "updated_at": model.updated_at
                    }
                    for model_id, model in self.models.items()
                },
                "state_history": self.state_history
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return False
            
    def load_state(self, file_path: str) -> bool:
        """加载状态"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 加载状态
            self.states = {
                state_id: InternalState(
                    id=state_id,
                    type=state["type"],
                    value=state["value"],
                    confidence=state["confidence"],
                    source=state["source"],
                    timestamp=state["timestamp"],
                    metadata=state["metadata"]
                )
                for state_id, state in data["states"].items()
            }
            
            # 加载模型
            self.models = {
                model_id: SelfModel(
                    id=model_id,
                    name=model["name"],
                    type=model["type"],
                    parameters=model["parameters"],
                    performance=model["performance"],
                    updated_at=model["updated_at"]
                )
                for model_id, model in data["models"].items()
            }
            
            # 加载历史
            self.state_history = data["state_history"]
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载状态失败: {str(e)}")
            return False 