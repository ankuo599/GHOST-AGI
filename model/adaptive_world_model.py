"""
GHOST-AGI 自适应世界模型

该模块实现环境建模和预测，使GHOST-AGI能够建立内部世界模型理解环境并预测未来状态变化。
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict

class AdaptiveWorldModel:
    """自适应世界模型，提供环境建模和预测功能"""
    
    def __init__(self, 
                 state_dimension: int = 100,
                 prediction_horizon: int = 5,
                 update_rate: float = 0.1,
                 logger: Optional[logging.Logger] = None):
        """
        初始化自适应世界模型
        
        Args:
            state_dimension: 状态维度
            prediction_horizon: 预测时域
            update_rate: 模型更新率
            logger: 日志记录器
        """
        # 模型参数
        self.state_dimension = state_dimension
        self.prediction_horizon = prediction_horizon
        self.update_rate = update_rate
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 世界模型组件
        self.state_representation = {}  # 当前状态表示
        self.entity_models = {}         # 实体模型
        self.causal_models = {}         # 因果模型
        self.transition_models = {}     # 状态转移模型
        self.reward_models = {}         # 奖励模型
        
        # 观察历史
        self.observation_history = []
        self.action_history = []
        self.state_history = []
        self.prediction_history = []
        
        # 不确定性估计
        self.uncertainty_estimates = {}
        self.confidence_levels = {}
        
        # 模型性能指标
        self.prediction_errors = defaultdict(list)
        self.model_accuracy = {}
        
        # 统计信息
        self.model_stats = {
            "observations": 0,
            "predictions": 0,
            "updates": 0,
            "entities_tracked": 0,
            "last_update": time.time()
        }
        
        self.logger.info("自适应世界模型初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AdaptiveWorldModel")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("world_model.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def update_model(self, observation: Dict[str, Any], action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        更新世界模型
        
        Args:
            observation: 观察到的环境状态
            action: 执行的动作（如果有）
            
        Returns:
            更新结果
        """
        self.logger.info("更新世界模型")
        
        # 记录观察和动作
        self.observation_history.append({
            "data": observation,
            "timestamp": time.time()
        })
        
        if action:
            self.action_history.append({
                "data": action,
                "timestamp": time.time()
            })
        
        # 提取实体和关系
        entities = self._extract_entities(observation)
        relations = self._extract_relations(observation)
        
        # 更新实体模型
        for entity_id, entity_data in entities.items():
            self._update_entity_model(entity_id, entity_data)
        
        # 更新因果模型
        self._update_causal_models(entities, relations, action)
        
        # 更新状态表示
        self._update_state_representation(observation, entities, relations)
        
        # 保存当前状态
        current_state = self.state_representation.copy()
        self.state_history.append({
            "state": current_state,
            "timestamp": time.time()
        })
        
        # 评估预测误差
        self._evaluate_predictions(current_state)
        
        # 更新统计
        self.model_stats["observations"] += 1
        self.model_stats["entities_tracked"] = len(self.entity_models)
        self.model_stats["updates"] += 1
        self.model_stats["last_update"] = time.time()
        
        return {
            "state_updated": True,
            "entities_updated": len(entities),
            "relations_updated": len(relations),
            "current_uncertainty": self._calculate_overall_uncertainty()
        }
    
    def _extract_entities(self, observation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """从观察中提取实体"""
        entities = {}
        
        # 提取观察中的实体
        if "entities" in observation:
            for entity in observation["entities"]:
                entity_id = entity.get("id", entity.get("name", f"entity_{len(entities)}"))
                entities[entity_id] = entity
        
        return entities
    
    def _extract_relations(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从观察中提取关系"""
        relations = []
        
        # 提取观察中的关系
        if "relations" in observation:
            relations = observation["relations"]
        
        return relations
    
    def _update_entity_model(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """更新实体模型"""
        if entity_id in self.entity_models:
            # 现有实体模型更新
            entity_model = self.entity_models[entity_id]
            
            # 更新属性
            if "attributes" in entity_data:
                if "attributes" not in entity_model:
                    entity_model["attributes"] = {}
                
                for attr_name, attr_value in entity_data["attributes"].items():
                    # 平滑更新
                    if attr_name in entity_model["attributes"]:
                        old_value = entity_model["attributes"][attr_name]
                        entity_model["attributes"][attr_name] = old_value * (1 - self.update_rate) + attr_value * self.update_rate
                    else:
                        entity_model["attributes"][attr_name] = attr_value
            
            # 更新状态和其他属性
            for key, value in entity_data.items():
                if key != "attributes" and key != "id":
                    entity_model[key] = value
            
            # 更新观察计数和时间戳
            entity_model["observation_count"] += 1
            entity_model["last_observed"] = time.time()
            
        else:
            # 创建新实体模型
            self.entity_models[entity_id] = {
                "id": entity_id,
                "first_observed": time.time(),
                "last_observed": time.time(),
                "observation_count": 1,
                "attributes": entity_data.get("attributes", {}),
                "type": entity_data.get("type", "unknown"),
                "confidence": 0.5  # 初始置信度
            }
            
            # 复制其他属性
            for key, value in entity_data.items():
                if key != "attributes" and key != "id":
                    self.entity_models[entity_id][key] = value
    
    def _update_causal_models(self, entities: Dict[str, Dict[str, Any]], 
                             relations: List[Dict[str, Any]], 
                             action: Optional[Dict[str, Any]]) -> None:
        """更新因果模型"""
        # 更新基于实体间关系的因果模型
        for relation in relations:
            if "source" in relation and "target" in relation and "type" in relation:
                relation_key = f"{relation['source']}_{relation['type']}_{relation['target']}"
                
                if relation_key not in self.causal_models:
                    # 创建新因果关系
                    self.causal_models[relation_key] = {
                        "source": relation["source"],
                        "target": relation["target"],
                        "type": relation["type"],
                        "strength": relation.get("strength", 0.5),
                        "observation_count": 1,
                        "first_observed": time.time(),
                        "last_observed": time.time(),
                        "confidence": 0.5,
                        "conditional_probs": {}
                    }
                else:
                    # 更新现有因果关系
                    causal_model = self.causal_models[relation_key]
                    causal_model["observation_count"] += 1
                    causal_model["last_observed"] = time.time()
                    
                    # 更新强度（如果提供）
                    if "strength" in relation:
                        old_strength = causal_model["strength"]
                        new_strength = relation["strength"]
                        causal_model["strength"] = old_strength * (1 - self.update_rate) + new_strength * self.update_rate
        
        # 如果有动作，更新动作-效果因果关系
        if action and self.action_history and self.state_history:
            action_id = action.get("id", action.get("type", "unknown_action"))
            prev_state = self.state_history[-2]["state"] if len(self.state_history) >= 2 else {}
            current_state = self.state_history[-1]["state"] if self.state_history else {}
            
            # 简化实现：仅记录动作和状态变化
            for entity_id in current_state.get("entities", {}):
                if entity_id in prev_state.get("entities", {}):
                    # 检测状态变化
                    action_effect_key = f"{action_id}_affects_{entity_id}"
                    
                    if action_effect_key not in self.causal_models:
                        self.causal_models[action_effect_key] = {
                            "source": action_id,
                            "target": entity_id,
                            "type": "action_effect",
                            "strength": 0.5,
                            "observation_count": 1,
                            "first_observed": time.time(),
                            "last_observed": time.time(),
                            "confidence": 0.5,
                            "effects": {}
                        }
                    else:
                        self.causal_models[action_effect_key]["observation_count"] += 1
                        self.causal_models[action_effect_key]["last_observed"] = time.time()
    
    def _update_state_representation(self, observation: Dict[str, Any], 
                                   entities: Dict[str, Dict[str, Any]], 
                                   relations: List[Dict[str, Any]]) -> None:
        """更新状态表示"""
        # 更新全局状态表示
        if not self.state_representation:
            self.state_representation = {
                "timestamp": time.time(),
                "entities": {},
                "relations": [],
                "global_attributes": {}
            }
        
        # 更新时间戳
        self.state_representation["timestamp"] = time.time()
        
        # 更新实体
        for entity_id, entity_data in entities.items():
            self.state_representation["entities"][entity_id] = entity_data
        
        # 更新关系
        self.state_representation["relations"] = relations
        
        # 更新全局属性
        if "global_attributes" in observation:
            for attr_name, attr_value in observation["global_attributes"].items():
                self.state_representation["global_attributes"][attr_name] = attr_value
    
    def _evaluate_predictions(self, current_state: Dict[str, Any]) -> None:
        """评估之前的预测准确性"""
        if not self.prediction_history:
            return
        
        # 获取最近的预测
        recent_predictions = [p for p in self.prediction_history 
                             if p["target_time"] <= time.time() and not p.get("evaluated", False)]
        
        for prediction in recent_predictions:
            # 标记为已评估
            prediction["evaluated"] = True
            
            # 计算预测误差
            predicted_state = prediction["predicted_state"]
            actual_state = current_state
            
            error = self._calculate_prediction_error(predicted_state, actual_state)
            
            # 记录误差
            model_id = prediction.get("model_id", "default")
            self.prediction_errors[model_id].append(error)
            
            # 更新准确性
            if model_id not in self.model_accuracy:
                self.model_accuracy[model_id] = 0.5  # 初始准确性
            
            # 平滑更新准确性 (error在0-1之间，越小越好)
            accuracy = 1.0 - error
            self.model_accuracy[model_id] = self.model_accuracy[model_id] * 0.9 + accuracy * 0.1
            
            # 记录评估结果
            prediction["error"] = error
            prediction["accuracy"] = accuracy
    
    def _calculate_prediction_error(self, predicted_state: Dict[str, Any], actual_state: Dict[str, Any]) -> float:
        """计算预测误差"""
        # 简化版本：比较实体属性
        total_error = 0.0
        comparison_count = 0
        
        # 比较实体
        pred_entities = predicted_state.get("entities", {})
        actual_entities = actual_state.get("entities", {})
        
        # 找到共同的实体
        common_entities = set(pred_entities.keys()) & set(actual_entities.keys())
        
        for entity_id in common_entities:
            pred_entity = pred_entities[entity_id]
            actual_entity = actual_entities[entity_id]
            
            # 比较属性
            pred_attrs = pred_entity.get("attributes", {})
            actual_attrs = actual_entity.get("attributes", {})
            
            common_attrs = set(pred_attrs.keys()) & set(actual_attrs.keys())
            
            for attr in common_attrs:
                if isinstance(pred_attrs[attr], (int, float)) and isinstance(actual_attrs[attr], (int, float)):
                    # 数值属性：计算相对误差
                    error = abs(pred_attrs[attr] - actual_attrs[attr]) / max(1.0, abs(actual_attrs[attr]))
                    total_error += min(1.0, error)  # 限制最大误差为1
                    comparison_count += 1
                elif pred_attrs[attr] != actual_attrs[attr]:
                    # 非数值属性：不匹配则误差为1
                    total_error += 1.0
                    comparison_count += 1
        
        # 避免除零错误
        if comparison_count == 0:
            return 0.0
        
        # 返回平均误差（0-1之间）
        return total_error / comparison_count
    
    def predict_future_state(self, steps: int = 1, action_sequence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        预测未来状态
        
        Args:
            steps: 预测步数
            action_sequence: 动作序列（如果有）
            
        Returns:
            预测结果
        """
        self.logger.info(f"预测未来状态: {steps} 步")
        
        if not self.state_representation:
            return {"error": "未初始化状态表示，无法进行预测"}
        
        # 限制预测步数
        steps = min(steps, self.prediction_horizon)
        
        # 初始状态为当前状态
        current_state = self.state_representation.copy()
        predicted_states = []
        
        # 逐步预测
        for step in range(steps):
            # 获取此步要执行的动作
            action = None
            if action_sequence and step < len(action_sequence):
                action = action_sequence[step]
            
            # 预测下一个状态
            next_state = self._predict_next_state(current_state, action)
            
            # 计算预测置信度
            confidence = self._calculate_prediction_confidence(current_state, next_state, step + 1)
            
            # 记录预测
            prediction_record = {
                "step": step + 1,
                "predicted_state": next_state,
                "action": action,
                "confidence": confidence,
                "timestamp": time.time(),
                "target_time": time.time() + (step + 1) * 3600,  # 假设每步对应1小时
                "model_id": "default",
                "evaluated": False
            }
            
            predicted_states.append(prediction_record)
            self.prediction_history.append(prediction_record)
            
            # 更新当前状态为预测的下一状态
            current_state = next_state
        
        # 更新统计
        self.model_stats["predictions"] += 1
        
        return {
            "initial_state": self.state_representation,
            "predicted_states": predicted_states,
            "steps": steps,
            "actions": action_sequence if action_sequence else []
        }
    
    def _predict_next_state(self, current_state: Dict[str, Any], action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """预测下一个状态"""
        # 复制当前状态
        next_state = {
            "timestamp": time.time() + 3600,  # 假设1小时后
            "entities": {},
            "relations": current_state.get("relations", []).copy(),
            "global_attributes": current_state.get("global_attributes", {}).copy()
        }
        
        # 预测每个实体的下一个状态
        for entity_id, entity_data in current_state.get("entities", {}).items():
            next_entity = entity_data.copy()
            
            # 深度复制属性
            if "attributes" in entity_data:
                next_entity["attributes"] = entity_data["attributes"].copy()
            
            # 如果有特定实体的转移模型，使用它进行预测
            if entity_id in self.transition_models:
                self._apply_transition_model(next_entity, self.transition_models[entity_id], action)
            else:
                # 否则使用简单规则进行预测
                self._apply_default_transitions(next_entity, action)
            
            # 应用因果效应
            self._apply_causal_effects(entity_id, next_entity, current_state, action)
            
            # 添加到下一个状态
            next_state["entities"][entity_id] = next_entity
        
        # 预测新实体的出现（简化版本）
        
        # 预测关系变化（简化版本）
        
        return next_state
    
    def _apply_transition_model(self, entity: Dict[str, Any], transition_model: Dict[str, Any], action: Optional[Dict[str, Any]]) -> None:
        """应用转移模型预测实体状态变化"""
        # 简化实现，实际系统需要更复杂的预测模型
        pass
    
    def _apply_default_transitions(self, entity: Dict[str, Any], action: Optional[Dict[str, Any]]) -> None:
        """应用默认转移规则"""
        # 简化实现：保持大多数属性不变，可能根据类型应用一些默认变化
        entity_type = entity.get("type", "unknown")
        
        # 示例：移动物体的位置变化
        if entity_type == "movable" and "attributes" in entity and "position" in entity["attributes"]:
            # 简单规则：如果有速度，则更新位置
            if "velocity" in entity["attributes"]:
                position = entity["attributes"]["position"]
                velocity = entity["attributes"]["velocity"]
                
                # 假设位置和速度是数值或列表
                if isinstance(position, list) and isinstance(velocity, list) and len(position) == len(velocity):
                    for i in range(len(position)):
                        position[i] += velocity[i]
                elif isinstance(position, (int, float)) and isinstance(velocity, (int, float)):
                    entity["attributes"]["position"] = position + velocity
    
    def _apply_causal_effects(self, entity_id: str, entity: Dict[str, Any], 
                             current_state: Dict[str, Any], 
                             action: Optional[Dict[str, Any]]) -> None:
        """应用因果效应"""
        # 应用实体间的因果关系
        for relation_key, causal_model in self.causal_models.items():
            if causal_model["target"] == entity_id:
                source_id = causal_model["source"]
                
                # 检查源实体是否存在
                if source_id in current_state.get("entities", {}):
                    source_entity = current_state["entities"][source_id]
                    
                    # 应用因果效应（简化实现）
                    self._apply_simple_causal_effect(entity, source_entity, causal_model)
        
        # 应用动作的因果效应
        if action:
            action_id = action.get("id", action.get("type", "unknown_action"))
            action_effect_key = f"{action_id}_affects_{entity_id}"
            
            if action_effect_key in self.causal_models:
                causal_model = self.causal_models[action_effect_key]
                
                # 应用动作效应（简化实现）
                self._apply_action_effect(entity, action, causal_model)
    
    def _apply_simple_causal_effect(self, target_entity: Dict[str, Any], 
                                  source_entity: Dict[str, Any], 
                                  causal_model: Dict[str, Any]) -> None:
        """应用简单因果效应"""
        # 简化实现
        pass
    
    def _apply_action_effect(self, entity: Dict[str, Any], 
                           action: Dict[str, Any], 
                           causal_model: Dict[str, Any]) -> None:
        """应用动作效应"""
        # 简化实现
        pass
    
    def _calculate_prediction_confidence(self, current_state: Dict[str, Any], 
                                      predicted_state: Dict[str, Any], 
                                      step: int) -> float:
        """计算预测置信度"""
        # 预测步数越多，置信度越低
        step_decay = max(0.0, 1.0 - 0.1 * step)
        
        # 模型准确性
        model_accuracy = self.model_accuracy.get("default", 0.5)
        
        # 数据充分性
        data_sufficiency = min(1.0, self.model_stats["observations"] / 100.0)
        
        # 整合因素
        confidence = step_decay * 0.4 + model_accuracy * 0.4 + data_sufficiency * 0.2
        
        return confidence
    
    def _calculate_overall_uncertainty(self) -> float:
        """计算总体不确定性"""
        # 简化计算：1减去平均模型准确性
        if not self.model_accuracy:
            return 0.5
        
        avg_accuracy = sum(self.model_accuracy.values()) / len(self.model_accuracy)
        return 1.0 - avg_accuracy
    
    def detect_anomalies(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测异常
        
        Args:
            observation: 当前观察
            
        Returns:
            异常检测结果
        """
        self.logger.info("检测异常")
        
        if not self.state_representation:
            return {"anomalies": [], "message": "未初始化状态表示，无法检测异常"}
        
        # 提取观察中的实体
        entities = self._extract_entities(observation)
        
        # 检测异常
        anomalies = []
        
        # 检测实体异常
        for entity_id, entity_data in entities.items():
            if entity_id in self.entity_models:
                # 比较当前属性与模型预期
                entity_anomalies = self._detect_entity_anomalies(entity_id, entity_data)
                anomalies.extend(entity_anomalies)
            else:
                # 新实体出现，可能是异常
                if self.model_stats["observations"] > 10:
                    anomalies.append({
                        "type": "new_entity",
                        "entity_id": entity_id,
                        "severity": 0.5,
                        "timestamp": time.time(),
                        "description": f"检测到新实体: {entity_id}"
                    })
        
        # 检测实体消失
        for entity_id in self.state_representation.get("entities", {}):
            if entity_id not in entities and time.time() - self.entity_models.get(entity_id, {}).get("last_observed", 0) < 86400:
                anomalies.append({
                    "type": "entity_disappeared",
                    "entity_id": entity_id,
                    "severity": 0.7,
                    "timestamp": time.time(),
                    "description": f"实体消失: {entity_id}"
                })
        
        # 检测全局状态异常
        global_anomalies = self._detect_global_anomalies(observation)
        anomalies.extend(global_anomalies)
        
        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "highest_severity": max([a["severity"] for a in anomalies]) if anomalies else 0.0
        }
    
    def _detect_entity_anomalies(self, entity_id: str, entity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测实体异常"""
        anomalies = []
        entity_model = self.entity_models[entity_id]
        
        # 检查属性异常
        if "attributes" in entity_data and "attributes" in entity_model:
            for attr_name, attr_value in entity_data["attributes"].items():
                if attr_name in entity_model["attributes"]:
                    expected_value = entity_model["attributes"][attr_name]
                    
                    # 对数值类型计算偏差
                    if isinstance(attr_value, (int, float)) and isinstance(expected_value, (int, float)):
                        deviation = abs(attr_value - expected_value) / max(1.0, abs(expected_value))
                        
                        # 如果偏差超过阈值，记为异常
                        if deviation > 0.5:
                            anomalies.append({
                                "type": "attribute_anomaly",
                                "entity_id": entity_id,
                                "attribute": attr_name,
                                "expected": expected_value,
                                "actual": attr_value,
                                "deviation": deviation,
                                "severity": min(1.0, deviation),
                                "timestamp": time.time(),
                                "description": f"属性异常: {entity_id}.{attr_name}"
                            })
                    # 对非数值类型检查是否一致
                    elif attr_value != expected_value:
                        anomalies.append({
                            "type": "attribute_anomaly",
                            "entity_id": entity_id,
                            "attribute": attr_name,
                            "expected": expected_value,
                            "actual": attr_value,
                            "severity": 0.8,
                            "timestamp": time.time(),
                            "description": f"属性异常: {entity_id}.{attr_name}"
                        })
        
        return anomalies
    
    def _detect_global_anomalies(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测全局状态异常"""
        anomalies = []
        
        # 检查全局属性
        if "global_attributes" in observation and "global_attributes" in self.state_representation:
            for attr_name, attr_value in observation["global_attributes"].items():
                if attr_name in self.state_representation["global_attributes"]:
                    expected_value = self.state_representation["global_attributes"][attr_name]
                    
                    # 对数值类型计算偏差
                    if isinstance(attr_value, (int, float)) and isinstance(expected_value, (int, float)):
                        deviation = abs(attr_value - expected_value) / max(1.0, abs(expected_value))
                        
                        # 如果偏差超过阈值，记为异常
                        if deviation > 0.5:
                            anomalies.append({
                                "type": "global_anomaly",
                                "attribute": attr_name,
                                "expected": expected_value,
                                "actual": attr_value,
                                "deviation": deviation,
                                "severity": min(1.0, deviation),
                                "timestamp": time.time(),
                                "description": f"全局属性异常: {attr_name}"
                            })
        
        return anomalies
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        获取模型统计
        
        Returns:
            统计信息
        """
        return {
            "observations": self.model_stats["observations"],
            "predictions": self.model_stats["predictions"],
            "updates": self.model_stats["updates"],
            "entities_tracked": self.model_stats["entities_tracked"],
            "average_accuracy": sum(self.model_accuracy.values()) / len(self.model_accuracy) if self.model_accuracy else 0.0,
            "uncertainty": self._calculate_overall_uncertainty(),
            "last_update": self.model_stats["last_update"]
        }
    
    def save_state(self, file_path: str) -> Dict[str, Any]:
        """
        保存模型状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存结果
        """
        try:
            state = {
                "model_parameters": {
                    "state_dimension": self.state_dimension,
                    "prediction_horizon": self.prediction_horizon,
                    "update_rate": self.update_rate
                },
                "model_stats": self.model_stats,
                "model_accuracy": self.model_accuracy,
                "saved_at": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"世界模型状态已保存到: {file_path}")
            
            return {"success": True, "file_path": file_path}
        
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return {"success": False, "error": str(e)} 