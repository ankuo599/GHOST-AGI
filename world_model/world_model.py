# -*- coding: utf-8 -*-
"""
世界模型 (World Model)

负责环境建模、状态预测和多模态信息整合
支持状态跟踪、环境模拟和交互历史记录
"""

import time
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import copy

class WorldModel:
    def __init__(self, memory_system=None):
        """
        初始化世界模型
        
        Args:
            memory_system: 记忆系统实例，用于获取历史数据
        """
        self.memory_system = memory_system
        self.current_state = {}
        self.state_history = []
        self.max_history_size = 100
        self.prediction_horizon = 3  # 预测步数
        self.confidence_threshold = 0.6  # 预测置信度阈值
        self.last_update = time.time()
        
        # 环境表示增强
        self.entity_registry = {}  # 实体注册表
        self.relation_map = {}     # 实体间关系
        self.causal_model = {}     # 因果关系模型
        self.interaction_history = []  # 交互历史
        
        # 多模态状态表示
        self.modality_processors = {
            "text": self._process_text_data,
            "visual": self._process_visual_data,
            "audio": self._process_audio_data,
            "numerical": self._process_numerical_data
        }
        
        # 预测模型
        self.prediction_models = {
            "linear": self._linear_prediction,
            "pattern": self._pattern_based_prediction,
            "causal": self._causal_prediction,
            "ensemble": self._ensemble_prediction
        }
        
        # 当前使用的预测模型
        self.active_prediction_model = "ensemble"
        
        # 初始化基础环境状态
        self._initialize_environment()
        
    def _initialize_environment(self):
        """
        初始化基础环境状态
        """
        # 设置基础环境状态
        self.current_state = {
            "timestamp": time.time(),
            "environment": {
                "type": "system",
                "status": "active",
                "start_time": time.time()
            },
            "entities": {},
            "relations": [],
            "events": [],
            "user_context": {
                "last_interaction": None,
                "interaction_count": 0,
                "preferences": {}
            }
        }
        
        # 记录初始状态
        self._record_state()
        
    def update_state(self, state_update: Dict[str, Any]):
        """
        更新当前世界状态
        
        Args:
            state_update (Dict[str, Any]): 状态更新
            
        Returns:
            Dict[str, Any]: 更新后的状态
        """
        # 深度更新当前状态
        self._deep_update(self.current_state, state_update)
        
        # 更新时间戳
        self.current_state["timestamp"] = time.time()
        self.last_update = time.time()
        
        # 记录状态历史
        self._record_state()
        
        return self.current_state
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        递归深度更新字典
        
        Args:
            target (Dict[str, Any]): 目标字典
            source (Dict[str, Any]): 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                
    def _record_state(self):
        """
        记录当前状态到历史
        """
        # 创建状态快照
        state_snapshot = copy.deepcopy(self.current_state)
        
        # 添加到历史
        self.state_history.append(state_snapshot)
        
        # 限制历史大小
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
            
    def register_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """
        注册实体到世界模型
        
        Args:
            entity_id (str): 实体ID
            entity_data (Dict[str, Any]): 实体数据
            
        Returns:
            bool: 是否成功注册
        """
        if not entity_id:
            entity_id = str(uuid.uuid4())
            
        # 添加基本属性
        if "type" not in entity_data:
            entity_data["type"] = "generic"
        if "created_at" not in entity_data:
            entity_data["created_at"] = time.time()
            
        # 注册到实体注册表
        self.entity_registry[entity_id] = entity_data
        
        # 更新当前状态
        if "entities" not in self.current_state:
            self.current_state["entities"] = {}
        self.current_state["entities"][entity_id] = entity_data
        
        return True
        
    def add_relation(self, source_id: str, target_id: str, relation_type: str, metadata: Dict[str, Any] = None):
        """
        添加实体间关系
        
        Args:
            source_id (str): 源实体ID
            target_id (str): 目标实体ID
            relation_type (str): 关系类型
            metadata (Dict[str, Any], optional): 关系元数据
            
        Returns:
            bool: 是否成功添加
        """
        if source_id not in self.entity_registry or target_id not in self.entity_registry:
            return False
            
        relation_id = str(uuid.uuid4())
        relation = {
            "id": relation_id,
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        
        # 添加到关系映射
        if source_id not in self.relation_map:
            self.relation_map[source_id] = {}
        if target_id not in self.relation_map[source_id]:
            self.relation_map[source_id][target_id] = []
            
        self.relation_map[source_id][target_id].append(relation)
        
        # 更新当前状态
        if "relations" not in self.current_state:
            self.current_state["relations"] = []
        self.current_state["relations"].append(relation)
        
        return True
        
    def record_interaction(self, interaction_type: str, content: Any, metadata: Dict[str, Any] = None):
        """
        记录交互历史
        
        Args:
            interaction_type (str): 交互类型
            content (Any): 交互内容
            metadata (Dict[str, Any], optional): 交互元数据
            
        Returns:
            Dict[str, Any]: 交互记录
        """
        interaction = {
            "id": str(uuid.uuid4()),
            "type": interaction_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # 添加到交互历史
        self.interaction_history.append(interaction)
        
        # 更新当前状态
        if "user_context" not in self.current_state:
            self.current_state["user_context"] = {}
        self.current_state["user_context"]["last_interaction"] = interaction
        self.current_state["user_context"]["interaction_count"] = \
            self.current_state["user_context"].get("interaction_count", 0) + 1
        
        # 添加到事件列表
        if "events" not in self.current_state:
            self.current_state["events"] = []
        self.current_state["events"].append({
            "type": "interaction",
            "interaction_id": interaction["id"],
            "timestamp": interaction["timestamp"]
        })
        
        return interaction
        
    def get_state_history(self, start_time: float = None, end_time: float = None, limit: int = None):
        """
        获取状态历史
        
        Args:
            start_time (float, optional): 开始时间戳
            end_time (float, optional): 结束时间戳
            limit (int, optional): 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 状态历史
        """
        results = []
        
        for state in self.state_history:
            timestamp = state.get("timestamp", 0)
            
            # 时间范围过滤
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
                
            results.append(state)
            
        # 限制结果数量
        if limit and len(results) > limit:
            results = results[-limit:]
            
        return results
        
    def simulate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        模拟执行动作并预测结果
        
        Args:
            action (Dict[str, Any]): 动作描述
            
        Returns:
            Dict[str, Any]: 模拟结果
        """
        # 创建当前状态的副本
        simulated_state = copy.deepcopy(self.current_state)
        
        # 获取动作类型和参数
        action_type = action.get("type")
        params = action.get("params", {})
        
        # 根据动作类型模拟结果
        if action_type == "create_entity":
            entity_id = params.get("id", str(uuid.uuid4()))
            entity_data = params.get("data", {})
            
            if "entities" not in simulated_state:
                simulated_state["entities"] = {}
            simulated_state["entities"][entity_id] = entity_data
            
            return {
                "status": "success",
                "action": action,
                "result": {
                    "entity_id": entity_id,
                    "created": True
                },
                "simulated_state": simulated_state
            }
        elif action_type == "update_entity":
            entity_id = params.get("id")
            updates = params.get("updates", {})
            
            if not entity_id or "entities" not in simulated_state or entity_id not in simulated_state["entities"]:
                return {
                    "status": "error",
                    "action": action,
                    "message": "实体不存在",
                    "simulated_state": simulated_state
                }
                
            # 更新实体
            for key, value in updates.items():
                simulated_state["entities"][entity_id][key] = value
                
            return {
                "status": "success",
                "action": action,
                "result": {
                    "entity_id": entity_id,
                    "updated": True
                },
                "simulated_state": simulated_state
            }
        elif action_type == "add_relation":
            source_id = params.get("source")
            target_id = params.get("target")
            relation_type = params.get("relation_type")
            
            if not source_id or not target_id or not relation_type:
                return {
                    "status": "error",
                    "action": action,
                    "message": "关系参数不完整",
                    "simulated_state": simulated_state
                }
                
            # 检查实体是否存在
            if "entities" not in simulated_state or \
               source_id not in simulated_state["entities"] or \
               target_id not in simulated_state["entities"]:
                return {
                    "status": "error",
                    "action": action,
                    "message": "实体不存在",
                    "simulated_state": simulated_state
                }
                
            # 添加关系
            relation = {
                "id": str(uuid.uuid4()),
                "source": source_id,
                "target": target_id,
                "type": relation_type,
                "created_at": time.time()
            }
            
            if "relations" not in simulated_state:
                simulated_state["relations"] = []
            simulated_state["relations"].append(relation)
            
            return {
                "status": "success",
                "action": action,
                "result": {
                    "relation_id": relation["id"],
                    "created": True
                },
                "simulated_state": simulated_state
            }
        else:
            # 默认情况：未知动作类型
            return {
                "status": "error",
                "action": action,
                "message": f"未知动作类型: {action_type}",
                "simulated_state": simulated_state
            }
        
        # 不确定性估计
        self.uncertainty_estimation = True
        
    def update_state(self, new_observations: Dict[str, Any]) -> bool:
        """
        更新世界状态
        
        Args:
            new_observations: 新的观察数据
            
        Returns:
            bool: 是否成功更新
        """
        if not new_observations:
            return False
            
        # 记录当前状态到历史
        if self.current_state:
            self.state_history.append({
                "state": self.current_state.copy(),
                "timestamp": time.time()
            })
            
            # 限制历史记录大小
            if len(self.state_history) > self.max_history_size:
                self.state_history = self.state_history[-self.max_history_size:]
        
        # 更新当前状态
        for key, value in new_observations.items():
            self.current_state[key] = value
            
        # 更新实体关系
        self._update_entity_relations(new_observations)
            
        self.last_update = time.time()
        return True
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前世界状态
        
        Returns:
            Dict[str, Any]: 当前状态
        """
        return self.current_state.copy()
    
    def predict_next_state(self, actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        预测下一个状态
        
        Args:
            actions: 计划执行的行动列表
            
        Returns:
            Dict[str, Any]: 预测的下一个状态
        """
        if not self.current_state:
            return {}
            
        # 简单预测模型，实际应用中可能需要更复杂的预测算法
        predicted_state = self.current_state.copy()
        
        # 如果有行动，根据行动更新预测
        if actions:
            for action in actions:
                action_type = action.get("action", "")
                params = action.get("params", {})
                
                # 根据行动类型预测状态变化
                self._apply_action_effects(predicted_state, action_type, params)
        
        # 添加预测置信度
        predicted_state["_prediction"] = {
            "confidence": self._calculate_prediction_confidence(),
            "timestamp": time.time()
        }
        
        return predicted_state
    
    def _apply_action_effects(self, state: Dict[str, Any], action_type: str, params: Dict[str, Any]) -> None:
        """
        应用行动效果到状态
        
        Args:
            state: 当前状态
            action_type: 行动类型
            params: 行动参数
        """
        # 这里应该实现针对不同行动类型的效果预测
        # 简单示例实现
        if action_type == "move":
            if "direction" in params:
                direction = params["direction"]
                if "position" in state:
                    position = state["position"].copy()
                    if direction == "north":
                        position[1] += 1
                    elif direction == "south":
                        position[1] -= 1
                    elif direction == "east":
                        position[0] += 1
                    elif direction == "west":
                        position[0] -= 1
                    state["position"] = position
        
        elif action_type == "interact":
            if "object" in params:
                target = params["object"]
                if "inventory" not in state:
                    state["inventory"] = []
                if target not in state["inventory"]:
                    state["inventory"].append(target)
    
    def _calculate_prediction_confidence(self) -> float:
        """
        计算预测置信度
        
        Returns:
            float: 置信度 (0-1)
        """
        # 简单置信度计算，基于历史数据量和最后更新时间
        history_factor = min(1.0, len(self.state_history) / 20.0)  # 历史数据因子
        
        time_factor = 1.0
        time_since_update = time.time() - self.last_update
        if time_since_update > 60:  # 如果超过1分钟没更新
            time_factor = max(0.5, 1.0 - (time_since_update - 60) / 3600)  # 最多降低到0.5
            
        return history_factor * time_factor * 0.9  # 基础置信度0.9
    
    def integrate_multimodal_data(self, visual_data: Dict[str, Any] = None, 
                                 audio_data: Dict[str, Any] = None, 
                                 text_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        整合多模态数据到世界模型
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            text_data: 文本数据
            
        Returns:
            Dict[str, Any]: 整合后的观察数据
        """
        integrated_data = {}
        
        # 整合视觉数据
        if visual_data:
            integrated_data["visual"] = visual_data
            
            # 提取视觉对象到环境对象列表
            if "objects" in visual_data:
                integrated_data["environment_objects"] = visual_data["objects"]
        
        # 整合音频数据
        if audio_data:
            integrated_data["audio"] = audio_data
            
            # 如果有语音转文本，添加到文本数据
            if "speech" in audio_data and audio_data["speech"]:
                if not text_data:
                    text_data = {}
                text_data["speech_text"] = audio_data["speech"]
        
        # 整合文本数据
        if text_data:
            integrated_data["text"] = text_data
        
        # 更新世界状态
        if integrated_data:
            self.update_state(integrated_data)
        
        return integrated_data
    
    def _process_text_data(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文本数据
        
        Args:
            text_data: 文本数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        processed_data = {"type": "text", "processed": True}
        
        # 提取关键信息
        if isinstance(text_data, dict):
            content = text_data.get("content", "")
            if content:
                # 简单的关键词提取
                words = content.lower().split()
                keywords = [word for word in words if len(word) > 3][:10]
                processed_data["keywords"] = keywords
                
                # 实体识别（简化版）
                entities = self._extract_entities(content)
                if entities:
                    processed_data["entities"] = entities
                    
                    # 更新实体注册表
                    for entity in entities:
                        if entity not in self.entity_registry:
                            self.entity_registry[entity] = {
                                "first_seen": time.time(),
                                "mentions": 1,
                                "contexts": [content[:50]]
                            }
                        else:
                            self.entity_registry[entity]["mentions"] += 1
                            if len(self.entity_registry[entity]["contexts"]) < 5:
                                self.entity_registry[entity]["contexts"].append(content[:50])
        
        return processed_data
    
    def _process_visual_data(self, visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视觉数据
        
        Args:
            visual_data: 视觉数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        processed_data = {"type": "visual", "processed": True}
        
        # 提取视觉对象和特征
        if isinstance(visual_data, dict):
            if "objects" in visual_data:
                processed_data["detected_objects"] = visual_data["objects"]
                
                # 更新实体注册表
                for obj in visual_data["objects"]:
                    obj_name = obj.get("name", "")
                    if obj_name:
                        if obj_name not in self.entity_registry:
                            self.entity_registry[obj_name] = {
                                "first_seen": time.time(),
                                "mentions": 1,
                                "visual": True,
                                "properties": obj.get("properties", {})
                            }
                        else:
                            self.entity_registry[obj_name]["mentions"] += 1
                            self.entity_registry[obj_name]["visual"] = True
        
        return processed_data
    
    def _process_audio_data(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理音频数据
        
        Args:
            audio_data: 音频数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        processed_data = {"type": "audio", "processed": True}
        
        # 处理音频内容
        if isinstance(audio_data, dict):
            if "speech" in audio_data:
                processed_data["speech_content"] = audio_data["speech"]
                
                # 如果有语音内容，也进行文本处理
                text_result = self._process_text_data({"content": audio_data["speech"]})
                processed_data.update(text_result)
        
        return processed_data
    
    def _process_numerical_data(self, numerical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数值数据
        
        Args:
            numerical_data: 数值数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        processed_data = {"type": "numerical", "processed": True}
        
        # 处理数值特征
        if isinstance(numerical_data, dict):
            # 计算基本统计量
            stats = {}
            for key, value in numerical_data.items():
                if isinstance(value, (int, float)):
                    stats[key] = value
            
            if stats:
                processed_data["statistics"] = stats
        
        return processed_data
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 实体列表
        """
        # 简单实现，实际应用中可能需要更复杂的NLP技术
        entities = []
        
        # 大写开头的词可能是实体
        words = text.split()
        for word in words:
            if len(word) > 1 and word[0].isupper():
                # 清理标点符号
                clean_word = word.strip('.,;:!?()"\'[]{}').strip()
                if clean_word and len(clean_word) > 1:
                    entities.append(clean_word)
        
        return list(set(entities))  # 去重
    
    def _linear_prediction(self, state_history: List[Dict[str, Any]], horizon: int = 1) -> List[Dict[str, Any]]:
        """
        基于线性趋势的预测
        
        Args:
            state_history: 状态历史
            horizon: 预测步数
            
        Returns:
            List[Dict[str, Any]]: 预测状态列表
        """
        if not state_history or horizon < 1:
            return []
            
        predictions = []
        current_state = state_history[-1]["state"] if isinstance(state_history[-1], dict) else state_history[-1]
        
        # 简单线性预测
        for step in range(horizon):
            predicted_state = current_state.copy()
            
            # 对数值型状态进行线性预测
            if len(state_history) >= 3:
                for key, value in current_state.items():
                    if isinstance(value, (int, float)):
                        # 获取历史值
                        history_values = []
                        for h in state_history[-3:]:
                            h_state = h["state"] if isinstance(h, dict) else h
                            if key in h_state and isinstance(h_state[key], (int, float)):
                                history_values.append(h_state[key])
                        
                        # 如果有足够的历史数据，计算趋势
                        if len(history_values) >= 2:
                            trend = (history_values[-1] - history_values[0]) / len(history_values)
                            predicted_state[key] = value + trend * (step + 1)
            
            # 添加预测置信度
            predicted_state["_prediction"] = {
                "method": "linear",
                "confidence": max(0.3, 0.8 - 0.1 * step),  # 随着步数增加，置信度降低
                "timestamp": time.time()
            }
            
            predictions.append(predicted_state)
            current_state = predicted_state.copy()
        
        return predictions
    
    def _pattern_based_prediction(self, state_history: List[Dict[str, Any]], horizon: int = 1) -> List[Dict[str, Any]]:
        """
        基于模式识别的预测
        
        Args:
            state_history: 状态历史
            horizon: 预测步数
            
        Returns:
            List[Dict[str, Any]]: 预测状态列表
        """
        # 模式识别预测（简化版）
        predictions = []
        current_state = state_history[-1]["state"] if isinstance(state_history[-1], dict) else state_history[-1]
        
        for step in range(horizon):
            predicted_state = current_state.copy()
            
            # 添加预测置信度
            predicted_state["_prediction"] = {
                "method": "pattern",
                "confidence": max(0.4, 0.7 - 0.1 * step),
                "timestamp": time.time()
            }
            
            predictions.append(predicted_state)
            current_state = predicted_state.copy()
        
        return predictions
    
    def _causal_prediction(self, state_history: List[Dict[str, Any]], horizon: int = 1) -> List[Dict[str, Any]]:
        """
        基于因果关系的预测
        
        Args:
            state_history: 状态历史
            horizon: 预测步数
            
        Returns:
            List[Dict[str, Any]]: 预测状态列表
        """
        # 因果关系预测（简化版）
        predictions = []
        current_state = state_history[-1]["state"] if isinstance(state_history[-1], dict) else state_history[-1]
        
        for step in range(horizon):
            predicted_state = current_state.copy()
            
            # 应用因果模型
            for cause, effects in self.causal_model.items():
                if cause in current_state:
                    cause_value = current_state[cause]
                    for effect, params in effects.items():
                        if "weight" in params and "delay" in params:
                            # 如果延迟步数匹配当前预测步数
                            if params["delay"] == step:
                                effect_value = cause_value * params["weight"]
                                predicted_state[effect] = effect_value
            
            # 添加预测置信度
            predicted_state["_prediction"] = {
                "method": "causal",
                "confidence": max(0.5, 0.9 - 0.1 * step),
                "timestamp": time.time()
            }
            
            predictions.append(predicted_state)
            current_state = predicted_state.copy()
        
        return predictions
    
    def _ensemble_prediction(self, state_history: List[Dict[str, Any]], horizon: int = 1) -> List[Dict[str, Any]]:
        """
        集成多种预测方法
        
        Args:
            state_history: 状态历史
            horizon: 预测步数
            
        Returns:
            List[Dict[str, Any]]: 预测状态列表
        """
        # 获取各种预测方法的结果
        linear_predictions = self._linear_prediction(state_history, horizon)
        pattern_predictions = self._pattern_based_prediction(state_history, horizon)
        causal_predictions = self._causal_prediction(state_history, horizon)
        
        # 集成预测结果
        ensemble_predictions = []
        
        for step in range(horizon):
            if step < len(linear_predictions) and step < len(pattern_predictions) and step < len(causal_predictions):
                linear_pred = linear_predictions[step]
                pattern_pred = pattern_predictions[step]
                causal_pred = causal_predictions[step]
                
                # 创建集成预测
                ensemble_state = {}
                
                # 合并所有预测的键
                all_keys = set()
                for pred in [linear_pred, pattern_pred, causal_pred]:
                    all_keys.update(pred.keys())
                
                # 移除预测元数据键
                if "_prediction" in all_keys:
                    all_keys.remove("_prediction")
                
                # 对每个键进行加权平均
                for key in all_keys:
                    values = []
                    weights = []
                    
                    # 收集各预测方法的值和权重
                    for pred, method_name in [(linear_pred, "linear"), (pattern_pred, "pattern"), (causal_pred, "causal")]:
                        if key in pred and isinstance(pred[key], (int, float)):
                            values.append(pred[key])
                            # 使用预测置信度作为权重
                            if "_prediction" in pred and "confidence" in pred["_prediction"]:
                                weights.append(pred["_prediction"]["confidence"])
                            else:
                                weights.append(0.5)  # 默认权重
                    
                    # 计算加权平均值
                    if values and weights:
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_weight
                            ensemble_state[key] = weighted_avg
                        else:
                            # 如果权重总和为0，使用简单平均
                            ensemble_state[key] = sum(values) / len(values)
                
                # 添加预测元数据
                ensemble_state["_prediction"] = {
                    "method": "ensemble",
                    "confidence": 0.8 - 0.1 * step,  # 随着步数增加，置信度降低
                    "timestamp": time.time(),
                    "component_methods": ["linear", "pattern", "causal"]
                }
                
                ensemble_predictions.append(ensemble_state)
        
        return ensemble_predictions
    
    def get_environment_description(self) -> str:
        """
        获取当前环境的文本描述
        
        Returns:
            str: 环境描述
        """
        if not self.current_state:
            return "环境状态未知"
        
        # 构建环境描述
        description = ["当前环境:"]
        
        # 添加位置信息
        if "position" in self.current_state:
            pos = self.current_state["position"]
            description.append(f"位置: ({pos[0]}, {pos[1]})")
        
        # 添加环境对象
        if "environment_objects" in self.current_state:
            objects = self.current_state["environment_objects"]
            if objects:
                description.append("环境中的对象:")
                for obj in objects:
                    description.append(f"  - {obj}")
            else:
                description.append("环境中没有可见对象")
        
        # 添加其他环境属性
        for key, value in self.current_state.items():
            if key not in ["position", "environment_objects", "visual", "audio", "text", "_prediction"]:
                description.append(f"{key}: {value}")
        
        return "\n".join(description)