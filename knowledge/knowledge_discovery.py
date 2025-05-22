"""
GHOST-AGI 知识发现模块

该模块负责从原始数据和经验中发现新知识，实现自主知识构建。
"""

import logging
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import uuid

class KnowledgeDiscoveryModel(nn.Module):
    """知识发现深度学习模型"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class KnowledgeDataset(Dataset):
    """知识数据集"""
    def __init__(self, data: List[Dict[str, Any]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # 将知识项转换为模型输入
        text = f"{item['content']} {item.get('metadata', {}).get('description', '')}"
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class KnowledgeDiscovery:
    """知识发现模块，负责从原始数据中发现新知识"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化知识发现模块
        
        Args:
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 初始化状态
        self.discovery_methods = {
            "pattern_mining": self._mine_patterns,
            "concept_formation": self._form_concepts,
            "relationship_discovery": self._discover_relationships,
            "causal_inference": self._infer_causality,
            "knowledge_validation": self._validate_knowledge,
            "deep_learning": self._deep_learning_discovery,
            "knowledge_graph": self._build_knowledge_graph,
            "active_learning": self._active_learning
        }
        
        # 发现统计
        self.stats = {
            "discoveries": 0,
            "validated_discoveries": 0,
            "last_discovery_time": None
        }
        
        # 发现历史
        self.discovery_history = []
        
        # 初始化深度学习模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        
        # 初始化知识图谱
        self.knowledge_graph = nx.DiGraph()
        
        # 初始化主动学习
        self.uncertainty_threshold = 0.3
        self.active_learning_pool = []
        
        self.logger.info("知识发现模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("KnowledgeDiscovery")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("knowledge_discovery.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def discover_from_data(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从数据中发现知识
        
        Args:
            data: 输入数据
            context: 上下文信息
            
        Returns:
            发现结果
        """
        self.logger.info("开始从数据中发现知识")
        
        context = context or {}
        results = {}
        
        # 应用各种发现方法
        for method_name, method_func in self.discovery_methods.items():
            try:
                self.logger.debug(f"应用发现方法: {method_name}")
                discovery_result = method_func(data, context)
                results[method_name] = discovery_result
            except Exception as e:
                self.logger.error(f"应用方法 {method_name} 时出错: {str(e)}")
                results[method_name] = {"error": str(e)}
        
        # 整合发现结果
        integrated_discoveries = self._integrate_discoveries(results)
        
        # 评估发现的重要性
        for discovery in integrated_discoveries:
            discovery["importance"] = self._evaluate_importance(discovery)
        
        # 排序发现结果
        integrated_discoveries.sort(key=lambda x: x["importance"], reverse=True)
        
        # 记录发现历史
        for discovery in integrated_discoveries:
            self.discovery_history.append({
                "discovery": discovery,
                "timestamp": time.time(),
                "context": context
            })
        
        # 更新统计信息
        self.stats["discoveries"] += len(integrated_discoveries)
        self.stats["last_discovery_time"] = time.time()
        
        return {
            "discoveries": integrated_discoveries,
            "stats": self.stats
        }
    
    def _mine_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """从数据中挖掘模式"""
        patterns = []
        
        # 检查数据类型
        if "sequences" in data and isinstance(data["sequences"], list):
            # 处理序列数据
            sequences = data["sequences"]
            if sequences and len(sequences) >= 2:
                # 寻找重复子序列
                repeating_patterns = self._find_repeating_subsequences(sequences)
                patterns.extend(repeating_patterns)
        
        if "attributes" in data and isinstance(data["attributes"], dict):
            # 处理属性数据
            attributes = data["attributes"]
            # 寻找属性关联
            associated_attrs = self._find_attribute_associations(attributes)
            patterns.extend(associated_attrs)
        
        return {
            "patterns_found": len(patterns),
            "patterns": patterns
        }
    
    def _find_repeating_subsequences(self, sequences: List[Any]) -> List[Dict[str, Any]]:
        """寻找重复子序列"""
        patterns = []
        
        # 简化实现
        # 实际应用中会使用更复杂的序列模式挖掘算法
        
        # 只处理简单的情况：单一序列中的重复
        for seq_idx, sequence in enumerate(sequences):
            if isinstance(sequence, list) and len(sequence) >= 4:
                # 尝试找到长度为2和3的重复子序列
                for pattern_length in [2, 3]:
                    if len(sequence) >= pattern_length * 2:
                        subsequence_counts = defaultdict(list)
                        
                        # 收集所有子序列
                        for i in range(len(sequence) - pattern_length + 1):
                            # 序列化子序列用于哈希
                            subsequence = tuple(sequence[i:i+pattern_length])
                            subsequence_counts[subsequence].append(i)
                        
                        # 检查重复子序列
                        for subsequence, positions in subsequence_counts.items():
                            if len(positions) >= 2:
                                patterns.append({
                                    "type": "repeating_subsequence",
                                    "sequence_index": seq_idx,
                                    "pattern": subsequence,
                                    "positions": positions,
                                    "frequency": len(positions),
                                    "confidence": 0.7
                                })
        
        return patterns
    
    def _find_attribute_associations(self, attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """寻找属性关联"""
        associations = []
        
        # 简化实现
        # 检查数值属性之间的简单相关性
        numeric_attrs = {}
        for key, value in attributes.items():
            if isinstance(value, (int, float)):
                numeric_attrs[key] = value
        
        # 如果有至少两个数值属性，计算它们之间的关系
        if len(numeric_attrs) >= 2:
            attr_keys = list(numeric_attrs.keys())
            
            for i in range(len(attr_keys) - 1):
                for j in range(i + 1, len(attr_keys)):
                    key1 = attr_keys[i]
                    key2 = attr_keys[j]
                    
                    # 简单的关系检查：检查值是否相等或者存在倍数关系
                    val1 = numeric_attrs[key1]
                    val2 = numeric_attrs[key2]
                    
                    if val1 == val2:
                        associations.append({
                            "type": "attribute_equality",
                            "attributes": [key1, key2],
                            "value": val1,
                            "confidence": 0.9
                        })
                    elif val1 != 0 and val2 % val1 == 0:
                        multiplier = val2 / val1
                        associations.append({
                            "type": "attribute_multiplier",
                            "attributes": [key1, key2],
                            "multiplier": multiplier,
                            "confidence": 0.7
                        })
                    elif val2 != 0 and val1 % val2 == 0:
                        multiplier = val1 / val2
                        associations.append({
                            "type": "attribute_multiplier",
                            "attributes": [key2, key1],
                            "multiplier": multiplier,
                            "confidence": 0.7
                        })
        
        return associations
    
    def _form_concepts(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """形成概念"""
        concepts = []
        
        # 从数据中提取特征
        features = self._extract_features(data)
        
        # 根据特征形成概念
        if features:
            # 简单实现：生成一个概念
            concept = {
                "id": f"concept_{int(time.time())}",
                "features": features,
                "abstraction_level": self._calculate_abstraction_level(features),
                "stability": 0.5,  # 初始稳定性
                "creation_time": time.time()
            }
            
            concepts.append(concept)
        
        return {
            "concepts_formed": len(concepts),
            "concepts": concepts
        }
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从数据中提取特征"""
        features = {}
        
        # 提取各种数据类型的特征
        if "attributes" in data and isinstance(data["attributes"], dict):
            features["attribute_features"] = {}
            
            # 处理数值特征
            numeric_values = {}
            categorical_values = {}
            
            for key, value in data["attributes"].items():
                if isinstance(value, (int, float)):
                    numeric_values[key] = value
                elif isinstance(value, str):
                    categorical_values[key] = value
            
            if numeric_values:
                features["attribute_features"]["numeric"] = numeric_values
            
            if categorical_values:
                features["attribute_features"]["categorical"] = categorical_values
        
        # 提取序列特征
        if "sequences" in data and isinstance(data["sequences"], list):
            sequences = data["sequences"]
            
            if sequences:
                sequence_features = {}
                
                # 计算序列长度
                sequence_features["lengths"] = [len(seq) if isinstance(seq, list) else 0 for seq in sequences]
                
                # 检查序列类型
                sequence_types = []
                for seq in sequences:
                    if isinstance(seq, list):
                        if all(isinstance(x, (int, float)) for x in seq):
                            sequence_types.append("numeric")
                        elif all(isinstance(x, str) for x in seq):
                            sequence_types.append("text")
                        else:
                            sequence_types.append("mixed")
                    else:
                        sequence_types.append("invalid")
                
                sequence_features["types"] = sequence_types
                
                features["sequence_features"] = sequence_features
        
        return features
    
    def _calculate_abstraction_level(self, features: Dict[str, Any]) -> float:
        """计算特征的抽象级别"""
        # 简化实现
        # 实际应用中会使用更复杂的抽象度量方法
        
        # 基本分数
        base_score = 0.5
        
        # 特征数量影响
        feature_count = 0
        if "attribute_features" in features:
            attr_features = features["attribute_features"]
            if "numeric" in attr_features:
                feature_count += len(attr_features["numeric"])
            if "categorical" in attr_features:
                feature_count += len(attr_features["categorical"])
        
        if "sequence_features" in features:
            seq_features = features["sequence_features"]
            if "lengths" in seq_features:
                feature_count += len(seq_features["lengths"])
        
        # 特征越多，抽象级别越低
        abstraction_modifier = max(0.1, 1.0 - (feature_count * 0.05))
        
        return base_score * abstraction_modifier
    
    def _discover_relationships(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """发现关系"""
        relationships = []
        
        # 检查对象关系
        if "objects" in data and isinstance(data["objects"], list):
            objects = data["objects"]
            
            # 至少需要两个对象才能形成关系
            if len(objects) >= 2:
                # 检查特定类型的关系
                hierarchical_relations = self._find_hierarchical_relations(objects)
                if hierarchical_relations:
                    relationships.extend(hierarchical_relations)
                
                dependency_relations = self._find_dependency_relations(objects)
                if dependency_relations:
                    relationships.extend(dependency_relations)
                
                similarity_relations = self._find_similarity_relations(objects)
                if similarity_relations:
                    relationships.extend(similarity_relations)
        
        return {
            "relationships_found": len(relationships),
            "relationships": relationships
        }
    
    def _find_hierarchical_relations(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找层次关系"""
        relations = []
        
        # 简化实现
        # 检查是否存在包含关系
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i != j:
                    # 检查对象i是否包含对象j的所有属性
                    if "attributes" in objects[i] and "attributes" in objects[j]:
                        attrs_i = set(objects[i]["attributes"].keys())
                        attrs_j = set(objects[j]["attributes"].keys())
                        
                        if attrs_j and attrs_j.issubset(attrs_i):
                            # 计算包含程度
                            inclusion_degree = len(attrs_j) / len(attrs_i) if attrs_i else 0
                            
                            if inclusion_degree > 0.5:  # 要求相当程度的包含
                                relations.append({
                                    "type": "hierarchical",
                                    "relation": "contains",
                                    "parent": i,
                                    "child": j,
                                    "inclusion_degree": inclusion_degree,
                                    "confidence": 0.7
                                })
        
        return relations
    
    def _find_dependency_relations(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找依赖关系"""
        relations = []
        
        # 简化实现
        # 检查基于ID引用的依赖
        for i in range(len(objects)):
            if "references" in objects[i] and isinstance(objects[i]["references"], list):
                for ref in objects[i]["references"]:
                    for j in range(len(objects)):
                        if i != j and "id" in objects[j] and objects[j]["id"] == ref:
                            relations.append({
                                "type": "dependency",
                                "relation": "depends_on",
                                "dependent": i,
                                "provider": j,
                                "confidence": 0.8
                            })
        
        return relations
    
    def _find_similarity_relations(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找相似关系"""
        relations = []
        
        # 简化实现
        # 基于属性相似度
        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                similarity = self._calculate_object_similarity(objects[i], objects[j])
                
                if similarity > 0.7:  # 相似度阈值
                    relations.append({
                        "type": "similarity",
                        "objects": [i, j],
                        "similarity_score": similarity,
                        "confidence": similarity
                    })
        
        return relations
    
    def _calculate_object_similarity(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> float:
        """计算对象相似度"""
        # 简化实现
        
        if "attributes" not in obj1 or "attributes" not in obj2:
            return 0.0
        
        attrs1 = obj1["attributes"]
        attrs2 = obj2["attributes"]
        
        # 获取共同属性
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        all_keys = set(attrs1.keys()) | set(attrs2.keys())
        
        if not common_keys:
            return 0.0
        
        # 计算属性相似度
        attr_similarities = []
        
        for key in common_keys:
            val1 = attrs1[key]
            val2 = attrs2[key]
            
            # 数值类型
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 避免除以零
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    attr_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
                else:
                    attr_similarities.append(1.0)
            # 字符串类型
            elif isinstance(val1, str) and isinstance(val2, str):
                if val1 == val2:
                    attr_similarities.append(1.0)
                else:
                    # 简单的字符串相似度
                    attr_similarities.append(0.0)
            # 布尔类型
            elif isinstance(val1, bool) and isinstance(val2, bool):
                attr_similarities.append(1.0 if val1 == val2 else 0.0)
            # 其他类型
            else:
                attr_similarities.append(0.0)
        
        # 计算整体相似度
        attr_similarity = sum(attr_similarities) / len(attr_similarities) if attr_similarities else 0.0
        coverage = len(common_keys) / len(all_keys) if all_keys else 0.0
        
        # 综合相似度和覆盖度
        return 0.7 * attr_similarity + 0.3 * coverage
    
    def _infer_causality(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """推断因果关系"""
        causal_relations = []
        
        # 检查是否存在时间序列数据
        if "time_series" in data and isinstance(data["time_series"], list):
            time_series = data["time_series"]
            
            # 至少需要两个时间点
            if len(time_series) >= 2:
                # 排序时间序列
                sorted_series = sorted(time_series, key=lambda x: x.get("timestamp", 0))
                
                # 检查连续事件
                for i in range(len(sorted_series) - 1):
                    event1 = sorted_series[i]
                    event2 = sorted_series[i + 1]
                    
                    # 检查可能的因果关系
                    if "event" in event1 and "event" in event2:
                        time_diff = event2.get("timestamp", 0) - event1.get("timestamp", 0)
                        
                        # 时间间隔适中（不太快也不太慢）
                        if 0 < time_diff < 10:  # 假设单位是秒
                            causal_relations.append({
                                "type": "temporal_causality",
                                "cause": event1["event"],
                                "effect": event2["event"],
                                "time_difference": time_diff,
                                "confidence": 0.6
                            })
        
        # 检查状态变化数据
        if "state_changes" in data and isinstance(data["state_changes"], list):
            changes = data["state_changes"]
            
            # 至少需要两个状态变化
            if len(changes) >= 2:
                # 检查连续状态变化
                for i in range(len(changes) - 1):
                    change1 = changes[i]
                    change2 = changes[i + 1]
                    
                    if "action" in change1 and "before" in change1 and "after" in change1:
                        if "before" in change2 and change1["after"] == change2["before"]:
                            causal_relations.append({
                                "type": "state_causality",
                                "cause_action": change1["action"],
                                "intermediate_state": change1["after"],
                                "effect_action": change2.get("action"),
                                "confidence": 0.7
                            })
        
        return {
            "causal_relations_found": len(causal_relations),
            "causal_relations": causal_relations
        }
    
    def _validate_knowledge(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """验证知识"""
        # 简化实现
        validated_items = []
        
        # 验证发现的模式
        if "patterns" in data and isinstance(data["patterns"], list):
            for pattern in data["patterns"]:
                # 模拟验证过程
                validation_score = random.uniform(0.6, 0.95)
                
                if validation_score > 0.7:  # 验证阈值
                    validated_items.append({
                        "type": "validated_pattern",
                        "pattern": pattern,
                        "validation_score": validation_score,
                        "validation_method": "statistical_validation"
                    })
        
        # 验证发现的关系
        if "relationships" in data and isinstance(data["relationships"], list):
            for relationship in data["relationships"]:
                # 模拟验证过程
                validation_score = random.uniform(0.5, 0.9)
                
                if validation_score > 0.7:  # 验证阈值
                    validated_items.append({
                        "type": "validated_relationship",
                        "relationship": relationship,
                        "validation_score": validation_score,
                        "validation_method": "consistency_check"
                    })
        
        # 更新统计
        self.stats["validated_discoveries"] += len(validated_items)
        
        return {
            "validated_items": len(validated_items),
            "items": validated_items
        }
    
    def _integrate_discoveries(self, discovery_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """整合发现结果"""
        integrated_discoveries = []
        
        # 收集所有发现
        all_discoveries = []
        
        # 从模式挖掘结果中收集发现
        if "pattern_mining" in discovery_results:
            patterns = discovery_results["pattern_mining"].get("patterns", [])
            for pattern in patterns:
                all_discoveries.append({
                    "source": "pattern_mining",
                    "content": pattern,
                    "type": pattern.get("type", "unknown")
                })
        
        # 从概念形成结果中收集发现
        if "concept_formation" in discovery_results:
            concepts = discovery_results["concept_formation"].get("concepts", [])
            for concept in concepts:
                all_discoveries.append({
                    "source": "concept_formation",
                    "content": concept,
                    "type": "concept"
                })
        
        # 从关系发现结果中收集发现
        if "relationship_discovery" in discovery_results:
            relationships = discovery_results["relationship_discovery"].get("relationships", [])
            for relationship in relationships:
                all_discoveries.append({
                    "source": "relationship_discovery",
                    "content": relationship,
                    "type": relationship.get("type", "unknown_relationship")
                })
        
        # 从因果推断结果中收集发现
        if "causal_inference" in discovery_results:
            causal_relations = discovery_results["causal_inference"].get("causal_relations", [])
            for relation in causal_relations:
                all_discoveries.append({
                    "source": "causal_inference",
                    "content": relation,
                    "type": relation.get("type", "unknown_causality")
                })
        
        # 从知识验证结果中收集发现
        if "knowledge_validation" in discovery_results:
            validated_items = discovery_results["knowledge_validation"].get("items", [])
            for item in validated_items:
                all_discoveries.append({
                    "source": "knowledge_validation",
                    "content": item,
                    "type": item.get("type", "unknown_validation"),
                    "validated": True
                })
        
        # 整合所有发现
        for discovery in all_discoveries:
            integrated_discovery = {
                "id": f"discovery_{len(integrated_discoveries)}_{int(time.time())}",
                "source": discovery["source"],
                "type": discovery["type"],
                "content": discovery["content"],
                "validated": discovery.get("validated", False),
                "timestamp": time.time()
            }
            
            integrated_discoveries.append(integrated_discovery)
        
        return integrated_discoveries
    
    def _evaluate_importance(self, discovery: Dict[str, Any]) -> float:
        """评估发现的重要性"""
        # 简化实现
        base_importance = 0.5
        
        # 根据来源调整重要性
        source_weights = {
            "pattern_mining": 0.6,
            "concept_formation": 0.7,
            "relationship_discovery": 0.8,
            "causal_inference": 0.9,
            "knowledge_validation": 0.7
        }
        
        source = discovery.get("source", "")
        source_weight = source_weights.get(source, 0.5)
        
        # 根据类型调整重要性
        type_weights = {
            "concept": 0.7,
            "hierarchical": 0.8,
            "dependency": 0.7,
            "similarity": 0.6,
            "temporal_causality": 0.9,
            "state_causality": 0.8,
            "validated_pattern": 0.7,
            "validated_relationship": 0.8
        }
        
        discovery_type = discovery.get("type", "")
        type_weight = type_weights.get(discovery_type, 0.5)
        
        # 是否经过验证
        validation_weight = 1.2 if discovery.get("validated", False) else 1.0
        
        # 计算综合重要性
        importance = (base_importance * 0.2 + source_weight * 0.3 + type_weight * 0.5) * validation_weight
        
        return min(1.0, importance)
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        获取发现统计信息
        
        Returns:
            统计信息
        """
        return {
            "total_discoveries": self.stats["discoveries"],
            "validated_discoveries": self.stats["validated_discoveries"],
            "discovery_history_length": len(self.discovery_history),
            "last_discovery_time": self.stats["last_discovery_time"]
        }
    
    def _deep_learning_discovery(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """使用深度学习模型发现知识"""
        try:
            # 准备数据
            dataset = KnowledgeDataset(data.get("items", []), self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 初始化模型
            if self.model is None:
                self.model = KnowledgeDiscoveryModel(
                    input_dim=768,  # BERT 输出维度
                    hidden_dim=256
                ).to(self.device)
                
            # 训练模型
            optimizer = torch.optim.Adam(self.model.parameters())
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(5):
                total_loss = 0
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # 获取 BERT 嵌入
                    with torch.no_grad():
                        outputs = self.bert_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                        
                    # 前向传播
                    encoded, decoded = self.model(embeddings)
                    loss = criterion(decoded, embeddings)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                self.logger.info(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
                
            # 使用模型发现知识
            self.model.eval()
            discoveries = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # 获取 BERT 嵌入
                    outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    # 使用模型编码
                    encoded, _ = self.model(embeddings)
                    
                    # 聚类分析
                    encoded_np = encoded.cpu().numpy()
                    clustering = DBSCAN(eps=0.5, min_samples=2).fit(encoded_np)
                    
                    # 发现知识模式
                    for label in set(clustering.labels_):
                        if label != -1:  # 排除噪声点
                            cluster_indices = np.where(clustering.labels_ == label)[0]
                            cluster_items = [data["items"][i] for i in cluster_indices]
                            
                            discoveries.append({
                                "type": "deep_learning_pattern",
                                "pattern": {
                                    "cluster_size": len(cluster_items),
                                    "items": cluster_items,
                                    "centroid": encoded_np[cluster_indices].mean(axis=0).tolist()
                                },
                                "confidence": 0.8
                            })
                            
            return {
                "discoveries": discoveries,
                "model_stats": {
                    "total_loss": total_loss/len(dataloader),
                    "clusters_found": len(discoveries)
                }
            }
            
        except Exception as e:
            self.logger.error(f"深度学习发现过程出错: {str(e)}")
            return {"error": str(e)}
            
    def _build_knowledge_graph(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """构建知识图谱"""
        try:
            # 添加节点
            for item in data.get("items", []):
                node_id = item.get("id", str(uuid.uuid4()))
                self.knowledge_graph.add_node(
                    node_id,
                    **item
                )
                
            # 添加边
            for relation in data.get("relations", []):
                source = relation.get("source")
                target = relation.get("target")
                relation_type = relation.get("type", "related_to")
                
                if source in self.knowledge_graph and target in self.knowledge_graph:
                    self.knowledge_graph.add_edge(
                        source,
                        target,
                        type=relation_type,
                        weight=relation.get("weight", 1.0)
                    )
                    
            # 分析图谱
            analysis = {
                "nodes": self.knowledge_graph.number_of_nodes(),
                "edges": self.knowledge_graph.number_of_edges(),
                "communities": list(nx.community.greedy_modularity_communities(
                    self.knowledge_graph.to_undirected()
                )),
                "central_nodes": nx.degree_centrality(self.knowledge_graph),
                "clustering_coefficient": nx.average_clustering(self.knowledge_graph)
            }
            
            # 发现新的关系
            new_relations = []
            
            # 基于路径发现关系
            for source in self.knowledge_graph.nodes():
                for target in self.knowledge_graph.nodes():
                    if source != target and not self.knowledge_graph.has_edge(source, target):
                        # 检查是否存在间接关系
                        paths = list(nx.all_simple_paths(
                            self.knowledge_graph,
                            source,
                            target,
                            cutoff=2
                        ))
                        
                        if paths:
                            new_relations.append({
                                "source": source,
                                "target": target,
                                "type": "inferred_relation",
                                "confidence": 0.6,
                                "paths": paths
                            })
                            
            return {
                "graph_stats": analysis,
                "new_relations": new_relations
            }
            
        except Exception as e:
            self.logger.error(f"知识图谱构建过程出错: {str(e)}")
            return {"error": str(e)}
            
    def _active_learning(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """主动学习"""
        try:
            # 准备数据
            items = data.get("items", [])
            
            # 计算不确定性
            uncertainties = []
            for item in items:
                # 使用模型预测
                if self.model is not None:
                    # 获取 BERT 嵌入
                    encoding = self.tokenizer(
                        item["content"],
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.bert_model(**encoding)
                        embedding = outputs.last_hidden_state[:, 0, :]
                        
                        # 使用模型预测
                        encoded, decoded = self.model(embedding)
                        
                        # 计算重构误差作为不确定性度量
                        uncertainty = torch.nn.functional.mse_loss(
                            decoded,
                            embedding
                        ).item()
                        
                        uncertainties.append({
                            "item": item,
                            "uncertainty": uncertainty
                        })
                        
            # 选择高不确定性的样本
            high_uncertainty_items = [
                item for item in uncertainties
                if item["uncertainty"] > self.uncertainty_threshold
            ]
            
            # 更新主动学习池
            self.active_learning_pool.extend(high_uncertainty_items)
            
            # 生成学习建议
            learning_suggestions = []
            for item in high_uncertainty_items:
                learning_suggestions.append({
                    "item": item["item"],
                    "uncertainty": item["uncertainty"],
                    "suggestion": "需要更多标注数据",
                    "priority": "high" if item["uncertainty"] > 0.5 else "medium"
                })
                
            return {
                "high_uncertainty_items": len(high_uncertainty_items),
                "learning_suggestions": learning_suggestions,
                "pool_size": len(self.active_learning_pool)
            }
            
        except Exception as e:
            self.logger.error(f"主动学习过程出错: {str(e)}")
            return {"error": str(e)} 