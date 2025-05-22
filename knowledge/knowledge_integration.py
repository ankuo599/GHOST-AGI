"""
GHOST-AGI 知识整合模块

该模块负责整合和组织来自不同来源的知识，形成一个连贯的知识网络。
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass
import json
from pathlib import Path
import hashlib

@dataclass
class KnowledgeVersion:
    """知识版本"""
    version_id: str
    timestamp: float
    changes: Dict[str, Any]
    author: str
    description: str
    parent_version: Optional[str] = None

class KnowledgeQuality:
    """知识质量评估"""
    def __init__(self):
        self.quality_metrics = {
            "completeness": self._evaluate_completeness,
            "consistency": self._evaluate_consistency,
            "relevance": self._evaluate_relevance,
            "reliability": self._evaluate_reliability
        }
        
    def _evaluate_completeness(self, knowledge: Dict[str, Any]) -> float:
        """评估知识完整性"""
        score = 0.0
        total_attributes = 0
        
        # 检查节点属性完整性
        for node in knowledge["nodes"]:
            required_attrs = {"id", "type", "content"}
            present_attrs = set(node.keys())
            score += len(required_attrs & present_attrs) / len(required_attrs)
            total_attributes += 1
            
        # 检查边属性完整性
        for edge in knowledge["edges"]:
            required_attrs = {"source", "target", "type"}
            present_attrs = set(edge.keys())
            score += len(required_attrs & present_attrs) / len(required_attrs)
            total_attributes += 1
            
        return score / total_attributes if total_attributes > 0 else 0.0
        
    def _evaluate_consistency(self, knowledge: Dict[str, Any]) -> float:
        """评估知识一致性"""
        score = 1.0
        
        # 检查节点ID唯一性
        node_ids = set()
        for node in knowledge["nodes"]:
            if node["id"] in node_ids:
                score *= 0.5
            node_ids.add(node["id"])
            
        # 检查边引用有效性
        for edge in knowledge["edges"]:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                score *= 0.5
                
        return score
        
    def _evaluate_relevance(self, knowledge: Dict[str, Any]) -> float:
        """评估知识相关性"""
        # 使用预训练模型计算语义相似度
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        # 计算节点间的平均相似度
        similarities = []
        for i, node1 in enumerate(knowledge["nodes"]):
            for node2 in knowledge["nodes"][i+1:]:
                # 获取节点文本
                text1 = f"{node1.get('content', '')} {node1.get('type', '')}"
                text2 = f"{node2.get('content', '')} {node2.get('type', '')}"
                
                # 计算相似度
                inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
                inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs1 = model(**inputs1)
                    outputs2 = model(**inputs2)
                    
                # 使用[CLS]标记的嵌入
                embedding1 = outputs1.last_hidden_state[:, 0, :]
                embedding2 = outputs2.last_hidden_state[:, 0, :]
                
                similarity = cosine_similarity(
                    embedding1.numpy(),
                    embedding2.numpy()
                )[0][0]
                
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 0.0
        
    def _evaluate_reliability(self, knowledge: Dict[str, Any]) -> float:
        """评估知识可靠性"""
        score = 1.0
        
        # 检查来源可靠性
        for node in knowledge["nodes"]:
            source = node.get("metadata", {}).get("source", "")
            if source in {"user_input", "inferred"}:
                score *= 0.8
            elif source in {"verified_source", "expert_verified"}:
                score *= 1.0
            else:
                score *= 0.9
                
        # 检查时间戳
        current_time = time.time()
        for node in knowledge["nodes"]:
            timestamp = node.get("metadata", {}).get("timestamp", current_time)
            age = current_time - timestamp
            if age > 365 * 24 * 3600:  # 超过一年
                score *= 0.9
                
        return score
        
    def evaluate(self, knowledge: Dict[str, Any]) -> Dict[str, float]:
        """评估知识质量"""
        scores = {}
        for metric_name, metric_func in self.quality_metrics.items():
            scores[metric_name] = metric_func(knowledge)
            
        # 计算总体质量分数
        scores["overall"] = np.mean(list(scores.values()))
        
        return scores

class KnowledgeIntegration:
    """知识整合模块，负责整合和组织知识"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化知识整合模块
        
        Args:
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 知识网络
        self.knowledge_network = {
            "nodes": {},  # 节点 (概念、实体等)
            "edges": defaultdict(list),  # 边 (关系)
            "metadata": {
                "creation_time": time.time(),
                "last_update": time.time(),
                "node_count": 0,
                "edge_count": 0
            }
        }
        
        # 冲突检测和解决记录
        self.conflict_history = []
        
        # 知识合并历史
        self.integration_history = []
        
        # 知识质量评估器
        self.quality_evaluator = KnowledgeQuality()
        
        # 版本控制
        self.version_history = []
        self.current_version = None
        
        self.logger.info("知识整合模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("KnowledgeIntegration")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("knowledge_integration.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def integrate_knowledge(self, new_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        整合新知识到知识网络
        
        Args:
            new_knowledge: 要整合的新知识
            
        Returns:
            整合结果
        """
        self.logger.info("开始整合新知识")
        
        # 验证和预处理知识
        processed_knowledge = self._preprocess_knowledge(new_knowledge)
        
        # 评估知识质量
        quality_scores = self.quality_evaluator.evaluate(processed_knowledge)
        
        # 检测冲突
        conflicts = self._detect_conflicts(processed_knowledge)
        
        # 解决冲突
        if conflicts:
            self.logger.info(f"发现 {len(conflicts)} 个知识冲突")
            resolved_knowledge = self._resolve_conflicts(processed_knowledge, conflicts)
        else:
            resolved_knowledge = processed_knowledge
        
        # 与现有知识合并
        merged_result = self._merge_knowledge(resolved_knowledge)
        
        # 创建新版本
        version = KnowledgeVersion(
            version_id=str(hashlib.sha256(str(time.time()).encode()).hexdigest()),
            timestamp=time.time(),
            changes={
                "integrated_nodes": merged_result["integrated_nodes"],
                "integrated_edges": merged_result["integrated_edges"],
                "conflicts_resolved": len(conflicts) if conflicts else 0
            },
            author=new_knowledge.get("source", "system"),
            description="知识整合更新",
            parent_version=self.current_version
        )
        
        self.version_history.append(version)
        self.current_version = version.version_id
        
        # 更新元数据
        self.knowledge_network["metadata"]["last_update"] = time.time()
        self.knowledge_network["metadata"]["node_count"] = len(self.knowledge_network["nodes"])
        self.knowledge_network["metadata"]["edge_count"] = sum(len(edges) for edges in self.knowledge_network["edges"].values())
        
        # 记录整合历史
        self.integration_history.append({
            "timestamp": time.time(),
            "knowledge_source": new_knowledge.get("source", "unknown"),
            "integrated_nodes": merged_result["integrated_nodes"],
            "integrated_edges": merged_result["integrated_edges"],
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": len(conflicts) if conflicts else 0,
            "quality_scores": quality_scores
        })
        
        return {
            "status": "success",
            "integrated_nodes": merged_result["integrated_nodes"],
            "integrated_edges": merged_result["integrated_edges"],
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": len(conflicts) if conflicts else 0,
            "quality_scores": quality_scores,
            "version": version.version_id,
            "network_stats": {
                "nodes": len(self.knowledge_network["nodes"]),
                "edges": sum(len(edges) for edges in self.knowledge_network["edges"].values())
            }
        }
    
    def _preprocess_knowledge(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """预处理和验证知识"""
        processed = {
            "nodes": [],
            "edges": [],
            "source": knowledge.get("source", "unknown"),
            "timestamp": knowledge.get("timestamp", time.time())
        }
        
        # 处理节点
        if "nodes" in knowledge and isinstance(knowledge["nodes"], list):
            for node in knowledge["nodes"]:
                # 确保节点有ID
                if "id" not in node:
                    node["id"] = f"node_{len(processed['nodes'])}_{int(time.time())}"
                
                # 添加元数据
                if "metadata" not in node:
                    node["metadata"] = {}
                
                node["metadata"]["source"] = knowledge.get("source", "unknown")
                node["metadata"]["added_at"] = time.time()
                
                processed["nodes"].append(node)
        
        # 处理边
        if "edges" in knowledge and isinstance(knowledge["edges"], list):
            for edge in knowledge["edges"]:
                # 确保边有源节点和目标节点
                if "source" not in edge or "target" not in edge:
                    self.logger.warning(f"跳过缺少源或目标的边: {edge}")
                    continue
                
                # 添加元数据
                if "metadata" not in edge:
                    edge["metadata"] = {}
                
                edge["metadata"]["source"] = knowledge.get("source", "unknown")
                edge["metadata"]["added_at"] = time.time()
                
                # 确保有关系类型
                if "type" not in edge:
                    edge["type"] = "generic_relation"
                
                processed["edges"].append(edge)
        
        return processed
    
    def _detect_conflicts(self, knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测知识冲突"""
        conflicts = []
        
        # 检查节点冲突
        for new_node in knowledge["nodes"]:
            node_id = new_node["id"]
            
            # 检查ID冲突
            if node_id in self.knowledge_network["nodes"]:
                existing_node = self.knowledge_network["nodes"][node_id]
                
                # 检查属性冲突
                if "attributes" in new_node and "attributes" in existing_node:
                    conflicting_attrs = {}
                    
                    for attr_key, attr_value in new_node["attributes"].items():
                        if attr_key in existing_node["attributes"] and existing_node["attributes"][attr_key] != attr_value:
                            conflicting_attrs[attr_key] = {
                                "existing": existing_node["attributes"][attr_key],
                                "new": attr_value
                            }
                    
                    if conflicting_attrs:
                        conflicts.append({
                            "type": "attribute_conflict",
                            "node_id": node_id,
                            "conflicting_attributes": conflicting_attrs,
                            "severity": "medium"
                        })
                
                # 检查类型冲突
                if "type" in new_node and "type" in existing_node and new_node["type"] != existing_node["type"]:
                    conflicts.append({
                        "type": "type_conflict",
                        "node_id": node_id,
                        "existing_type": existing_node["type"],
                        "new_type": new_node["type"],
                        "severity": "high"
                    })
        
        # 检查边冲突
        for new_edge in knowledge["edges"]:
            source_id = new_edge["source"]
            target_id = new_edge["target"]
            edge_type = new_edge.get("type", "generic_relation")
            
            # 检查是否已存在相同起点和终点的边
            if source_id in self.knowledge_network["edges"]:
                for existing_edge in self.knowledge_network["edges"][source_id]:
                    if existing_edge["target"] == target_id:
                        # 检查边类型冲突
                        if existing_edge.get("type") != edge_type:
                            conflicts.append({
                                "type": "edge_type_conflict",
                                "source_id": source_id,
                                "target_id": target_id,
                                "existing_type": existing_edge.get("type"),
                                "new_type": edge_type,
                                "severity": "medium"
                            })
                        
                        # 检查边属性冲突
                        if "attributes" in new_edge and "attributes" in existing_edge:
                            conflicting_attrs = {}
                            
                            for attr_key, attr_value in new_edge["attributes"].items():
                                if attr_key in existing_edge["attributes"] and existing_edge["attributes"][attr_key] != attr_value:
                                    conflicting_attrs[attr_key] = {
                                        "existing": existing_edge["attributes"][attr_key],
                                        "new": attr_value
                                    }
                            
                            if conflicting_attrs:
                                conflicts.append({
                                    "type": "edge_attribute_conflict",
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "conflicting_attributes": conflicting_attrs,
                                    "severity": "low"
                                })
        
        # 记录冲突
        for conflict in conflicts:
            self.conflict_history.append({
                "conflict": conflict,
                "timestamp": time.time(),
                "knowledge_source": knowledge.get("source")
            })
        
        return conflicts
    
    def _resolve_conflicts(self, knowledge: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解决知识冲突"""
        resolved_knowledge = {
            "nodes": knowledge["nodes"].copy(),
            "edges": knowledge["edges"].copy(),
            "source": knowledge.get("source"),
            "timestamp": knowledge.get("timestamp")
        }
        
        # 处理每个冲突
        for conflict in conflicts:
            conflict_type = conflict["type"]
            
            if conflict_type == "attribute_conflict":
                node_id = conflict["node_id"]
                
                # 查找节点
                node_index = next((i for i, node in enumerate(resolved_knowledge["nodes"]) if node["id"] == node_id), -1)
                
                if node_index >= 0:
                    node = resolved_knowledge["nodes"][node_index]
                    
                    # 解决属性冲突
                    for attr_key, conflict_info in conflict["conflicting_attributes"].items():
                        # 获取当前节点和现有节点
                        existing_value = conflict_info["existing"]
                        new_value = conflict_info["new"]
                        
                        # 冲突解决策略：保留更新的值，如果可能的话融合它们
                        if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                            # 数值取平均
                            resolved_value = (existing_value + new_value) / 2
                        elif isinstance(existing_value, list) and isinstance(new_value, list):
                            # 列表合并
                            resolved_value = list(set(existing_value + new_value))
                        elif isinstance(existing_value, dict) and isinstance(new_value, dict):
                            # 字典合并
                            resolved_value = {**existing_value, **new_value}
                        else:
                            # 默认使用新值，同时保留冲突记录
                            resolved_value = new_value
                            
                            # 添加冲突记录
                            if "conflict_history" not in node["attributes"]:
                                node["attributes"]["conflict_history"] = {}
                            
                            if attr_key not in node["attributes"]["conflict_history"]:
                                node["attributes"]["conflict_history"][attr_key] = []
                            
                            node["attributes"]["conflict_history"][attr_key].append({
                                "timestamp": time.time(),
                                "value": existing_value,
                                "source": self.knowledge_network["nodes"][node_id].get("metadata", {}).get("source")
                            })
                        
                        # 更新属性值
                        node["attributes"][attr_key] = resolved_value
            
            elif conflict_type == "type_conflict":
                node_id = conflict["node_id"]
                
                # 查找节点
                node_index = next((i for i, node in enumerate(resolved_knowledge["nodes"]) if node["id"] == node_id), -1)
                
                if node_index >= 0:
                    # 冲突解决策略：保留两种类型
                    node = resolved_knowledge["nodes"][node_index]
                    existing_type = conflict["existing_type"]
                    new_type = conflict["new_type"]
                    
                    # 创建类型列表
                    if isinstance(node["type"], list):
                        if new_type not in node["type"]:
                            node["type"].append(new_type)
                    else:
                        node["type"] = [node["type"], new_type]
            
            elif conflict_type in ["edge_type_conflict", "edge_attribute_conflict"]:
                source_id = conflict["source_id"]
                target_id = conflict["target_id"]
                
                # 查找边
                edge_index = next((i for i, edge in enumerate(resolved_knowledge["edges"]) 
                                 if edge["source"] == source_id and edge["target"] == target_id), -1)
                
                if edge_index >= 0:
                    edge = resolved_knowledge["edges"][edge_index]
                    
                    if conflict_type == "edge_type_conflict":
                        # 冲突解决策略：保留两种类型
                        existing_type = conflict["existing_type"]
                        new_type = conflict["new_type"]
                        
                        # 创建类型列表
                        if "type" in edge:
                            if isinstance(edge["type"], list):
                                if new_type not in edge["type"]:
                                    edge["type"].append(new_type)
                            else:
                                edge["type"] = [edge["type"], new_type]
                        else:
                            edge["type"] = new_type
                    
                    elif conflict_type == "edge_attribute_conflict":
                        # 解决边属性冲突
                        for attr_key, conflict_info in conflict["conflicting_attributes"].items():
                            # 合并策略类似于节点属性
                            existing_value = conflict_info["existing"]
                            new_value = conflict_info["new"]
                            
                            # 根据类型选择合并策略
                            if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                                resolved_value = (existing_value + new_value) / 2
                            elif isinstance(existing_value, list) and isinstance(new_value, list):
                                resolved_value = list(set(existing_value + new_value))
                            elif isinstance(existing_value, dict) and isinstance(new_value, dict):
                                resolved_value = {**existing_value, **new_value}
                            else:
                                resolved_value = new_value
                            
                            # 更新属性
                            if "attributes" not in edge:
                                edge["attributes"] = {}
                            
                            edge["attributes"][attr_key] = resolved_value
        
        return resolved_knowledge
    
    def _merge_knowledge(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """合并知识到知识网络"""
        integrated_nodes = []
        integrated_edges = []
        
        # 合并节点
        for node in knowledge["nodes"]:
            node_id = node["id"]
            
            if node_id in self.knowledge_network["nodes"]:
                # 更新现有节点
                existing_node = self.knowledge_network["nodes"][node_id]
                
                # 更新属性
                if "attributes" in node:
                    if "attributes" not in existing_node:
                        existing_node["attributes"] = {}
                    
                    for attr_key, attr_value in node["attributes"].items():
                        existing_node["attributes"][attr_key] = attr_value
                
                # 更新其他字段
                for key, value in node.items():
                    if key not in ["id", "attributes", "metadata"]:
                        existing_node[key] = value
                
                # 更新元数据
                existing_node["metadata"]["updated_at"] = time.time()
                existing_node["metadata"]["last_source"] = node["metadata"]["source"]
                
                integrated_nodes.append({"id": node_id, "action": "updated"})
            else:
                # 添加新节点
                self.knowledge_network["nodes"][node_id] = node
                integrated_nodes.append({"id": node_id, "action": "added"})
        
        # 合并边
        for edge in knowledge["edges"]:
            source_id = edge["source"]
            target_id = edge["target"]
            
            # 确保源节点和目标节点存在
            if source_id not in self.knowledge_network["nodes"] or target_id not in self.knowledge_network["nodes"]:
                self.logger.warning(f"跳过边 {source_id} -> {target_id}，因为源节点或目标节点不存在")
                continue
            
            # 检查是否已存在相同的边
            existing_edge_index = -1
            for i, existing_edge in enumerate(self.knowledge_network["edges"][source_id]):
                if existing_edge["target"] == target_id:
                    existing_edge_index = i
                    break
            
            if existing_edge_index >= 0:
                # 更新现有边
                existing_edge = self.knowledge_network["edges"][source_id][existing_edge_index]
                
                # 更新属性
                if "attributes" in edge:
                    if "attributes" not in existing_edge:
                        existing_edge["attributes"] = {}
                    
                    for attr_key, attr_value in edge["attributes"].items():
                        existing_edge["attributes"][attr_key] = attr_value
                
                # 更新其他字段
                for key, value in edge.items():
                    if key not in ["source", "target", "attributes", "metadata"]:
                        existing_edge[key] = value
                
                # 更新元数据
                existing_edge["metadata"]["updated_at"] = time.time()
                existing_edge["metadata"]["last_source"] = edge["metadata"]["source"]
                
                integrated_edges.append({
                    "source": source_id, 
                    "target": target_id, 
                    "action": "updated"
                })
            else:
                # 添加新边
                self.knowledge_network["edges"][source_id].append(edge)
                integrated_edges.append({
                    "source": source_id, 
                    "target": target_id, 
                    "action": "added"
                })
        
        return {
            "integrated_nodes": integrated_nodes,
            "integrated_edges": integrated_edges
        }
    
    def get_knowledge_by_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        按查询检索知识
        
        Args:
            query: 查询参数
            
        Returns:
            查询结果
        """
        results = {
            "nodes": [],
            "edges": [],
            "query": query
        }
        
        # 处理节点查询
        if "node_ids" in query:
            for node_id in query["node_ids"]:
                if node_id in self.knowledge_network["nodes"]:
                    results["nodes"].append(self.knowledge_network["nodes"][node_id])
        
        # 处理属性查询
        if "node_attributes" in query:
            for attr_key, attr_value in query["node_attributes"].items():
                for node_id, node in self.knowledge_network["nodes"].items():
                    if "attributes" in node and attr_key in node["attributes"]:
                        if node["attributes"][attr_key] == attr_value:
                            if node not in results["nodes"]:
                                results["nodes"].append(node)
        
        # 处理类型查询
        if "node_types" in query:
            for node_type in query["node_types"]:
                for node_id, node in self.knowledge_network["nodes"].items():
                    if "type" in node:
                        if isinstance(node["type"], list):
                            if node_type in node["type"] and node not in results["nodes"]:
                                results["nodes"].append(node)
                        elif node["type"] == node_type and node not in results["nodes"]:
                            results["nodes"].append(node)
        
        # 处理边查询
        if "source_id" in query and "target_id" in query:
            source_id = query["source_id"]
            target_id = query["target_id"]
            
            if source_id in self.knowledge_network["edges"]:
                for edge in self.knowledge_network["edges"][source_id]:
                    if edge["target"] == target_id:
                        results["edges"].append(edge)
        
        # 处理相邻节点查询
        if "neighbors_of" in query:
            node_id = query["neighbors_of"]
            
            if node_id in self.knowledge_network["nodes"]:
                # 查找出边
                if node_id in self.knowledge_network["edges"]:
                    for edge in self.knowledge_network["edges"][node_id]:
                        results["edges"].append(edge)
                        target_id = edge["target"]
                        if target_id in self.knowledge_network["nodes"]:
                            target_node = self.knowledge_network["nodes"][target_id]
                            if target_node not in results["nodes"]:
                                results["nodes"].append(target_node)
                
                # 查找入边
                for source_id, edges in self.knowledge_network["edges"].items():
                    for edge in edges:
                        if edge["target"] == node_id:
                            results["edges"].append(edge)
                            if source_id in self.knowledge_network["nodes"]:
                                source_node = self.knowledge_network["nodes"][source_id]
                                if source_node not in results["nodes"]:
                                    results["nodes"].append(source_node)
        
        # 添加统计信息
        results["stats"] = {
            "node_count": len(results["nodes"]),
            "edge_count": len(results["edges"])
        }
        
        return results
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取知识网络统计信息
        
        Returns:
            统计信息
        """
        # 计算节点统计
        node_types = {}
        node_creation_times = []
        
        for node_id, node in self.knowledge_network["nodes"].items():
            # 统计节点类型
            if "type" in node:
                node_type = node["type"]
                if isinstance(node_type, list):
                    for t in node_type:
                        node_types[t] = node_types.get(t, 0) + 1
                else:
                    node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # 收集创建时间
            if "metadata" in node and "added_at" in node["metadata"]:
                node_creation_times.append(node["metadata"]["added_at"])
        
        # 计算边统计
        edge_types = {}
        edge_creation_times = []
        
        for source_id, edges in self.knowledge_network["edges"].items():
            for edge in edges:
                # 统计边类型
                if "type" in edge:
                    edge_type = edge["type"]
                    if isinstance(edge_type, list):
                        for t in edge_type:
                            edge_types[t] = edge_types.get(t, 0) + 1
                    else:
                        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                
                # 收集创建时间
                if "metadata" in edge and "added_at" in edge["metadata"]:
                    edge_creation_times.append(edge["metadata"]["added_at"])
        
        # 计算节点度分布
        degree_distribution = {}
        for node_id in self.knowledge_network["nodes"]:
            # 出度
            out_degree = len(self.knowledge_network["edges"].get(node_id, []))
            
            # 入度
            in_degree = 0
            for source_id, edges in self.knowledge_network["edges"].items():
                for edge in edges:
                    if edge["target"] == node_id:
                        in_degree += 1
            
            # 总度
            total_degree = out_degree + in_degree
            degree_distribution[total_degree] = degree_distribution.get(total_degree, 0) + 1
        
        return {
            "general": {
                "node_count": len(self.knowledge_network["nodes"]),
                "edge_count": sum(len(edges) for edges in self.knowledge_network["edges"].values()),
                "creation_time": self.knowledge_network["metadata"]["creation_time"],
                "last_update": self.knowledge_network["metadata"]["last_update"],
                "integration_operations": len(self.integration_history),
                "conflict_count": len(self.conflict_history)
            },
            "nodes": {
                "type_distribution": node_types,
                "creation_time_distribution": self._get_time_distribution(node_creation_times) if node_creation_times else {}
            },
            "edges": {
                "type_distribution": edge_types,
                "creation_time_distribution": self._get_time_distribution(edge_creation_times) if edge_creation_times else {}
            },
            "network": {
                "degree_distribution": degree_distribution
            }
        }
    
    def _get_time_distribution(self, timestamps: List[float]) -> Dict[str, int]:
        """获取时间分布"""
        if not timestamps:
            return {}
        
        # 简化实现：按天分组
        day_counts = defaultdict(int)
        
        for ts in timestamps:
            # 转换为天
            day = int(ts / (24 * 3600))
            day_counts[day] += 1
        
        return dict(day_counts)
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """获取版本历史"""
        return [
            {
                "version_id": v.version_id,
                "timestamp": v.timestamp,
                "changes": v.changes,
                "author": v.author,
                "description": v.description,
                "parent_version": v.parent_version
            }
            for v in self.version_history
        ]
        
    def rollback_to_version(self, version_id: str) -> bool:
        """回滚到指定版本"""
        try:
            # 找到目标版本
            target_version = None
            for version in self.version_history:
                if version.version_id == version_id:
                    target_version = version
                    break
                    
            if not target_version:
                self.logger.error(f"未找到版本: {version_id}")
                return False
                
            # 回滚更改
            for change in target_version.changes.get("integrated_nodes", []):
                if change["action"] == "add":
                    self.knowledge_network["nodes"].pop(change["node_id"], None)
                elif change["action"] == "update":
                    if change["node_id"] in self.knowledge_network["nodes"]:
                        self.knowledge_network["nodes"][change["node_id"]] = change["old_state"]
                        
            for change in target_version.changes.get("integrated_edges", []):
                if change["action"] == "add":
                    self.knowledge_network["edges"][change["source"]].remove(change["edge"])
                elif change["action"] == "update":
                    if change["source"] in self.knowledge_network["edges"]:
                        self.knowledge_network["edges"][change["source"]] = change["old_state"]
                        
            # 更新当前版本
            self.current_version = version_id
            
            return True
            
        except Exception as e:
            self.logger.error(f"回滚版本失败: {str(e)}")
            return False 