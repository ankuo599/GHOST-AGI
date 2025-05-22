# -*- coding: utf-8 -*-
"""
知识迁移模块 (Knowledge Transfer)

负责实现跨领域知识迁移，提高系统学习效率和知识通用性
"""

import os
import time
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import networkx as nx

@dataclass
class TransferMetrics:
    """迁移评估指标"""
    accuracy: float
    f1_score: float
    transfer_loss: float
    domain_adaptation_score: float
    knowledge_preservation: float

class TransferLearningModel(nn.Module):
    """迁移学习模型"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, alpha=0.0):
        features = self.feature_extractor(x)
        
        # 梯度反转层
        reverse_features = GradientReversal.apply(features, alpha)
        
        domain_output = self.domain_classifier(reverse_features)
        task_output = self.task_classifier(features)
        
        return domain_output, task_output, features

class GradientReversal(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class KnowledgeDistillation:
    """知识蒸馏"""
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = 2.0
        
    def distill(self, input_data: torch.Tensor) -> torch.Tensor:
        """执行知识蒸馏"""
        with torch.no_grad():
            teacher_output = self.teacher_model(input_data)
            
        student_output = self.student_model(input_data)
        
        # 计算蒸馏损失
        distillation_loss = nn.KLDivLoss()(
            torch.log_softmax(student_output / self.temperature, dim=1),
            torch.softmax(teacher_output / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        return distillation_loss

class KnowledgeTransfer:
    """知识迁移系统，实现跨领域知识迁移和应用"""
    
    def __init__(self, memory_system=None, event_system=None):
        """
        初始化知识迁移系统
        
        Args:
            memory_system: 记忆系统，用于存取知识
            event_system: 事件系统，用于通知知识迁移事件
        """
        self.memory_system = memory_system
        self.event_system = event_system
        self.logger = logging.getLogger("KnowledgeTransfer")
        
        # 知识领域映射
        self.domain_mappings = {}
        
        # 迁移历史
        self.transfer_history = []
        
        # 迁移成功率统计
        self.transfer_stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "domains": {}
        }
        
        # 初始化迁移学习模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transfer_model = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        
        # 知识蒸馏
        self.teacher_model = None
        self.student_model = None
        self.distillation = None
        
        self.logger.info("知识迁移系统初始化完成")
        
    def register_domain(self, domain_name: str, schema: Dict[str, Any]) -> bool:
        """
        注册知识领域
        
        Args:
            domain_name: 领域名称
            schema: 领域结构定义
            
        Returns:
            bool: 是否成功注册
        """
        try:
            if domain_name in self.domain_mappings:
                self.logger.warning(f"领域 '{domain_name}' 已存在，将被覆盖")
                
            self.domain_mappings[domain_name] = {
                "schema": schema,
                "registered_at": time.time(),
                "concepts": schema.get("concepts", {}),
                "relations": schema.get("relations", {}),
                "axioms": schema.get("axioms", [])
            }
            
            # 初始化领域统计
            if domain_name not in self.transfer_stats["domains"]:
                self.transfer_stats["domains"][domain_name] = {
                    "source_transfers": 0,
                    "target_transfers": 0,
                    "success_rate": 0.0
                }
                
            # 发布事件
            if self.event_system:
                self.event_system.publish("knowledge_transfer.domain_registered", {
                    "domain_name": domain_name,
                    "concept_count": len(schema.get("concepts", {})),
                    "relation_count": len(schema.get("relations", {}))
                })
                
            return True
        except Exception as e:
            self.logger.error(f"注册领域失败: {str(e)}")
            return False
            
    def create_domain_mapping(self, source_domain: str, target_domain: str, 
                            concept_mappings: Dict[str, str], relation_mappings: Dict[str, str] = None) -> Dict[str, Any]:
        """
        创建领域间的映射关系
        
        Args:
            source_domain: 源领域名称
            target_domain: 目标领域名称
            concept_mappings: 概念映射关系
            relation_mappings: 关系映射关系
            
        Returns:
            Dict: 映射结果
        """
        try:
            # 检查领域是否存在
            if source_domain not in self.domain_mappings:
                return {
                    "status": "error",
                    "message": f"源领域 '{source_domain}' 不存在"
                }
                
            if target_domain not in self.domain_mappings:
                return {
                    "status": "error",
                    "message": f"目标领域 '{target_domain}' 不存在"
                }
                
            # 创建映射关系
            mapping_id = f"{source_domain}_{target_domain}_{int(time.time())}"
            
            # 验证概念映射
            source_concepts = self.domain_mappings[source_domain]["concepts"]
            target_concepts = self.domain_mappings[target_domain]["concepts"]
            
            invalid_mappings = []
            for src_concept, tgt_concept in concept_mappings.items():
                if src_concept not in source_concepts:
                    invalid_mappings.append(f"源概念 '{src_concept}' 不存在于领域 '{source_domain}'")
                if tgt_concept not in target_concepts:
                    invalid_mappings.append(f"目标概念 '{tgt_concept}' 不存在于领域 '{target_domain}'")
                    
            if invalid_mappings:
                return {
                    "status": "error",
                    "message": "无效的概念映射",
                    "invalid_mappings": invalid_mappings
                }
                
            # 处理关系映射
            relation_mappings = relation_mappings or {}
            
            # 创建映射
            mapping = {
                "id": mapping_id,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "concept_mappings": concept_mappings,
                "relation_mappings": relation_mappings,
                "created_at": time.time()
            }
            
            # 将映射添加到领域中
            if "mappings" not in self.domain_mappings[source_domain]:
                self.domain_mappings[source_domain]["mappings"] = {}
                
            self.domain_mappings[source_domain]["mappings"][target_domain] = mapping
            
            # 发布事件
            if self.event_system:
                self.event_system.publish("knowledge_transfer.mapping_created", {
                    "mapping_id": mapping_id,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "concept_count": len(concept_mappings),
                    "relation_count": len(relation_mappings)
                })
                
            return {
                "status": "success",
                "mapping_id": mapping_id,
                "mapping": mapping
            }
        except Exception as e:
            self.logger.error(f"创建领域映射失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def transfer_knowledge(self, source_domain: str, target_domain: str, 
                         knowledge: Dict[str, Any], mapping_id: str = None) -> Dict[str, Any]:
        """
        将知识从源领域迁移到目标领域
        
        Args:
            source_domain: 源领域名称
            target_domain: 目标领域名称
            knowledge: 要迁移的知识
            mapping_id: 映射ID，如果为None则使用最新的映射
            
        Returns:
            Dict: 迁移结果
        """
        try:
            # 更新统计信息
            self.transfer_stats["total_attempts"] += 1
            self.transfer_stats["domains"][source_domain]["source_transfers"] += 1
            self.transfer_stats["domains"][target_domain]["target_transfers"] += 1
            
            # 获取映射
            mapping = None
            
            if mapping_id:
                # 查找指定ID的映射
                for domain_data in self.domain_mappings.values():
                    if "mappings" in domain_data:
                        for domain_mapping in domain_data["mappings"].values():
                            if domain_mapping["id"] == mapping_id:
                                mapping = domain_mapping
                                break
            else:
                # 使用源领域到目标领域的最新映射
                if source_domain in self.domain_mappings and "mappings" in self.domain_mappings[source_domain]:
                    if target_domain in self.domain_mappings[source_domain]["mappings"]:
                        mapping = self.domain_mappings[source_domain]["mappings"][target_domain]
                        
            if not mapping:
                self.transfer_stats["failed"] += 1
                return {
                    "status": "error",
                    "message": f"未找到从 '{source_domain}' 到 '{target_domain}' 的映射关系"
                }
                
            # 获取映射关系
            concept_mappings = mapping["concept_mappings"]
            relation_mappings = mapping["relation_mappings"]
            
            # 执行概念映射
            transformed_knowledge = {}
            
            # 处理概念
            if "concepts" in knowledge:
                transformed_knowledge["concepts"] = {}
                
                for concept, data in knowledge["concepts"].items():
                    # 查找对应的目标概念
                    if concept in concept_mappings:
                        target_concept = concept_mappings[concept]
                        transformed_knowledge["concepts"][target_concept] = data
                    else:
                        # 无映射关系的概念，保留原名
                        transformed_knowledge["concepts"][concept] = data
                        
            # 处理关系
            if "relations" in knowledge:
                transformed_knowledge["relations"] = []
                
                for relation in knowledge["relations"]:
                    relation_type = relation.get("type")
                    source = relation.get("source")
                    target = relation.get("target")
                    
                    # 映射关系类型
                    mapped_type = relation_mappings.get(relation_type, relation_type)
                    
                    # 映射源概念
                    mapped_source = concept_mappings.get(source, source)
                    
                    # 映射目标概念
                    mapped_target = concept_mappings.get(target, target)
                    
                    # 创建映射后的关系
                    mapped_relation = {
                        "type": mapped_type,
                        "source": mapped_source,
                        "target": mapped_target
                    }
                    
                    # 复制其他属性
                    for key, value in relation.items():
                        if key not in ["type", "source", "target"]:
                            mapped_relation[key] = value
                            
                    transformed_knowledge["relations"].append(mapped_relation)
                    
            # 处理其他知识属性
            for key, value in knowledge.items():
                if key not in ["concepts", "relations"]:
                    transformed_knowledge[key] = value
                    
            # 添加迁移元数据
            transformed_knowledge["metadata"] = {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "transfer_timestamp": time.time(),
                "mapping_id": mapping["id"]
            }
            
            # 记录迁移
            transfer_record = {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "mapping_id": mapping["id"],
                "timestamp": time.time(),
                "success": True
            }
            
            self.transfer_history.append(transfer_record)
            
            # 限制历史记录大小
            max_history = 1000
            if len(self.transfer_history) > max_history:
                self.transfer_history = self.transfer_history[-max_history:]
                
            # 更新统计信息
            self.transfer_stats["successful"] += 1
            
            # 计算成功率
            for domain in [source_domain, target_domain]:
                domain_data = self.transfer_stats["domains"][domain]
                total_transfers = domain_data["source_transfers"] + domain_data["target_transfers"]
                if total_transfers > 0:
                    success_rate = self.transfer_stats["successful"] / self.transfer_stats["total_attempts"]
                    domain_data["success_rate"] = success_rate
                    
            # 发布事件
            if self.event_system:
                self.event_system.publish("knowledge_transfer.completed", {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "mapping_id": mapping["id"],
                    "success": True
                })
                
            return {
                "status": "success",
                "transformed_knowledge": transformed_knowledge,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "mapping_id": mapping["id"]
            }
        except Exception as e:
            # 记录失败
            transfer_record = {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "mapping_id": mapping["id"] if mapping else None,
                "timestamp": time.time(),
                "success": False,
                "error": str(e)
            }
            
            self.transfer_history.append(transfer_record)
            
            # 更新统计信息
            self.transfer_stats["failed"] += 1
            
            self.logger.error(f"知识迁移失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def create_unified_representation(self, knowledge: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        创建统一知识表示
        
        Args:
            knowledge: 领域知识
            domain: 知识所属领域
            
        Returns:
            Dict: 统一表示
        """
        try:
            # 检查领域是否存在
            if domain not in self.domain_mappings:
                return {
                    "status": "error",
                    "message": f"领域 '{domain}' 不存在"
                }
                
            # 获取领域结构
            domain_schema = self.domain_mappings[domain]["schema"]
            
            # 创建统一表示
            unified = {
                "domain": domain,
                "created_at": time.time(),
                "concepts": {},
                "relations": [],
                "metadata": {
                    "original_domain": domain,
                    "domain_schema": domain_schema.get("name", domain)
                }
            }
            
            # 处理概念
            if "concepts" in knowledge:
                for concept, data in knowledge["concepts"].items():
                    unified["concepts"][concept] = {
                        "id": concept,
                        "domain": domain,
                        "data": data
                    }
                    
            # 处理关系
            if "relations" in knowledge:
                for relation in knowledge["relations"]:
                    unified_relation = {
                        "type": relation.get("type"),
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "domain": domain
                    }
                    
                    # 复制其他属性
                    for key, value in relation.items():
                        if key not in ["type", "source", "target"]:
                            unified_relation[key] = value
                            
                    unified["relations"].append(unified_relation)
                    
            # 处理其他知识属性
            for key, value in knowledge.items():
                if key not in ["concepts", "relations", "metadata"]:
                    unified[key] = value
                    
            return {
                "status": "success",
                "unified_representation": unified
            }
        except Exception as e:
            self.logger.error(f"创建统一表示失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def merge_knowledge(self, knowledge_sets: List[Dict[str, Any]], 
                      merge_strategy: str = "union") -> Dict[str, Any]:
        """
        合并多个知识集
        
        Args:
            knowledge_sets: 要合并的知识集列表
            merge_strategy: 合并策略 ('union', 'intersection', 'priority')
            
        Returns:
            Dict: 合并结果
        """
        try:
            if not knowledge_sets:
                return {
                    "status": "error",
                    "message": "没有提供要合并的知识集"
                }
                
            # 创建统一表示
            unified_sets = []
            
            for knowledge in knowledge_sets:
                domain = knowledge.get("metadata", {}).get("source_domain") or knowledge.get("domain")
                
                if not domain:
                    return {
                        "status": "error",
                        "message": "知识集缺少领域信息"
                    }
                    
                # 将知识转换为统一表示
                unified_result = self.create_unified_representation(knowledge, domain)
                
                if unified_result["status"] == "success":
                    unified_sets.append(unified_result["unified_representation"])
                else:
                    return unified_result
                    
            # 执行合并
            if merge_strategy == "union":
                merged = self._merge_union(unified_sets)
            elif merge_strategy == "intersection":
                merged = self._merge_intersection(unified_sets)
            elif merge_strategy == "priority":
                merged = self._merge_priority(unified_sets)
            else:
                return {
                    "status": "error",
                    "message": f"不支持的合并策略: {merge_strategy}"
                }
                
            # 添加合并元数据
            merged["metadata"] = {
                "merge_strategy": merge_strategy,
                "source_domains": [knowledge.get("domain") for knowledge in unified_sets],
                "merge_timestamp": time.time(),
                "knowledge_count": len(knowledge_sets)
            }
            
            # 发布事件
            if self.event_system:
                self.event_system.publish("knowledge_transfer.knowledge_merged", {
                    "strategy": merge_strategy,
                    "source_count": len(knowledge_sets),
                    "concept_count": len(merged.get("concepts", {})),
                    "relation_count": len(merged.get("relations", []))
                })
                
            return {
                "status": "success",
                "merged_knowledge": merged
            }
        except Exception as e:
            self.logger.error(f"合并知识失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _merge_union(self, knowledge_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """联合合并策略"""
        merged = {
            "concepts": {},
            "relations": []
        }
        
        # 合并概念 (联合所有概念)
        for knowledge in knowledge_sets:
            for concept_id, concept_data in knowledge.get("concepts", {}).items():
                if concept_id not in merged["concepts"]:
                    merged["concepts"][concept_id] = concept_data
                else:
                    # 合并概念数据
                    for key, value in concept_data.items():
                        if key not in merged["concepts"][concept_id]:
                            merged["concepts"][concept_id][key] = value
                            
        # 合并关系 (合并所有唯一关系)
        relation_set = set()
        
        for knowledge in knowledge_sets:
            for relation in knowledge.get("relations", []):
                relation_key = (relation.get("type"), relation.get("source"), relation.get("target"))
                
                if relation_key not in relation_set:
                    relation_set.add(relation_key)
                    merged["relations"].append(relation)
                    
        return merged
        
    def _merge_intersection(self, knowledge_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """交集合并策略"""
        if not knowledge_sets:
            return {"concepts": {}, "relations": []}
            
        # 从第一个知识集开始
        merged = {
            "concepts": dict(knowledge_sets[0].get("concepts", {})),
            "relations": list(knowledge_sets[0].get("relations", []))
        }
        
        # 对于每个后续知识集，只保留交集
        for knowledge in knowledge_sets[1:]:
            # 处理概念交集
            concept_ids = set(merged["concepts"].keys()) & set(knowledge.get("concepts", {}).keys())
            merged["concepts"] = {
                concept_id: merged["concepts"][concept_id]
                for concept_id in concept_ids
            }
            
            # 处理关系交集
            relation_keys_1 = {
                (r.get("type"), r.get("source"), r.get("target"))
                for r in merged["relations"]
            }
            
            relation_keys_2 = {
                (r.get("type"), r.get("source"), r.get("target"))
                for r in knowledge.get("relations", [])
            }
            
            relation_keys = relation_keys_1 & relation_keys_2
            
            merged["relations"] = [
                r for r in merged["relations"]
                if (r.get("type"), r.get("source"), r.get("target")) in relation_keys
            ]
            
        return merged
        
    def _merge_priority(self, knowledge_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优先级合并策略 (后面的知识集优先)"""
        merged = {
            "concepts": {},
            "relations": []
        }
        
        # 按照优先级顺序合并
        for knowledge in knowledge_sets:
            # 合并概念
            for concept_id, concept_data in knowledge.get("concepts", {}).items():
                merged["concepts"][concept_id] = concept_data
                
            # 合并关系
            relation_keys = {
                (r.get("type"), r.get("source"), r.get("target"))
                for r in merged["relations"]
            }
            
            for relation in knowledge.get("relations", []):
                relation_key = (relation.get("type"), relation.get("source"), relation.get("target"))
                
                # 如果关系已存在，移除旧的
                if relation_key in relation_keys:
                    merged["relations"] = [
                        r for r in merged["relations"]
                        if (r.get("type"), r.get("source"), r.get("target")) != relation_key
                    ]
                    
                # 添加新关系
                merged["relations"].append(relation)
                relation_keys.add(relation_key)
                
        return merged
        
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        获取知识迁移统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 计算总体成功率
        if self.transfer_stats["total_attempts"] > 0:
            success_rate = self.transfer_stats["successful"] / self.transfer_stats["total_attempts"]
        else:
            success_rate = 0.0
            
        stats = {
            "total_attempts": self.transfer_stats["total_attempts"],
            "successful": self.transfer_stats["successful"],
            "failed": self.transfer_stats["failed"],
            "success_rate": success_rate,
            "domains": self.transfer_stats["domains"],
            "timestamp": time.time()
        }
        
        return stats
        
    def get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """
        获取领域知识
        
        Args:
            domain: 领域名称
            
        Returns:
            Dict: 领域知识
        """
        try:
            if domain not in self.domain_mappings:
                return {
                    "status": "error",
                    "message": f"领域 '{domain}' 不存在"
                }
                
            domain_data = self.domain_mappings[domain]
            
            # 从记忆系统获取领域知识
            knowledge = {
                "domain": domain,
                "concepts": domain_data.get("concepts", {}),
                "relations": [],
                "axioms": domain_data.get("axioms", []),
                "timestamp": time.time()
            }
            
            # 如果有记忆系统，尝试检索更多知识
            if self.memory_system:
                memory_query = {
                    "type": "domain_knowledge",
                    "domain": domain
                }
                
                memory_results = self.memory_system.search(memory_query)
                
                if memory_results:
                    # 整合记忆结果
                    for result in memory_results:
                        result_data = result.get("data", {})
                        
                        # 整合概念
                        if "concepts" in result_data:
                            for concept_id, concept_data in result_data["concepts"].items():
                                if concept_id not in knowledge["concepts"]:
                                    knowledge["concepts"][concept_id] = concept_data
                                    
                        # 整合关系
                        if "relations" in result_data:
                            # 创建已有关系的集合
                            existing_relations = {
                                (r.get("type"), r.get("source"), r.get("target"))
                                for r in knowledge["relations"]
                            }
                            
                            for relation in result_data["relations"]:
                                relation_key = (relation.get("type"), relation.get("source"), relation.get("target"))
                                
                                if relation_key not in existing_relations:
                                    knowledge["relations"].append(relation)
                                    existing_relations.add(relation_key)
                                    
            return {
                "status": "success",
                "knowledge": knowledge
            }
        except Exception as e:
            self.logger.error(f"获取领域知识失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def apply_knowledge(self, knowledge: Dict[str, Any], target_system: Any) -> Dict[str, Any]:
        """
        将知识应用到目标系统
        
        Args:
            knowledge: 要应用的知识
            target_system: 目标系统对象
            
        Returns:
            Dict: 应用结果
        """
        try:
            if not target_system:
                return {
                    "status": "error",
                    "message": "未提供目标系统"
                }
                
            # 检查目标系统是否支持知识应用
            if not hasattr(target_system, "apply_knowledge"):
                return {
                    "status": "error",
                    "message": "目标系统不支持知识应用"
                }
                
            # 应用知识
            result = target_system.apply_knowledge(knowledge)
            
            # 记录应用
            application_record = {
                "target_system": target_system.__class__.__name__,
                "knowledge_domain": knowledge.get("domain"),
                "timestamp": time.time(),
                "success": result.get("status") == "success"
            }
            
            # 发布事件
            if self.event_system:
                self.event_system.publish("knowledge_transfer.knowledge_applied", {
                    "target_system": target_system.__class__.__name__,
                    "knowledge_domain": knowledge.get("domain"),
                    "success": result.get("status") == "success"
                })
                
            return result
        except Exception as e:
            self.logger.error(f"应用知识失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def train_transfer_model(self, source_data: List[Dict[str, Any]], 
                           target_data: List[Dict[str, Any]],
                           epochs: int = 10) -> Dict[str, Any]:
        """训练迁移学习模型"""
        try:
            # 准备数据
            source_dataset = self._prepare_dataset(source_data)
            target_dataset = self._prepare_dataset(target_data)
            
            source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
            target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
            
            # 初始化模型
            if self.transfer_model is None:
                self.transfer_model = TransferLearningModel(
                    input_dim=768,  # BERT 输出维度
                    hidden_dim=256,
                    num_domains=2
                ).to(self.device)
                
            # 优化器
            optimizer = torch.optim.Adam(self.transfer_model.parameters())
            
            # 损失函数
            domain_criterion = nn.CrossEntropyLoss()
            task_criterion = nn.BCEWithLogitsLoss()
            
            # 训练循环
            for epoch in range(epochs):
                self.transfer_model.train()
                total_loss = 0
                
                for (source_batch, target_batch) in zip(source_loader, target_loader):
                    # 源域数据
                    source_inputs = source_batch["input_ids"].to(self.device)
                    source_masks = source_batch["attention_mask"].to(self.device)
                    source_labels = source_batch["labels"].to(self.device)
                    
                    # 目标域数据
                    target_inputs = target_batch["input_ids"].to(self.device)
                    target_masks = target_batch["attention_mask"].to(self.device)
                    
                    # 获取 BERT 嵌入
                    with torch.no_grad():
                        source_outputs = self.bert_model(
                            input_ids=source_inputs,
                            attention_mask=source_masks
                        )
                        target_outputs = self.bert_model(
                            input_ids=target_inputs,
                            attention_mask=target_masks
                        )
                        
                    source_embeddings = source_outputs.last_hidden_state[:, 0, :]
                    target_embeddings = target_outputs.last_hidden_state[:, 0, :]
                    
                    # 合并源域和目标域数据
                    combined_embeddings = torch.cat([source_embeddings, target_embeddings])
                    domain_labels = torch.cat([
                        torch.zeros(len(source_embeddings)),
                        torch.ones(len(target_embeddings))
                    ]).to(self.device)
                    
                    # 前向传播
                    domain_output, task_output, _ = self.transfer_model(
                        combined_embeddings,
                        alpha=2.0 / (1.0 + np.exp(-10 * epoch / epochs))
                    )
                    
                    # 计算损失
                    domain_loss = domain_criterion(domain_output, domain_labels.long())
                    task_loss = task_criterion(task_output[:len(source_embeddings)], source_labels)
                    
                    loss = domain_loss + task_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                self.logger.info(f"Epoch {epoch+1}, Loss: {total_loss/len(source_loader)}")
                
            return {
                "status": "success",
                "final_loss": total_loss/len(source_loader)
            }
            
        except Exception as e:
            self.logger.error(f"训练迁移模型失败: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def distill_knowledge(self, teacher_data: List[Dict[str, Any]], 
                         student_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行知识蒸馏"""
        try:
            # 准备数据
            teacher_dataset = self._prepare_dataset(teacher_data)
            student_dataset = self._prepare_dataset(student_data)
            
            teacher_loader = DataLoader(teacher_dataset, batch_size=32, shuffle=True)
            student_loader = DataLoader(student_dataset, batch_size=32, shuffle=True)
            
            # 初始化模型
            if self.teacher_model is None:
                self.teacher_model = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                ).to(self.device)
                
            if self.student_model is None:
                self.student_model = nn.Sequential(
                    nn.Linear(768, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                ).to(self.device)
                
            # 初始化蒸馏器
            self.distillation = KnowledgeDistillation(
                self.teacher_model,
                self.student_model
            )
            
            # 训练教师模型
            teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters())
            teacher_criterion = nn.BCEWithLogitsLoss()
            
            self.teacher_model.train()
            for epoch in range(5):
                total_loss = 0
                for batch in teacher_loader:
                    inputs = batch["input_ids"].to(self.device)
                    masks = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # 获取 BERT 嵌入
                    with torch.no_grad():
                        outputs = self.bert_model(
                            input_ids=inputs,
                            attention_mask=masks
                        )
                        embeddings = outputs.last_hidden_state[:, 0, :]
                        
                    # 前向传播
                    teacher_output = self.teacher_model(embeddings)
                    loss = teacher_criterion(teacher_output, labels)
                    
                    # 反向传播
                    teacher_optimizer.zero_grad()
                    loss.backward()
                    teacher_optimizer.step()
                    
                    total_loss += loss.item()
                    
                self.logger.info(f"Teacher Epoch {epoch+1}, Loss: {total_loss/len(teacher_loader)}")
                
            # 蒸馏知识到学生模型
            student_optimizer = torch.optim.Adam(self.student_model.parameters())
            
            self.student_model.train()
            for epoch in range(10):
                total_loss = 0
                for batch in student_loader:
                    inputs = batch["input_ids"].to(self.device)
                    masks = batch["attention_mask"].to(self.device)
                    
                    # 获取 BERT 嵌入
                    with torch.no_grad():
                        outputs = self.bert_model(
                            input_ids=inputs,
                            attention_mask=masks
                        )
                        embeddings = outputs.last_hidden_state[:, 0, :]
                        
                    # 蒸馏
                    distillation_loss = self.distillation.distill(embeddings)
                    
                    # 反向传播
                    student_optimizer.zero_grad()
                    distillation_loss.backward()
                    student_optimizer.step()
                    
                    total_loss += distillation_loss.item()
                    
                self.logger.info(f"Student Epoch {epoch+1}, Loss: {total_loss/len(student_loader)}")
                
            return {
                "status": "success",
                "teacher_loss": total_loss/len(teacher_loader),
                "student_loss": total_loss/len(student_loader)
            }
            
        except Exception as e:
            self.logger.error(f"知识蒸馏失败: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def evaluate_transfer(self, source_data: List[Dict[str, Any]], 
                         target_data: List[Dict[str, Any]]) -> TransferMetrics:
        """评估迁移效果"""
        try:
            # 准备数据
            source_dataset = self._prepare_dataset(source_data)
            target_dataset = self._prepare_dataset(target_data)
            
            source_loader = DataLoader(source_dataset, batch_size=32)
            target_loader = DataLoader(target_dataset, batch_size=32)
            
            # 评估源域性能
            source_accuracy, source_f1 = self._evaluate_domain(source_loader)
            
            # 评估目标域性能
            target_accuracy, target_f1 = self._evaluate_domain(target_loader)
            
            # 计算迁移损失
            transfer_loss = self._calculate_transfer_loss(source_loader, target_loader)
            
            # 计算领域适应分数
            domain_adaptation = self._calculate_domain_adaptation(source_loader, target_loader)
            
            # 计算知识保持度
            knowledge_preservation = self._calculate_knowledge_preservation(
                source_loader, target_loader
            )
            
            return TransferMetrics(
                accuracy=(source_accuracy + target_accuracy) / 2,
                f1_score=(source_f1 + target_f1) / 2,
                transfer_loss=transfer_loss,
                domain_adaptation_score=domain_adaptation,
                knowledge_preservation=knowledge_preservation
            )
            
        except Exception as e:
            self.logger.error(f"评估迁移效果失败: {str(e)}")
            return TransferMetrics(0.0, 0.0, float('inf'), 0.0, 0.0)
            
    def _evaluate_domain(self, data_loader: DataLoader) -> Tuple[float, float]:
        """评估单个领域的性能"""
        self.transfer_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                
                # 获取 BERT 嵌入
                outputs = self.bert_model(
                    input_ids=inputs,
                    attention_mask=masks
                )
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # 预测
                _, task_output, _ = self.transfer_model(embeddings)
                preds = (torch.sigmoid(task_output) > 0.5).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        return (
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds)
        )
        
    def _calculate_transfer_loss(self, source_loader: DataLoader, 
                               target_loader: DataLoader) -> float:
        """计算迁移损失"""
        self.transfer_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for source_batch, target_batch in zip(source_loader, target_loader):
                # 源域数据
                source_inputs = source_batch["input_ids"].to(self.device)
                source_masks = source_batch["attention_mask"].to(self.device)
                
                # 目标域数据
                target_inputs = target_batch["input_ids"].to(self.device)
                target_masks = target_batch["attention_mask"].to(self.device)
                
                # 获取 BERT 嵌入
                source_outputs = self.bert_model(
                    input_ids=source_inputs,
                    attention_mask=source_masks
                )
                target_outputs = self.bert_model(
                    input_ids=target_inputs,
                    attention_mask=target_masks
                )
                
                source_embeddings = source_outputs.last_hidden_state[:, 0, :]
                target_embeddings = target_outputs.last_hidden_state[:, 0, :]
                
                # 计算特征分布差异
                source_mean = source_embeddings.mean(dim=0)
                target_mean = target_embeddings.mean(dim=0)
                source_std = source_embeddings.std(dim=0)
                target_std = target_embeddings.std(dim=0)
                
                mean_diff = torch.norm(source_mean - target_mean)
                std_diff = torch.norm(source_std - target_std)
                
                total_loss += mean_diff + std_diff
                
        return total_loss.item() / len(source_loader)
        
    def _calculate_domain_adaptation(self, source_loader: DataLoader, 
                                   target_loader: DataLoader) -> float:
        """计算领域适应分数"""
        self.transfer_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for source_batch, target_batch in zip(source_loader, target_loader):
                # 源域数据
                source_inputs = source_batch["input_ids"].to(self.device)
                source_masks = source_batch["attention_mask"].to(self.device)
                
                # 目标域数据
                target_inputs = target_batch["input_ids"].to(self.device)
                target_masks = target_batch["attention_mask"].to(self.device)
                
                # 获取 BERT 嵌入
                source_outputs = self.bert_model(
                    input_ids=source_inputs,
                    attention_mask=source_masks
                )
                target_outputs = self.bert_model(
                    input_ids=target_inputs,
                    attention_mask=target_masks
                )
                
                source_embeddings = source_outputs.last_hidden_state[:, 0, :]
                target_embeddings = target_outputs.last_hidden_state[:, 0, :]
                
                # 合并数据
                combined_embeddings = torch.cat([source_embeddings, target_embeddings])
                domain_labels = torch.cat([
                    torch.zeros(len(source_embeddings)),
                    torch.ones(len(target_embeddings))
                ]).to(self.device)
                
                # 预测领域
                domain_output, _, _ = self.transfer_model(combined_embeddings)
                preds = torch.argmax(domain_output, dim=1)
                
                correct += (preds == domain_labels).sum().item()
                total += len(domain_labels)
                
        return correct / total if total > 0 else 0.0
        
    def _calculate_knowledge_preservation(self, source_loader: DataLoader, 
                                        target_loader: DataLoader) -> float:
        """计算知识保持度"""
        self.transfer_model.eval()
        source_features = []
        target_features = []
        
        with torch.no_grad():
            # 收集源域特征
            for batch in source_loader:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                
                outputs = self.bert_model(
                    input_ids=inputs,
                    attention_mask=masks
                )
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                _, _, features = self.transfer_model(embeddings)
                source_features.append(features.cpu().numpy())
                
            # 收集目标域特征
            for batch in target_loader:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                
                outputs = self.bert_model(
                    input_ids=inputs,
                    attention_mask=masks
                )
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                _, _, features = self.transfer_model(embeddings)
                target_features.append(features.cpu().numpy())
                
        # 计算特征相似度
        source_features = np.concatenate(source_features)
        target_features = np.concatenate(target_features)
        
        similarity = np.mean([
            np.mean(cosine_similarity(source_features, target_features))
        ])
        
        return float(similarity) 