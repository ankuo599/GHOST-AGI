# -*- coding: utf-8 -*-
"""
零样本学习模块 (Zero-Shot Learning Module)

实现从少量或零样本中学习的能力，支持元学习、模式提取和知识迁移
"""

import time
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from collections import defaultdict
import math
import json
import os
import uuid
import random
from scipy.spatial import distance

class ZeroShotLearningModule:
    def __init__(self, memory_system=None, vector_store=None, event_system=None):
        """
        初始化零样本学习模块
        
        Args:
            memory_system: 记忆系统实例
            vector_store: 向量存储实例
            event_system: 事件系统实例
        """
        self.memory_system = memory_system
        self.vector_store = vector_store
        self.event_system = event_system
        
        # 元学习参数
        self.meta_models = {}
        self.learning_strategies = []
        self.task_performances = {}
        
        # 模式提取
        self.pattern_cache = {}
        self.pattern_confidences = {}
        
        # 知识迁移
        self.domain_mappings = {}
        self.transfer_success_rates = {}
        
        # 相似性阈值
        self.similarity_threshold = 0.75
        
        # 初始化基本学习策略
        self._initialize_learning_strategies()
        
    def _initialize_learning_strategies(self):
        """初始化基本学习策略"""
        # 基本的学习策略
        self.learning_strategies = [
            {
                "name": "conceptual_similarity",
                "description": "通过概念相似性进行推断",
                "params": {"similarity_threshold": 0.8}
            },
            {
                "name": "relational_transfer",
                "description": "通过关系映射进行知识迁移",
                "params": {"min_confidence": 0.7}
            },
            {
                "name": "analogical_reasoning",
                "description": "通过类比进行推理",
                "params": {"vector_combination": True}
            },
            {
                "name": "hierarchical_inference",
                "description": "通过层次结构进行归纳和演绎",
                "params": {"inheritance_strength": 0.9}
            }
        ]
        
    def zero_shot_inference(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行零样本推理
        
        Args:
            query: 查询信息，包含任务类型和相关数据
            
        Returns:
            Dict: 推理结果
        """
        task_type = query.get("task_type", "classification")
        data = query.get("data", {})
        context = query.get("context", {})
        
        # 根据任务类型选择合适的推理方法
        if task_type == "classification":
            result = self._zero_shot_classification(data, context)
        elif task_type == "generation":
            result = self._zero_shot_generation(data, context)
        elif task_type == "relation_prediction":
            result = self._zero_shot_relation_prediction(data, context)
        elif task_type == "analogical_reasoning":
            result = self._zero_shot_analogy(data, context)
        else:
            # 默认使用最通用的方法
            result = self._general_zero_shot_inference(data, context)
            
        # 记录推理结果用于后续改进
        self._record_inference_result(task_type, data, result)
        
        return result
    
    def _zero_shot_classification(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """零样本分类"""
        # 获取目标类别
        target_classes = context.get("target_classes", [])
        if not target_classes:
            return {"status": "error", "message": "没有提供目标类别"}
            
        # 获取输入特征或描述
        input_data = data.get("input", "")
        if not input_data:
            return {"status": "error", "message": "没有提供输入数据"}
            
        # 如果有向量存储，使用语义相似度
        if self.vector_store:
            # 将输入和目标类别转换为向量
            input_vector = self.vector_store._text_to_embedding(input_data)
            
            class_vectors = {}
            for cls in target_classes:
                class_vector = self.vector_store._text_to_embedding(cls)
                class_vectors[cls] = class_vector
                
            # 计算相似度并选择最匹配的类别
            similarities = {}
            for cls, vector in class_vectors.items():
                similarity = 1 - np.linalg.norm(input_vector - vector)
                similarities[cls] = similarity
                
            # 返回所有类别及其相似度分数
            sorted_classes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "status": "success",
                "predicted_class": sorted_classes[0][0],
                "confidence": float(sorted_classes[0][1]),
                "all_classes": [{"class": cls, "score": float(score)} for cls, score in sorted_classes]
            }
        else:
            # 降级方法：使用字符串匹配
            max_overlap = 0
            best_class = None
            
            for cls in target_classes:
                # 简单的词重叠度量
                cls_words = set(cls.lower().split())
                input_words = set(input_data.lower().split())
                overlap = len(cls_words.intersection(input_words))
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_class = cls
                    
            # 如果没有重叠，随机选择
            if best_class is None:
                best_class = random.choice(target_classes)
                confidence = 1.0 / len(target_classes)
            else:
                # 粗略的置信度
                confidence = max_overlap / max(1, len(input_data.split()))
                
            return {
                "status": "success",
                "predicted_class": best_class,
                "confidence": confidence
            }
            
    def _zero_shot_generation(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """零样本生成"""
        # 获取输入提示
        prompt = data.get("prompt", "")
        if not prompt:
            return {"status": "error", "message": "没有提供生成提示"}
            
        # 获取约束条件
        constraints = context.get("constraints", {})
        
        # 从记忆中检索相关知识
        if self.memory_system:
            relevant_memories = self.memory_system.search_by_content(prompt, limit=5)
            
            # 分析记忆内容，提取有用信息
            extracted_info = []
            
            for memory in relevant_memories:
                content = memory.get("content", "")
                if content:
                    extracted_info.append(content)
                    
            # 基于提取的信息生成响应
            if extracted_info:
                # 简单的模板填充生成
                response = f"基于我了解的相关信息: {'; '.join(extracted_info[:3])}"
                
                return {
                    "status": "success",
                    "generated_text": response,
                    "source_info": extracted_info
                }
        
        # 如果没有记忆系统或未找到相关记忆，使用模式提取
        patterns = self._extract_patterns_from_prompt(prompt)
        
        if patterns:
            # 选择最匹配的模式
            best_pattern = patterns[0]
            
            # 根据模式生成回应
            response = f"根据'{best_pattern['pattern']}'模式生成的回应"
            
            return {
                "status": "success",
                "generated_text": response,
                "pattern_used": best_pattern
            }
        
        # 如果没有找到模式，返回一个基本回应
        return {
            "status": "limited",
            "message": "无足够信息进行高质量生成",
            "generated_text": "我需要更多信息来提供完整回答。"
        }
        
    def _zero_shot_relation_prediction(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """零样本关系预测"""
        entity1 = data.get("entity1")
        entity2 = data.get("entity2")
        
        if not entity1 or not entity2:
            return {"status": "error", "message": "需要提供两个实体用于关系预测"}
            
        # 尝试在向量存储的知识图中推断关系
        if self.vector_store and hasattr(self.vector_store, "infer_relations"):
            # 检查实体是否存在于知识图中
            # 如果不存在，先创建临时实体
            entity1_id = f"entity:{entity1.lower().replace(' ', '_')}"
            entity2_id = f"entity:{entity2.lower().replace(' ', '_')}"
            
            # 检查并添加实体
            for entity, entity_id in [(entity1, entity1_id), (entity2, entity2_id)]:
                if entity_id not in self.vector_store.concept_vectors:
                    # 临时添加，不保存
                    self.vector_store.add_concept(entity, properties={"temporary": True})
            
            # 推断关系
            relations = self.vector_store.infer_relations(entity1_id, entity2_id)
            
            if relations:
                return {
                    "status": "success",
                    "entity1": entity1,
                    "entity2": entity2,
                    "predicted_relations": relations
                }
                
        # 如果没有知识图或未找到关系，使用一般方法
        # 这里可以使用更复杂的启发式方法，但作为示例，我们返回一个默认关系
        return {
            "status": "limited",
            "entity1": entity1,
            "entity2": entity2,
            "predicted_relations": [{
                "relation": "related_to",
                "confidence": 0.5,
                "inferred": True
            }]
        }
        
    def _zero_shot_analogy(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行零样本类比推理，形式如A:B::C:?
        
        Args:
            data: 类比推理数据
            context: 上下文信息
            
        Returns:
            Dict: 推理结果
        """
        # 获取类比三元组
        a = data.get("term_a")
        b = data.get("term_b")
        c = data.get("term_c")
        
        if not a or not b or not c:
            return {"status": "error", "message": "类比推理需要三个项: A, B, C"}
            
        # 尝试向量存储的类比功能
        if self.vector_store and hasattr(self.vector_store, "find_analogies"):
            try:
                analogy_results = self.vector_store.find_analogies(a, b, c, n=5)
                
                if analogy_results:
                    return {
                        "status": "success",
                        "analogy_query": f"{a}:{b}::{c}:?",
                        "results": analogy_results,
                        "best_match": analogy_results[0]["term"],
                        "confidence": analogy_results[0]["score"]
                    }
            except Exception as e:
                # 如果类比查找失败，回退到备用方法
                pass
                
        # 备用方法：尝试使用关系推理
        try:
            # 将A, B, C转换为向量空间中的点
            a_id = self._get_or_create_concept_id(a)
            b_id = self._get_or_create_concept_id(b)
            c_id = self._get_or_create_concept_id(c)
            
            # 寻找AB之间的关系
            if hasattr(self.vector_store, "infer_relations"):
                ab_relations = self.vector_store.infer_relations(a_id, b_id)
                if ab_relations:
                    # 获取最可能的关系
                    primary_relation = ab_relations[0]["relation"]
                    confidence = ab_relations[0]["confidence"]
                    
                    # 找到与C具有相同关系的概念
                    related = self.vector_store.find_related_concepts(c_id, [primary_relation], max_depth=1)
                    
                    if related:
                        return {
                            "status": "success",
                            "analogy_query": f"{a}:{b}::{c}:?",
                            "inferred_relation": primary_relation,
                            "results": [{"term": r["name"], "score": confidence * 0.8} for r in related[:5]],
                            "best_match": related[0]["name"],
                            "confidence": confidence * 0.8,
                            "method": "relation_inference"
                        }
            
            # 向量空间类比（备选方法）
            # D = C + (B - A)
            if hasattr(self.vector_store, "concept_vectors"):
                # 获取向量
                a_vec = self.vector_store.concept_vectors.get(a_id)
                b_vec = self.vector_store.concept_vectors.get(b_id)
                c_vec = self.vector_store.concept_vectors.get(c_id)
                
                if a_vec is not None and b_vec is not None and c_vec is not None:
                    # 计算目标向量 D = C + (B - A)
                    target_vec = c_vec + (b_vec - a_vec)
                    
                    # 搜索最接近目标向量的概念
                    results = []
                    for concept_id, vec in self.vector_store.concept_vectors.items():
                        if concept_id in [a_id, b_id, c_id]:
                            continue  # 跳过输入概念
                            
                        # 计算相似度
                        sim = 1 - distance.cosine(target_vec, vec)
                        if sim > 0.5:  # 只考虑相似度足够高的
                            concept_name = ""
                            if concept_id in self.vector_store.knowledge_graph.nodes:
                                concept_name = self.vector_store.knowledge_graph.nodes[concept_id].get("name", concept_id)
                            else:
                                concept_name = concept_id.split(":")[-1].replace("_", " ")
                                
                            results.append({"term": concept_name, "score": float(sim)})
                    
                    if results:
                        # 按相似度排序
                        results.sort(key=lambda x: x["score"], reverse=True)
                        
                        return {
                            "status": "success",
                            "analogy_query": f"{a}:{b}::{c}:?",
                            "results": results[:5],
                            "best_match": results[0]["term"],
                            "confidence": results[0]["score"],
                            "method": "vector_analogy"
                        }
        except Exception as e:
            # 所有高级方法都失败了，回退到字符串模式
            pass
            
        # 最基础的字符串替换方法
        try:
            # 寻找A和B之间的字符串关系，并应用到C
            a_lower = a.lower()
            b_lower = b.lower()
            c_lower = c.lower()
            
            # 简单规则：如果B是A的复数形式
            if b_lower == a_lower + 's':
                return {
                    "status": "success",
                    "analogy_query": f"{a}:{b}::{c}:?",
                    "results": [{"term": c_lower + 's', "score": 0.6}],
                    "best_match": c_lower + 's',
                    "confidence": 0.6,
                    "method": "string_pattern"
                }
                
            # 简单规则：如果B是A的过去式
            if b_lower == a_lower + 'ed':
                return {
                    "status": "success",
                    "analogy_query": f"{a}:{b}::{c}:?",
                    "results": [{"term": c_lower + 'ed', "score": 0.6}],
                    "best_match": c_lower + 'ed',
                    "confidence": 0.6,
                    "method": "string_pattern"
                }
                
            # 无法找到模式
            return {
                "status": "limited",
                "message": "无法找到有效的类比模式",
                "analogy_query": f"{a}:{b}::{c}:?",
                "confidence": 0.1
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"类比推理失败: {str(e)}",
                "analogy_query": f"{a}:{b}::{c}:?",
            }

    def _get_or_create_concept_id(self, concept_name: str) -> str:
        """
        获取或创建概念ID
        
        Args:
            concept_name: 概念名称
            
        Returns:
            str: 概念ID
        """
        concept_id = f"concept:{concept_name.lower().replace(' ', '_')}"
        
        # 检查向量存储中是否存在该概念
        if self.vector_store and hasattr(self.vector_store, "concept_vectors"):
            if concept_id not in self.vector_store.concept_vectors:
                # 创建新概念
                if hasattr(self.vector_store, "add_concept"):
                    try:
                        self.vector_store.add_concept(concept_name)
                    except:
                        # 如果添加失败，仍然返回ID
                        pass
        
        return concept_id
        
    def _general_zero_shot_inference(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """通用零样本推理方法"""
        # 结合多种策略
        query = data.get("query", "")
        if not query:
            return {"status": "error", "message": "没有提供查询信息"}
            
        # 应用每种学习策略，收集结果
        strategy_results = []
        
        for strategy in self.learning_strategies:
            strategy_name = strategy["name"]
            params = strategy["params"]
            
            if strategy_name == "conceptual_similarity":
                result = self._apply_conceptual_similarity(query, params)
            elif strategy_name == "relational_transfer":
                result = self._apply_relational_transfer(query, params)
            elif strategy_name == "analogical_reasoning":
                result = self._apply_analogical_reasoning(query, params)
            elif strategy_name == "hierarchical_inference":
                result = self._apply_hierarchical_inference(query, params)
            else:
                continue
                
            if result and result.get("status") == "success":
                strategy_results.append({
                    "strategy": strategy_name,
                    "result": result,
                    "confidence": result.get("confidence", 0.5)
                })
                
        # 如果有有效结果，根据置信度选择最佳结果
        if strategy_results:
            best_result = max(strategy_results, key=lambda x: x["confidence"])
            
            return {
                "status": "success",
                "selected_strategy": best_result["strategy"],
                "result": best_result["result"],
                "confidence": best_result["confidence"],
                "all_strategies": strategy_results
            }
            
        # 如果没有有效结果，返回有限的结果
        return {
            "status": "limited",
            "message": "无法应用任何零样本学习策略",
            "query": query
        }
        
    def _extract_patterns_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """从提示中提取模式"""
        patterns = []
        
        # 简单的模式匹配示例
        if "如何" in prompt or "怎样" in prompt:
            patterns.append({
                "pattern": "how_to",
                "confidence": 0.8,
                "template": "要做{X}，你需要首先{Y}，然后{Z}"
            })
            
        if "为什么" in prompt:
            patterns.append({
                "pattern": "explanation",
                "confidence": 0.7,
                "template": "{X}的原因是{Y}"
            })
            
        if "比较" in prompt or "区别" in prompt:
            patterns.append({
                "pattern": "comparison",
                "confidence": 0.75,
                "template": "{X}和{Y}的主要区别在于{Z}"
            })
            
        # 返回按置信度排序的模式
        return sorted(patterns, key=lambda x: x["confidence"], reverse=True)
        
    def _apply_conceptual_similarity(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用概念相似性策略"""
        threshold = params.get("similarity_threshold", 0.8)
        
        if not self.vector_store:
            return {"status": "error", "message": "概念相似性策略需要向量存储"}
            
        # 将查询转换为向量
        query_vector = self.vector_store._text_to_embedding(query)
        
        # 搜索相似概念
        results = self.vector_store.search(query_vector, k=5)
        
        # 检查最相似的结果是否超过阈值
        if results and results[0]["score"] >= threshold:
            top_result = results[0]
            
            return {
                "status": "success",
                "matched_concept": top_result.get("id", "unknown"),
                "similarity": top_result.get("score", 0),
                "confidence": top_result.get("score", 0),
                "all_matches": results
            }
            
        return {"status": "failure", "message": "未找到足够相似的概念"}
        
    def _apply_relational_transfer(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用关系迁移策略"""
        min_confidence = params.get("min_confidence", 0.7)
        
        # 这里应该根据查询提取实体和潜在关系
        # 简化版：假设查询形式为"EntityA与EntityB的关系是什么？"
        import re
        pattern = r"(.+)与(.+)的关系是什么"
        match = re.search(pattern, query)
        
        if match:
            entity1 = match.group(1)
            entity2 = match.group(2)
            
            # 调用关系预测
            relation_data = {
                "entity1": entity1,
                "entity2": entity2
            }
            
            result = self._zero_shot_relation_prediction(relation_data, {})
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "entity1": entity1,
                    "entity2": entity2,
                    "relations": result["predicted_relations"],
                    "confidence": max([r.get("confidence", 0) for r in result["predicted_relations"]])
                }
                
        return {"status": "failure", "message": "无法应用关系迁移"}
        
    def _apply_analogical_reasoning(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用类比推理策略"""
        # 提取类比查询
        import re
        pattern = r"(.+)对于(.+)就像(.+)对于什么"
        match = re.search(pattern, query)
        
        if match:
            a = match.group(1)
            b = match.group(2)
            c = match.group(3)
            
            # 调用类比查询
            analogy_data = {
                "term_a": a,
                "term_b": b,
                "term_c": c
            }
            
            result = self._zero_shot_analogy(analogy_data, {})
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "analogy": f"{a}:{b}::{c}:{result['results'][0]['name']}",
                    "result": result["results"][0]["name"],
                    "confidence": result["results"][0]["similarity"]
                }
                
        return {"status": "failure", "message": "无法应用类比推理"}
        
    def _apply_hierarchical_inference(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用层次推理策略"""
        inheritance_strength = params.get("inheritance_strength", 0.9)
        
        # 提取分类查询
        import re
        pattern = r"(.+)是(.+)吗"
        match = re.search(pattern, query)
        
        if match:
            entity = match.group(1)
            category = match.group(2)
            
            # 使用向量存储的知识图谱
            if self.vector_store and hasattr(self.vector_store, "knowledge_graph"):
                entity_id = f"concept:{entity.lower().replace(' ', '_')}"
                category_id = f"concept:{category.lower().replace(' ', '_')}"
                
                # 检查直接关系
                if self.vector_store.knowledge_graph.has_edge(entity_id, category_id):
                    edge_data = self.vector_store.knowledge_graph.get_edge_data(entity_id, category_id)
                    if edge_data.get("type") == "is_a":
                        return {
                            "status": "success",
                            "result": True,
                            "relationship": "direct",
                            "confidence": 1.0
                        }
                        
                # 检查间接关系（祖先）
                ancestors = self.vector_store._get_ancestors(entity_id)
                if category_id in ancestors:
                    # 计算置信度：随着层级增加而降低
                    path_length = self.vector_store._get_path_length(entity_id, category_id)
                    confidence = inheritance_strength ** (path_length - 1)
                    
                    return {
                        "status": "success",
                        "result": True,
                        "relationship": "ancestor",
                        "path_length": path_length,
                        "confidence": confidence
                    }
                    
                # 检查相似性
                if entity_id in self.vector_store.concept_vectors and category_id in self.vector_store.concept_vectors:
                    entity_vec = self.vector_store.concept_vectors[entity_id]
                    category_vec = self.vector_store.concept_vectors[category_id]
                    
                    similarity = 1 - np.linalg.norm(entity_vec - category_vec)
                    
                    if similarity > 0.8:
                        return {
                            "status": "success",
                            "result": "similar",
                            "relationship": "similar",
                            "similarity": similarity,
                            "confidence": similarity * 0.7  # 相似性推断的置信度要低一些
                        }
                        
        return {"status": "failure", "message": "无法应用层次推理"}
        
    def _record_inference_result(self, task_type: str, data: Dict[str, Any], result: Dict[str, Any]):
        """记录推理结果，用于学习和改进"""
        # 记录任务性能
        if task_type not in self.task_performances:
            self.task_performances[task_type] = {
                "total": 0,
                "success": 0,
                "confidence_sum": 0,
                "recent_results": []
            }
            
        # 更新统计
        perf = self.task_performances[task_type]
        perf["total"] += 1
        
        if result.get("status") == "success":
            perf["success"] += 1
            
        confidence = result.get("confidence", 0)
        perf["confidence_sum"] += confidence
        
        # 保留最近的结果
        perf["recent_results"].append({
            "data": data,
            "result": result,
            "timestamp": time.time()
        })
        
        # 只保留最近10个结果
        if len(perf["recent_results"]) > 10:
            perf["recent_results"] = perf["recent_results"][-10:]
            
        # 发布事件
        if self.event_system:
            self.event_system.publish("learning.zero_shot_inference", {
                "task_type": task_type,
                "success": result.get("status") == "success",
                "confidence": confidence
            })
            
    def meta_learn(self) -> Dict[str, Any]:
        """
        执行元学习，分析之前的推理结果并改进学习策略
        
        Returns:
            Dict: 元学习结果
        """
        if not self.task_performances:
            return {"status": "no_data", "message": "没有足够的历史数据用于元学习"}
            
        improvements = []
        
        # 分析每个任务类型的成功率
        for task_type, records in self.task_performances.items():
            if not records:
                continue
                
            # 计算成功率
            success_count = sum(1 for r in records if r.get("success", False))
            success_rate = success_count / len(records)
            
            if success_rate < 0.7:  # 如果成功率低于70%，尝试改进
                # 分析失败案例
                failures = [r for r in records if not r.get("success", False)]
                
                # 对失败案例进行分析
                failure_analysis = self._analyze_failures(task_type, failures)
                
                if failure_analysis.get("patterns"):
                    # 根据失败模式调整策略
                    for pattern in failure_analysis["patterns"]:
                        pattern_type = pattern.get("type")
                        
                        # 根据不同类型的失败模式应用不同的改进
                        if pattern_type == "similarity_threshold_too_high":
                            # 降低相似度阈值
                            old_threshold = self.similarity_threshold
                            new_threshold = max(0.5, old_threshold - 0.1)  # 降低阈值，但不低于0.5
                            self.similarity_threshold = new_threshold
                            
                            improvements.append({
                                "task_type": task_type,
                                "improvement": "降低相似度阈值",
                                "old_value": old_threshold,
                                "new_value": new_threshold,
                                "reason": "太多样本未能达到匹配阈值"
                            })
                            
                        elif pattern_type == "wrong_strategy_selection":
                            # 分析哪种策略更适合该任务
                            best_strategy = pattern.get("best_strategy")
                            if best_strategy:
                                # 调整策略选择逻辑
                                for strategy in self.learning_strategies:
                                    if strategy["name"] == best_strategy:
                                        # 增加该策略的权重
                                        if "weight" not in strategy:
                                            strategy["weight"] = 1.0
                                        strategy["weight"] = min(2.0, strategy["weight"] + 0.2)
                                        
                                        improvements.append({
                                            "task_type": task_type,
                                            "improvement": f"提高策略 '{best_strategy}' 的权重",
                                            "new_weight": strategy["weight"],
                                            "reason": "该策略在类似样本上表现更好"
                                        })
                                        
                        elif pattern_type == "insufficient_knowledge":
                            # 记录领域知识不足的问题
                            domain = pattern.get("domain", "general")
                            improvements.append({
                                "task_type": task_type,
                                "improvement": "标记知识缺口",
                                "domain": domain,
                                "reason": "该领域的概念关系不足"
                            })
                            
                            # 可以触发自动知识扩充流程
                            if self.event_system:
                                self.event_system.publish("learning.knowledge_gap", {
                                    "domain": domain,
                                    "task_type": task_type,
                                    "confidence": pattern.get("confidence", 0.7)
                                })
                            
        # 检查是否有类比推理模式可以提取
        analogy_records = [r for task_records in self.task_performances.values() 
                          for r in task_records if r.get("task_subtype") == "analogy"]
                          
        if len(analogy_records) >= 5:
            # 尝试从成功的类比中学习模式
            successful_analogies = [r for r in analogy_records if r.get("success", False)]
            
            if successful_analogies:
                # 提取类比模式
                common_patterns = self._extract_analogy_patterns(successful_analogies)
                
                if common_patterns:
                    # 添加到学习策略
                    self.analogy_patterns = common_patterns
                    
                    improvements.append({
                        "task_type": "analogy",
                        "improvement": "提取类比模式",
                        "patterns_count": len(common_patterns),
                        "reason": "从成功推理中学习"
                    })
        
        # 保存改进记录
        meta_learning_result = {
            "status": "success" if improvements else "no_improvement",
            "improvements": improvements,
            "timestamp": time.time()
        }
        
        return meta_learning_result
        
    def _extract_analogy_patterns(self, successful_analogies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从成功的类比推理中提取模式
        
        Args:
            successful_analogies: 成功的类比记录
            
        Returns:
            List: 提取的模式
        """
        patterns = []
        
        # 按关系类型分组
        relation_groups = defaultdict(list)
        
        for record in successful_analogies:
            query = record.get("query", {})
            result = record.get("result", {})
            
            # 提取A:B::C:D中的各个术语
            a = query.get("data", {}).get("term_a", "")
            b = query.get("data", {}).get("term_b", "")
            c = query.get("data", {}).get("term_c", "")
            d = result.get("best_match", "")
            
            if not (a and b and c and d):
                continue
                
            # 提取使用的方法
            method = result.get("method", "unknown")
            relation_groups[method].append({"a": a, "b": b, "c": c, "d": d})
            
        # 分析每种方法的模式
        for method, examples in relation_groups.items():
            if len(examples) < 2:  # 至少需要2个例子
                continue
                
            if method == "relation_inference":
                # 分析关系推理类比
                inferred_relations = defaultdict(int)
                for record in successful_analogies:
                    if record.get("result", {}).get("method") == "relation_inference":
                        relation = record.get("result", {}).get("inferred_relation", "")
                        if relation:
                            inferred_relations[relation] += 1
                            
                # 提取最常见关系
                common_relations = sorted(inferred_relations.items(), key=lambda x: x[1], reverse=True)
                if common_relations:
                    patterns.append({
                        "type": "relation_pattern",
                        "method": "relation_inference",
                        "common_relations": common_relations[:3],
                        "example_count": len(examples),
                        "confidence": common_relations[0][1] / sum(count for _, count in common_relations)
                    })
                    
            elif method == "vector_analogy":
                # 向量类比模式分析
                # 计算向量偏移的一致性
                if self.vector_store and hasattr(self.vector_store, "concept_vectors"):
                    offset_similarities = []
                    
                    for i in range(len(examples)):
                        ex1 = examples[i]
                        a1_id = self._get_or_create_concept_id(ex1["a"])
                        b1_id = self._get_or_create_concept_id(ex1["b"])
                        
                        if a1_id in self.vector_store.concept_vectors and b1_id in self.vector_store.concept_vectors:
                            offset1 = self.vector_store.concept_vectors[b1_id] - self.vector_store.concept_vectors[a1_id]
                            
                            for j in range(i+1, len(examples)):
                                ex2 = examples[j]
                                c2_id = self._get_or_create_concept_id(ex2["c"])
                                d2_id = self._get_or_create_concept_id(ex2["d"])
                                
                                if c2_id in self.vector_store.concept_vectors and d2_id in self.vector_store.concept_vectors:
                                    offset2 = self.vector_store.concept_vectors[d2_id] - self.vector_store.concept_vectors[c2_id]
                                    
                                    # 计算偏移向量的相似度
                                    sim = 1 - distance.cosine(offset1, offset2)
                                    offset_similarities.append(sim)
                    
                    if offset_similarities:
                        avg_sim = sum(offset_similarities) / len(offset_similarities)
                        if avg_sim > 0.5:  # 偏移向量相似度较高
                            patterns.append({
                                "type": "vector_offset_pattern",
                                "method": "vector_analogy",
                                "average_offset_similarity": avg_sim,
                                "example_count": len(examples),
                                "confidence": avg_sim
                            })
            
            elif method == "string_pattern":
                # 字符串模式分析
                string_patterns = defaultdict(int)
                
                for ex in examples:
                    a, b = ex["a"], ex["b"]
                    c, d = ex["c"], ex["d"]
                    
                    # 检查简单的字符串变换
                    if b == a + "s" and d == c + "s":
                        string_patterns["plural"] += 1
                    elif b == a + "ed" and d == c + "ed":
                        string_patterns["past_tense"] += 1
                    elif b == a + "ing" and d == c + "ing":
                        string_patterns["gerund"] += 1
                        
                if string_patterns:
                    top_pattern = max(string_patterns.items(), key=lambda x: x[1])
                    if top_pattern[1] >= 2:  # 至少出现两次
                        patterns.append({
                            "type": "string_transformation",
                            "method": "string_pattern",
                            "pattern": top_pattern[0],
                            "frequency": top_pattern[1],
                            "example_count": len(examples),
                            "confidence": top_pattern[1] / len(examples)
                        })
                        
        return patterns
        
    def _analyze_failures(self, task_type: str, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析失败案例，提取共同特征
        
        Args:
            task_type: 任务类型
            failures: 失败记录
            
        Returns:
            Dict: 分析结果
        """
        if not failures:
            return {"patterns": []}
            
        patterns = []
        
        # 计算各种可能的失败原因
        reasons = defaultdict(int)
        
        for failure in failures:
            # 提取可能的原因
            result = failure.get("result", {})
            error_msg = result.get("message", "")
            
            if "相似度" in error_msg or "阈值" in error_msg or result.get("confidence", 0) < self.similarity_threshold:
                reasons["similarity_threshold_too_high"] += 1
                
            if "策略" in error_msg or "方法" in error_msg:
                reasons["wrong_strategy_selection"] += 1
                
            if "知识" in error_msg or "概念" in error_msg:
                reasons["insufficient_knowledge"] += 1
                
        # 确定主要失败模式
        total = len(failures)
        for reason, count in reasons.items():
            if count / total > 0.3:  # 如果超过30%的失败有相同原因
                if reason == "similarity_threshold_too_high":
                    patterns.append({
                        "type": "similarity_threshold_too_high",
                        "frequency": count,
                        "confidence": count / total
                    })
                elif reason == "wrong_strategy_selection":
                    # 分析哪个策略可能更合适
                    strategy_performance = defaultdict(list)
                    
                    for failure in failures:
                        # 检查是否有策略尝试记录
                        strategy_trials = failure.get("strategy_trials", [])
                        for trial in strategy_trials:
                            strategy = trial.get("strategy")
                            score = trial.get("score", 0)
                            if strategy:
                                strategy_performance[strategy].append(score)
                    
                    # 找出平均得分最高的策略
                    best_strategy = None
                    best_score = -1
                    
                    for strategy, scores in strategy_performance.items():
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            if avg_score > best_score:
                                best_score = avg_score
                                best_strategy = strategy
                    
                    patterns.append({
                        "type": "wrong_strategy_selection",
                        "frequency": count,
                        "confidence": count / total,
                        "best_strategy": best_strategy
                    })
                elif reason == "insufficient_knowledge":
                    # 分析是哪个知识领域不足
                    domain_gaps = defaultdict(int)
                    
                    for failure in failures:
                        query = failure.get("query", {}).get("data", {})
                        
                        # 提取可能的领域
                        text_fields = []
                        if task_type == "classification":
                            text_fields.append(query.get("input", ""))
                        elif task_type == "generation":
                            text_fields.append(query.get("prompt", ""))
                        elif task_type == "relation_prediction":
                            text_fields.append(query.get("entity1", ""))
                            text_fields.append(query.get("entity2", ""))
                        elif task_type == "analogical_reasoning":
                            text_fields.append(query.get("term_a", ""))
                            text_fields.append(query.get("term_b", ""))
                            text_fields.append(query.get("term_c", ""))
                            
                        # 简单的领域检测
                        text = " ".join([t for t in text_fields if t])
                        if "科技" in text or "技术" in text or "电脑" in text:
                            domain_gaps["technology"] += 1
                        elif "医学" in text or "健康" in text or "疾病" in text:
                            domain_gaps["medical"] += 1
                        elif "金融" in text or "经济" in text or "投资" in text:
                            domain_gaps["finance"] += 1
                        else:
                            domain_gaps["general"] += 1
                            
                    # 找出最缺乏的领域
                    top_domain = max(domain_gaps.items(), key=lambda x: x[1])
                    
                    patterns.append({
                        "type": "insufficient_knowledge",
                        "frequency": count,
                        "confidence": count / total,
                        "domain": top_domain[0]
                    })
                    
        return {"patterns": patterns}
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = {
            "learning_strategies": len(self.learning_strategies),
            "strategies": [s["name"] for s in self.learning_strategies],
            "task_performance": {}
        }
        
        # 添加任务性能统计
        for task, perf in self.task_performances.items():
            if perf["total"] > 0:
                stats["task_performance"][task] = {
                    "total_attempts": perf["total"],
                    "success_count": perf["success"],
                    "success_rate": perf["success"] / perf["total"],
                    "avg_confidence": perf["confidence_sum"] / perf["total"]
                }
                
        return stats 