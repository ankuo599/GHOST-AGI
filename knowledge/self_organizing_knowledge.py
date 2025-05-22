"""
自组织知识结构 (Self-Organizing Knowledge)

提供智能的知识组织和管理机制，根据概念关联性自动组织知识结构。
支持知识的自动分类、聚类和关联，实现知识的高效检索和推理。
"""

import time
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from collections import defaultdict
import re

class SelfOrganizingKnowledge:
    """自组织知识结构，负责知识的智能组织和管理"""
    
    def __init__(self, memory_system=None, logger=None):
        """
        初始化自组织知识结构
        
        Args:
            memory_system: 记忆系统
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 系统组件
        self.memory_system = memory_system
        
        # 概念存储
        self.concepts = {}  # {concept_id: concept_data}
        
        # 关系存储
        self.relations = defaultdict(set)  # {concept_id: {(related_id, relation_type)}}
        
        # 层次结构存储
        self.hierarchies = {}  # {hierarchy_id: hierarchy_data}
        
        # 概念向量缓存
        self.concept_vectors = {}  # {concept_id: vector}
        
        # 概念索引
        self.concept_index = {
            "by_name": defaultdict(list),
            "by_tag": defaultdict(list),
            "by_domain": defaultdict(list),
            "by_type": defaultdict(list)
        }
        
        # 统计信息
        self.stats = {
            "concept_count": 0,
            "relation_count": 0,
            "hierarchy_count": 0,
            "reorganizations": 0,
            "last_reorganized": None
        }
        
        # 概念元数据
        self.concept_metadata = {}  # {concept_id: metadata}
        
        # 配置
        self.config = {
            "vector_dim": 100,
            "similarity_threshold": 0.7,
            "auto_reorganize": True,
            "reorganize_threshold": 50,  # 新增概念数量触发重组织
            "max_relation_distance": 0.8,
            "min_cluster_size": 3,
            "concept_size_limit": 1024 * 1024,  # 1MB
            "enable_cross_domain": True
        }
        
        self.logger.info("自组织知识结构初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("SelfOrganizingKnowledge")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("knowledge.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def add_concept(self, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加概念
        
        Args:
            concept_data: 概念数据
            
        Returns:
            Dict: 添加结果
        """
        # 验证概念数据
        if not self._validate_concept_data(concept_data):
            return {
                "status": "error",
                "message": "概念数据无效"
            }
            
        # 生成概念ID（如果没有提供）
        if "id" not in concept_data:
            concept_data["id"] = str(uuid.uuid4())
            
        concept_id = concept_data["id"]
        
        # 检查概念是否已存在
        if concept_id in self.concepts:
            return {
                "status": "error",
                "message": f"概念已存在: {concept_id}"
            }
            
        # 添加创建时间
        if "created_at" not in concept_data:
            concept_data["created_at"] = time.time()
            
        # 添加默认值
        if "tags" not in concept_data:
            concept_data["tags"] = []
            
        if "domain" not in concept_data:
            concept_data["domain"] = "general"
            
        if "type" not in concept_data:
            concept_data["type"] = "generic"
            
        # 存储概念
        self.concepts[concept_id] = concept_data
        
        # 更新索引
        self._update_concept_index(concept_id, concept_data)
        
        # 生成概念向量
        self.concept_vectors[concept_id] = self._generate_concept_vector(concept_data)
        
        # 更新统计信息
        self.stats["concept_count"] += 1
        
        # 初始化元数据
        self.concept_metadata[concept_id] = {
            "usage_count": 0,
            "last_accessed": time.time(),
            "creation_time": time.time(),
            "related_concepts": set(),
            "similarity_scores": {}
        }
        
        # 检查是否需要重组织
        self._check_reorganization_needed()
        
        # 查找相似概念并建立关系
        similar_concepts = self._find_similar_concepts(concept_id)
        for similar in similar_concepts:
            sim_id = similar["id"]
            similarity = similar["similarity"]
            
            # 记录相似度
            self.concept_metadata[concept_id]["similarity_scores"][sim_id] = similarity
            self.concept_metadata[sim_id]["similarity_scores"][concept_id] = similarity
            
            # 根据相似度确定关系类型
            relation_type = self._determine_relation_type(concept_id, sim_id, similarity)
            
            # 建立关系
            self.add_relation(concept_id, sim_id, relation_type)
        
        self.logger.info(f"已添加概念: {concept_id} ({concept_data.get('name', 'unnamed')})")
        
        return {
            "status": "success",
            "concept_id": concept_id,
            "similar_concepts": [s["id"] for s in similar_concepts]
        }
    
    def add_relation(self, source_id: str, target_id: str, 
                    relation_type: str, confidence: float = 0.8) -> Dict[str, Any]:
        """
        添加概念关系
        
        Args:
            source_id: 源概念ID
            target_id: 目标概念ID
            relation_type: 关系类型
            confidence: 关系置信度
            
        Returns:
            Dict: 添加结果
        """
        # 检查概念是否存在
        if source_id not in self.concepts:
            return {
                "status": "error",
                "message": f"源概念不存在: {source_id}"
            }
            
        if target_id not in self.concepts:
            return {
                "status": "error",
                "message": f"目标概念不存在: {target_id}"
            }
        
        # 添加关系
        relation_data = (target_id, relation_type, confidence)
        self.relations[source_id].add(relation_data)
        
        # 添加反向关系（根据关系类型确定）
        inverse_type = self._get_inverse_relation_type(relation_type)
        if inverse_type:
            inverse_relation = (source_id, inverse_type, confidence)
            self.relations[target_id].add(inverse_relation)
        
        # 更新元数据
        self.concept_metadata[source_id]["related_concepts"].add(target_id)
        self.concept_metadata[target_id]["related_concepts"].add(source_id)
        
        # 更新统计信息
        self.stats["relation_count"] += 1
        
        self.logger.info(f"已添加关系: {source_id} --[{relation_type}]--> {target_id}")
        
        return {
            "status": "success",
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "inverse_type": inverse_type
        }
    
    def create_hierarchy(self, root_id: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建概念层次结构
        
        Args:
            root_id: 根概念ID
            structure: 层次结构描述
            
        Returns:
            Dict: 创建结果
        """
        # 检查根概念是否存在
        if root_id not in self.concepts:
            return {
                "status": "error",
                "message": f"根概念不存在: {root_id}"
            }
        
        # 创建层次结构ID
        hierarchy_id = str(uuid.uuid4())
        
        # 创建层次结构记录
        hierarchy_data = {
            "id": hierarchy_id,
            "name": structure.get("name", f"Hierarchy_{hierarchy_id}"),
            "root_id": root_id,
            "created_at": time.time(),
            "structure": structure,
            "depth": self._calculate_hierarchy_depth(structure),
            "concept_count": self._count_hierarchy_concepts(structure)
        }
        
        # 存储层次结构
        self.hierarchies[hierarchy_id] = hierarchy_data
        
        # 处理层次关系
        self._process_hierarchy_relations(root_id, structure)
        
        # 更新统计信息
        self.stats["hierarchy_count"] += 1
        
        self.logger.info(f"已创建层次结构: {hierarchy_id}, 根概念: {root_id}")
        
        return {
            "status": "success",
            "hierarchy_id": hierarchy_id,
            "root_id": root_id,
            "depth": hierarchy_data["depth"],
            "concept_count": hierarchy_data["concept_count"]
        }
    
    def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        获取概念详情
        
        Args:
            concept_id: 概念ID
            
        Returns:
            Dict: 概念详情
        """
        # 检查概念是否存在
        if concept_id not in self.concepts:
            return {
                "status": "error",
                "message": f"概念不存在: {concept_id}"
            }
            
        # 获取概念数据
        concept_data = self.concepts[concept_id]
        
        # 获取关系
        relations = self._find_relations(concept_id)
        
        # 获取所属层次结构
        hierarchies = []
        for h_id, h_data in self.hierarchies.items():
            if concept_id == h_data["root_id"] or self._is_in_hierarchy(concept_id, h_data["structure"]):
                hierarchies.append({
                    "id": h_id,
                    "name": h_data["name"],
                    "is_root": concept_id == h_data["root_id"]
                })
        
        # 更新访问统计
        self.concept_metadata[concept_id]["usage_count"] += 1
        self.concept_metadata[concept_id]["last_accessed"] = time.time()
        
        # 构建响应
        result = {
            "status": "success",
            "concept": concept_data,
            "relations": relations,
            "hierarchies": hierarchies,
            "metadata": {
                "usage_count": self.concept_metadata[concept_id]["usage_count"],
                "creation_time": self.concept_metadata[concept_id]["creation_time"],
                "last_accessed": self.concept_metadata[concept_id]["last_accessed"],
                "related_count": len(self.concept_metadata[concept_id]["related_concepts"])
            }
        }
        
        return result
    
    def search_concepts(self, query: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """
        搜索概念
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            
        Returns:
            Dict: 搜索结果
        """
        results = []
        total_matches = 0
        
        # 根据查询类型执行不同的搜索
        if "text" in query:
            # 文本搜索
            text_results = self._search_by_text(query["text"], limit)
            results.extend(text_results)
            total_matches += len(text_results)
            
        if "vector" in query:
            # 向量搜索
            vector_results = self._search_by_vector(query["vector"], limit)
            results.extend(vector_results)
            total_matches += len(vector_results)
            
        if "domain" in query:
            # 领域搜索
            domain_results = self._search_by_domain(query["domain"], limit)
            results.extend(domain_results)
            total_matches += len(domain_results)
            
        if "tag" in query:
            # 标签搜索
            tag_results = self._search_by_tag(query["tag"], limit)
            results.extend(tag_results)
            total_matches += len(tag_results)
            
        if "similar_to" in query:
            # 相似性搜索
            similar_results = self._search_similar_concepts(query["similar_to"], limit)
            results.extend(similar_results)
            total_matches += len(similar_results)
            
        if "relation" in query:
            # 关系搜索
            relation_results = self._search_by_relation(query["relation"], limit)
            results.extend(relation_results)
            total_matches += len(relation_results)
        
        # 去重并限制结果数量
        unique_results = []
        seen_ids = set()
        
        for result in results:
            concept_id = result["id"]
            if concept_id not in seen_ids:
                seen_ids.add(concept_id)
                unique_results.append(result)
                
                # 更新概念访问统计
                self.concept_metadata[concept_id]["usage_count"] += 1
                self.concept_metadata[concept_id]["last_accessed"] = time.time()
                
        # 最终结果排序和限制
        final_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)[:limit]
        
        return {
            "status": "success",
            "query": query,
            "results": final_results,
            "total_matches": total_matches,
            "unique_matches": len(unique_results),
            "returned_count": len(final_results)
        }
    
    def reorganize_knowledge(self) -> Dict[str, Any]:
        """
        重组织知识结构
        
        Returns:
            Dict: 重组织结果
        """
        self.logger.info("开始重组织知识结构...")
        
        start_time = time.time()
        
        result = {
            "success": False,
            "started_at": start_time,
            "stages": [],
            "clusters_formed": 0,
            "relations_added": 0,
                        "relations_removed": 0
        }
        
        try:
            # 阶段1: 概念聚类
            self.logger.info("执行概念聚类...")
            clusters = self._cluster_concepts()
            
            stage_result = {
                "name": "concept_clustering",
                "cluster_count": len(clusters),
                "clusters": [{"size": len(c), "concepts": c[:5]} for c in clusters]  # 仅包含前5个概念作为示例
            }
            result["stages"].append(stage_result)
            result["clusters_formed"] = len(clusters)
            
            # 阶段2: 优化关系
            self.logger.info("优化概念关系...")
            relation_changes = self._optimize_relations()
            
            stage_result = {
                "name": "relation_optimization",
                "added": relation_changes["added"],
                "removed": relation_changes["removed"],
                "updated": relation_changes["updated"]
            }
            result["stages"].append(stage_result)
            result["relations_added"] = relation_changes["added"]
            result["relations_removed"] = relation_changes["removed"]
            
            # 阶段3: 更新层次结构
            self.logger.info("更新层次结构...")
            hierarchy_changes = self._update_hierarchies(clusters)
            
            stage_result = {
                "name": "hierarchy_update",
                "created": hierarchy_changes["created"],
                "updated": hierarchy_changes["updated"],
                "removed": hierarchy_changes["removed"]
            }
            result["stages"].append(stage_result)
            
            # 阶段4: 概念聚类优化
            self.logger.info("优化概念聚类...")
            cluster_optimizations = {}
            for i, cluster in enumerate(clusters):
                if len(cluster) > 3:  # 只优化足够大的聚类
                    cluster_result = self._optimize_cluster(cluster)
                    cluster_optimizations[f"cluster_{i}"] = cluster_result
            
            stage_result = {
                "name": "cluster_optimization",
                "optimized_clusters": len(cluster_optimizations),
                "details": cluster_optimizations
            }
            result["stages"].append(stage_result)
            
            # 阶段5: 跨领域知识关联
            if self.config["enable_cross_domain"]:
                self.logger.info("创建跨领域知识关联...")
                cross_domain = self._create_cross_domain_connections()
                
                stage_result = {
                    "name": "cross_domain_connection",
                    "connections": cross_domain["connections"],
                    "domains": cross_domain["domains"]
                }
                result["stages"].append(stage_result)
            
            # 阶段6: 检测和合并冗余概念
            self.logger.info("检测和合并冗余概念...")
            redundancy = self._detect_merge_redundancy()
            
            stage_result = {
                "name": "redundancy_detection",
                "detected": redundancy["detected"],
                "merged": redundancy["merged"]
            }
            result["stages"].append(stage_result)
            
            # 阶段7: 自动分类概念
            self.logger.info("自动分类概念...")
            classification = self._auto_classify_concepts()
            
            stage_result = {
                "name": "concept_classification",
                "classified": classification["classified"],
                "categories": classification["categories"]
            }
            result["stages"].append(stage_result)
            
            # 更新重组织统计信息
            self.stats["reorganizations"] += 1
            self.stats["last_reorganized"] = time.time()
            
            result["success"] = True
            result["completed_at"] = time.time()
            result["duration"] = result["completed_at"] - start_time
            
            self.logger.info(f"知识重组织完成，耗时 {result['duration']:.2f} 秒")
            
        except Exception as e:
            self.logger.error(f"知识重组织过程异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            result["success"] = False
            result["error"] = str(e)
            result["completed_at"] = time.time()
            result["duration"] = result["completed_at"] - start_time
        
        return result
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        获取知识统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 概念统计
        concept_stats = {
            "total": self.stats["concept_count"],
            "by_domain": defaultdict(int),
            "by_type": defaultdict(int),
            "by_tag": defaultdict(int),
            "created_last_day": 0,
            "created_last_week": 0,
            "created_last_month": 0
        }
        
        # 关系统计
        relation_stats = {
            "total": self.stats["relation_count"],
            "by_type": defaultdict(int),
            "avg_per_concept": 0,
            "max_per_concept": 0
        }
        
        # 层次结构统计
        hierarchy_stats = {
            "total": self.stats["hierarchy_count"],
            "avg_depth": 0,
            "max_depth": 0,
            "avg_concepts": 0
        }
        
        # 使用统计
        usage_stats = {
            "most_used_concepts": [],
            "least_used_concepts": [],
            "recent_accessed": []
        }
        
        now = time.time()
        one_day_ago = now - 86400
        one_week_ago = now - 604800
        one_month_ago = now - 2592000
        
        # 计算概念相关统计
        for concept_id, concept in self.concepts.items():
            # 领域统计
            domain = concept.get("domain", "general")
            concept_stats["by_domain"][domain] += 1
            
            # 类型统计
            concept_type = concept.get("type", "generic")
            concept_stats["by_type"][concept_type] += 1
            
            # 标签统计
            for tag in concept.get("tags", []):
                concept_stats["by_tag"][tag] += 1
                
            # 创建时间统计
            created_at = concept.get("created_at", 0)
            if created_at > one_day_ago:
                concept_stats["created_last_day"] += 1
            if created_at > one_week_ago:
                concept_stats["created_last_week"] += 1
            if created_at > one_month_ago:
                concept_stats["created_last_month"] += 1
                
            # 关系统计
            if concept_id in self.relations:
                rel_count = len(self.relations[concept_id])
                relation_stats["max_per_concept"] = max(relation_stats["max_per_concept"], rel_count)
                
                for _, rel_type, _ in self.relations[concept_id]:
                    relation_stats["by_type"][rel_type] += 1
        
        # 计算平均关系数
        if concept_stats["total"] > 0:
            relation_stats["avg_per_concept"] = relation_stats["total"] / concept_stats["total"]
            
        # 计算层次结构统计
        depths = []
        concept_counts = []
        
        for h_id, h_data in self.hierarchies.items():
            depth = h_data.get("depth", 0)
            depths.append(depth)
            hierarchy_stats["max_depth"] = max(hierarchy_stats["max_depth"], depth)
            
            concept_count = h_data.get("concept_count", 0)
            concept_counts.append(concept_count)
        
        if depths:
            hierarchy_stats["avg_depth"] = sum(depths) / len(depths)
        if concept_counts:
            hierarchy_stats["avg_concepts"] = sum(concept_counts) / len(concept_counts)
            
        # 使用统计
        usage_data = [(cid, self.concept_metadata[cid]["usage_count"]) for cid in self.concepts]
        usage_data.sort(key=lambda x: x[1], reverse=True)
        
        # 最常用概念
        for cid, count in usage_data[:10]:
            usage_stats["most_used_concepts"].append({
                "id": cid,
                "name": self.concepts[cid]["name"],
                "usage_count": count
            })
            
        # 最少用概念
        for cid, count in usage_data[-10:]:
            usage_stats["least_used_concepts"].append({
                "id": cid,
                "name": self.concepts[cid]["name"],
                "usage_count": count
            })
            
        # 最近访问概念
        recent_data = [(cid, self.concept_metadata[cid]["last_accessed"]) for cid in self.concepts]
        recent_data.sort(key=lambda x: x[1], reverse=True)
        
        for cid, last_access in recent_data[:10]:
            usage_stats["recent_accessed"].append({
                "id": cid,
                "name": self.concepts[cid]["name"],
                "last_accessed": last_access
            })
        
        return {
            "timestamp": time.time(),
            "concept_stats": concept_stats,
            "relation_stats": relation_stats,
            "hierarchy_stats": hierarchy_stats,
            "usage_stats": usage_stats,
            "reorganizations": self.stats["reorganizations"],
            "last_reorganized": self.stats["last_reorganized"]
        }
    
    def export_knowledge_graph(self, format_type: str = "json") -> Dict[str, Any]:
        """
        导出知识图谱
        
        Args:
            format_type: 导出格式类型 (json, cytoscape, networkx)
            
        Returns:
            Dict: 导出结果
        """
        # 基本知识图谱结构
        knowledge_graph = {
                "metadata": {
                    "exported_at": time.time(),
                "concept_count": self.stats["concept_count"],
                "relation_count": self.stats["relation_count"],
                "hierarchy_count": self.stats["hierarchy_count"]
            },
            "concepts": {},
            "relations": [],
            "hierarchies": {}
        }
        
        # 导出概念
        for concept_id, concept in self.concepts.items():
            # 复制概念，但排除过大的字段
            concept_copy = {}
            for key, value in concept.items():
                # 跳过大型数据字段
                if key not in ["raw_data", "embeddings", "large_content"] and (
                    not isinstance(value, (str, list, dict)) or 
                    len(json.dumps(value)) < 10000  # 限制大小
                ):
                    concept_copy[key] = value
                    
            knowledge_graph["concepts"][concept_id] = concept_copy
            
                    # 导出关系
            for source_id, relations in self.relations.items():
                for target_id, relation_type, confidence in relations:
                    knowledge_graph["relations"].append({
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "confidence": confidence
                })
                
        # 导出层次结构
        for hierarchy_id, hierarchy in self.hierarchies.items():
            # 简化层次结构表示
            simplified_hierarchy = {
                "id": hierarchy["id"],
                "name": hierarchy.get("name", ""),
                "root_id": hierarchy["root_id"],
                "depth": hierarchy.get("depth", 0)
            }
            knowledge_graph["hierarchies"][hierarchy_id] = simplified_hierarchy
            
        # 根据请求格式转换
        if format_type == "json":
            return {
                "status": "success",
                "format": "json",
                "data": knowledge_graph
            }
        elif format_type == "cytoscape":
            # 转换为Cytoscape格式
            cytoscape_data = self._convert_to_cytoscape(knowledge_graph)
            return {
                "status": "success",
                "format": "cytoscape",
                "data": cytoscape_data
            }
        elif format_type == "networkx":
            # 转换为NetworkX格式(简化表示)
            networkx_data = {
                "nodes": list(knowledge_graph["concepts"].keys()),
                "edges": [(r["source"], r["target"], {"type": r["type"], "weight": r["confidence"]}) 
                         for r in knowledge_graph["relations"]]
            }
            return {
                "status": "success",
                "format": "networkx",
                "data": networkx_data
            }
        else:
            return {
                "status": "error",
                "message": f"不支持的导出格式: {format_type}"
            }
    
    def import_knowledge_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入知识图谱
        
        Args:
            graph_data: 知识图谱数据
            
        Returns:
            Dict: 导入结果
        """
        if not isinstance(graph_data, dict):
            return {
                "status": "error",
                "message": "导入数据格式无效"
            }
            
        result = {
            "status": "success",
            "imported_concepts": 0,
            "imported_relations": 0,
            "imported_hierarchies": 0,
            "errors": [],
            "warnings": []
        }
        
        # 导入概念
        concepts = graph_data.get("concepts", {})
        for concept_id, concept_data in concepts.items():
            try:
                # 检查概念ID是否已存在
                if concept_id in self.concepts:
                    result["warnings"].append(f"概念ID已存在，将更新: {concept_id}")
                    # 更新概念
                    self.concepts[concept_id].update(concept_data)
                else:
                    # 添加概念
                    concept_data["id"] = concept_id
                    self.add_concept(concept_data)
                    result["imported_concepts"] += 1
            except Exception as e:
                result["errors"].append(f"导入概念 {concept_id} 失败: {str(e)}")
                
        # 导入关系
        relations = graph_data.get("relations", [])
        for relation in relations:
            try:
                source_id = relation.get("source")
                target_id = relation.get("target")
                relation_type = relation.get("type")
                confidence = relation.get("confidence", 1.0)
                
                if not (source_id and target_id and relation_type):
                    result["warnings"].append(f"关系数据不完整，跳过: {relation}")
                    continue
                    
                # 检查概念是否存在
                if source_id not in self.concepts:
                        result["warnings"].append(f"关系源概念不存在，跳过: {source_id}")
                continue
                
                if target_id not in self.concepts:
                    result["warnings"].append(f"关系目标概念不存在，跳过: {target_id}")
                    continue
                    
                # 添加关系
                self.add_relation(source_id, target_id, relation_type, confidence)
                result["imported_relations"] += 1
            except Exception as e:
                result["errors"].append(f"导入关系失败: {str(e)}")
                
        # 导入层次结构
        hierarchies = graph_data.get("hierarchies", {})
        for hierarchy_id, hierarchy_data in hierarchies.items():
            try:
                root_id = hierarchy_data.get("root_id")
                structure = hierarchy_data.get("structure", {})
                
                if not root_id:
                    result["warnings"].append(f"层次结构根概念缺失，跳过: {hierarchy_id}")
                    continue
                    
                if root_id not in self.concepts:
                    result["warnings"].append(f"层次结构根概念不存在，跳过: {root_id}")
                    continue
                    
                # 创建层次结构
                self.create_hierarchy(root_id, structure)
                result["imported_hierarchies"] += 1
            except Exception as e:
                result["errors"].append(f"导入层次结构失败: {str(e)}")
                
        # 更新统计信息
        self.stats["concept_count"] = len(self.concepts)
        self.stats["relation_count"] = sum(len(relations) for relations in self.relations.values())
        self.stats["hierarchy_count"] = len(self.hierarchies)
        
        # 如果导入了足够多的内容，触发知识重组织
        if result["imported_concepts"] > self.config["reorganize_threshold"]:
            self.logger.info(f"导入了大量概念 ({result['imported_concepts']}), 触发知识重组织")
            try:
                reorganize_result = self.reorganize_knowledge()
                result["reorganized"] = reorganize_result["success"]
            except Exception as e:
                result["warnings"].append(f"知识重组织失败: {str(e)}")
                result["reorganized"] = False
        
        return result
    
    def _generate_concept_vector(self, concept_data: Dict[str, Any]) -> np.ndarray:
        """生成概念向量表示"""
        # 简化实现：模拟基于概念属性的向量生成
        vector_dim = self.config["vector_dim"]
        vector = np.random.randn(vector_dim)
        
        # 实际实现中，这里应该使用嵌入模型生成语义向量
        return vector / np.linalg.norm(vector)  # 归一化
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算向量相似度"""
        # 余弦相似度
        return float(np.dot(vec1, vec2))
    
    def _find_similar_concepts(self, concept_id: str, threshold: float = None) -> List[Dict[str, Any]]:
        """查找与给定概念相似的概念"""
        if threshold is None:
            threshold = self.config["similarity_threshold"]
            
        if concept_id not in self.concepts or concept_id not in self.concept_vectors:
            return []
            
        concept_vector = self.concept_vectors[concept_id]
        similarities = []
        
        for other_id, other_vector in self.concept_vectors.items():
            if other_id != concept_id:
                similarity = self._calculate_similarity(concept_vector, other_vector)
            
            if similarity >= threshold:
                    similarities.append({
                    "id": other_id,
                    "name": self.concepts[other_id]["name"],
                    "similarity": similarity
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities
    
    def _find_relations(self, concept_id: str):
        """查找概念的关系"""
        if concept_id not in self.relations:
            return []
            
        outgoing_relations = []
        for target_id, rel_type, confidence in self.relations[concept_id]:
            if target_id in self.concepts:
                outgoing_relations.append({
                    "target_id": target_id,
                    "target_name": self.concepts[target_id]["name"],
                    "type": rel_type,
                    "confidence": confidence,
                    "direction": "outgoing"
                })
                
        # 查找入向关系
        incoming_relations = []
        for source_id, source_relations in self.relations.items():
            if source_id != concept_id:
                for target_id, rel_type, confidence in source_relations:
                    if target_id == concept_id:
                        incoming_relations.append({
                            "source_id": source_id,
                            "source_name": self.concepts[source_id]["name"],
                            "type": rel_type,
                            "confidence": confidence,
                            "direction": "incoming"
                        })
                        
        return outgoing_relations + incoming_relations
    
    def _has_common_attributes(self, concept_id1: str, concept_id2: str) -> bool:
        """检查两个概念是否有共同属性"""
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            return False
            
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        
        # 检查标签重叠
        tags1 = set(concept1.get("tags", []))
        tags2 = set(concept2.get("tags", []))
        
        # 检查属性重叠
        attrs1 = set(concept1.get("attributes", {}).keys())
        attrs2 = set(concept2.get("attributes", {}).keys())
        
        return (tags1 & tags2) or (attrs1 & attrs2)
    
    def _process_hierarchy_relations(self, parent_id: str, structure: Dict[str, Any], 
                                  rel_type: str = "has_subconcept"):
        """处理层次结构中的关系"""
        if "children" not in structure:
            return
            
        for child in structure["children"]:
            child_id = child.get("id")
            
            if not child_id or child_id not in self.concepts:
                continue
                
            # 添加父子关系
            self.add_relation(parent_id, child_id, rel_type)
                
                # 递归处理子节点
            self._process_hierarchy_relations(child_id, child)
    
    def _check_reorganization_needed(self, force: bool = False):
        """检查是否需要重组织知识"""
        if force or (
            self.config["auto_reorganize"] and 
            self.stats["concept_count"] % self.config["reorganize_threshold"] == 0
        ):
            self.logger.info("触发自动知识重组织")
            try:
                self.reorganize_knowledge()
            except Exception as e:
                self.logger.error(f"自动知识重组织失败: {str(e)}")
    
    def _cluster_concepts(self) -> List[List[str]]:
        """将概念聚类"""
        # 计算相似度矩阵
        concept_ids = list(self.concepts.keys())
        similarity_matrix = {}
        
        for i, id1 in enumerate(concept_ids):
            vector1 = self.concept_vectors[id1]
            
            for j in range(i+1, len(concept_ids)):
                id2 = concept_ids[j]
                vector2 = self.concept_vectors[id2]
                
                similarity = self._calculate_similarity(vector1, vector2)
                similarity_matrix[(id1, id2)] = similarity
                similarity_matrix[(id2, id1)] = similarity
        
        # 贪婪聚类 (简化实现)
        clusters = []
        remaining = set(concept_ids)
        
        while remaining:
            # 选择一个种子概念
            seed = next(iter(remaining))
            remaining.remove(seed)
            
            # 创建新的聚类
            cluster = [seed]
            
            # 添加相似概念
            candidates = list(remaining)
            for candidate in candidates:
                # 检查与聚类中所有概念的平均相似度
                avg_similarity = sum(similarity_matrix.get((candidate, c), 0) for c in cluster) / len(cluster)
                
                if avg_similarity >= self.config["similarity_threshold"]:
                    cluster.append(candidate)
                    remaining.remove(candidate)
            
                clusters.append(cluster)
        
        # 合并过小的聚类
        min_size = self.config["min_cluster_size"]
        i = 0
        while i < len(clusters):
            if len(clusters[i]) < min_size:
                # 尝试合并到最相似的聚类
                best_match = -1
                best_similarity = -1
                
                for j, other_cluster in enumerate(clusters):
                    if i != j:
                        # 计算聚类间平均相似度
                        sim_sum = 0
                        count = 0
                        
                        for c1 in clusters[i]:
                            for c2 in other_cluster:
                                sim_sum += similarity_matrix.get((c1, c2), 0)
                                count += 1
                                
                        if count > 0:
                            avg_sim = sim_sum / count
                            if avg_sim > best_similarity:
                                best_similarity = avg_sim
                                best_match = j
                
                if best_match >= 0 and best_similarity > 0.3:  # 合并阈值
                    # 合并到最佳匹配的聚类
                    clusters[best_match].extend(clusters[i])
                    clusters.pop(i)
                else:
                    i += 1
            else:
                i += 1
        
        return clusters
    
    def _optimize_relations(self) -> Dict[str, int]:
        """优化概念间关系"""
        changes = {
            "added": 0,
            "removed": 0,
            "modified": 0
        }
        
        # 寻找缺失的传递关系
        for source_id in self.relations:
            # 获取源概念的所有直接关系
            direct_relations = set()
            for target_id, rel_type, _ in self.relations[source_id]:
                direct_relations.add((target_id, rel_type))
            
            # 检查二级关系
            second_level_relations = []
            for target_id, rel_type1, conf1 in self.relations[source_id]:
                for second_target, rel_type2, conf2 in self.relations.get(target_id, []):
                    if second_target != source_id:
                        # 推断传递关系类型
                        transitive_type = self._infer_transitive_relation(rel_type1, rel_type2)
                        if transitive_type:
                            transitive_conf = conf1 * conf2 * 0.8  # 降低传递关系的置信度
                            if transitive_conf >= self.config["relation_confidence_threshold"]:
                                second_level_relations.append((second_target, transitive_type, transitive_conf))
            
            # 添加缺失的传递关系
            for second_target, trans_type, trans_conf in second_level_relations:
                if (second_target, trans_type) not in direct_relations:
                    self.add_relation(source_id, second_target, trans_type, trans_conf)
                    changes["added"] += 1
        
        # 移除冗余关系 (实际应通过更复杂的算法)
        
        return changes
    
    def _infer_transitive_relation(self, rel_type1: str, rel_type2: str) -> Optional[str]:
        """推断传递关系类型"""
        # 简化实现: 定义一些基本规则
        
        # 相同关系类型可能产生相同的传递关系
        if rel_type1 == rel_type2 and rel_type1 in {"is_a", "part_of", "has_property"}:
            return rel_type1
            
        # 特定组合规则
        if rel_type1 == "is_a" and rel_type2 == "is_a":
            return "is_a"  # 传递性: A是B，B是C，则A是C
            
        if rel_type1 == "part_of" and rel_type2 == "part_of":
            return "part_of"  # 传递性: A是B的一部分，B是C的一部分，则A是C的一部分
            
        if rel_type1 == "is_a" and rel_type2 == "has_property":
            return "has_property"  # A是B，B有属性C，则A有属性C
            
        # 其他情况无法明确推断
        return None
    
    def _update_hierarchies(self, clusters: List[List[str]]) -> Dict[str, int]:
        """更新层次结构"""
        changes = {
            "created": 0,
            "updated": 0
        }
        
        # 对每个聚类创建或更新层次结构
        for cluster in clusters:
            if not cluster:
                continue
                
            # 选择聚类中最重要的概念作为根
            root_id = self._find_cluster_root(cluster)
            
            # 构建层次结构
            structure = self._build_hierarchy_structure(root_id, cluster, depth=0)
            
            # 创建新的层次结构
            self.create_hierarchy(root_id, structure)
            changes["created"] += 1
        
        return changes
    
    def _find_cluster_root(self, cluster: List[str]) -> str:
        """找到聚类的根概念"""
        if not cluster:
            return None
            
        # 简化实现: 选择关系最多或重要性最高的概念
        concept_scores = []
        
        for concept_id in cluster:
            # 关系数量
            relation_count = len(self.relations.get(concept_id, []))
            
            # 重要性分数
            importance = self.concept_metadata.get(concept_id, {}).get("importance", 0.5)
            
            # 综合评分
            score = (relation_count * 0.7) + (importance * 0.3)
            concept_scores.append((concept_id, score))
        
        # 选择得分最高的概念
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        return concept_scores[0][0] if concept_scores else cluster[0]
    
    def _build_hierarchy_structure(self, root_id: str, cluster: List[str], depth: int) -> Dict[str, Any]:
        """构建层次结构"""
        if depth >= self.config["max_hierarchy_depth"]:
            return {}
            
        # 创建根节点结构
        structure = {
            "id": root_id,
            "name": self.concepts[root_id]["name"] if root_id in self.concepts else f"Concept-{root_id[:8]}",
            "children": []
        }
        
        # 寻找子概念
        child_candidates = []
        for target_id, rel_type, confidence in self.relations.get(root_id, []):
            if target_id in cluster and target_id != root_id:
                if rel_type in {"has_subconcept", "contains", "broader_than"}:
                    child_candidates.append((target_id, confidence))
        
        # 按置信度排序
        child_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 添加子节点，并递归构建其子结构
        processed = {root_id}
        for child_id, _ in child_candidates:
            if child_id not in processed:
                child_structure = self._build_hierarchy_structure(child_id, cluster, depth + 1)
                if child_structure:
                    structure["children"].append(child_structure)
                processed.add(child_id)
        
        # 处理剩余概念
        remaining = [c for c in cluster if c not in processed]
        for concept_id in remaining:
            if len(structure["children"]) < 10:  # 限制子节点数量
                structure["children"].append({
                    "id": concept_id,
                    "name": self.concepts[concept_id]["name"] if concept_id in self.concepts else f"Concept-{concept_id[:8]}"
                })
                processed.add(concept_id)
        
        return structure 
    
    def _optimize_cluster(self, cluster: List[str]) -> Dict[str, int]:
        """优化聚类内部的概念关系"""
        changes = {
            "relations_added": 0,
            "similarity_updated": 0,
            "redundancy_detected": 0
        }
        
        # 如果聚类太小，不需要优化
        if len(cluster) < 3:
            return changes
            
        # 计算聚类内部的相似度矩阵
        similarity_matrix = {}
        for i, id1 in enumerate(cluster):
            for j in range(i+1, len(cluster)):
                id2 = cluster[j]
                sim = self._calculate_similarity(
                    self.concept_vectors[id1],
                    self.concept_vectors[id2]
                )
                similarity_matrix[(id1, id2)] = sim
                similarity_matrix[(id2, id1)] = sim
                
        # 识别聚类中心
        center_id = self._identify_cluster_center(cluster, similarity_matrix)
        
        if not center_id:
            # 如果没有明确的中心，使用第一个概念
            center_id = cluster[0]
            
        # 计算与中心的平均相似度
        avg_center_sim = sum(similarity_matrix.get((center_id, c_id), 0) for c_id in cluster if c_id != center_id) / (len(cluster) - 1)
        
        # 建立与中心概念的关系
        for concept_id in cluster:
            if concept_id != center_id:
                sim = similarity_matrix.get((center_id, concept_id), 0)
                
                # 已经存在的关系
                existing_relations = [
                    (rel_type, conf) for target, rel_type, conf in self.relations.get(center_id, set())
                    if target == concept_id
                ]
                
                if not existing_relations:
                    # 添加新关系
                    relation_type = self._determine_relation_type(center_id, concept_id, sim)
                    self.add_relation(center_id, concept_id, relation_type, sim)
                    changes["relations_added"] += 1
                else:
                    # 更新现有关系的置信度
                    for rel_type, old_conf in existing_relations:
                        # 如果新相似度显著不同，更新置信度
                        if abs(sim - old_conf) > 0.2:
                            # 移除旧关系
                            self.relations[center_id] = {
                                (target, r_type, conf) for target, r_type, conf in self.relations[center_id]
                                if target != concept_id or r_type != rel_type
                            }
                            
                            # 添加新关系
                            self.relations[center_id].add((concept_id, rel_type, sim))
                            changes["similarity_updated"] += 1
        
        # 检测冗余或重复概念
        for i, id1 in enumerate(cluster):
            for j in range(i+1, len(cluster)):
                id2 = cluster[j]
                sim = similarity_matrix.get((id1, id2), 0)
                
                # 如果两个概念极其相似，标记为可能冗余
                if sim > 0.95:
                    # 在元数据中标记潜在冗余
                    self.concept_metadata[id1]["potential_duplicate"] = id2
                    self.concept_metadata[id2]["potential_duplicate"] = id1
                    changes["redundancy_detected"] += 1
        
        return changes
    
    def _determine_relation_type(self, concept_id1: str, concept_id2: str, similarity: float) -> str:
        """根据概念特征和相似度确定关系类型"""
        # 默认关系类型
        default_type = "related_to"
        
        if similarity > 0.9:
            return "similar_to"
            
        # 获取概念类型
        type1 = self.concepts[concept_id1].get("type", "")
        type2 = self.concepts[concept_id2].get("type", "")
        
        # 基于类型确定关系
        type_based_relations = {
            ("category", "entity"): "contains",
            ("entity", "category"): "belongs_to",
            ("whole", "part"): "has_part",
            ("part", "whole"): "part_of",
            ("class", "instance"): "has_instance",
            ("instance", "class"): "instance_of"
        }
        
        relation = type_based_relations.get((type1, type2))
        if relation:
            return relation
            
        # 检查是否有共同属性或标签确定关系
        if self._has_common_attributes(concept_id1, concept_id2):
            return "shares_attributes_with"
            
        return default_type
    
    def _identify_cluster_center(self, cluster: List[str], similarity_matrix: Dict[Tuple[str, str], float]) -> Optional[str]:
        """识别聚类中心概念"""
        if not cluster:
            return None
            
        # 计算每个概念与其他概念的平均相似度
        avg_similarities = {}
        
        for concept_id in cluster:
            sim_sum = sum(similarity_matrix.get((concept_id, other_id), 0) 
                         for other_id in cluster if other_id != concept_id)
            avg_similarities[concept_id] = sim_sum / max(1, len(cluster) - 1)
            
        # 返回平均相似度最高的概念
        if not avg_similarities:
            return None
            
        return max(avg_similarities.items(), key=lambda x: x[1])[0]
    
    def _create_cross_domain_connections(self) -> Dict[str, Any]:
        """创建跨领域知识连接"""
        if not self.config["enable_cross_domain"]:
            return {"connections": 0, "domains": 0}
            
        result = {
            "connections": 0,
            "domains": set(),
            "domain_pairs": []
        }
        
        # 按领域分组概念
        domain_concepts = defaultdict(list)
        
        for concept_id, concept in self.concepts.items():
            domain = concept.get("domain", "general")
            domain_concepts[domain].append(concept_id)
            
        # 记录领域数量
        result["domains"] = len(domain_concepts)
        
        # 如果只有一个领域，不需要跨域连接
        if len(domain_concepts) <= 1:
            return result
            
        # 计算领域之间的相似概念
        domains = list(domain_concepts.keys())
        
        for i, domain1 in enumerate(domains):
            concepts1 = domain_concepts[domain1]
            
            for j in range(i+1, len(domains)):
                domain2 = domains[j]
                concepts2 = domain_concepts[domain2]
                
                # 查找两个领域之间的相似概念
                cross_connections = []
                
                for id1 in concepts1:
                    vector1 = self.concept_vectors[id1]
                    
                    for id2 in concepts2:
                        vector2 = self.concept_vectors[id2]
                        
                        similarity = self._calculate_similarity(vector1, vector2)
                        
                        # 如果相似度超过阈值，建立跨领域连接
                        if similarity > self.config["similarity_threshold"]:
                            cross_connections.append((id1, id2, similarity))
                
                # 按相似度排序，保留前N个连接
                cross_connections.sort(key=lambda x: x[2], reverse=True)
                top_connections = cross_connections[:5]  # 每对领域最多保留5个连接
                
                # 创建关系
                for id1, id2, similarity in top_connections:
                    relation_type = "cross_domain_similar"
                    self.add_relation(id1, id2, relation_type, similarity)
                    result["connections"] += 1
                
                if top_connections:
                    result["domain_pairs"].append((domain1, domain2, len(top_connections)))
        
        return result
    
    def _infer_concept_domain(self, concept: Dict[str, Any]) -> str:
        """推断概念所属领域"""
        # 如果已明确指定领域，直接返回
        if "domain" in concept:
            return concept["domain"]
            
        # 从标签推断
        if "tags" in concept and concept["tags"]:
            # 领域关键词映射
            domain_keywords = {
                "science": ["physics", "chemistry", "biology", "scientific", "experiment", "theory"],
                "technology": ["software", "hardware", "computer", "algorithm", "data", "digital"],
                "math": ["mathematics", "geometry", "algebra", "calculation", "equation", "formula"],
                "arts": ["art", "painting", "music", "literature", "creative", "aesthetic"],
                "philosophy": ["philosophy", "ethics", "logic", "metaphysics", "epistemology"],
                "social": ["society", "culture", "communication", "relationship", "community"]
            }
            
            # 计算标签与各领域的匹配度
            domain_scores = defaultdict(int)
            
            for tag in concept["tags"]:
                tag_lower = tag.lower()
                for domain, keywords in domain_keywords.items():
                    for keyword in keywords:
                        if keyword in tag_lower:
                            domain_scores[domain] += 1
            
            # 返回最匹配的领域
            if domain_scores:
                return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 从名称和描述推断
        name = concept.get("name", "").lower()
        description = concept.get("description", "").lower()
        text = name + " " + description
        
        domain_indicators = {
            "science": ["science", "scientific", "experiment", "laboratory", "research", "theory"],
            "technology": ["technology", "software", "hardware", "digital", "computer", "internet"],
            "math": ["mathematics", "math", "calculation", "number", "equation", "theorem"],
            "arts": ["art", "artistic", "creative", "beauty", "aesthetic", "expression"],
            "philosophy": ["philosophy", "philosophical", "thinking", "concept", "abstract"],
            "social": ["social", "society", "community", "relationship", "communication"]
        }
        
        # 计算文本与各领域的匹配度
        domain_scores = defaultdict(int)
        
        for domain, indicators in domain_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    domain_scores[domain] += 1
        
        # 返回最匹配的领域或默认领域
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
            
        return "general"  # 默认领域
    
    def _detect_merge_redundancy(self) -> Dict[str, Any]:
        """检测和合并冗余概念"""
        result = {
            "detected": 0,
            "merged": 0,
            "merged_pairs": []
        }
        
        # 按相似度识别冗余
        redundant_pairs = []
        
        # 首先从元数据中查找已标记的潜在冗余
        for concept_id, metadata in self.concept_metadata.items():
            if "potential_duplicate" in metadata:
                duplicate_id = metadata["potential_duplicate"]
                
                # 确保两个概念都存在
                if concept_id in self.concepts and duplicate_id in self.concepts:
                    # 再次确认相似度
                    if (concept_id in self.concept_vectors and duplicate_id in self.concept_vectors):
                        similarity = self._calculate_similarity(
                            self.concept_vectors[concept_id],
                            self.concept_vectors[duplicate_id]
                        )
                        
                        if similarity > 0.95:  # 高度相似
                            redundant_pairs.append((concept_id, duplicate_id, similarity))
        
                    # 基于检测策略执行冗余检测
            concept_ids = list(self.concepts.keys())
            
            if detection_strategy == "basic":
                # 基本策略 - 只检测向量相似度和精确重复
                for i in range(len(concept_ids)):
                    for j in range(i+1, len(concept_ids)):
                        id1 = concept_ids[i]
                        id2 = concept_ids[j]
                        
                        # 检查是否已在冗余对列表中
                        if any((id1, id2, _) in redundant_pairs or (id2, id1, _) in redundant_pairs):
                            continue
                        
                        # 检查是否是精确重复
                        if self._is_exact_duplicate(self.concepts[id1], self.concepts[id2]):
                            redundant_pairs.append((id1, id2, 1.0))
                            continue
                        
                        # 计算向量相似度
                        if (id1 in self.concept_vectors and id2 in self.concept_vectors):
                            similarity = self._calculate_similarity(
                                self.concept_vectors[id1],
                                self.concept_vectors[id2]
                            )
                            
                            if similarity > similarity_threshold:
                                redundant_pairs.append((id1, id2, similarity))
            
            elif detection_strategy == "strict":
                # 严格策略 - 多重条件验证，避免错误合并
                for i in range(len(concept_ids)):
                    for j in range(i+1, len(concept_ids)):
                        id1 = concept_ids[i]
                        id2 = concept_ids[j]
                        
                        # 检查是否已在冗余对列表中
                        if any((id1, id2, _) in redundant_pairs or (id2, id1, _) in redundant_pairs):
                            continue
                        
                        # 精确重复检查
                        if self._is_exact_duplicate(self.concepts[id1], self.concepts[id2]):
                            redundant_pairs.append((id1, id2, 1.0))
                            continue
                        
                        # 计算三种不同的相似度指标
                        # 1. 向量相似度
                        vector_similarity = 0.0
                        if (id1 in self.concept_vectors and id2 in self.concept_vectors):
                            vector_similarity = self._calculate_similarity(
                                self.concept_vectors[id1],
                                self.concept_vectors[id2]
                            )
                        
                        # 2. 属性相似度
                        attribute_similarity = self._calculate_attribute_similarity(id1, id2)
                        
                        # 3. 关系结构相似度
                        structure_similarity = self._calculate_relation_structure_similarity(id1, id2)
                        
                        # 综合评分 - 加权平均
                        composite_similarity = (
                            vector_similarity * 0.4 + 
                            attribute_similarity * 0.4 + 
                            structure_similarity * 0.2
                        )
                        
                        # 只有在综合评分高且至少两项指标都高的情况下才认为是冗余
                        if composite_similarity > similarity_threshold:
                            high_scores = 0
                            if vector_similarity > similarity_threshold: high_scores += 1
                            if attribute_similarity > similarity_threshold: high_scores += 1
                            if structure_similarity > similarity_threshold: high_scores += 1
                            
                            if high_scores >= 2:
                                redundant_pairs.append((id1, id2, composite_similarity))
                                # 记录详细的相似度指标
                                result.setdefault("similarity_details", {})[f"{id1}_{id2}"] = {
                                    "vector": vector_similarity,
                                    "attribute": attribute_similarity,
                                    "structure": structure_similarity,
                                    "composite": composite_similarity
                                }
            
            else:  # 综合策略(默认)
                # 综合策略 - 结合多种检测方法
                for i in range(len(concept_ids)):
                    for j in range(i+1, len(concept_ids)):
                        id1 = concept_ids[i]
                        id2 = concept_ids[j]
                        
                        # 检查是否已在冗余对列表中
                        if any((id1, id2, _) in redundant_pairs or (id2, id1, _) in redundant_pairs):
                            continue
                        
                        # 检查是否是精确重复
                        if self._is_exact_duplicate(self.concepts[id1], self.concepts[id2]):
                            redundant_pairs.append((id1, id2, 1.0))
                            continue
                        
                        # 检查是否是超集关系
                        if self._is_superset(self.concepts[id1], self.concepts[id2]):
                            redundant_pairs.append((id2, id1, 0.98))  # id1是id2的超集
                            continue
                        elif self._is_superset(self.concepts[id2], self.concepts[id1]):
                            redundant_pairs.append((id1, id2, 0.98))  # id2是id1的超集
                            continue
                        
                        # 计算相似度
                        if (id1 in self.concept_vectors and id2 in self.concept_vectors):
                            similarity = self._calculate_similarity(
                                self.concept_vectors[id1],
                                self.concept_vectors[id2]
                            )
                            
                            # 高度相似的概念
                            if similarity > 0.95:
                                redundant_pairs.append((id1, id2, similarity))
                            # 中等相似度的概念，进行额外检查
                            elif similarity > similarity_threshold:
                                # 检查属性相似度
                                attribute_similarity = self._calculate_attribute_similarity(id1, id2)
                                if attribute_similarity > 0.8:  # 属性也高度相似
                                    composite_similarity = (similarity + attribute_similarity) / 2
                                    redundant_pairs.append((id1, id2, composite_similarity))
            
            # 更新检测结果
            result["detected"] = len(redundant_pairs)
            result["detection_strategy"] = detection_strategy
        
        # 合并冗余概念
        for source_id, target_id, similarity in redundant_pairs:
            # 确保两个概念都还存在（可能在之前的合并中已被删除）
            if source_id in self.concepts and target_id in self.concepts:
                # 合并概念
                self._merge_concepts(source_id, target_id)
                result["merged"] += 1
                result["merged_pairs"].append((source_id, target_id))
        
        return result
    
    def _is_exact_duplicate(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> bool:
        """检查两个概念是否完全相同"""
        # 检查名称
        if concept1.get("name", "") != concept2.get("name", ""):
            return False
            
        # 检查描述
        if concept1.get("description", "") != concept2.get("description", ""):
            return False
            
        # 检查类型
        if concept1.get("type", "") != concept2.get("type", ""):
            return False
            
        # 检查标签集合
        tags1 = set(concept1.get("tags", []))
        tags2 = set(concept2.get("tags", []))
        
        if tags1 != tags2:
            return False
            
        # 检查属性
        attrs1 = concept1.get("attributes", {})
        attrs2 = concept2.get("attributes", {})
        
        if set(attrs1.keys()) != set(attrs2.keys()):
            return False
            
        for key in attrs1:
            if attrs1[key] != attrs2[key]:
                return False
                
        return True
    
    def _is_superset(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> bool:
        """检查概念1是否是概念2的超集"""
        # 检查名称和描述是否包含
        name1 = concept1.get("name", "").lower()
        name2 = concept2.get("name", "").lower()
        desc1 = concept1.get("description", "").lower()
        desc2 = concept2.get("description", "").lower()
        
        # 如果概念2的名称或描述完全不在概念1中，则不是超集
        if name2 not in name1 and name2 not in desc1:
            return False
            
        # 检查标签
        tags1 = set(concept1.get("tags", []))
        tags2 = set(concept2.get("tags", []))
        
        if not tags2.issubset(tags1):
            return False
            
        # 检查属性
        attrs1 = concept1.get("attributes", {})
        attrs2 = concept2.get("attributes", {})
        
        for key, value in attrs2.items():
            if key not in attrs1:
                return False
                
        return True
    
    def _merge_concepts(self, source_id: str, target_id: str):
        """将源概念合并到目标概念"""
        if source_id not in self.concepts or target_id not in self.concepts:
            return
            
        source = self.concepts[source_id]
        target = self.concepts[target_id]
        
        # 合并标签
        if "tags" in source:
            if "tags" not in target:
                target["tags"] = []
            target["tags"] = list(set(target["tags"] + source["tags"]))
            
        # 合并属性
        if "attributes" in source:
            if "attributes" not in target:
                target["attributes"] = {}
            for key, value in source["attributes"].items():
                if key not in target["attributes"]:
                    target["attributes"][key] = value
        
        # 合并关系
        if source_id in self.relations:
            if target_id not in self.relations:
                self.relations[target_id] = set()
                
            # 转移所有关系
            for rel_target, rel_type, confidence in self.relations[source_id]:
                # 避免自我引用
                if rel_target != target_id:
                    self.relations[target_id].add((rel_target, rel_type, confidence))
                    
                    # 更新反向关系
                    if rel_target in self.relations:
                        # 移除指向源概念的关系
                        self.relations[rel_target] = {
                            (t, rt, c) for t, rt, c in self.relations[rel_target]
                            if t != source_id
                        }
                        # 添加指向目标概念的关系
                        inverse_type = self._get_inverse_relation_type(rel_type)
                        if inverse_type:
                            self.relations[rel_target].add((target_id, inverse_type, confidence))
            
            # 删除源概念的关系
            del self.relations[source_id]
        
        # 更新元数据
        if source_id in self.concept_metadata and target_id in self.concept_metadata:
            # 合并使用计数
            self.concept_metadata[target_id]["usage_count"] += self.concept_metadata[source_id]["usage_count"]
            
            # 合并相关概念
            if "related_concepts" in self.concept_metadata[source_id]:
                if "related_concepts" not in self.concept_metadata[target_id]:
                    self.concept_metadata[target_id]["related_concepts"] = set()
                
                self.concept_metadata[target_id]["related_concepts"].update(
                    self.concept_metadata[source_id]["related_concepts"]
                )
                
                # 移除自引用
                if target_id in self.concept_metadata[target_id]["related_concepts"]:
                    self.concept_metadata[target_id]["related_concepts"].remove(target_id)
            
            # 删除源概念元数据
            del self.concept_metadata[source_id]
        
        # 删除源概念
        del self.concepts[source_id]
        if source_id in self.concept_vectors:
            del self.concept_vectors[source_id]
            
        # 更新索引
        for index_type, index in self.concept_index.items():
            for key, concepts in list(index.items()):
                if source_id in concepts:
                    index[key].remove(source_id)
                    
        # 更新统计信息
        self.stats["concept_count"] = len(self.concepts)
        
        self.logger.info(f"已合并概念: {source_id} -> {target_id}")
    
    def _auto_classify_concepts(self) -> Dict[str, Any]:
        """自动分类概念"""
        result = {
            "classified": 0,
            "categories": defaultdict(int)
        }
        
        # 聚类并提取主题
        concept_clusters = self._cluster_concepts()
        
        # 为每个聚类生成类别和标签
        for cluster in concept_clusters:
            if len(cluster) < 3:  # 忽略太小的聚类
                continue
                
            # 收集聚类中所有概念的文本
            cluster_text = ""
            for concept_id in cluster:
                concept = self.concepts[concept_id]
                cluster_text += concept.get("name", "") + " "
                cluster_text += concept.get("description", "") + " "
                
                # 添加已有标签
                for tag in concept.get("tags", []):
                    cluster_text += tag + " "
            
            # 提取关键词
            keywords = self._extract_keywords(cluster_text)
            
            if not keywords:
                continue
                
            # 用前3个关键词作为类别
            category = "_".join(keywords[:min(3, len(keywords))])
            
            # 为聚类中的每个概念添加标签
            for concept_id in cluster:
                concept = self.concepts[concept_id]
                
                # 确保有标签字段
                if "tags" not in concept:
                    concept["tags"] = []
                    
                # 添加新标签，避免重复
                new_tags = [kw for kw in keywords if kw not in concept["tags"]]
                if new_tags:
                    concept["tags"].extend(new_tags[:5])  # 最多添加5个新标签
                    result["classified"] += 1
                    result["categories"][category] += 1
                    
                    # 更新索引
                    for tag in new_tags:
                        self.concept_index["by_tag"][tag].append(concept_id)
        
        # 从关系中推断标签
        for concept_id in self.concepts:
            related_tags = self._infer_tags_from_relations(concept_id)
            
            if related_tags:
                concept = self.concepts[concept_id]
                
                # 确保有标签字段
                if "tags" not in concept:
                    concept["tags"] = []
                    
                # 添加新标签，避免重复
                new_tags = [tag for tag in related_tags if tag not in concept["tags"]]
                if new_tags:
                    concept["tags"].extend(new_tags)
                    result["classified"] += 1
                    
                    # 更新索引
                    for tag in new_tags:
                        self.concept_index["by_tag"][tag].append(concept_id)
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简化实现：使用词频统计提取关键词
        # 在实际系统中，可以使用TF-IDF或更复杂的关键词提取算法
        
        # 清理文本
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        words = text.split()
        
        # 移除常见停用词
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "in", "of", "to", "for", "with", "on", "at"}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 计算词频
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
            
        # 按频率排序
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前N个关键词
        return [word for word, count in sorted_words[:10]]
    
    def _infer_tags_from_relations(self, concept_id: str) -> List[str]:
        """根据关系推断标签"""
        if concept_id not in self.relations:
            return []
            
        related_tags = []
        
        # 收集相关概念的标签
        for target_id, relation_type, confidence in self.relations[concept_id]:
            if target_id in self.concepts and "tags" in self.concepts[target_id]:
                # 根据关系类型和置信度选择标签
                if relation_type in ["is_a", "instance_of", "similar_to"] and confidence > 0.8:
                    related_tags.extend(self.concepts[target_id]["tags"])
                elif relation_type in ["related_to", "shares_attributes_with"] and confidence > 0.9:
                    related_tags.extend(self.concepts[target_id]["tags"])
        
        # 计算标签频率
        tag_count = defaultdict(int)
        for tag in related_tags:
            tag_count[tag] += 1
            
        # 选择频率最高的标签
        sorted_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        
        return [tag for tag, count in sorted_tags if count > 1]  # 只选择出现多次的标签
    
    def _update_concept_index(self, concept_id: str, concept_data: Dict[str, Any]):
        """更新概念索引"""
        # 索引名称
        name = concept_data.get("name", "").lower()
        if name:
            self.concept_index["by_name"][name].append(concept_id)
            
        # 索引标签
        for tag in concept_data.get("tags", []):
            self.concept_index["by_tag"][tag].append(concept_id)
            
        # 索引领域
        domain = concept_data.get("domain", "general")
        self.concept_index["by_domain"][domain].append(concept_id)
        
        # 索引类型
        concept_type = concept_data.get("type", "generic")
        self.concept_index["by_type"][concept_type].append(concept_id)
    
    def _validate_concept_data(self, concept_data: Dict[str, Any]) -> bool:
        """验证概念数据"""
        # 必需字段
        if "name" not in concept_data or not concept_data["name"]:
            return False
            
        # 检查内容大小
        try:
            size = len(json.dumps(concept_data))
            if size > self.config["concept_size_limit"]:
                return False
        except:
            return False
            
        return True
    
    def _calculate_hierarchy_depth(self, structure: Dict[str, Any]) -> int:
        """计算层次结构深度"""
        if "children" not in structure or not structure["children"]:
            return 1
            
        child_depths = [
            self._calculate_hierarchy_depth(child) 
            for child in structure["children"]
        ]
        
        return 1 + max(child_depths)
    
    def _count_hierarchy_concepts(self, structure: Dict[str, Any]) -> int:
        """计算层次结构中的概念数量"""
        count = 1  # 当前节点
        
        if "children" in structure:
            for child in structure["children"]:
                count += self._count_hierarchy_concepts(child)
                
        return count
    
    def _is_in_hierarchy(self, concept_id: str, structure: Dict[str, Any]) -> bool:
        """检查概念是否在层次结构中"""
        if structure.get("id") == concept_id:
            return True
            
        if "children" in structure:
            for child in structure["children"]:
                if self._is_in_hierarchy(concept_id, child):
                    return True
                    
        return False
    
    def _get_inverse_relation_type(self, relation_type: str) -> Optional[str]:
        """获取关系的反向类型"""
        # 关系反向映射
        inverse_relations = {
            "contains": "contained_in",
            "contained_in": "contains",
            "has_part": "part_of",
            "part_of": "has_part",
            "parent_of": "child_of",
            "child_of": "parent_of",
            "similar_to": "similar_to",  # 对称关系
            "related_to": "related_to",  # 对称关系
            "is_a": "has_type",
            "has_type": "is_a",
            "has_instance": "instance_of",
            "instance_of": "has_instance"
        }
        
        return inverse_relations.get(relation_type)
    
    def _convert_to_cytoscape(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """转换为Cytoscape格式"""
        nodes = []
        edges = []
        
        # 转换节点
        for concept_id, concept in knowledge_graph["concepts"].items():
            nodes.append({
                "data": {
                    "id": concept_id,
                    "name": concept.get("name", ""),
                    "type": concept.get("type", ""),
                    "domain": concept.get("domain", ""),
                    "tags": concept.get("tags", [])
                }
            })
            
        # 转换边
        for relation in knowledge_graph["relations"]:
            edge_id = f"{relation['source']}_{relation['type']}_{relation['target']}"
            edges.append({
                "data": {
                    "id": edge_id,
                    "source": relation["source"],
                    "target": relation["target"],
                    "type": relation["type"],
                    "weight": relation["confidence"]
                }
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        } 
    
    def update_concept_relations(self, concept_id: str, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        动态更新概念关联，根据最新知识状态重新评估概念关系
        
        Args:
            concept_id: 概念ID
            force_rebuild: 是否强制重建所有关系
            
        Returns:
            Dict: 更新结果
        """
        self.logger.info(f"更新概念关联: {concept_id}")
        
        result = {
            "status": "error",
            "updated_relations": 0,
            "added_relations": 0,
            "removed_relations": 0,
            "changed_relation_types": 0
        }
        
        # 检查概念是否存在
        if concept_id not in self.concepts:
            result["message"] = f"概念不存在: {concept_id}"
            return result
            
        concept = self.concepts[concept_id]
        
        # 获取当前关系
        current_relations = set()
        for relation_data in self.relations.get(concept_id, set()):
            target_id, relation_type, _ = relation_data
            current_relations.add((target_id, relation_type))
        
        # 如果强制重建，清除所有现有关系
        if force_rebuild:
            for relation_data in list(self.relations.get(concept_id, set())):
                target_id, relation_type, confidence = relation_data
                # 也需要移除反向关系
                self._remove_relation(concept_id, target_id, relation_type)
                result["removed_relations"] += 1
            
            current_relations = set()
        
        # 计算与所有其他概念的最新关系
        new_relations = set()
        all_concepts = set(self.concepts.keys())
        
        # 移除自身
        all_concepts.discard(concept_id)
        
        # 计算新的关系集
        for other_id in all_concepts:
            # 计算两个概念的相似度
            similarity = self._calculate_similarity(
                self.concept_vectors.get(concept_id, np.zeros(self.config["vector_dim"])),
                self.concept_vectors.get(other_id, np.zeros(self.config["vector_dim"]))
            )
            
            # 检查是否超过相似度阈值
            if similarity >= self.config["similarity_threshold"]:
                # 确定关系类型
                relation_type = self._determine_relation_type(concept_id, other_id, similarity)
                new_relations.add((other_id, relation_type))
                
                # 查找现有关系中的配对
                existing_relation = None
                for rel in current_relations:
                    if rel[0] == other_id:
                        existing_relation = rel
                        break
                
                if existing_relation is None:
                    # 添加新关系
                    self.add_relation(concept_id, other_id, relation_type, similarity)
                    result["added_relations"] += 1
                elif existing_relation[1] != relation_type:
                    # 关系类型变化，更新关系
                    self._remove_relation(concept_id, other_id, existing_relation[1])
                    self.add_relation(concept_id, other_id, relation_type, similarity)
                    result["changed_relation_types"] += 1
        
        # 移除不再满足条件的旧关系
        for old_relation in current_relations:
            target_id, old_type = old_relation
            if old_relation not in new_relations:
                self._remove_relation(concept_id, target_id, old_type)
                result["removed_relations"] += 1
        
        # 更新元数据
        self.concept_metadata[concept_id]["last_relation_update"] = time.time()
        
        # 更新统计
        result["status"] = "success"
        result["updated_relations"] = result["added_relations"] + result["removed_relations"] + result["changed_relation_types"]
        
        self.logger.info(f"概念关联更新完成: {concept_id}, 添加:{result['added_relations']}, 移除:{result['removed_relations']}, 更改:{result['changed_relation_types']}")
        
        return result
        
    def _remove_relation(self, source_id: str, target_id: str, relation_type: str):
        """移除一个关系及其反向关系"""
        # 移除正向关系
        relations_to_remove = set()
        for rel in self.relations.get(source_id, set()):
            rel_target, rel_type, _ = rel
            if rel_target == target_id and rel_type == relation_type:
                relations_to_remove.add(rel)
                
        if source_id in self.relations:
            self.relations[source_id] -= relations_to_remove
            
        # 移除反向关系
        inverse_type = self._get_inverse_relation_type(relation_type)
        if inverse_type:
            inverse_to_remove = set()
            for rel in self.relations.get(target_id, set()):
                rel_target, rel_type, _ = rel
                if rel_target == source_id and rel_type == inverse_type:
                    inverse_to_remove.add(rel)
                    
            if target_id in self.relations:
                self.relations[target_id] -= inverse_to_remove
        
        # 更新元数据
        if source_id in self.concept_metadata and "related_concepts" in self.concept_metadata[source_id]:
            self.concept_metadata[source_id]["related_concepts"].discard(target_id)
            
        if target_id in self.concept_metadata and "related_concepts" in self.concept_metadata[target_id]:
            self.concept_metadata[target_id]["related_concepts"].discard(source_id)
            
    def batch_update_relations(self, max_concepts: int = 50) -> Dict[str, Any]:
        """
        批量更新概念关联，优先处理长时间未更新的概念
        
        Args:
            max_concepts: 最大处理概念数量
            
        Returns:
            Dict: 批处理结果
        """
        self.logger.info(f"开始批量更新概念关联，最大处理数: {max_concepts}")
        
        result = {
            "status": "success",
            "processed_concepts": 0,
            "total_updates": 0,
            "total_added": 0,
            "total_removed": 0,
            "total_changed": 0
        }
        
        # 获取所有概念ID
        all_concept_ids = list(self.concepts.keys())
        
        # 按最后更新时间排序
        concept_update_times = []
        for concept_id in all_concept_ids:
            last_update = self.concept_metadata.get(concept_id, {}).get("last_relation_update", 0)
            concept_update_times.append((concept_id, last_update))
            
        # 排序，优先处理最旧的
        concept_update_times.sort(key=lambda x: x[1])
        
        # 限制处理数量
        concepts_to_process = [item[0] for item in concept_update_times[:max_concepts]]
        
        # 批量更新
        for concept_id in concepts_to_process:
            update_result = self.update_concept_relations(concept_id)
            
            if update_result["status"] == "success":
                result["processed_concepts"] += 1
                result["total_updates"] += update_result["updated_relations"]
                result["total_added"] += update_result["added_relations"]
                result["total_removed"] += update_result["removed_relations"] 
                result["total_changed"] += update_result["changed_relation_types"]
        
        self.logger.info(f"批量关联更新完成: 处理了 {result['processed_concepts']} 个概念, 总更新 {result['total_updates']}")
        
        return result
    
    def transfer_knowledge_across_domains(self, source_domain: str, target_domain: str, 
                                    similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        在不同领域之间迁移知识，找出可复用的概念和关系
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            similarity_threshold: 概念相似度阈值
            
        Returns:
            Dict: 迁移结果
        """
        self.logger.info(f"开始跨领域知识迁移: {source_domain} -> {target_domain}")
        
        result = {
            "status": "error",
            "transferred_concepts": 0,
            "transferred_relations": 0,
            "mapping": {},
            "new_concepts": []
        }
        
        try:
            # 获取源领域和目标领域的概念
            source_concepts = self._get_domain_concepts(source_domain)
            target_concepts = self._get_domain_concepts(target_domain)
            
            if not source_concepts:
                result["message"] = f"源领域 {source_domain} 没有概念"
                return result
                
            self.logger.info(f"源领域 {source_domain} 有 {len(source_concepts)} 个概念")
            self.logger.info(f"目标领域 {target_domain} 有 {len(target_concepts)} 个概念")
            
            # 概念映射: {source_id: target_id}
            concept_mapping = {}
            
            # 首先匹配现有概念
            for source_id, source_concept in source_concepts.items():
                best_match = None
                best_similarity = 0
                
                # 查找最佳匹配
                for target_id, target_concept in target_concepts.items():
                    if "vector" in self.concept_vectors and source_id in self.concept_vectors and target_id in self.concept_vectors:
                        # 使用向量计算相似度
                        similarity = self._calculate_similarity(
                            self.concept_vectors[source_id],
                            self.concept_vectors[target_id]
                        )
                    else:
                        # 使用属性计算相似度
                        similarity = self._calculate_concept_attribute_similarity(source_concept, target_concept)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = target_id
                
                # 如果找到足够相似的概念，建立映射
                if best_match is not None and best_similarity >= similarity_threshold:
                    concept_mapping[source_id] = best_match
            
            # 创建未匹配概念的副本到目标领域
            new_concepts = []
            for source_id, source_concept in source_concepts.items():
                if source_id not in concept_mapping:
                    # 创建新概念的副本
                    new_concept = self._create_domain_adapted_concept(source_concept, target_domain)
                    
                    # 添加到目标域
                    add_result = self.add_concept(new_concept)
                    
                    if add_result["status"] == "success":
                        new_id = add_result["concept_id"]
                        concept_mapping[source_id] = new_id
                        new_concepts.append(new_id)
            
            # 迁移关系
            transferred_relations = 0
            for source_id, source_concept in source_concepts.items():
                if source_id in concept_mapping:
                    # 获取源概念的关系
                    source_relations = self.relations.get(source_id, set())
                    
                    # 对每个关系，如果源和目标都在映射中，创建对应的关系
                    for relation in source_relations:
                        rel_target_id, rel_type, confidence = relation
                        
                        if rel_target_id in concept_mapping:
                            # 添加目标域中对应概念间的关系
                            target_source_id = concept_mapping[source_id]
                            target_target_id = concept_mapping[rel_target_id]
                            
                            # 检查关系是否已存在
                            relation_exists = False
                            for existing_rel in self.relations.get(target_source_id, set()):
                                if existing_rel[0] == target_target_id and existing_rel[1] == rel_type:
                                    relation_exists = True
                                    break
                            
                            # 如果关系不存在，创建它
                            if not relation_exists:
                                add_rel_result = self.add_relation(
                                    target_source_id, 
                                    target_target_id,
                                    rel_type,
                                    confidence * 0.9  # 略微降低置信度
                                )
                                
                                if add_rel_result["status"] == "success":
                                    transferred_relations += 1
            
            # 返回迁移结果
            result["status"] = "success"
            result["transferred_concepts"] = len(concept_mapping)
            result["transferred_relations"] = transferred_relations
            result["mapping"] = concept_mapping
            result["new_concepts"] = new_concepts
            
            self.logger.info(f"跨领域知识迁移完成: 迁移了 {len(concept_mapping)} 个概念和 {transferred_relations} 个关系")
            
        except Exception as e:
            self.logger.error(f"跨领域知识迁移异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            result["message"] = str(e)
        
        return result
    
    def _get_domain_concepts(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """获取指定领域的所有概念"""
        domain_concepts = {}
        
        for concept_id, concept in self.concepts.items():
            concept_domain = concept.get("domain", "general")
            if concept_domain == domain:
                domain_concepts[concept_id] = concept
        
        return domain_concepts
    
    def _calculate_concept_attribute_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """基于属性计算两个概念的相似度"""
        # 比较名称
        name_similarity = 0.0
        if "name" in concept1 and "name" in concept2:
            name1 = concept1["name"].lower()
            name2 = concept2["name"].lower()
            
            # 简单的名称相似度计算
            if name1 == name2:
                name_similarity = 1.0
            elif name1 in name2 or name2 in name1:
                name_similarity = 0.8
            else:
                # 词汇重叠比例
                words1 = set(name1.split())
                words2 = set(name2.split())
                if words1 and words2:
                    common_words = words1.intersection(words2)
                    name_similarity = len(common_words) / max(len(words1), len(words2))
        
        # 比较类型
        type_similarity = 0.0
        if "type" in concept1 and "type" in concept2:
            if concept1["type"] == concept2["type"]:
                type_similarity = 1.0
        
        # 比较标签
        tag_similarity = 0.0
        if "tags" in concept1 and "tags" in concept2:
            tags1 = set(concept1["tags"])
            tags2 = set(concept2["tags"])
            if tags1 and tags2:
                common_tags = tags1.intersection(tags2)
                tag_similarity = len(common_tags) / max(len(tags1), len(tags2))
        
        # 比较属性
        attribute_similarity = 0.0
        if "attributes" in concept1 and "attributes" in concept2:
            attrs1 = concept1["attributes"]
            attrs2 = concept2["attributes"]
            
            common_keys = set(attrs1.keys()).intersection(set(attrs2.keys()))
            all_keys = set(attrs1.keys()).union(set(attrs2.keys()))
            
            if all_keys:
                # 计算属性键相似度
                key_similarity = len(common_keys) / len(all_keys)
                
                # 计算属性值相似度
                value_similarity = 0.0
                if common_keys:
                    matching_values = sum(1 for k in common_keys if str(attrs1[k]) == str(attrs2[k]))
                    value_similarity = matching_values / len(common_keys)
                
                attribute_similarity = (key_similarity + value_similarity) / 2
        
        # 加权组合
        weights = {
            "name": 0.4,
            "type": 0.3,
            "tag": 0.2,
            "attribute": 0.1
        }
        
        similarity = (
            weights["name"] * name_similarity +
            weights["type"] * type_similarity +
            weights["tag"] * tag_similarity +
            weights["attribute"] * attribute_similarity
        )
        
        return similarity
    
    def _create_domain_adapted_concept(self, source_concept: Dict[str, Any], target_domain: str) -> Dict[str, Any]:
        """创建一个源概念的域适应副本"""
        # 复制概念
        new_concept = source_concept.copy()
        
        # 移除ID (将自动生成新ID)
        if "id" in new_concept:
            del new_concept["id"]
        
        # 更新领域
        new_concept["domain"] = target_domain
        
        # 调整名称以表明这是跨域迁移的概念
        if "name" in new_concept:
            original_name = new_concept["name"]
            new_concept["name"] = f"{original_name} ({target_domain})"
        
        # 添加来源元数据
        if "metadata" not in new_concept:
            new_concept["metadata"] = {}
        
        new_concept["metadata"]["transferred_from_domain"] = source_concept.get("domain", "unknown")
        new_concept["metadata"]["original_concept_id"] = source_concept.get("id", "unknown")
        new_concept["metadata"]["transfer_timestamp"] = time.time()
        
        # 调整置信度 (略微降低)
        if "confidence" in new_concept:
            new_concept["confidence"] = max(0.1, new_concept["confidence"] * 0.9)
        else:
            new_concept["confidence"] = 0.7  # 默认值
        
        # 添加迁移标签
        if "tags" not in new_concept:
            new_concept["tags"] = []
        
        new_concept["tags"].append("domain_transferred")
        
        return new_concept
    
    def find_cross_domain_analogies(self, concept_id: str, target_domains: List[str] = None,
                                  top_k: int = 5) -> Dict[str, Any]:
        """
        查找跨领域的类比概念，识别不同领域中的功能等效概念
        
        Args:
            concept_id: 源概念ID
            target_domains: 目标领域列表，如果不指定则搜索所有非源概念领域
            top_k: 每个领域返回的最佳匹配数量
            
        Returns:
            Dict: 类比结果
        """
        self.logger.info(f"查找跨领域类比: {concept_id}")
        
        result = {
            "status": "error",
            "analogies": {},
            "source_concept": None
        }
        
        # 检查概念是否存在
        if concept_id not in self.concepts:
            result["message"] = f"概念不存在: {concept_id}"
            return result
            
        source_concept = self.concepts[concept_id]
        result["source_concept"] = {
            "id": concept_id,
            "name": source_concept.get("name", ""),
            "domain": source_concept.get("domain", "general")
        }
        
        source_domain = source_concept.get("domain", "general")
        
        # 确定目标领域
        if target_domains is None:
            # 获取所有非源概念领域
            all_domains = set()
            for concept in self.concepts.values():
                all_domains.add(concept.get("domain", "general"))
            
            target_domains = list(all_domains - {source_domain})
        
        analogies = {}
        
        # 对每个目标领域搜索类比概念
        for domain in target_domains:
            domain_concepts = self._get_domain_concepts(domain)
            
            # 计算与每个目标领域概念的相似度
            similarities = []
            for target_id, target_concept in domain_concepts.items():
                # 使用结构相似度而非内容相似度
                similarity = self._calculate_structural_similarity(concept_id, target_id)
                similarities.append((target_id, similarity))
            
            # 排序并获取top_k结果
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_k]
            
            # 添加到结果
            if top_matches:
                analogies[domain] = []
                for match_id, similarity in top_matches:
                    match_concept = self.concepts[match_id]
                    analogies[domain].append({
                        "id": match_id,
                        "name": match_concept.get("name", ""),
                        "similarity": similarity,
                        "confidence": min(1.0, similarity * 1.2)  # 调整后的置信度
                    })
        
        result["status"] = "success"
        result["analogies"] = analogies
        
        self.logger.info(f"找到跨领域类比: {concept_id} -> {sum(len(matches) for matches in analogies.values())} 个匹配")
        
        return result
    
    def _calculate_structural_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        计算两个概念的结构相似度，基于它们的关系模式而非内容
        """
        # 获取概念的关系结构
        relations1 = self._get_concept_relation_structure(concept_id1)
        relations2 = self._get_concept_relation_structure(concept_id2)
        
        # 总相似度分数
        total_similarity = 0.0
        
        # 比较关系类型分布
        relation_types1 = relations1["outgoing_types"]
        relation_types2 = relations2["outgoing_types"]
        
        # 计算关系类型的Jaccard相似度
        all_types = set(relation_types1.keys()) | set(relation_types2.keys())
        if all_types:
            common_types = set(relation_types1.keys()) & set(relation_types2.keys())
            type_similarity = len(common_types) / len(all_types)
            
            # 对于共同类型，比较数量相似度
            count_similarity = 0.0
            if common_types:
                type_ratios = []
                for rel_type in common_types:
                    count1 = relation_types1[rel_type]
                    count2 = relation_types2[rel_type]
                    
                    # 计算比例相似度 (较小值/较大值)
                    ratio = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 1.0
                    type_ratios.append(ratio)
                
                # 平均比例相似度
                count_similarity = sum(type_ratios) / len(type_ratios)
            
            type_distribution_similarity = 0.7 * type_similarity + 0.3 * count_similarity
            total_similarity += 0.4 * type_distribution_similarity
        
        # 比较关系的拓扑结构
        structure_similarity = self._compare_relation_topology(relations1, relations2)
        total_similarity += 0.6 * structure_similarity
        
        return total_similarity
    
    def _get_concept_relation_structure(self, concept_id: str) -> Dict[str, Any]:
        """
        获取概念的关系结构信息
        """
        # 关系类型计数
        outgoing_types = defaultdict(int)
        incoming_types = defaultdict(int)
        
        # 连接的概念集
        outgoing_concepts = set()
        incoming_concepts = set()
        
        # 处理出向关系
        for relation_data in self.relations.get(concept_id, set()):
            target_id, relation_type, _ = relation_data
            outgoing_types[relation_type] += 1
            outgoing_concepts.add(target_id)
        
        # 处理入向关系 (查找所有指向此概念的关系)
        for source_id, relations in self.relations.items():
            if source_id == concept_id:
                continue
                
            for relation_data in relations:
                target_id, relation_type, _ = relation_data
                if target_id == concept_id:
                    incoming_types[relation_type] += 1
                    incoming_concepts.add(source_id)
        
        # 二级关系 (概念的邻居之间的关系)
        neighbor_relations = 0
        connected_neighbors = 0
        
        # 检查所有出向概念之间是否存在关系
        all_neighbors = outgoing_concepts | incoming_concepts
        checked_pairs = set()
        
        for neighbor1 in all_neighbors:
            for neighbor2 in all_neighbors:
                if neighbor1 == neighbor2 or (neighbor1, neighbor2) in checked_pairs or (neighbor2, neighbor1) in checked_pairs:
                    continue
                    
                checked_pairs.add((neighbor1, neighbor2))
                
                # 检查邻居之间是否有直接关系
                has_relation = False
                for relation_data in self.relations.get(neighbor1, set()):
                    if relation_data[0] == neighbor2:
                        has_relation = True
                        neighbor_relations += 1
                        break
                
                if has_relation:
                    connected_neighbors += 1
        
        # 计算网络密度 (如果有足够的邻居)
        network_density = 0.0
        if len(all_neighbors) > 1:
            # 完全连接图中的理论最大边数
            max_possible_edges = len(all_neighbors) * (len(all_neighbors) - 1) / 2
            if max_possible_edges > 0:
                network_density = connected_neighbors / max_possible_edges
        
        return {
            "outgoing_types": dict(outgoing_types),
            "incoming_types": dict(incoming_types),
            "outgoing_count": len(outgoing_concepts),
            "incoming_count": len(incoming_concepts),
            "total_neighbors": len(all_neighbors),
            "neighbor_relations": neighbor_relations,
            "network_density": network_density
        }
        
    def _compare_relation_topology(self, structure1: Dict[str, Any], structure2: Dict[str, Any]) -> float:
        """
        比较两个关系结构的拓扑相似度
        """
        # 比较多个拓扑特征
        
        # 1. 出入度比例相似度
        degree_ratio_sim = 1.0
        ratio1 = structure1["outgoing_count"] / max(1, structure1["incoming_count"])
        ratio2 = structure2["outgoing_count"] / max(1, structure2["incoming_count"])
        
        if ratio1 > 0 and ratio2 > 0:
            degree_ratio_sim = min(ratio1, ratio2) / max(ratio1, ratio2)
        
        # 2. 网络密度相似度
        density_sim = 1.0 - abs(structure1["network_density"] - structure2["network_density"])
        
        # 3. 总邻居数量相似度
        neighbor_count_sim = 1.0
        if max(structure1["total_neighbors"], structure2["total_neighbors"]) > 0:
            neighbor_count_sim = min(structure1["total_neighbors"], structure2["total_neighbors"]) / max(structure1["total_neighbors"], structure2["total_neighbors"])
        
        # 4. 邻居间关系数量相似度
        neighbor_rel_sim = 1.0
        if max(structure1["neighbor_relations"], structure2["neighbor_relations"]) > 0:
            neighbor_rel_sim = min(structure1["neighbor_relations"], structure2["neighbor_relations"]) / max(structure1["neighbor_relations"], structure2["neighbor_relations"])
        
        # 加权组合得到最终相似度
        weights = {
            "degree_ratio": 0.25,
            "density": 0.3,
            "neighbor_count": 0.2,
            "neighbor_relations": 0.25
        }
        
        topology_sim = (
            weights["degree_ratio"] * degree_ratio_sim +
            weights["density"] * density_sim +
            weights["neighbor_count"] * neighbor_count_sim +
            weights["neighbor_relations"] * neighbor_rel_sim
        )
        
        return topology_sim
    
    def detect_and_merge_redundancies(self, similarity_threshold: float = 0.85, 
                                    auto_merge: bool = True,
                                    detection_strategy: str = "comprehensive",
                                    merge_strategy: str = "intelligent") -> Dict[str, Any]:
        """
        增强版知识冗余检测与合并，支持精确冗余识别和智能合并策略
        
        Args:
            similarity_threshold: 认为概念冗余的相似度阈值
            auto_merge: 是否自动合并冗余概念
            detection_strategy: 检测策略(basic/comprehensive/strict)
            merge_strategy: 合并策略(simple/intelligent/guided)
            
        Returns:
            Dict: 检测与合并结果
        """
        
        # 以下是辅助函数实现
        
    def _intelligent_merge_concepts(self, primary_id: str, secondary_id: str, similarity: float) -> Dict[str, Any]:
        """
        使用智能策略合并概念，根据概念特性权重保留更重要的信息
        
        Args:
            primary_id: 主要概念ID（保留）
            secondary_id: 次要概念ID（将被合并）
            similarity: 两个概念的相似度
            
        Returns:
            Dict: 合并结果
        """
        result = {
            "status": "error",
            "message": "",
            "affected_relations": 0
        }
        
        # 检查概念是否存在
        if primary_id not in self.concepts:
            result["message"] = f"主概念不存在: {primary_id}"
            return result
            
        if secondary_id not in self.concepts:
            result["message"] = f"次要概念不存在: {secondary_id}"
            return result
            
        if primary_id == secondary_id:
            result["message"] = "不能自我合并"
            return result
        
        try:
            # 获取两个概念
            primary = self.concepts[primary_id]
            secondary = self.concepts[secondary_id]
            
            # 计算概念重要性
            primary_importance = self._calculate_concept_importance(primary)
            secondary_importance = self._calculate_concept_importance(secondary)
            
            # 如果次要概念实际更重要，交换角色
            if secondary_importance > primary_importance * 1.5:
                self.logger.info(f"次要概念更重要，交换合并角色: {secondary_id} -> {primary_id}")
                primary_id, secondary_id = secondary_id, primary_id
                primary, secondary = secondary, primary
            
            # 智能合并属性，保留更重要/更新的信息
            self._weighted_merge_attributes(primary, secondary, similarity)
            
            # 合并关系
            affected_relations = self._migrate_concept_relations(primary_id, secondary_id)
            
            # 处理引用次要概念的关系
            for source_id, relations in self.relations.items():
                if source_id == primary_id or source_id == secondary_id:
                    continue
                    
                updated_relations = set()
                for relation in relations:
                    target_id, rel_type, confidence = relation
                    
                    # 如果目标是次要概念，重定向到主要概念
                    if target_id == secondary_id:
                        # 检查是否已存在到主要概念的相同关系
                        has_primary_relation = False
                        for rel in relations:
                            if rel[0] == primary_id and rel[1] == rel_type:
                                has_primary_relation = True
                                break
                        
                        if not has_primary_relation:
                            updated_relations.add((primary_id, rel_type, confidence))
                            affected_relations += 1
                    else:
                        updated_relations.add(relation)
                
                # 更新关系集
                self.relations[source_id] = updated_relations
            
            # 记录合并历史
            if not hasattr(self, "merge_history"):
                self.merge_history = []
                
            self.merge_history.append({
                "timestamp": time.time(),
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "similarity": similarity,
                "strategy": "intelligent",
                "affected_relations": affected_relations
            })
            
            # 添加合并来源标记
            primary.setdefault("merged_from", []).append({
                "id": secondary_id,
                "timestamp": time.time(),
                "similarity": similarity
            })
            
            # 更新概念向量 - 融合两个向量
            if hasattr(self, 'concept_vectors') and primary_id in self.concept_vectors and secondary_id in self.concept_vectors:
                # 根据重要性加权融合向量
                primary_weight = primary_importance / (primary_importance + secondary_importance)
                secondary_weight = 1.0 - primary_weight
                
                combined_vector = (
                    primary_weight * self.concept_vectors[primary_id] + 
                    secondary_weight * self.concept_vectors[secondary_id]
                )
                
                # 归一化
                norm = np.linalg.norm(combined_vector)
                if norm > 0:
                    combined_vector = combined_vector / norm
                    
                self.concept_vectors[primary_id] = combined_vector
            
            # 删除次要概念
            self._remove_from_index(secondary_id)
            del self.concepts[secondary_id]
            if secondary_id in self.relations:
                del self.relations[secondary_id]
            if hasattr(self, 'concept_vectors') and secondary_id in self.concept_vectors:
                del self.concept_vectors[secondary_id]
            
            result["status"] = "success"
            result["affected_relations"] = affected_relations
            result["primary_concept"] = primary_id
            result["secondary_concept"] = secondary_id
            
        except Exception as e:
            self.logger.error(f"智能合并概念异常: {str(e)}")
            result["message"] = str(e)
        
        return result

    def _weighted_merge_attributes(self, primary: Dict[str, Any], secondary: Dict[str, Any], similarity: float):
        """
        基于权重智能合并概念属性，保留更重要的信息
        
        Args:
            primary: 主要概念（将被修改）
            secondary: 次要概念
            similarity: 两个概念的相似度
        """
        # 合并基本信息
        # 名称和描述 - 如果次要概念的更长更详细，考虑使用它
        if "name" in secondary and len(secondary.get("name", "")) > len(primary.get("name", "")) * 1.5:
            primary["name"] = secondary["name"]
            
        if "description" in secondary and len(secondary.get("description", "")) > len(primary.get("description", "")) * 1.2:
            primary["description"] = secondary["description"]
            
        # 合并类型和领域
        if "type" not in primary and "type" in secondary:
            primary["type"] = secondary["type"]
            
        if "domain" not in primary and "domain" in secondary:
            primary["domain"] = secondary["domain"]
        
        # 合并标签
        if "tags" in secondary:
            if "tags" not in primary:
                primary["tags"] = []
            
            # 添加新标签
            primary["tags"].extend([tag for tag in secondary["tags"] if tag not in primary["tags"]])
        
        # 合并属性
        if "attributes" in secondary:
            if "attributes" not in primary:
                primary["attributes"] = {}
                
            for key, value in secondary["attributes"].items():
                # 如果主要概念没有这个属性，直接添加
                if key not in primary["attributes"]:
                    primary["attributes"][key] = value
                # 如果都有，根据规则选择
                else:
                    # 数值属性 - 选择数据更新的版本
                    if isinstance(value, (int, float)) and isinstance(primary["attributes"][key], (int, float)):
                        # 检查哪个更新
                        primary_updated = primary.get("last_updated", 0)
                        secondary_updated = secondary.get("last_updated", 0)
                        
                        if secondary_updated > primary_updated:
                            primary["attributes"][key] = value
                    # 字符串属性 - 选择更详细的版本
                    elif isinstance(value, str) and isinstance(primary["attributes"][key], str):
                        if len(value) > len(primary["attributes"][key]) * 1.3:
                            primary["attributes"][key] = value
                    # 列表属性 - 合并唯一值
                    elif isinstance(value, list) and isinstance(primary["attributes"][key], list):
                        primary["attributes"][key] = list(set(primary["attributes"][key] + value))
                    # 字典属性 - 递归合并
                    elif isinstance(value, dict) and isinstance(primary["attributes"][key], dict):
                        self._merge_dict_values(primary["attributes"][key], value)
        
        # 元数据合并
        if "metadata" in secondary:
            if "metadata" not in primary:
                primary["metadata"] = {}
                
            # 合并元数据字段
            for key, value in secondary["metadata"].items():
                if key not in primary["metadata"]:
                    primary["metadata"][key] = value
                # 特殊处理用法统计
                elif key == "usage_count":
                    primary["metadata"][key] = primary["metadata"][key] + value
                elif key == "last_accessed" and value > primary["metadata"].get(key, 0):
                    primary["metadata"][key] = value
        
        # 更新合并时间戳
        primary["last_updated"] = time.time()
        primary["merged_at"] = time.time()

    def _merge_dict_values(self, target: Dict[str, Any], source: Dict[str, Any]):
        """递归合并字典值"""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                self._merge_dict_values(target[key], value)
            elif isinstance(value, list) and isinstance(target[key], list):
                target[key] = list(set(target[key] + value))

    def _calculate_attribute_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        计算两个概念的属性相似度
        
        Args:
            concept_id1: 第一个概念ID
            concept_id2: 第二个概念ID
            
        Returns:
            float: 属性相似度分数(0-1)
        """
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            return 0.0
            
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        
        # 基本相似度
        similarity = 0.0
        
        # 1. 名称相似度
        name1 = concept1.get("name", "").lower()
        name2 = concept2.get("name", "").lower()
        
        if name1 and name2:
            # 计算名称相似度 (简单方法)
            if name1 == name2:
                similarity += 0.3  # 完全相同的名称
            elif name1 in name2 or name2 in name1:
                similarity += 0.2  # 包含关系
            else:
                # 计算编辑距离
                max_len = max(len(name1), len(name2))
                if max_len > 0:
                    from difflib import SequenceMatcher
                    ratio = SequenceMatcher(None, name1, name2).ratio()
                    if ratio > 0.7:
                        similarity += 0.15 * ratio
        
        # 2. 类型相似度
        if concept1.get("type") == concept2.get("type"):
            similarity += 0.1
        
        # 3. 标签相似度
        tags1 = set(concept1.get("tags", []))
        tags2 = set(concept2.get("tags", []))
        
        if tags1 and tags2:
            # 计算Jaccard相似度
            intersection = len(tags1.intersection(tags2))
            union = len(tags1.union(tags2))
            
            if union > 0:
                tag_similarity = intersection / union
                similarity += 0.1 * tag_similarity
        
        # 4. 属性相似度
        attrs1 = concept1.get("attributes", {})
        attrs2 = concept2.get("attributes", {})
        
        if attrs1 and attrs2:
            # 计算属性键的重叠度
            keys1 = set(attrs1.keys())
            keys2 = set(attrs2.keys())
            
            key_intersection = len(keys1.intersection(keys2))
            key_union = len(keys1.union(keys2))
            
            if key_union > 0:
                key_similarity = key_intersection / key_union
                similarity += 0.2 * key_similarity
                
                # 对于共有的属性，比较值
                common_keys = keys1.intersection(keys2)
                if common_keys:
                    value_matches = 0
                    for key in common_keys:
                        val1 = attrs1[key]
                        val2 = attrs2[key]
                        
                        # 比较值
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            # 数值比较 - 如果在20%范围内，认为相似
                            if val1 == 0 and val2 == 0:
                                value_matches += 1
                            elif val1 != 0 and abs((val2 - val1) / val1) < 0.2:
                                value_matches += 1
                        elif val1 == val2:
                            value_matches += 1
                    
                    if common_keys:
                        value_similarity = value_matches / len(common_keys)
                        similarity += 0.2 * value_similarity
        
        # 5. 关系结构相似度
        relations1 = self.relations.get(concept_id1, set())
        relations2 = self.relations.get(concept_id2, set())
        
        if relations1 and relations2:
            # 提取关系类型集合
            rel_types1 = {rel[1] for rel in relations1}
            rel_types2 = {rel[1] for rel in relations2}
            
            # 计算关系类型重叠度
            type_intersection = len(rel_types1.intersection(rel_types2))
            type_union = len(rel_types1.union(rel_types2))
            
            if type_union > 0:
                relation_similarity = type_intersection / type_union
                similarity += 0.1 * relation_similarity
        
        # 确保相似度在0-1范围内
        return min(1.0, max(0.0, similarity))

    def _calculate_relation_structure_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        计算两个概念的关系结构相似度
        
        Args:
            concept_id1: 第一个概念ID
            concept_id2: 第二个概念ID
            
        Returns:
            float: 关系结构相似度(0-1)
        """
        if concept_id1 not in self.relations and concept_id2 not in self.relations:
            return 1.0  # 都没有关系结构，认为相似
        elif concept_id1 not in self.relations or concept_id2 not in self.relations:
            return 0.0  # 一个有关系一个没有，认为不相似
        
        # 获取概念的关系
        relations1 = self.relations[concept_id1]
        relations2 = self.relations[concept_id2]
        
        # 提取关系类型和目标类型的组合
        rel_patterns1 = set()
        for target_id, rel_type, _ in relations1:
            if target_id in self.concepts:
                target_type = self.concepts[target_id].get("type", "unknown")
                rel_patterns1.add((rel_type, target_type))
        
        rel_patterns2 = set()
        for target_id, rel_type, _ in relations2:
            if target_id in self.concepts:
                target_type = self.concepts[target_id].get("type", "unknown")
                rel_patterns2.add((rel_type, target_type))
        
        # 计算模式相似度
        if not rel_patterns1 or not rel_patterns2:
            return 0.0
            
        intersection = len(rel_patterns1.intersection(rel_patterns2))
        union = len(rel_patterns1.union(rel_patterns2))
        
        return intersection / union if union > 0 else 0.0

    def _get_merge_rules(self) -> List[Dict[str, Any]]:
        """
        获取合并规则集
        
        Returns:
            List[Dict[str, Any]]: 合并规则列表
        """
        # 默认合并规则集
        return [
            {
                "name": "exact_duplicate",
                "condition": lambda p, s: self._is_exact_duplicate(self.concepts[p], self.concepts[s]),
                "action": "merge_all",
                "priority": 10
            },
            {
                "name": "superset_relation",
                "condition": lambda p, s: self._is_superset(self.concepts[p], self.concepts[s]),
                "action": "keep_primary",
                "priority": 8
            },
            {
                "name": "subset_relation",
                "condition": lambda p, s: self._is_superset(self.concepts[s], self.concepts[p]),
                "action": "keep_secondary",
                "priority": 8
            },
            {
                "name": "same_name_different_domain",
                "condition": lambda p, s: (
                    self.concepts[p].get("name", "") == self.concepts[s].get("name", "") and
                    self.concepts[p].get("domain", "") != self.concepts[s].get("domain", "")
                ),
                "action": "create_cross_domain_relation",
                "priority": 5
            },
            {
                "name": "high_similarity_same_type",
                "condition": lambda p, s: (
                    self._calculate_similarity(
                        self.concept_vectors.get(p, np.zeros(self.config["vector_dim"])),
                        self.concept_vectors.get(s, np.zeros(self.config["vector_dim"]))
                    ) > 0.9 and
                    self.concepts[p].get("type", "") == self.concepts[s].get("type", "")
                ),
                "action": "merge_weighted",
                "priority": 7
            }
        ]

    def _find_applicable_merge_rule(self, primary_id: str, secondary_id: str, 
                                rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        查找适用于给定概念对的合并规则
        
        Args:
            primary_id: 主要概念ID
            secondary_id: 次要概念ID
            rules: 规则列表
            
        Returns:
            Optional[Dict[str, Any]]: 找到的规则，没找到返回None
        """
        applicable_rules = []
        
        for rule in rules:
            condition_fn = rule["condition"]
            
            try:
                if condition_fn(primary_id, secondary_id):
                    applicable_rules.append(rule)
            except Exception as e:
                self.logger.warning(f"检查规则 {rule['name']} 时出错: {str(e)}")
        
        # 按优先级排序并返回最高优先级的规则
        if applicable_rules:
            return sorted(applicable_rules, key=lambda r: r["priority"], reverse=True)[0]
            
        return None

    def _apply_merge_rule(self, primary_id: str, secondary_id: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用合并规则
        
        Args:
            primary_id: 主要概念ID
            secondary_id: 次要概念ID
            rule: 要应用的规则
            
        Returns:
            Dict: 合并结果
        """
        action = rule["action"]
        result = {
            "status": "error",
            "message": f"未知合并动作: {action}",
            "affected_relations": 0
        }
        
        if action == "merge_all":
            # 完全合并 - 使用标准合并功能
            return self._merge_redundant_concepts(primary_id, secondary_id)
            
        elif action == "keep_primary":
            # 保留主要概念，迁移次要概念的关系
            return self._merge_redundant_concepts(primary_id, secondary_id)
            
        elif action == "keep_secondary":
            # 保留次要概念，迁移主要概念的关系(交换角色)
            return self._merge_redundant_concepts(secondary_id, primary_id)
            
        elif action == "merge_weighted":
            # 加权合并 - 使用智能合并功能
            # 计算相似度
            similarity = 0.0
            if primary_id in self.concept_vectors and secondary_id in self.concept_vectors:
                similarity = self._calculate_similarity(
                    self.concept_vectors[primary_id],
                    self.concept_vectors[secondary_id]
                )
            return self._intelligent_merge_concepts(primary_id, secondary_id, similarity)
            
        elif action == "create_cross_domain_relation":
            # 创建跨领域关系而不合并
            primary_domain = self.concepts[primary_id].get("domain", "general")
            secondary_domain = self.concepts[secondary_id].get("domain", "general")
            
            # 添加跨域关系
            self.add_relation(primary_id, secondary_id, "cross_domain_equivalent", 0.9)
            
            result = {
                "status": "success",
                "message": f"创建了跨域关系而非合并",
                "affected_relations": 1,
                "relation_type": "cross_domain_equivalent",
                "domains": [primary_domain, secondary_domain]
            }
        
        return result
        self.logger.info(f"开始检测知识冗余，相似度阈值: {similarity_threshold}")
        
        result = {
            "status": "success",
            "redundancy_groups": [],
            "merged_concepts": 0,
            "affected_relations": 0
        }
        
        try:
            # 获取所有概念ID
            concept_ids = list(self.concepts.keys())
            
            # 对每个概念，寻找高度相似的其他概念
            redundancy_groups = []
            processed_ids = set()
            
            for i, concept_id in enumerate(concept_ids):
                # 如果已经处理过（作为另一组的一部分），则跳过
                if concept_id in processed_ids:
                    continue
                    
                # 获取当前概念
                concept = self.concepts[concept_id]
                
                # 查找与当前概念相似的概念
                similar_concepts = []
                for j, other_id in enumerate(concept_ids[i+1:], i+1):
                    # 计算相似度
                    similarity = self._calculate_concept_similarity(concept_id, other_id)
                    
                    if similarity >= similarity_threshold:
                        similar_concepts.append({
                            "id": other_id,
                            "similarity": similarity,
                            "concept": self.concepts[other_id]
                        })
                
                # 如果找到相似概念，形成冗余组
                if similar_concepts:
                    redundancy_group = {
                        "primary": {
                            "id": concept_id,
                            "name": concept.get("name", ""),
                            "type": concept.get("type", "generic"),
                            "domain": concept.get("domain", "general")
                        },
                        "similar": similar_concepts,
                        "merged": False
                    }
                    
                    # 将所有相似概念标记为已处理
                    processed_ids.add(concept_id)
                    for similar in similar_concepts:
                        processed_ids.add(similar["id"])
                    
                    redundancy_groups.append(redundancy_group)
            
            # 记录找到的冗余组
            result["redundancy_groups"] = redundancy_groups
            result["total_redundancy_groups"] = len(redundancy_groups)
            
            # 自动合并冗余概念
            if auto_merge and redundancy_groups:
                merged_count = 0
                affected_relations = 0
                
                # 智能合并策略
                if merge_strategy == "intelligent":
                    # 按领域分组冗余组
                    domain_groups = defaultdict(list)
                    for group in redundancy_groups:
                        domain = group["primary"].get("domain", "general")
                        domain_groups[domain].append(group)
                        
                    # 对每个领域单独处理
                    for domain, groups in domain_groups.items():
                        # 按概念类型排序
                        typed_groups = defaultdict(list)
                        for group in groups:
                            concept_type = group["primary"].get("type", "generic")
                            typed_groups[concept_type].append(group)
                        
                        # 针对每种类型应用合并
                        for concept_type, type_groups in typed_groups.items():
                            # 先处理相似度最高的组
                            sorted_groups = sorted(type_groups, 
                                                 key=lambda g: max([s["similarity"] for s in g["similar"]] if g["similar"] else [0]), 
                                                 reverse=True)
                            
                            for group in sorted_groups:
                                primary_id = group["primary"]["id"]
                                # 按相似度排序相似概念
                                similar_concepts = sorted(group["similar"], key=lambda s: s["similarity"], reverse=True)
                                
                                # 使用加权属性策略进行合并
                                for similar in similar_concepts:
                                    merge_result = self._intelligent_merge_concepts(
                                        primary_id, similar["id"], similar["similarity"]
                                    )
                                    
                                    if merge_result["status"] == "success":
                                        merged_count += 1
                                        affected_relations += merge_result["affected_relations"]
                                        group["merged"] = True
                                        group["merge_strategy"] = "intelligent"
                
                # 引导式合并策略
                elif merge_strategy == "guided":
                    # 使用合并规则引导合并过程
                    merge_rules = self._get_merge_rules()
                    
                    for group in redundancy_groups:
                        primary_id = group["primary"]["id"]
                        similar_concepts = group["similar"]
                        
                        for similar in similar_concepts:
                            # 查找适用的规则
                            rule = self._find_applicable_merge_rule(
                                primary_id, similar["id"], merge_rules
                            )
                            
                            if rule:
                                # 应用规则进行合并
                                merge_result = self._apply_merge_rule(
                                    primary_id, similar["id"], rule
                                )
                                
                                if merge_result["status"] == "success":
                                    merged_count += 1
                                    affected_relations += merge_result["affected_relations"]
                                    group["merged"] = True
                                    group["merge_strategy"] = "guided"
                                    group["applied_rule"] = rule["name"]
                
                # 简单合并策略(默认)
                else:
                    for group in redundancy_groups:
                        primary_id = group["primary"]["id"]
                        similar_concepts = group["similar"]
                        
                        # 合并每个相似概念到主概念
                        for similar in similar_concepts:
                            merge_result = self._merge_redundant_concepts(primary_id, similar["id"])
                            
                            if merge_result["status"] == "success":
                                merged_count += 1
                                affected_relations += merge_result["affected_relations"]
                                group["merged"] = True
                                group["merge_strategy"] = "simple"
                
                result["merged_concepts"] = merged_count
                result["affected_relations"] = affected_relations
                result["merge_strategy"] = merge_strategy
                
                # 更新统计信息
                self.stats["concept_count"] = len(self.concepts)
                
                # 记录合并操作
                self.logger.info(f"已合并 {merged_count} 个冗余概念, 影响了 {affected_relations} 个关系")
        
        except Exception as e:
            self.logger.error(f"知识冗余检测异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            result["status"] = "error"
            result["message"] = str(e)
        
        return result
    
    def _calculate_concept_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        计算两个概念的综合相似度
        
        考虑以下因素:
        1. 向量相似度
        2. 属性相似度
        3. 关系结构相似度
        """
        # 权重配置
        weights = {
            "vector": 0.4,      # 向量相似度权重
            "attribute": 0.4,   # 属性相似度权重
            "structure": 0.2    # 结构相似度权重
        }
        
        # 向量相似度
        vector_similarity = 0.0
        if (concept_id1 in self.concept_vectors and 
            concept_id2 in self.concept_vectors):
            vector_similarity = self._calculate_similarity(
                self.concept_vectors[concept_id1],
                self.concept_vectors[concept_id2]
            )
        
        # 属性相似度
        attribute_similarity = self._calculate_concept_attribute_similarity(
            self.concepts[concept_id1],
            self.concepts[concept_id2]
        )
        
        # 结构相似度
        structure_similarity = 0.0
        try:
            structure_similarity = self._calculate_structural_similarity(concept_id1, concept_id2)
        except Exception as e:
            self.logger.warning(f"计算结构相似度异常: {str(e)}")
        
        # 加权综合相似度
        similarity = (
            weights["vector"] * vector_similarity +
            weights["attribute"] * attribute_similarity +
            weights["structure"] * structure_similarity
        )
        
        return similarity
    
    def _merge_redundant_concepts(self, primary_id: str, secondary_id: str) -> Dict[str, Any]:
        """
        合并冗余概念，将次要概念的信息合并到主要概念中
        
        Args:
            primary_id: 主要概念ID（保留）
            secondary_id: 次要概念ID（将被合并）
            
        Returns:
            Dict: 合并结果
        """
        result = {
            "status": "error",
            "message": "",
            "affected_relations": 0
        }
        
        # 检查概念是否存在
        if primary_id not in self.concepts:
            result["message"] = f"主概念不存在: {primary_id}"
            return result
            
        if secondary_id not in self.concepts:
            result["message"] = f"次要概念不存在: {secondary_id}"
            return result
            
        if primary_id == secondary_id:
            result["message"] = "不能自我合并"
            return result
        
        try:
            # 获取概念
            primary = self.concepts[primary_id]
            secondary = self.concepts[secondary_id]
            
            # 合并属性 - 采用补充而非覆盖策略
            self._merge_concept_attributes(primary, secondary)
            
            # 合并标签
            if "tags" in secondary:
                if "tags" not in primary:
                    primary["tags"] = []
                primary["tags"] = list(set(primary["tags"] + secondary["tags"]))
            
            # 合并元数据
            if "metadata" in secondary:
                if "metadata" not in primary:
                    primary["metadata"] = {}
                    
                # 记录合并信息
                if "merged_from" not in primary["metadata"]:
                    primary["metadata"]["merged_from"] = []
                    
                primary["metadata"]["merged_from"].append({
                    "id": secondary_id,
                    "name": secondary.get("name", ""),
                    "timestamp": time.time()
                })
                
                # 合并其他元数据
                for key, value in secondary["metadata"].items():
                    if key not in primary["metadata"] and key != "merged_from":
                        primary["metadata"][key] = value
            
            # 更新向量
            if primary_id in self.concept_vectors and secondary_id in self.concept_vectors:
                # 加权平均两个向量
                primary_vec = self.concept_vectors[primary_id]
                secondary_vec = self.concept_vectors[secondary_id]
                
                # 使用 0.7:0.3 的权重
                import numpy as np
                merged_vec = 0.7 * primary_vec + 0.3 * secondary_vec
                
                # 归一化
                norm = np.linalg.norm(merged_vec)
                if norm > 0:
                    merged_vec = merged_vec / norm
                    
                self.concept_vectors[primary_id] = merged_vec
            
            # 处理关系迁移
            affected_relations = self._migrate_concept_relations(primary_id, secondary_id)
            result["affected_relations"] = affected_relations
            
            # 更新索引
            self._update_concept_index(primary_id, primary)
            
            # 从索引中移除次要概念
            self._remove_from_index(secondary_id)
            
            # 记录次要概念被合并的信息
            if "merged_into" not in secondary:
                secondary["merged_into"] = primary_id
                secondary["merge_timestamp"] = time.time()
            
            # 保留次要概念的引用但标记为已合并
            merged_info = {
                "id": secondary_id,
                "merged_into": primary_id,
                "merge_timestamp": time.time(),
                "name": secondary.get("name", ""),
                "type": secondary.get("type", "generic")
            }
            
            # 从活动概念中移除次要概念
            del self.concepts[secondary_id]
            
            # 更新元数据
            self.concept_metadata[primary_id]["last_modified"] = time.time()
            self.concept_metadata[primary_id]["merge_count"] = self.concept_metadata[primary_id].get("merge_count", 0) + 1
            
            # 更新统计信息
            self.stats["concept_count"] -= 1
            
            # 记录合并事件
            self.logger.info(f"已合并概念: {secondary_id} -> {primary_id}")
            
            result["status"] = "success"
            
        except Exception as e:
            self.logger.error(f"合并概念异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            result["message"] = str(e)
        
        return result
    
    def _merge_concept_attributes(self, primary: Dict[str, Any], secondary: Dict[str, Any]):
        """合并概念属性，采用补充策略"""
        # 合并基本属性
        for key in ["description", "source", "url", "examples"]:
            if key in secondary and key not in primary:
                primary[key] = secondary[key]
        
        # 合并特征
        if "features" in secondary:
            if "features" not in primary:
                primary["features"] = {}
                
            for feat_key, feat_value in secondary["features"].items():
                if feat_key not in primary["features"]:
                    primary["features"][feat_key] = feat_value
        
        # 合并属性
        if "attributes" in secondary:
            if "attributes" not in primary:
                primary["attributes"] = {}
                
            for attr_key, attr_value in secondary["attributes"].items():
                if attr_key not in primary["attributes"]:
                    primary["attributes"][attr_key] = attr_value
        
        # 更新创建/修改时间
        if "created_at" in secondary and ("created_at" not in primary or secondary["created_at"] < primary["created_at"]):
            primary["created_at"] = secondary["created_at"]
            
        primary["last_modified"] = time.time()
    
    def _migrate_concept_relations(self, primary_id: str, secondary_id: str) -> int:
        """迁移次要概念的所有关系到主要概念"""
        affected_relations = 0
        
        # 处理出向关系
        if secondary_id in self.relations:
            for relation in list(self.relations[secondary_id]):
                target_id, rel_type, confidence = relation
                
                # 避免自环
                if target_id == primary_id:
                    continue
                
                # 检查是否已存在相同关系
                relation_exists = False
                for existing_rel in self.relations.get(primary_id, set()):
                    if existing_rel[0] == target_id and existing_rel[1] == rel_type:
                        relation_exists = True
                        break
                
                # 如果不存在，添加新关系
                if not relation_exists:
                    if primary_id not in self.relations:
                        self.relations[primary_id] = set()
                    
                    self.relations[primary_id].add((target_id, rel_type, confidence))
                    affected_relations += 1
                    
                    # 更新目标概念的反向关系
                    self._update_inverse_relation(target_id, secondary_id, primary_id, rel_type)
            
            # 清除次要概念的关系
            del self.relations[secondary_id]
        
        # 处理入向关系（指向次要概念的关系）
        for source_id, relations in self.relations.items():
            if source_id == primary_id or source_id == secondary_id:
                continue
                
            updated_relations = set()
            relations_to_add = set()
            
            for relation in relations:
                target_id, rel_type, confidence = relation
                
                # 如果关系指向次要概念，将其重定向到主要概念
                if target_id == secondary_id:
                    # 检查是否已存在到主要概念的相同关系
                    has_primary_relation = False
                    for rel in relations:
                        if rel[0] == primary_id and rel[1] == rel_type:
                            has_primary_relation = True
                            break
                    
                    if not has_primary_relation:
                        # 添加到主要概念的关系
                        relations_to_add.add((primary_id, rel_type, confidence))
                        affected_relations += 1
                else:
                    # 保留其他关系
                    updated_relations.add(relation)
            
            # 更新关系集
            updated_relations.update(relations_to_add)
            self.relations[source_id] = updated_relations
        
        return affected_relations
    
    def _update_inverse_relation(self, target_id: str, old_source_id: str, new_source_id: str, relation_type: str):
        """更新反向关系的源"""
        inverse_type = self._get_inverse_relation_type(relation_type)
        if not inverse_type:
            return
            
        # 查找并更新目标概念的反向关系
        if target_id in self.relations:
            updated_relations = set()
            
            for relation in self.relations[target_id]:
                rel_target, rel_type, confidence = relation
                
                # 如果是指向旧源的反向关系，重定向到新源
                if rel_target == old_source_id and rel_type == inverse_type:
                    updated_relations.add((new_source_id, rel_type, confidence))
                else:
                    updated_relations.add(relation)
            
            self.relations[target_id] = updated_relations
    
    def _remove_from_index(self, concept_id: str):
        """从索引中移除概念"""
        concept = self.concepts.get(concept_id)
        if not concept:
            return
            
        # 从名称索引中移除
        if "name" in concept:
            name = concept["name"].lower()
            if name in self.concept_index["by_name"]:
                self.concept_index["by_name"][name] = [cid for cid in self.concept_index["by_name"][name] if cid != concept_id]
                
                if not self.concept_index["by_name"][name]:
                    del self.concept_index["by_name"][name]
        
        # 从标签索引中移除
        if "tags" in concept:
            for tag in concept["tags"]:
                tag = tag.lower()
                if tag in self.concept_index["by_tag"]:
                    self.concept_index["by_tag"][tag] = [cid for cid in self.concept_index["by_tag"][tag] if cid != concept_id]
                    
                    if not self.concept_index["by_tag"][tag]:
                        del self.concept_index["by_tag"][tag]
        
        # 从领域索引中移除
        if "domain" in concept:
            domain = concept["domain"].lower()
            if domain in self.concept_index["by_domain"]:
                self.concept_index["by_domain"][domain] = [cid for cid in self.concept_index["by_domain"][domain] if cid != concept_id]
                
                if not self.concept_index["by_domain"][domain]:
                    del self.concept_index["by_domain"][domain]
        
        # 从类型索引中移除
        if "type" in concept:
            type_name = concept["type"].lower()
            if type_name in self.concept_index["by_type"]:
                self.concept_index["by_type"][type_name] = [cid for cid in self.concept_index["by_type"][type_name] if cid != concept_id]
                
                if not self.concept_index["by_type"][type_name]:
                    del self.concept_index["by_type"][type_name]
    
    def evolve_knowledge_graph(self, iterations: int = 1, evolution_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        自动演化知识图谱，优化结构和关系，发现新模式
        
        Args:
            iterations: 演化迭代次数
            evolution_params: 演化参数配置
            
        Returns:
            Dict: 演化结果
        """
        self.logger.info(f"开始知识图谱演化，迭代次数: {iterations}")
        
        # 默认演化参数
        default_params = {
            "consolidation_threshold": 0.8,   # 概念合并阈值
            "relation_inference_threshold": 0.7,  # 关系推断阈值
            "max_cluster_concepts": 100,      # 每次处理的最大概念数量
            "enable_hierarchy_formation": True,  # 是否启用层次结构形成
            "enable_domain_bridging": True,    # 是否启用跨域桥接
            "min_category_size": 3            # 最小类别大小
        }
        
        # 更新参数
        params = default_params.copy()
        if evolution_params:
            params.update(evolution_params)
        
        # 初始化结果
        result = {
            "success": True,
            "iterations_completed": 0,
            "concepts_affected": 0,
            "relations_added": 0,
            "relations_removed": 0,
            "hierarchies_formed": 0,
            "clusters_detected": 0,
            "cross_domain_bridges": 0,
            "emerging_categories": [],
            "evolution_stages": []
        }
        
        try:
            # 执行多次迭代
            for iteration in range(iterations):
                self.logger.info(f"开始演化迭代 {iteration+1}/{iterations}")
                
                # 记录此次迭代的结果
                iteration_result = {
                    "iteration": iteration + 1,
                    "stages": [],
                    "concepts_affected": 0,
                    "relations_modified": 0
                }
                
                # 阶段1: 概念聚类和合并
                cluster_result = self._evolution_cluster_and_consolidate(params)
                iteration_result["stages"].append({
                    "name": "cluster_and_consolidate",
                    "details": cluster_result
                })
                
                # 阶段2: 关系优化与推断
                relation_result = self._evolution_optimize_relations(params)
                iteration_result["stages"].append({
                    "name": "relation_optimization",
                    "details": relation_result
                })
                
                # 阶段3: 层次结构形成（如果启用）
                if params["enable_hierarchy_formation"]:
                    hierarchy_result = self._evolution_form_hierarchies(params)
                    iteration_result["stages"].append({
                        "name": "hierarchy_formation",
                        "details": hierarchy_result
                    })
                    result["hierarchies_formed"] += hierarchy_result["hierarchies_formed"]
                
                # 阶段4: 跨领域桥接（如果启用）
                if params["enable_domain_bridging"]:
                    bridge_result = self._evolution_create_domain_bridges(params)
                    iteration_result["stages"].append({
                        "name": "domain_bridging",
                        "details": bridge_result
                    })
                    result["cross_domain_bridges"] += bridge_result["bridges_created"]
                
                # 阶段5: 发现新兴类别
                category_result = self._evolution_discover_categories(params)
                iteration_result["stages"].append({
                    "name": "category_discovery",
                    "details": category_result
                })
                result["emerging_categories"].extend(category_result["new_categories"])
                
                # 更新迭代统计
                iteration_result["concepts_affected"] = (
                    cluster_result["concepts_affected"] +
                    relation_result["concepts_affected"] +
                    category_result["concepts_affected"]
                )
                
                iteration_result["relations_modified"] = (
                    relation_result["relations_added"] +
                    relation_result["relations_removed"]
                )
                
                # 添加到总结果
                result["evolution_stages"].append(iteration_result)
                result["concepts_affected"] += iteration_result["concepts_affected"]
                result["relations_added"] += relation_result["relations_added"]
                result["relations_removed"] += relation_result["relations_removed"]
                result["clusters_detected"] += cluster_result["clusters_detected"]
                
                # 更新完成的迭代次数
                result["iterations_completed"] += 1
                
                self.logger.info(f"完成演化迭代 {iteration+1}/{iterations}")
            
            # 更新统计信息
            self.stats["reorganizations"] += 1
            self.stats["last_reorganized"] = time.time()
            
            self.logger.info(f"知识图谱演化完成: 影响了 {result['concepts_affected']} 个概念, "
                         f"增加了 {result['relations_added']} 个关系, 移除了 {result['relations_removed']} 个关系")
            
        except Exception as e:
            self.logger.error(f"知识图谱演化异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _evolution_cluster_and_consolidate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """知识演化：概念聚类和合并"""
        result = {
            "clusters_detected": 0,
            "concepts_affected": 0,
            "concepts_merged": 0,
            "concept_count_before": len(self.concepts)
        }
        
        # 分批处理概念，避免一次处理过多
        concept_ids = list(self.concepts.keys())
        max_concepts = min(params["max_cluster_concepts"], len(concept_ids))
        
        # 选择最近更新或使用频率最高的概念
        concept_priorities = []
        for cid in concept_ids:
            # 计算优先级分数
            last_accessed = self.concept_metadata.get(cid, {}).get("last_accessed", 0)
            usage_count = self.concept_metadata.get(cid, {}).get("usage_count", 0)
            related_count = len(self.relations.get(cid, set()))
            
            # 综合分数 (时间衰减 + 使用频率 + 关系数量)
            time_factor = 1.0 / (1.0 + 0.1 * (time.time() - last_accessed) / 86400)  # 时间衰减（天）
            priority = 0.5 * time_factor + 0.3 * min(1.0, usage_count / 100) + 0.2 * min(1.0, related_count / 20)
            
            concept_priorities.append((cid, priority))
        
        # 按优先级排序并选择前N个
        concept_priorities.sort(key=lambda x: x[1], reverse=True)
        selected_concepts = [item[0] for item in concept_priorities[:max_concepts]]
        
        # 执行聚类
        clusters = self._cluster_selected_concepts(selected_concepts)
        result["clusters_detected"] = len(clusters)
        
        # 对每个聚类进行合并处理
        for cluster in clusters:
            # 如果聚类太小，跳过
            if len(cluster) < 2:
                continue
                
            # 计算聚类内部相似度
            consolidation_candidates = self._find_consolidation_candidates(cluster, params["consolidation_threshold"])
            
            # 合并高度相似的概念
            for primary_id, secondary_ids in consolidation_candidates.items():
                for secondary_id in secondary_ids:
                    # 合并概念
                    merge_result = self._merge_redundant_concepts(primary_id, secondary_id)
                    
                    if merge_result["status"] == "success":
                        result["concepts_merged"] += 1
                        
            # 标记受影响的概念
            result["concepts_affected"] += len(cluster)
        
        # 记录聚类结果
        result["concept_count_after"] = len(self.concepts)
        result["reduction_percentage"] = 0
        if result["concept_count_before"] > 0:
            result["reduction_percentage"] = 100 * (result["concept_count_before"] - result["concept_count_after"]) / result["concept_count_before"]
        
        return result
    
    def _cluster_selected_concepts(self, concept_ids: List[str]) -> List[List[str]]:
        """对选定的概念执行聚类分析"""
        # 准备相似度矩阵
        n = len(concept_ids)
        similarity_matrix = np.zeros((n, n))
        
        # 计算概念间相似度
        for i in range(n):
            for j in range(i+1, n):
                sim = self._calculate_concept_similarity(concept_ids[i], concept_ids[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
        # 对角线设为1.0（自身相似度）
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # 尝试使用scikit-learn进行聚类
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # 确定合适的聚类数
            max_clusters = min(n // 2, 20)  # 最多20个聚类，且每个至少2个概念
            if max_clusters < 2:
                max_clusters = 2
                
            # 基于相似度进行层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None,  # 不指定聚类数，使用距离阈值
                distance_threshold=0.3,  # 相似度阈值转为距离阈值 (1-相似度)
                affinity='precomputed',  # 使用预计算的相似度矩阵
                linkage='average',       # 使用平均连接
                compute_distances=True,
                metric='precomputed'
            )
            
            # 转换为距离矩阵 (1-相似度)
            distance_matrix = 1.0 - similarity_matrix
            
            # 执行聚类
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # 将聚类结果转换为概念组
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(concept_ids[i])
                
            return list(clusters.values())
                
        except ImportError:
            # 如果没有sklearn，使用简单的相似度聚类
            return self._simple_similarity_clustering(concept_ids, similarity_matrix)
    
    def _simple_similarity_clustering(self, concept_ids: List[str], similarity_matrix: np.ndarray) -> List[List[str]]:
        """简单的基于相似度的聚类算法"""
        n = len(concept_ids)
        if n == 0:
            return []
            
        # 未分配的概念索引
        unassigned = set(range(n))
        clusters = []
        
        while unassigned:
            # 选择一个未分配的概念作为新聚类的种子
            seed = next(iter(unassigned))
            cluster = [seed]
            unassigned.remove(seed)
            
            # 找出与种子相似的概念
            for i in unassigned.copy():
                if similarity_matrix[seed, i] >= 0.7:  # 相似度阈值
                    cluster.append(i)
                    unassigned.remove(i)
            
            # 添加聚类（转换为概念ID）
            clusters.append([concept_ids[idx] for idx in cluster])
        
        return clusters
    
    def _find_consolidation_candidates(self, cluster: List[str], threshold: float) -> Dict[str, List[str]]:
        """在聚类中查找可合并的概念对"""
        # 合并候选 {primary_id: [secondary_ids]}
        candidates = defaultdict(list)
        
        # 计算聚类中每对概念的相似度
        for i, concept_id1 in enumerate(cluster):
            for concept_id2 in cluster[i+1:]:
                sim = self._calculate_concept_similarity(concept_id1, concept_id2)
                
                # 如果超过合并阈值
                if sim >= threshold:
                    # 决定哪个是主要概念（保留哪个）
                    primary, secondary = self._determine_merge_priority(concept_id1, concept_id2)
                    candidates[primary].append(secondary)
        
        return candidates
    
    def _determine_merge_priority(self, concept_id1: str, concept_id2: str) -> Tuple[str, str]:
        """确定两个概念合并时哪个应该作为主要概念（保留）"""
        # 评分标准：关系数量、访问频率、创建时间、概念完整度
        
        # 1. 获取概念
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        meta1 = self.concept_metadata.get(concept_id1, {})
        meta2 = self.concept_metadata.get(concept_id2, {})
        
        # 2. 计算各项指标
        # 关系数量
        relations1 = len(self.relations.get(concept_id1, set()))
        relations2 = len(self.relations.get(concept_id2, set()))
        
        # 使用频率
        usage1 = meta1.get("usage_count", 0)
        usage2 = meta2.get("usage_count", 0)
        
        # 创建时间（较旧的优先）
        age1 = meta1.get("creation_time", concept1.get("created_at", time.time()))
        age2 = meta2.get("creation_time", concept2.get("created_at", time.time()))
        age_score1 = 1.0 if age1 < age2 else 0.0
        age_score2 = 1.0 if age2 < age1 else 0.0
        
        # 概念完整度
        completeness1 = self._calculate_concept_completeness(concept1)
        completeness2 = self._calculate_concept_completeness(concept2)
        
        # 3. 加权评分
        weights = {
            "relations": 0.35,
            "usage": 0.25,
            "age": 0.15,
            "completeness": 0.25
        }
        
        # 归一化关系和使用频率得分
        max_relations = max(1, max(relations1, relations2))
        max_usage = max(1, max(usage1, usage2))
        
        rel_score1 = relations1 / max_relations
        rel_score2 = relations2 / max_relations
        usage_score1 = usage1 / max_usage
        usage_score2 = usage2 / max_usage
        
        # 计算总分
        score1 = (
            weights["relations"] * rel_score1 +
            weights["usage"] * usage_score1 +
            weights["age"] * age_score1 +
            weights["completeness"] * completeness1
        )
        
        score2 = (
            weights["relations"] * rel_score2 +
            weights["usage"] * usage_score2 +
            weights["age"] * age_score2 +
            weights["completeness"] * completeness2
        )
        
        # 返回得分较高的作为主要概念
        if score1 >= score2:
            return concept_id1, concept_id2
        else:
            return concept_id2, concept_id1
    
    def _calculate_concept_completeness(self, concept: Dict[str, Any]) -> float:
        """计算概念的完整度评分"""
        completeness = 0.0
        total_weight = 0.0
        
        # 检查关键字段存在性
        fields = {
            "name": 0.2,
            "description": 0.3,
            "type": 0.15,
            "features": 0.25,
            "tags": 0.1
        }
        
        for field, weight in fields.items():
            if field in concept and concept[field]:
                # 对于列表或字典类型，还要考虑非空内容
                if isinstance(concept[field], (list, dict)) and not concept[field]:
                    continue
                    
                completeness += weight
                
            total_weight += weight
        
        # 标准化结果
        if total_weight > 0:
            completeness /= total_weight
            
        return completeness
    
    def _evolution_optimize_relations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """知识演化：关系优化与推断"""
        result = {
            "relations_added": 0,
            "relations_removed": 0,
            "concepts_affected": 0
        }
        
        # 获取所有概念ID
        concept_ids = list(self.concepts.keys())
        concepts_affected = set()
        
        # 1. 移除冗余关系
        redundant_count = 0
        for concept_id in concept_ids:
            redundant_relations = self._remove_redundant_relations(concept_id)
            if redundant_relations > 0:
                concepts_affected.add(concept_id)
                result["relations_removed"] += redundant_relations
        
        # 2. 推断潜在关系
        for i, concept_id1 in enumerate(concept_ids):
            for concept_id2 in concept_ids[i+1:]:
                # 检查两个概念当前是否有直接关系
                has_direct_relation = False
                for relation in self.relations.get(concept_id1, set()):
                    if relation[0] == concept_id2:
                        has_direct_relation = True
                        break
                        
                if not has_direct_relation:
                    # 尝试推断关系
                    inferred = self._infer_potential_relation(concept_id1, concept_id2, params["relation_inference_threshold"])
                    if inferred:
                        concepts_affected.add(concept_id1)
                        concepts_affected.add(concept_id2)
                        result["relations_added"] += 1
        
        # 3. 增强关系的语义精确性
        refined_count = self._refine_relation_semantics(concept_ids)
        result["relations_refined"] = refined_count
        
        # 更新受影响的概念数
        result["concepts_affected"] = len(concepts_affected)
        
        return result
    
    def _remove_redundant_relations(self, concept_id: str) -> int:
        """移除概念的冗余关系"""
        if concept_id not in self.relations:
            return 0
            
        relations = self.relations[concept_id]
        redundant_relations = set()
        
        # 检查传递性关系
        # 如果 A->B, B->C, A->C 存在，且 A->C 是传递性关系，则可能冗余
        relation_list = list(relations)
        for rel1_idx, rel1 in enumerate(relation_list):
            target1, type1, _ = rel1
            
            # 查找target1的关系
            for rel2 in self.relations.get(target1, set()):
                target2, type2, _ = rel2
                
                # 查找是否有直接关系到target2
                for rel3_idx, rel3 in enumerate(relation_list):
                    if rel3_idx == rel1_idx:  # 跳过相同关系
                        continue
                        
                    target3, type3, _ = rel3
                    if target3 == target2:
                        # 检查是否是传递性关系
                        if self._is_transitive_relation(type1, type2, type3):
                            redundant_relations.add(rel3)
        
        # 更新关系集
        if redundant_relations:
            self.relations[concept_id] = relations - redundant_relations
            
        return len(redundant_relations)
    
    def _is_transitive_relation(self, type1: str, type2: str, type3: str) -> bool:
        """判断是否是传递性关系"""
        # 传递性关系规则
        transitive_rules = {
            ("part_of", "part_of"): "part_of",
            ("contains", "contains"): "contains",
            ("subclass_of", "subclass_of"): "subclass_of",
            ("superclass_of", "superclass_of"): "superclass_of",
            ("precedes", "precedes"): "precedes",
            ("follows", "follows"): "follows"
        }
        
        return transitive_rules.get((type1, type2)) == type3
    
    def _infer_potential_relation(self, concept_id1: str, concept_id2: str, threshold: float) -> bool:
        """推断两个概念之间潜在的关系"""
        # 获取概念
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        
        # 首先检查直接相关性
        similarity = self._calculate_concept_similarity(concept_id1, concept_id2)
        
        # 如果相似度高，但尚未建立直接关系，创建相似关系
        if similarity >= threshold:
            relation_type = self._determine_relation_type(concept_id1, concept_id2, similarity)
            add_result = self.add_relation(concept_id1, concept_id2, relation_type, similarity)
            return add_result["status"] == "success"
            
        # 尝试通过共同邻居推断关系
        common_neighbors = self._find_common_neighbors(concept_id1, concept_id2)
        if len(common_neighbors) >= 2:  # 至少有2个共同邻居
            # 基于共同邻居的连接强度计算隐含关系
            rel_confidence = min(1.0, 0.3 + 0.1 * len(common_neighbors))
            
            # 确定关系类型
            relation_type = "related_to"  # 默认关系类型
            
            # 分析通过共同邻居推断的可能关系类型
            type_counts = defaultdict(int)
            for neighbor in common_neighbors:
                # 获取与共同邻居的关系类型
                rel_type1 = self._get_relation_type(concept_id1, neighbor)
                rel_type2 = self._get_relation_type(concept_id2, neighbor)
                
                # 基于这些关系类型推断可能的类型
                inferred_type = self._infer_relation_from_common(rel_type1, rel_type2)
                if inferred_type:
                    type_counts[inferred_type] += 1
            
            # 选择最频繁的推断类型（如果有）
            if type_counts:
                relation_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # 添加推断的关系
            add_result = self.add_relation(concept_id1, concept_id2, relation_type, rel_confidence)
            return add_result["status"] == "success"
            
        return False
    
    def _get_relation_type(self, source_id: str, target_id: str) -> Optional[str]:
        """获取从源概念到目标概念的关系类型"""
        for relation in self.relations.get(source_id, set()):
            if relation[0] == target_id:
                return relation[1]
        return None
    
    def _find_common_neighbors(self, concept_id1: str, concept_id2: str) -> List[str]:
        """查找两个概念的共同邻居"""
        # 获取概念1的所有关联概念
        neighbors1 = set()
        for relation in self.relations.get(concept_id1, set()):
            neighbors1.add(relation[0])
            
        # 检查指向概念1的关系
        for source_id, relations in self.relations.items():
            for relation in relations:
                if relation[0] == concept_id1:
                    neighbors1.add(source_id)
                    break
        
        # 获取概念2的所有关联概念
        neighbors2 = set()
        for relation in self.relations.get(concept_id2, set()):
            neighbors2.add(relation[0])
            
        # 检查指向概念2的关系
        for source_id, relations in self.relations.items():
            for relation in relations:
                if relation[0] == concept_id2:
                    neighbors2.add(source_id)
                    break
        
        # 找出交集（排除两个概念本身）
        common = neighbors1 & neighbors2
        common.discard(concept_id1)
        common.discard(concept_id2)
        
        return list(common)
    
    def _infer_relation_from_common(self, rel_type1: Optional[str], rel_type2: Optional[str]) -> Optional[str]:
        """根据与共同邻居的关系类型推断关系"""
        if not rel_type1 or not rel_type2:
            return None
            
        # 关系推断规则
        inference_rules = {
            # 相同类型
            ("part_of", "part_of"): "related_to",
            ("contains", "contains"): "related_to",
            ("subclass_of", "subclass_of"): "similar_to",
            ("superclass_of", "superclass_of"): "similar_to",
            
            # 反向类型（可能表示同级关系）
            ("part_of", "contains"): None,  # 一个是另一个的一部分，不应直接关联
            ("subclass_of", "superclass_of"): None,  # 层次关系中可能没有直接联系
            
            # 特殊组合
            ("used_by", "used_by"): "alternative_to",
            ("affects", "affects"): "related_effect"
        }
        
        # 尝试两种顺序
        inferred = inference_rules.get((rel_type1, rel_type2))
        if inferred is None:
            inferred = inference_rules.get((rel_type2, rel_type1))
            
        # 默认关系类型
        return inferred or "related_to"
    
    def _refine_relation_semantics(self, concept_ids: List[str]) -> int:
        """优化关系的语义精确性"""
        refined_count = 0
        
        # 查找一般关系（如"related_to"）并尝试优化为更精确的关系
        for concept_id in concept_ids:
            if concept_id not in self.relations:
                continue
                
            relations_to_refine = []
            
            # 查找需要优化的关系
            for relation in self.relations[concept_id]:
                target_id, rel_type, confidence = relation
                
                # 检查是否是一般关系类型
                if rel_type in ["related_to", "connected_to", "associated_with"]:
                    relations_to_refine.append((target_id, rel_type, confidence))
            
            # 优化找到的关系
            if relations_to_refine:
                for target_id, old_type, old_confidence in relations_to_refine:
                    # 尝试确定更精确的关系类型
                    new_type = self._determine_specific_relation_type(concept_id, target_id)
                    
                    if new_type and new_type != old_type:
                        # 移除旧关系
                        self.relations[concept_id].remove((target_id, old_type, old_confidence))
                        
                        # 添加新关系
                        self.relations[concept_id].add((target_id, new_type, old_confidence))
                        
                        # 更新反向关系
                        self._update_inverse_relation(target_id, concept_id, concept_id, new_type)
                        
                        refined_count += 1
        
        return refined_count
    
    def _determine_specific_relation_type(self, concept_id1: str, concept_id2: str) -> Optional[str]:
        """确定两个概念之间更精确的关系类型"""
        # 获取概念
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        
        # 检查类型特定关系
        type1 = concept1.get("type", "generic")
        type2 = concept2.get("type", "generic")
        
        # 类型组合规则
        type_relation_rules = {
            ("object", "property"): "has_property",
            ("property", "object"): "property_of",
            ("object", "action"): "can_undergo",
            ("action", "object"): "applies_to",
            ("category", "concept"): "contains",
            ("concept", "category"): "belongs_to",
            ("process", "step"): "includes_step",
            ("step", "process"): "part_of_process",
            ("cause", "effect"): "causes",
            ("effect", "cause"): "caused_by"
        }
        
        # 检查类型规则
        specific_type = type_relation_rules.get((type1, type2))
        if specific_type:
            return specific_type
            
        # 基于属性推断关系
        if "attributes" in concept1 and "attributes" in concept2:
            attrs1 = set(concept1["attributes"].keys())
            attrs2 = set(concept2["attributes"].keys())
            
            # 如果一个概念的属性是另一个的超集
            if attrs1.issuperset(attrs2) and len(attrs1) > len(attrs2):
                return "generalizes"
            elif attrs2.issuperset(attrs1) and len(attrs2) > len(attrs1):
                return "specializes"
        
        # 基于特征推断关系
        if "features" in concept1 and "features" in concept2:
            return self._infer_relation_from_features(concept1["features"], concept2["features"])
            
        # 无法确定更精确的类型
        return None
    
    def _infer_relation_from_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Optional[str]:
        """根据概念特征推断关系类型"""
        # 检查特征重叠
        keys1 = set(features1.keys())
        keys2 = set(features2.keys())
        
        common_keys = keys1 & keys2
        
        # 如果没有共同特征，返回None
        if not common_keys:
            return None
            
        # 检查值相等的特征数量
        matching_values = 0
        for key in common_keys:
            if features1[key] == features2[key]:
                matching_values += 1
                
        # 根据匹配程度确定关系
        match_ratio = matching_values / len(common_keys)
        
        if match_ratio > 0.8:
            return "similar_to"
        elif match_ratio > 0.5:
            return "related_to"
        elif keys1.issuperset(keys2):
            return "generalizes"
        elif keys2.issuperset(keys1):
            return "specializes"
            
        return "shares_features"
    
    def update_concept_relations_dynamic(self, concept_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        动态更新概念关联，基于上下文和使用情况实时调整关系强度和相关性
        
        Args:
            concept_id: 要更新的概念ID
            context: 上下文信息，包含当前活跃概念、使用模式等
            
        Returns:
            Dict: 更新结果
        """
        if concept_id not in self.concepts:
            return {
                "status": "error",
                "message": f"概念不存在: {concept_id}"
            }
            
        result = {
            "status": "success",
            "concept_id": concept_id,
            "new_relations": 0,
            "removed_relations": 0,
            "strengthened_relations": 0,
            "weakened_relations": 0,
            "relation_strength_changes": {}
        }
        
        # 获取概念数据
        concept = self.concepts[concept_id]
        
        # 提取上下文信息
        context = context or {}
        active_concepts = context.get("active_concepts", [])
        interaction_type = context.get("interaction_type", "view")  # view, edit, query, etc.
        domain_focus = context.get("domain_focus", concept.get("domain", "general"))
        
        try:
            # 根据上下文，更新现有关系强度
            if active_concepts:
                # 当前与其他活跃概念同时出现，增强关系
                for active_id in active_concepts:
                    if active_id == concept_id or active_id not in self.concepts:
                        continue
                        
                    # 获取现有关系
                    existing_rel_type = self._get_relation_type(concept_id, active_id)
                    inverse_rel_type = self._get_relation_type(active_id, concept_id)
                    
                    if existing_rel_type:
                        # 获取关系元数据
                        if not hasattr(self, "relation_metadata"):
                            self.relation_metadata = defaultdict(lambda: defaultdict(dict))
                        
                        rel_meta = self.relation_metadata[concept_id][active_id]
                        
                        # 初始化或更新强度
                        if "strength" not in rel_meta:
                            rel_meta["strength"] = 0.5  # 初始默认强度
                            rel_meta["co_occurrence"] = 0
                            rel_meta["last_updated"] = time.time()
                        
                        # 更新共现次数和强度
                        rel_meta["co_occurrence"] += 1
                        
                        # 计算新强度 - 基于交互类型不同权重
                        interaction_weight = {
                            "view": 0.01,     # 简单查看，微弱增强
                            "edit": 0.03,     # 编辑，适度增强
                            "query": 0.02,    # 查询，中等增强
                            "create": 0.05,   # 创建，强增强
                            "analyze": 0.04   # 分析，较强增强
                        }.get(interaction_type, 0.01)
                        
                        # 增强强度，但有上限
                        old_strength = rel_meta["strength"]
                        new_strength = min(1.0, old_strength + interaction_weight)
                        rel_meta["strength"] = new_strength
                        rel_meta["last_updated"] = time.time()
                        
                        # 记录变化
                        if new_strength > old_strength:
                            result["strengthened_relations"] += 1
                            result["relation_strength_changes"][active_id] = {
                                "old": old_strength,
                                "new": new_strength,
                                "change": new_strength - old_strength
                            }
                    else:
                        # 没有现有关系，但共现频繁，可能需要创建新关系
                        
                        # 计算相似度
                        similarity = self._calculate_similarity(
                            self.concept_vectors[concept_id],
                            self.concept_vectors[active_id]
                        )
                        
                        # 根据相似度和上下文决定是否创建新关系
                        co_occurrence_weight = context.get("co_occurrence_weight", 0.7)
                        relation_threshold = self.config["similarity_threshold"] * (1 - co_occurrence_weight)
                        
                        if similarity > relation_threshold:
                            # 确定关系类型
                            relation_type = self._determine_relation_type_with_context(
                                concept_id, active_id, similarity, context
                            )
                            
                            # 创建新的关系
                            self.add_relation(concept_id, active_id, relation_type)
                            
                            # 设置初始关系强度
                            initial_strength = min(0.5 + similarity * 0.3, 0.7)  # 初始强度基于相似度
                            self.relation_metadata[concept_id][active_id]["strength"] = initial_strength
                            self.relation_metadata[concept_id][active_id]["co_occurrence"] = 1
                            self.relation_metadata[concept_id][active_id]["last_updated"] = time.time()
                            self.relation_metadata[concept_id][active_id]["created_by"] = "dynamic_update"
                            
                            result["new_relations"] += 1
                            result["relation_strength_changes"][active_id] = {
                                "old": 0.0,
                                "new": initial_strength,
                                "change": initial_strength
                            }
            
            # 弱化长期未使用的关系
            current_time = time.time()
            decay_threshold = context.get("decay_threshold", 30 * 24 * 60 * 60)  # 默认30天
            decay_rate = context.get("decay_rate", 0.05)  # 默认衰减率
            
            if hasattr(self, "relation_metadata"):
                for related_id in list(self.relation_metadata[concept_id].keys()):
                    rel_meta = self.relation_metadata[concept_id][related_id]
                    
                    if "last_updated" in rel_meta:
                        time_since_update = current_time - rel_meta["last_updated"]
                        
                        # 如果长时间未更新，降低强度
                        if time_since_update > decay_threshold:
                            old_strength = rel_meta["strength"]
                            decay_amount = decay_rate * (time_since_update / decay_threshold)
                            new_strength = max(0.1, old_strength - decay_amount)  # 不低于0.1
                            
                            rel_meta["strength"] = new_strength
                            
                            if new_strength < old_strength:
                                result["weakened_relations"] += 1
                                result["relation_strength_changes"][related_id] = {
                                    "old": old_strength,
                                    "new": new_strength,
                                    "change": new_strength - old_strength
                                }
                            
                            # 如果关系强度太低，考虑移除
                            if new_strength <= 0.15 and rel_meta.get("co_occurrence", 0) < 3:
                                # 获取关系类型
                                rel_type = self._get_relation_type(concept_id, related_id)
                                if rel_type:
                                    self._remove_relation(concept_id, related_id, rel_type)
                                    result["removed_relations"] += 1
            
            # 推断潜在的新关系，基于知识图谱结构分析
            new_inferred_relations = self._infer_potential_relations_dynamic(concept_id, context)
            result["inferred_relations"] = len(new_inferred_relations)
            
            # 更新概念的使用统计
            self.concept_metadata[concept_id]["usage_count"] += 1
            self.concept_metadata[concept_id]["last_accessed"] = current_time
            
            if "access_patterns" not in self.concept_metadata[concept_id]:
                self.concept_metadata[concept_id]["access_patterns"] = defaultdict(int)
                
            self.concept_metadata[concept_id]["access_patterns"][interaction_type] += 1
            
            # 如果有领域重点，记录跨领域使用
            if domain_focus != concept.get("domain", "general"):
                if "cross_domain_usage" not in self.concept_metadata[concept_id]:
                    self.concept_metadata[concept_id]["cross_domain_usage"] = defaultdict(int)
                    
                self.concept_metadata[concept_id]["cross_domain_usage"][domain_focus] += 1
                
                # 如果跨领域使用频繁，考虑添加多领域标签
                if self.concept_metadata[concept_id]["cross_domain_usage"][domain_focus] > 5:
                    if "multi_domain" not in concept:
                        concept["multi_domain"] = []
                        
                    if domain_focus not in concept["multi_domain"]:
                        concept["multi_domain"].append(domain_focus)
                        self.logger.info(f"概念 {concept_id} 添加了多领域标签: {domain_focus}")
        except Exception as e:
            self.logger.error(f"动态更新概念关系出错: {str(e)}")
            result["status"] = "error"
            result["message"] = f"更新关系时发生错误: {str(e)}"
        
        return result
        
    def _infer_potential_relations_dynamic(self, concept_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于知识图谱结构动态推断潜在的关系
        
        Args:
            concept_id: 概念ID
            context: 上下文信息
            
        Returns:
            List[Dict[str, Any]]: 推断的关系列表
        """
        inferred_relations = []
        
        try:
            # 获取概念的一阶和二阶邻居
            first_order = set()
            for related_id, rel_type in self._find_relations(concept_id):
                first_order.add(related_id)
                
            # 对于每个一阶邻居，查找其邻居（即二阶邻居）
            second_order = set()
            for neighbor in first_order:
                for related_id, rel_type in self._find_relations(neighbor):
                    if related_id != concept_id and related_id not in first_order:
                        second_order.add(related_id)
            
            # 分析二阶邻居，寻找可能的直接关系
            for second_neighbor in second_order:
                # 计算与中心概念的相似度
                similarity = self._calculate_similarity(
                    self.concept_vectors[concept_id],
                    self.concept_vectors[second_neighbor]
                )
                
                # 获取通向该二阶邻居的所有路径
                paths = []
                for first_neighbor in first_order:
                    rel_type1 = self._get_relation_type(concept_id, first_neighbor)
                    rel_type2 = self._get_relation_type(first_neighbor, second_neighbor)
                    
                    if rel_type1 and rel_type2:
                        paths.append((first_neighbor, rel_type1, rel_type2))
                
                # 判断是否需要创建直接关系
                if similarity > self.config["similarity_threshold"] * 0.8 and paths:
                    # 尝试推断关系类型
                    inferred_types = []
                    for path in paths:
                        _, rel_type1, rel_type2 = path
                        inferred = self._infer_transitive_relation(rel_type1, rel_type2)
                        if inferred:
                            inferred_types.append(inferred)
                    
                    # 使用最常见的推断类型
                    if inferred_types:
                        relation_counts = defaultdict(int)
                        for rel_type in inferred_types:
                            relation_counts[rel_type] += 1
                            
                        most_common = max(relation_counts.items(), key=lambda x: x[1])
                        inferred_type = most_common[0]
                        
                        # 创建推断的关系
                        self.add_relation(concept_id, second_neighbor, inferred_type, 0.7)
                        
                        # 添加关系元数据
                        if not hasattr(self, "relation_metadata"):
                            self.relation_metadata = defaultdict(lambda: defaultdict(dict))
                            
                        self.relation_metadata[concept_id][second_neighbor] = {
                            "strength": 0.6,  # 初始强度适中
                            "co_occurrence": 0,
                            "last_updated": time.time(),
                            "created_by": "inference",
                            "inference_paths": paths,
                            "inferred_from_types": inferred_types
                        }
                        
                        # 记录推断的关系
                        inferred_relations.append({
                            "source": concept_id,
                            "target": second_neighbor,
                            "relation_type": inferred_type,
                            "inference_paths": paths,
                            "similarity": similarity
                        })
            
            # 根据上下文寻找可能的全新关系
            if context and "semantic_relevance" in context:
                semantic_candidates = context["semantic_relevance"]
                
                for candidate_id, relevance in semantic_candidates.items():
                    if (candidate_id not in first_order and 
                        candidate_id not in second_order and 
                        candidate_id != concept_id and
                        candidate_id in self.concepts):
                        
                        # 评估相关性是否足够高
                        if relevance > 0.7:
                            # 计算实际相似度
                            similarity = self._calculate_similarity(
                                self.concept_vectors[concept_id],
                                self.concept_vectors[candidate_id]
                            )
                            
                            # 如果相似度也足够高，创建关系
                            if similarity > self.config["similarity_threshold"] * 0.75:
                                # 确定关系类型
                                relation_type = self._determine_relation_type_with_context(
                                    concept_id, candidate_id, similarity, context
                                )
                                
                                # 创建关系
                                self.add_relation(concept_id, candidate_id, relation_type, 0.65)
                                
                                # 添加关系元数据
                                if not hasattr(self, "relation_metadata"):
                                    self.relation_metadata = defaultdict(lambda: defaultdict(dict))
                                    
                                self.relation_metadata[concept_id][candidate_id] = {
                                    "strength": 0.5,  # 初始强度中等
                                    "co_occurrence": 0,
                                    "last_updated": time.time(),
                                    "created_by": "semantic_relevance",
                                    "relevance_score": relevance
                                }
                                
                                # 记录创建的关系
                                inferred_relations.append({
                                    "source": concept_id,
                                    "target": candidate_id,
                                    "relation_type": relation_type,
                                    "relevance": relevance,
                                    "similarity": similarity
                                })
        except Exception as e:
            self.logger.error(f"推断潜在关系时发生错误: {str(e)}")
            
        return inferred_relations
    
    def _determine_relation_type_with_context(self, concept_id1: str, concept_id2: str, 
                                         similarity: float, context: Dict[str, Any]) -> str:
        """
        基于上下文增强的关系类型判断
        
        Args:
            concept_id1: 第一个概念ID
            concept_id2: 第二个概念ID
            similarity: 概念间相似度
            context: 上下文信息
            
        Returns:
            str: 关系类型
        """
        # 基础关系判断
        basic_relation = self._determine_relation_type(concept_id1, concept_id2, similarity)
        
        # 如果没有上下文，使用基础判断
        if not context:
            return basic_relation
            
        concept1 = self.concepts[concept_id1]
        concept2 = self.concepts[concept_id2]
        
        # 提取上下文信息
        interaction_type = context.get("interaction_type", "view")
        recent_relations = context.get("recent_relations", [])
        shared_context = context.get("shared_context", {})
        
        # 分析两个概念是否可能有特定的关系模式
        if interaction_type == "analyze" or interaction_type == "compare":
            # 可能是比较或分析关系
            if concept1.get("type") == concept2.get("type"):
                return "compare_with"
                
        # 检查时序关系
        if "temporal_sequence" in context:
            sequence = context["temporal_sequence"]
            if concept_id1 in sequence and concept_id2 in sequence:
                idx1 = sequence.index(concept_id1)
                idx2 = sequence.index(concept_id2)
                
                if idx1 < idx2:
                    return "precedes"
                elif idx1 > idx2:
                    return "follows"
        
        # 检查领域关系
        domain1 = concept1.get("domain", "general")
        domain2 = concept2.get("domain", "general")
        
        if domain1 != domain2:
            # 跨领域概念
            if "is_analogy" in shared_context and shared_context["is_analogy"]:
                return "analogous_to"
                
        # 检查概念类型相关的关系
        type1 = concept1.get("type", "generic")
        type2 = concept2.get("type", "generic")
        
        if type1 == "process" and type2 == "resource":
            return "uses"
        elif type1 == "problem" and type2 == "solution":
            return "solved_by"
        elif type1 == "entity" and type2 == "attribute":
            return "has_property"
        
        # 使用最近的关系模式
        if recent_relations and len(recent_relations) >= 3:
            # 查找最常见的关系类型
            relation_counts = defaultdict(int)
            for rel in recent_relations:
                relation_counts[rel] += 1
                
            most_common = max(relation_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 2:  # 至少出现两次
                return most_common[0]
        
        # 默认使用基础关系判断
        return basic_relation
    
    def update_relation_strength(self, source_id: str, target_id: str, 
                              strength_change: float, reason: str = "manual", 
                              usage_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        增强版关系强度更新，基于使用情况实时调整关系强度
        
        Args:
            source_id: 源概念ID
            target_id: 目标概念ID
            strength_change: 强度变化量(正值增强，负值减弱)
            reason: 更新原因
            usage_context: 使用上下文信息
            
        Returns:
            Dict: 更新结果
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return {
                "status": "error",
                "message": "源概念或目标概念不存在"
            }
            
        # 获取现有关系类型
        relation_type = self._get_relation_type(source_id, target_id)
        
        if not relation_type:
            return {
                "status": "error",
                "message": "概念间没有现有关系"
            }
            
        # 确保关系元数据存在
        if not hasattr(self, "relation_metadata"):
            self.relation_metadata = defaultdict(lambda: defaultdict(dict))
            
        # 获取或初始化关系元数据
        if not self.relation_metadata[source_id].get(target_id):
            self.relation_metadata[source_id][target_id] = {
                "strength": 0.5,  # 默认强度
                "co_occurrence": 0,
                "last_updated": time.time(),
                "creation_time": time.time(),
                "update_count": 0,
                "usage_patterns": defaultdict(int)
            }
    
        metadata = self.relation_metadata[source_id][target_id]
    
        # 更新使用模式统计
        if usage_context:
            # 记录使用场景
            context_type = usage_context.get("context_type", "general")
            metadata["usage_patterns"][context_type] += 1
            
            # 记录使用频率
            if "usage_frequency" not in metadata:
                metadata["usage_frequency"] = []
                
            metadata["usage_frequency"].append(time.time())
            
            # 只保留最近100次使用记录
            if len(metadata["usage_frequency"]) > 100:
                metadata["usage_frequency"] = metadata["usage_frequency"][-100:]
                
            # 基于使用上下文调整强度变化
            if context_type == "critical_operation":
                # 关键操作，增强变化
                strength_change *= 1.5
            elif context_type == "exploratory":
                # 探索性使用，减弱变化
                strength_change *= 0.7
            
            # 基于使用频率调整
            recent_time = 30 * 24 * 60 * 60  # 30天
            current_time = time.time()
            recent_usages = [t for t in metadata["usage_frequency"] if current_time - t < recent_time]
            
            # 使用频率增强
            frequency_factor = min(2.0, 1.0 + len(recent_usages) / 50.0)
            strength_change *= frequency_factor
    
        # 更新强度
        old_strength = metadata.get("strength", 0.5)
    
        # 应用非线性调整 - 强度越高越难增加，越低越难减少
        if strength_change > 0:
            # 增强时，随着强度增高变得更难增加
            adjustment_factor = 1.0 - (old_strength ** 2) * 0.5
            adjusted_change = strength_change * adjustment_factor
        else:
            # 减弱时，随着强度降低变得更难减少
            adjustment_factor = 1.0 - ((1.0 - old_strength) ** 2) * 0.5
            adjusted_change = strength_change * adjustment_factor
    
        new_strength = max(0.1, min(1.0, old_strength + adjusted_change))  # 保持在0.1-1.0范围内
    
        metadata["strength"] = new_strength
        metadata["last_updated"] = time.time()
        metadata["update_count"] += 1
    
        if "update_history" not in metadata:
            metadata["update_history"] = []
            
        # 记录更新历史
        metadata["update_history"].append({
            "timestamp": time.time(),
            "old_strength": old_strength,
            "new_strength": new_strength,
            "raw_change": strength_change,
            "adjusted_change": adjusted_change,
            "reason": reason,
            "context": usage_context.get("context_type", "none") if usage_context else "none"
        })
    
        # 限制历史记录长度
        if len(metadata["update_history"]) > 20:
            metadata["update_history"] = metadata["update_history"][-20:]
    
        # 更新关系元数据中的稳定性指标
        stability = self._calculate_relation_stability(metadata)
        metadata["stability"] = stability
    
        # 如果强度过低，考虑移除关系
        if new_strength <= 0.15:
            should_remove = False
            
            # 检查关系的稳定性、共现次数和历史
            co_occurrence = metadata.get("co_occurrence", 0)
            history_length = len(metadata.get("update_history", []))
        
            # 低强度 + 低稳定性 + 低共现 + 持续减弱 = 移除关系
            if stability < 0.3 and co_occurrence < 3 and history_length > 2:
                recent_changes = [h["adjusted_change"] for h in metadata["update_history"][-3:]]
                if all(change <= 0 for change in recent_changes):
                    should_remove = True
        
            if should_remove:
                self._remove_relation(source_id, target_id, relation_type)
                
                # 记录移除原因
                if not hasattr(self, "removed_relations_log"):
                    self.removed_relations_log = []
                    
                self.removed_relations_log.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "removed_at": time.time(),
                    "final_strength": new_strength,
                    "stability": stability,
                    "co_occurrence": co_occurrence,
                    "reason": "low_strength_unstable"
                })
                
                return {
                    "status": "success",
                    "old_strength": old_strength,
                    "new_strength": 0,
                    "relation_removed": True,
                    "removal_reason": "low_strength_unstable"
                }
    
        # 更新关系的置信度以反映强度
        # 找到并更新关系的置信度值
        updated = False
        if source_id in self.relations:
            updated_relations = set()
            for rel in self.relations[source_id]:
                if rel[0] == target_id and rel[1] == relation_type:
                    # 更新置信度为新强度
                    updated_relations.add((target_id, relation_type, new_strength))
                    updated = True
                else:
                    updated_relations.add(rel)
            
            if updated:
                self.relations[source_id] = updated_relations
    
        return {
            "status": "success",
            "old_strength": old_strength,
            "new_strength": new_strength,
            "raw_change": strength_change,
            "adjusted_change": adjusted_change,
            "stability": stability,
            "update_count": metadata["update_count"]
        }

    def _calculate_relation_stability(self, relation_metadata: Dict[str, Any]) -> float:
        """
        计算关系稳定性
        
        Args:
            relation_metadata: 关系元数据
            
        Returns:
            float: 稳定性评分(0-1)
        """
        stability = 0.5  # 默认中等稳定性
    
        # 检查历史更新
        history = relation_metadata.get("update_history", [])
        if not history:
            return stability
    
        # 1. 计算强度变化的一致性
        if len(history) >= 3:
            changes = [entry["adjusted_change"] for entry in history[-3:]]
            # 检查变化方向是否一致
            if all(change > 0 for change in changes) or all(change < 0 for change in changes):
                # 一致的变化表示更稳定的趋势
                stability += 0.1
            elif all(abs(change) < 0.05 for change in changes):
                # 微小变化表示稳定
                stability += 0.2
            else:
                # 变化不一致表示不稳定
                stability -= 0.1
    
        # 2. 考虑关系年龄
        if "creation_time" in relation_metadata:
            age_days = (time.time() - relation_metadata["creation_time"]) / (24 * 60 * 60)
            # 随着关系年龄增长，基础稳定性提高
            age_factor = min(0.3, age_days / 30.0 * 0.3)  # 最多贡献0.3分
            stability += age_factor
    
        # 3. 考虑使用频率
        usage_frequency = relation_metadata.get("usage_frequency", [])
        if usage_frequency:
            # 计算30天内的使用次数
            recent_time = 30 * 24 * 60 * 60  # 30天
            current_time = time.time()
            recent_usages = [t for t in usage_frequency if current_time - t < recent_time]
            
            # 频繁使用的关系更稳定
            usage_factor = min(0.2, len(recent_usages) / 20.0 * 0.2)  # 最多贡献0.2分
            stability += usage_factor
    
        # 4. 考虑使用场景多样性
        usage_patterns = relation_metadata.get("usage_patterns", {})
        if usage_patterns:
            diversity = len(usage_patterns) / 5.0  # 假设最多5种使用场景
            diversity_factor = min(0.2, diversity * 0.2)  # 最多贡献0.2分
            stability += diversity_factor
    
        # 确保稳定性在0-1范围内
        return max(0.0, min(1.0, stability))
    
    def enhanced_cross_domain_knowledge_transfer(self, source_domain: str, target_domain: str, 
                                          adaptation_level: str = "moderate", 
                                          concept_types: List[str] = None,
                                          mapping_strategy: str = "similarity",
                                          preserve_structure: bool = True) -> Dict[str, Any]:
        """
        增强版跨领域知识迁移，具有更高级的知识适应和映射能力
        
        Args:
            source_domain: 源知识领域
            target_domain: 目标知识领域
            adaptation_level: 适应级别(minimal/moderate/aggressive)
            concept_types: 要迁移的概念类型列表，为None时迁移所有类型
            mapping_strategy: 概念映射策略(similarity/structural/hybrid)
            preserve_structure: 是否保留源领域的知识结构
            
        Returns:
            Dict: 迁移结果
        """
        result = {
            "status": "success",
            "transferred_concepts": [],
            "adapted_concepts": [],
            "created_analogies": [],
            "created_mappings": [],
            "preserved_structures": [],
            "adaptation_metadata": {},
            "stats": {
                "total_transferred": 0,
                "total_adapted": 0,
                "analogies_created": 0,
                "cross_mappings": 0,
                "adaptation_conflicts": 0,
                "ignored_concepts": 0,
                "structure_elements_preserved": 0
            }
        }
        
        try:
            # 获取源领域概念
            source_concepts = self._get_domain_concepts(source_domain)
            
            if not source_concepts:
                return {
                    "status": "error",
                    "message": f"源领域 {source_domain} 不包含任何概念"
                }
                
            # 获取目标领域概念
            target_concepts = self._get_domain_concepts(target_domain)
            
            # 1. 根据概念类型过滤
            if concept_types:
                filtered_source = {}
                for cid, concept in source_concepts.items():
                    if concept.get("type") in concept_types:
                        filtered_source[cid] = concept
                source_concepts = filtered_source
            
            if not source_concepts:
                return {
                    "status": "error",
                    "message": f"源领域 {source_domain} 没有匹配指定类型的概念"
                }
            
            # 2. 分析概念兼容性和迁移优先级
            transfer_candidates = self._analyze_transfer_compatibility(source_concepts, target_domain, target_concepts)
            
            # 3. 按优先级排序
            sorted_candidates = sorted(transfer_candidates.items(), key=lambda x: x[1].get("priority", 0), reverse=True)
            
            # 4. 执行知识迁移
            for concept_id, transfer_info in sorted_candidates:
                source_concept = source_concepts[concept_id]
                
                # 检查目标领域是否已有相似概念
                conflict = transfer_info.get("conflict")
                if conflict:
                    # 存在冲突，可能需要合并或适应
                    if adaptation_level == "minimal":
                        # 最小适应级别: 跳过冲突的概念
                        result["stats"]["ignored_concepts"] += 1
                        continue
                    
                    # 创建目标领域的适应版本
                    adapted_concept = self._create_adapted_concept(
                        source_concept, 
                        target_domain, 
                        transfer_info,
                        conflict,
                        adaptation_level
                    )
                    
                    # 添加到知识库
                    add_result = self.add_concept(adapted_concept)
                    
                    if add_result["status"] == "success":
                        result["adapted_concepts"].append({
                            "original_id": concept_id,
                            "adapted_id": add_result["concept_id"],
                            "conflict_id": conflict["concept_id"],
                            "adaptation_level": adaptation_level
                        })
                        result["stats"]["total_adapted"] += 1
                        
                        # 创建与冲突概念的映射关系
                        self.add_relation(
                            add_result["concept_id"], 
                            conflict["concept_id"], 
                            "domain_adapted_version_of",
                            0.9
                        )
                        
                        # 创建与源概念的关系
                        self.add_relation(
                            add_result["concept_id"], 
                            concept_id, 
                            "adapted_from",
                            0.95
                        )
                        
                        result["created_mappings"].append({
                            "source": add_result["concept_id"],
                            "target": conflict["concept_id"],
                            "type": "domain_adapted_version_of"
                        })
                        
                        result["stats"]["cross_mappings"] += 1
                    else:
                        result["stats"]["adaptation_conflicts"] += 1
                else:
                    # 没有冲突，直接迁移概念
                    transferred_concept = self._create_domain_adapted_concept(source_concept, target_domain)
                    
                    # 添加到知识库
                    add_result = self.add_concept(transferred_concept)
                    
                    if add_result["status"] == "success":
                        result["transferred_concepts"].append({
                            "original_id": concept_id,
                            "transferred_id": add_result["concept_id"]
                        })
                        result["stats"]["total_transferred"] += 1
                        
                        # 创建与源概念的关系
                        self.add_relation(
                            add_result["concept_id"], 
                            concept_id, 
                            "cross_domain_equivalent",
                            0.9
                        )
            
            # 5. 创建领域间类比
            analogies = self._create_domain_analogies(source_domain, target_domain)
            
            result["created_analogies"] = analogies
            result["stats"]["analogies_created"] = len(analogies)
            
            # 6. 保留知识结构（如果需要）
            if preserve_structure:
                structure_results = self._preserve_domain_structure(source_domain, target_domain, result)
                result["preserved_structures"] = structure_results["preserved_structures"]
                result["stats"]["structure_elements_preserved"] = structure_results["count"]
            
            # 7. 增强目标领域知识结构 
            self._enhance_domain_structure(target_domain)
            
            # 8. 记录适应过程元数据
            result["adaptation_metadata"] = {
                "source_domain_size": len(self._get_domain_concepts(source_domain)),
                "target_domain_size_before": len(target_concepts),
                "target_domain_size_after": len(self._get_domain_concepts(target_domain)),
                "adaptation_level": adaptation_level,
                "mapping_strategy": mapping_strategy,
                "timestamp": time.time(),
                "success_rate": self._calculate_transfer_success_rate(result),
                "adaptation_quality": self._evaluate_adaptation_quality(source_domain, target_domain)
            }
            
        except Exception as e:
            self.logger.error(f"跨领域知识迁移异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            result["status"] = "error"
            result["message"] = f"知识迁移过程中出错: {str(e)}"
            
        return result
        
    def _preserve_domain_structure(self, source_domain: str, target_domain: str, 
                                 transfer_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        保留源领域的知识结构，应用到目标领域的迁移概念上
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            transfer_result: 迁移结果
            
        Returns:
            Dict: 结构保留结果
        """
        result = {
            "preserved_structures": [],
            "count": 0
        }
        
        # 获取源领域的层次结构
        source_hierarchies = []
        for concept_id, concept in self.concepts.items():
            if concept.get("domain") == source_domain and concept.get("is_hierarchy_root", False):
                hierarchy = self._get_hierarchy_structure(concept_id)
                if hierarchy:
                    source_hierarchies.append(hierarchy)
        
        # 创建概念ID映射（源领域ID -> 目标领域ID）
        concept_mapping = {}
        
        # 从迁移的概念中提取映射关系
        for transfer in transfer_result["transferred_concepts"]:
            concept_mapping[transfer["original_id"]] = transfer["transferred_id"]
            
        for adaptation in transfer_result["adapted_concepts"]:
            concept_mapping[adaptation["original_id"]] = adaptation["adapted_id"]
        
        # 遍历每个源领域层次结构
        for hierarchy in source_hierarchies:
            # 创建目标领域对应的层次结构
            mapped_structure = self._map_hierarchy_structure(hierarchy, concept_mapping, target_domain)
            
            if mapped_structure and mapped_structure["mapped_nodes"] > 0:
                # 在目标领域创建层次结构
                root_id = mapped_structure["root_id"]
                structure = mapped_structure["structure"]
                
                if root_id and root_id in self.concepts:
                    # 创建层次结构
                    self.create_hierarchy(root_id, structure)
                    
                    # 记录保留的结构
                    result["preserved_structures"].append({
                        "source_root": hierarchy["root_id"],
                        "target_root": root_id,
                        "node_count": mapped_structure["mapped_nodes"],
                        "mapping_coverage": mapped_structure["coverage"]
                    })
                    
                    result["count"] += mapped_structure["mapped_nodes"]
        
        # 保留重要关系结构
        preserved_relations = self._preserve_relation_patterns(source_domain, target_domain, concept_mapping)
        result["count"] += preserved_relations
        
        return result
        
    def _map_hierarchy_structure(self, hierarchy: Dict[str, Any], 
                              concept_mapping: Dict[str, str],
                              target_domain: str) -> Dict[str, Any]:
        """
        将源领域的层次结构映射到目标领域
        
        Args:
            hierarchy: 源层次结构
            concept_mapping: 概念ID映射
            target_domain: 目标领域
            
        Returns:
            Dict: 映射的层次结构
        """
        # 初始化结果
        result = {
            "root_id": None,
            "structure": {},
            "mapped_nodes": 0,
            "total_nodes": 0,
            "coverage": 0.0
        }
        
        source_root_id = hierarchy["root_id"]
        
        # 检查根概念是否有映射
        if source_root_id not in concept_mapping:
            return result
            
        result["root_id"] = concept_mapping[source_root_id]
        result["total_nodes"] = hierarchy["total_nodes"]
        
        # 递归映射层次结构
        def map_structure(source_structure):
            mapped = {}
            node_count = 0
            
            # 处理子节点
            if "children" in source_structure:
                mapped["children"] = []
                
                for child in source_structure["children"]:
                    child_id = child["id"]
                    
                    # 检查子概念是否有映射
                    if child_id in concept_mapping:
                        mapped_child = {
                            "id": concept_mapping[child_id],
                            "name": self.concepts[concept_mapping[child_id]].get("name", "")
                        }
                        
                        # 递归处理子节点的子节点
                        if "children" in child:
                            child_result = map_structure(child)
                            if child_result["children"]:
                                mapped_child["children"] = child_result["children"]
                            node_count += child_result["count"]
                        
                        mapped["children"].append(mapped_child)
                        node_count += 1
            
            return {"children": mapped.get("children", []), "count": node_count}
        
        # 映射整个结构
        structure_result = map_structure(hierarchy["structure"])
        result["structure"] = {"children": structure_result["children"]}
        result["mapped_nodes"] = structure_result["count"] + 1  # +1 表示根节点
        
        # 计算覆盖率
        if result["total_nodes"] > 0:
            result["coverage"] = result["mapped_nodes"] / result["total_nodes"]
            
        return result
        
    def _preserve_relation_patterns(self, source_domain: str, target_domain: str, 
                                concept_mapping: Dict[str, str]) -> int:
        """
        保留源领域中的重要关系模式
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            concept_mapping: 概念ID映射
            
        Returns:
            int: 保留的关系数量
        """
        preserved_count = 0
        
        # 遍历源领域中的每个概念关系
        for source_id, relations in self.relations.items():
            # 确认源概念属于源领域并且已映射到目标领域
            source_concept = self.concepts.get(source_id)
            if not source_concept or source_concept.get("domain") != source_domain or source_id not in concept_mapping:
                continue
                
            mapped_source_id = concept_mapping[source_id]
            
            # 处理每个关系
            for relation in relations:
                target_id, rel_type, confidence = relation
                
                # 确认目标概念属于源领域并且已映射到目标领域
                target_concept = self.concepts.get(target_id)
                if not target_concept or target_concept.get("domain") != source_domain or target_id not in concept_mapping:
                    continue
                    
                mapped_target_id = concept_mapping[target_id]
                
                # 检查目标领域中是否已存在此关系
                existing_relation = False
                if mapped_source_id in self.relations:
                    for rel in self.relations[mapped_source_id]:
                        if rel[0] == mapped_target_id and rel[1] == rel_type:
                            existing_relation = True
                            break
                
                # 如果关系不存在，则创建
                if not existing_relation:
                    self.add_relation(mapped_source_id, mapped_target_id, rel_type, confidence)
                    preserved_count += 1
        
        return preserved_count
        
    def _calculate_transfer_success_rate(self, result: Dict[str, Any]) -> float:
        """
        计算知识迁移的成功率
        
        Args:
            result: 迁移结果
            
        Returns:
            float: 成功率(0-1)
        """
        stats = result["stats"]
        total_attempted = stats["total_transferred"] + stats["total_adapted"] + stats["ignored_concepts"]
        
        if total_attempted == 0:
            return 0.0
            
        successful = stats["total_transferred"] + stats["total_adapted"]
        return successful / total_attempted
        
    def _evaluate_adaptation_quality(self, source_domain: str, target_domain: str) -> float:
        """
        评估适应质量
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            
        Returns:
            float: 适应质量评分(0-1)
        """
        # 获取领域概念
        source_concepts = self._get_domain_concepts(source_domain)
        target_concepts = self._get_domain_concepts(target_domain)
        
        # 如果目标领域为空，无法评估质量
        if not target_concepts:
            return 0.0
            
        # 计算结构一致性
        structural_coherence = self._evaluate_domain_structural_coherence(target_domain)
        
        # 计算领域特征保留度
        domain_feature_retention = self._evaluate_domain_feature_retention(
            source_domain, target_domain
        )
        
        # 综合评分
        quality = 0.5 * structural_coherence + 0.5 * domain_feature_retention
        return min(1.0, max(0.0, quality))
        
    def _evaluate_domain_structural_coherence(self, domain: str) -> float:
        """评估领域的结构一致性"""
        domain_concepts = self._get_domain_concepts(domain)
        
        if not domain_concepts:
            return 0.0
            
        # 计算领域内部关系密度
        relation_count = 0
        for concept_id in domain_concepts:
            if concept_id in self.relations:
                for rel in self.relations[concept_id]:
                    if rel[0] in domain_concepts:
                        relation_count += 1
        
        # 计算最大可能关系数
        max_relations = len(domain_concepts) * (len(domain_concepts) - 1)
        
        if max_relations == 0:
            return 0.0
            
        # 理想密度为0.1-0.3，过高或过低都不好
        density = relation_count / max_relations
        
        if density < 0.05:
            return density * 10  # 低于理想范围
        elif density <= 0.3:
            return min(1.0, density / 0.3)  # 理想范围内
        else:
            return max(0.3, 1.0 - (density - 0.3))  # 高于理想范围
        
    def _evaluate_domain_feature_retention(self, source_domain: str, target_domain: str) -> float:
        """评估目标领域保留源领域特征的程度"""
        source_concepts = self._get_domain_concepts(source_domain)
        target_concepts = self._get_domain_concepts(target_domain)
        
        if not source_concepts or not target_concepts:
            return 0.0
            
        # 获取源领域的主要特征
        source_attributes = set()
        for concept in source_concepts.values():
            if "attributes" in concept:
                source_attributes.update(concept["attributes"].keys())
                
        if not source_attributes:
            return 0.5  # 没有特征可以比较
        
        # 检查目标领域保留了多少特征
        target_attributes = set()
        for concept in target_concepts.values():
            if "attributes" in concept:
                target_attributes.update(concept["attributes"].keys())
        
        # 计算特征保留率
        common_attributes = source_attributes.intersection(target_attributes)
        retention_rate = len(common_attributes) / len(source_attributes)
        
        return retention_rate
        
    def _analyze_transfer_compatibility(self, source_concepts: Dict[str, Dict[str, Any]], 
                                      target_domain: str, 
                                      target_concepts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        分析概念迁移兼容性和优先级
        
        Args:
            source_concepts: 源领域概念
            target_domain: 目标领域
            target_concepts: 目标领域概念
            
        Returns:
            Dict: 概念迁移分析结果
        """
        transfer_info = {}
        
        for concept_id, concept in source_concepts.items():
            # 初始化
            transfer_info[concept_id] = {
                "priority": 0.5,  # 默认中等优先级
                "conflict": None,
                "compatibility": 1.0,
                "needed_adaptations": []
            }
            
            # 检查目标领域的相似概念
            similar_concepts = []
            for target_id, target_concept in target_concepts.items():
                similarity = self._calculate_concept_similarity(concept_id, target_id)
                if similarity > 0.6:  # 相似度阈值
                    similar_concepts.append({
                        "concept_id": target_id,
                        "similarity": similarity,
                        "concept": target_concept
                    })
            
            # 如果存在相似概念，记录冲突
            if similar_concepts:
                # 按相似度排序
                similar_concepts.sort(key=lambda x: x["similarity"], reverse=True)
                transfer_info[concept_id]["conflict"] = similar_concepts[0]
                
                # 降低优先级
                conflict_similarity = similar_concepts[0]["similarity"]
                transfer_info[concept_id]["priority"] = 0.3 + (1.0 - conflict_similarity) * 0.5
                
                # 分析需要适应的字段
                transfer_info[concept_id]["needed_adaptations"] = self._analyze_adaptation_needs(
                    concept, 
                    similar_concepts[0]["concept"]
                )
            else:
                # 没有冲突，分析概念与目标领域的兼容性
                domain_compatibility = self._analyze_domain_compatibility(concept, target_domain)
                transfer_info[concept_id]["compatibility"] = domain_compatibility
                
                # 兼容性高的概念优先级提高
                transfer_info[concept_id]["priority"] = 0.5 + domain_compatibility * 0.5
                
                # 根据概念的完整性和重要性调整优先级
                completeness = self._calculate_concept_completeness(concept)
                importance = concept.get("importance", 0.5)
                
                # 完整性和重要性高的概念优先迁移
                priority_boost = (completeness * 0.5 + importance * 0.5) * 0.2
                transfer_info[concept_id]["priority"] += priority_boost
        
        return transfer_info
        
    def _analyze_adaptation_needs(self, source_concept: Dict[str, Any], 
                                target_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析概念适应需求
        
        Args:
            source_concept: 源概念
            target_concept: 目标概念
            
        Returns:
            List: 需要适应的字段列表
        """
        adaptation_needs = []
        
        # 比较关键字段
        key_fields = ["name", "description", "type", "features", "attributes", "behavior"]
        
        for field in key_fields:
            if field in source_concept and field in target_concept:
                source_value = source_concept[field]
                target_value = target_concept[field]
                
                if isinstance(source_value, dict) and isinstance(target_value, dict):
                    # 比较字典类型字段
                    diff_keys = set(source_value.keys()) - set(target_value.keys())
                    conflict_keys = []
                    
                    for k in set(source_value.keys()) & set(target_value.keys()):
                        if source_value[k] != target_value[k]:
                            conflict_keys.append(k)
                    
                    if diff_keys or conflict_keys:
                        adaptation_needs.append({
                            "field": field,
                            "type": "dict",
                            "different_keys": list(diff_keys),
                            "conflicting_keys": conflict_keys,
                            "severity": 0.5 if diff_keys else 0.8 if conflict_keys else 0.2
                        })
                elif isinstance(source_value, list) and isinstance(target_value, list):
                    # 比较列表类型字段
                    diff_items = set(str(x) for x in source_value) - set(str(x) for x in target_value)
                    if diff_items:
                        adaptation_needs.append({
                            "field": field,
                            "type": "list",
                            "different_items": len(diff_items),
                            "severity": min(1.0, len(diff_items) / max(1, len(source_value)))
                        })
                elif source_value != target_value:
                    # 比较其他类型字段
                    adaptation_needs.append({
                        "field": field,
                        "type": "value",
                        "source_value": source_value,
                        "target_value": target_value,
                        "severity": 0.7  # 默认值差异的严重程度
                    })
                    
        # 检查额外字段
        source_fields = set(source_concept.keys())
        target_fields = set(target_concept.keys())
        
        extra_fields = source_fields - target_fields
        if extra_fields:
            adaptation_needs.append({
                "field": "structure",
                "type": "extra_fields",
                "fields": list(extra_fields),
                "severity": 0.3
            })
            
        return adaptation_needs
        