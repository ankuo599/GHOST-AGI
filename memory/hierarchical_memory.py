"""
GHOST-AGI 分层记忆与知识整合系统

该模块实现多层次记忆结构，提供高效的知识存储、检索和整合能力。
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict

class HierarchicalMemory:
    """分层记忆系统，提供多层次记忆结构和知识整合能力"""
    
    def __init__(self, 
                 episodic_capacity: int = 1000,
                 semantic_capacity: int = 5000, 
                 working_memory_size: int = 7,
                 logger: Optional[logging.Logger] = None):
        """
        初始化分层记忆系统
        
        Args:
            episodic_capacity: 情景记忆容量
            semantic_capacity: 语义记忆容量
            working_memory_size: 工作记忆大小
            logger: 日志记录器
        """
        # 记忆参数
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = semantic_capacity
        self.working_memory_size = working_memory_size
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 记忆层次结构
        self.working_memory = []   # 工作记忆
        self.episodic_memory = []  # 情景记忆
        self.semantic_memory = {}  # 语义记忆
        self.procedural_memory = {} # 程序记忆
        
        # 知识图谱
        self.knowledge_graph = {
            "entities": {},
            "relations": [],
            "concepts": {}
        }
        
        # 检索索引
        self.memory_indices = {
            "temporal": {},  # 时间索引
            "semantic": {},  # 语义索引
            "entity": {},    # 实体索引
            "spatial": {}    # 空间索引
        }
        
        # 记忆动态属性
        self.activation_levels = {}  # 激活水平
        self.retrieval_counts = defaultdict(int)  # 检索计数
        self.last_access_time = {}  # 最后访问时间
        self.emotional_tags = {}  # 情感标签
        
        # 知识整合参数
        self.integration_threshold = 0.7  # 整合阈值
        self.abstraction_levels = 5  # 抽象层次数
        
        # 统计信息
        self.memory_stats = {
            "total_episodic": 0,
            "total_semantic": 0,
            "total_procedural": 0,
            "retrievals": 0,
            "integrations": 0,
            "last_cleanup": time.time()
        }
        
        self.logger.info("分层记忆系统初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("HierarchicalMemory")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("hierarchical_memory.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def add_to_working_memory(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加项目到工作记忆
        
        Args:
            item: 记忆项目
            
        Returns:
            添加结果
        """
        self.logger.debug("添加项目到工作记忆")
        
        # 确保项目有唯一ID
        if "id" not in item:
            item["id"] = f"wm_{int(time.time())}_{len(self.working_memory)}"
        
        # 确保项目有时间戳
        if "timestamp" not in item:
            item["timestamp"] = time.time()
        
        # 添加到工作记忆
        self.working_memory.append(item)
        
        # 如果超出容量，移除最旧项目
        if len(self.working_memory) > self.working_memory_size:
            removed = self.working_memory.pop(0)
            self._consolidate_to_episodic(removed)  # 将移除的项目整合到情景记忆
        
        # 更新激活水平
        self.activation_levels[item["id"]] = 1.0
        self.last_access_time[item["id"]] = time.time()
        
        return {"status": "success", "item_id": item["id"]}
    
    def _consolidate_to_episodic(self, item: Dict[str, Any]) -> None:
        """将工作记忆项目整合到情景记忆"""
        # 添加情景记忆特有属性
        if "episodic_context" not in item:
            item["episodic_context"] = {
                "time": time.time(),
                "related_items": [i["id"] for i in self.working_memory if i["id"] != item["id"]]
            }
        
        # 重新生成ID
        original_id = item["id"]
        item["id"] = f"ep_{int(time.time())}_{len(self.episodic_memory)}"
        
        # 添加到情景记忆
        self.episodic_memory.append(item)
        self.memory_stats["total_episodic"] += 1
        
        # 更新索引
        self._update_indices(item)
        
        # 更新激活水平和访问时间
        self.activation_levels[item["id"]] = self.activation_levels.pop(original_id, 0.5)
        self.last_access_time[item["id"]] = time.time()
        
        # 检查情景记忆容量
        if len(self.episodic_memory) > self.episodic_capacity:
            self._cleanup_episodic_memory()
    
    def _update_indices(self, item: Dict[str, Any]) -> None:
        """更新记忆索引"""
        item_id = item["id"]
        
        # 时间索引
        timestamp = item.get("timestamp", time.time())
        time_key = int(timestamp / 3600) * 3600  # 按小时分组
        if time_key not in self.memory_indices["temporal"]:
            self.memory_indices["temporal"][time_key] = []
        self.memory_indices["temporal"][time_key].append(item_id)
        
        # 语义索引
        if "tags" in item:
            for tag in item["tags"]:
                if tag not in self.memory_indices["semantic"]:
                    self.memory_indices["semantic"][tag] = []
                self.memory_indices["semantic"][tag].append(item_id)
        
        # 实体索引
        if "entities" in item:
            for entity in item["entities"]:
                entity_id = entity.get("id", entity.get("name", ""))
                if entity_id:
                    if entity_id not in self.memory_indices["entity"]:
                        self.memory_indices["entity"][entity_id] = []
                    self.memory_indices["entity"][entity_id].append(item_id)
    
    def _cleanup_episodic_memory(self) -> None:
        """清理情景记忆，移除不重要的记忆"""
        # 计算每个记忆项的重要性分数
        importance_scores = {}
        current_time = time.time()
        
        for i, item in enumerate(self.episodic_memory):
            item_id = item["id"]
            
            # 计算时间因子（较新的记忆更重要）
            time_factor = 1.0 - min(1.0, (current_time - item.get("timestamp", 0)) / (30 * 24 * 3600))
            
            # 计算访问因子（访问越频繁越重要）
            access_factor = min(1.0, self.retrieval_counts[item_id] / 10.0)
            
            # 计算情感因子（情感标记的记忆更重要）
            emotional_factor = 1.0 if item_id in self.emotional_tags else 0.5
            
            # 计算连接因子（在知识图谱中有更多连接的记忆更重要）
            connection_count = 0
            if "entities" in item:
                for entity in item["entities"]:
                    entity_id = entity.get("id", entity.get("name", ""))
                    if entity_id in self.memory_indices["entity"]:
                        connection_count += len(self.memory_indices["entity"][entity_id])
            connection_factor = min(1.0, connection_count / 20.0)
            
            # 整合分数
            importance = (time_factor * 0.3 + 
                         access_factor * 0.3 + 
                         emotional_factor * 0.2 + 
                         connection_factor * 0.2)
            
            importance_scores[i] = importance
        
        # 按重要性排序
        sorted_indices = sorted(importance_scores.keys(), key=lambda i: importance_scores[i])
        
        # 移除最不重要的记忆
        items_to_remove = max(1, int(self.episodic_capacity * 0.1))  # 移除约10%的记忆
        for idx in sorted_indices[:items_to_remove]:
            item = self.episodic_memory[idx]
            
            # 考虑整合到语义记忆
            self._consider_semantic_integration(item)
            
            # 从索引中移除
            for index_type in self.memory_indices:
                for key in list(self.memory_indices[index_type].keys()):
                    if item["id"] in self.memory_indices[index_type][key]:
                        self.memory_indices[index_type][key].remove(item["id"])
        
        # 移除选定的记忆
        self.episodic_memory = [item for i, item in enumerate(self.episodic_memory) if i not in sorted_indices[:items_to_remove]]
        
        self.memory_stats["last_cleanup"] = time.time()
        self.logger.info(f"清理了 {items_to_remove} 个情景记忆项目")
    
    def _consider_semantic_integration(self, item: Dict[str, Any]) -> None:
        """考虑将情景记忆整合到语义记忆"""
        # 简化实现：检查是否达到检索阈值
        if self.retrieval_counts[item["id"]] >= 3:
            self._integrate_to_semantic(item)
    
    def _integrate_to_semantic(self, item: Dict[str, Any]) -> None:
        """将记忆整合到语义记忆"""
        # 提取关键信息
        concepts = item.get("concepts", [])
        entities = item.get("entities", [])
        relations = item.get("relations", [])
        
        # 整合实体到知识图谱
        for entity in entities:
            entity_id = entity.get("id", entity.get("name", ""))
            if entity_id:
                if entity_id not in self.knowledge_graph["entities"]:
                    self.knowledge_graph["entities"][entity_id] = entity
                else:
                    # 合并属性
                    for key, value in entity.items():
                        if key not in self.knowledge_graph["entities"][entity_id]:
                            self.knowledge_graph["entities"][entity_id][key] = value
        
        # 整合关系到知识图谱
        for relation in relations:
            if "source" in relation and "target" in relation and "type" in relation:
                self.knowledge_graph["relations"].append(relation)
        
        # 整合概念到语义记忆
        for concept in concepts:
            concept_id = concept.get("id", concept.get("name", ""))
            if concept_id:
                if concept_id not in self.semantic_memory:
                    # 新概念
                    self.semantic_memory[concept_id] = {
                        "id": concept_id,
                        "name": concept.get("name", concept_id),
                        "attributes": concept.get("attributes", {}),
                        "instance_count": 1,
                        "first_encountered": time.time(),
                        "last_updated": time.time(),
                        "abstraction_level": concept.get("abstraction_level", 1),
                        "related_concepts": concept.get("related_concepts", []),
                        "examples": [item["id"]]
                    }
                else:
                    # 更新现有概念
                    existing = self.semantic_memory[concept_id]
                    existing["instance_count"] += 1
                    existing["last_updated"] = time.time()
                    
                    # 合并属性
                    for key, value in concept.get("attributes", {}).items():
                        if key not in existing["attributes"]:
                            existing["attributes"][key] = value
                    
                    # 添加到示例
                    if item["id"] not in existing["examples"]:
                        existing["examples"].append(item["id"])
                    
                    # 合并相关概念
                    for related in concept.get("related_concepts", []):
                        if related not in existing["related_concepts"]:
                            existing["related_concepts"].append(related)
                
                self.memory_stats["total_semantic"] += 1
        
        self.memory_stats["integrations"] += 1
        self.logger.debug(f"将记忆项目 {item['id']} 整合到语义记忆")
    
    def retrieve_from_memory(self, 
                           query: Dict[str, Any],
                           memory_type: str = "all",
                           limit: int = 10) -> Dict[str, Any]:
        """
        从记忆中检索
        
        Args:
            query: 查询条件
            memory_type: 记忆类型 ('working', 'episodic', 'semantic', 'all')
            limit: 结果数量限制
            
        Returns:
            检索结果
        """
        self.logger.info(f"从记忆中检索: 类型={memory_type}, 查询={query}")
        
        results = []
        
        # 检索工作记忆
        if memory_type in ["working", "all"]:
            working_results = self._search_working_memory(query)
            results.extend(working_results)
        
        # 检索情景记忆
        if memory_type in ["episodic", "all"] and len(results) < limit:
            episodic_results = self._search_episodic_memory(query, limit - len(results))
            results.extend(episodic_results)
        
        # 检索语义记忆
        if memory_type in ["semantic", "all"] and len(results) < limit:
            semantic_results = self._search_semantic_memory(query, limit - len(results))
            results.extend(semantic_results)
        
        # 更新检索统计和激活水平
        for item in results:
            item_id = item["id"]
            self.retrieval_counts[item_id] += 1
            self.last_access_time[item_id] = time.time()
            self.activation_levels[item_id] = min(1.0, self.activation_levels.get(item_id, 0.0) + 0.2)
        
        self.memory_stats["retrievals"] += 1
        
        return {
            "query": query,
            "memory_type": memory_type,
            "result_count": len(results),
            "results": results[:limit]
        }
    
    def _search_working_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """搜索工作记忆"""
        results = []
        
        for item in self.working_memory:
            if self._match_query(item, query):
                results.append(item)
        
        return results
    
    def _search_episodic_memory(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """搜索情景记忆"""
        # 使用索引优化搜索
        candidate_ids = set()
        
        # 按时间范围筛选
        if "time_range" in query:
            start_time, end_time = query["time_range"]
            start_key = int(start_time / 3600) * 3600
            end_key = int(end_time / 3600) * 3600
            
            for time_key in range(start_key, end_key + 3600, 3600):
                if time_key in self.memory_indices["temporal"]:
                    candidate_ids.update(self.memory_indices["temporal"][time_key])
        
        # 按标签筛选
        if "tags" in query:
            tag_ids = set()
            for tag in query["tags"]:
                if tag in self.memory_indices["semantic"]:
                    tag_ids.update(self.memory_indices["semantic"][tag])
            
            if candidate_ids:
                candidate_ids.intersection_update(tag_ids)
            else:
                candidate_ids = tag_ids
        
        # 按实体筛选
        if "entities" in query:
            entity_ids = set()
            for entity in query["entities"]:
                entity_id = entity.get("id", entity.get("name", ""))
                if entity_id in self.memory_indices["entity"]:
                    entity_ids.update(self.memory_indices["entity"][entity_id])
            
            if candidate_ids:
                candidate_ids.intersection_update(entity_ids)
            else:
                candidate_ids = entity_ids
        
        # 如果有候选ID，从中筛选
        results = []
        if candidate_ids:
            for item in self.episodic_memory:
                if item["id"] in candidate_ids and self._match_query(item, query):
                    results.append(item)
                    if len(results) >= limit:
                        break
        else:
            # 线性搜索
            for item in self.episodic_memory:
                if self._match_query(item, query):
                    results.append(item)
                    if len(results) >= limit:
                        break
        
        return results
    
    def _search_semantic_memory(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """搜索语义记忆"""
        results = []
        
        # 按概念搜索
        if "concepts" in query:
            for concept_id in query["concepts"]:
                if concept_id in self.semantic_memory:
                    results.append(self.semantic_memory[concept_id])
                    if len(results) >= limit:
                        break
        
        # 按属性搜索
        if "attributes" in query and len(results) < limit:
            for concept_id, concept in self.semantic_memory.items():
                match = True
                for attr_key, attr_value in query["attributes"].items():
                    if attr_key not in concept.get("attributes", {}) or concept["attributes"][attr_key] != attr_value:
                        match = False
                        break
                
                if match:
                    results.append(concept)
                    if len(results) >= limit:
                        break
        
        return results
    
    def _match_query(self, item: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """检查项目是否匹配查询条件"""
        for key, value in query.items():
            # 跳过特殊查询键
            if key in ["time_range", "limit", "memory_type"]:
                continue
            
            # 检查字段是否存在
            if key not in item:
                return False
            
            # 列表字段的包含检查
            if isinstance(value, list) and isinstance(item[key], list):
                # 检查是否有任何一个值匹配
                if not any(v in item[key] for v in value):
                    return False
            # 字典字段的键值检查
            elif isinstance(value, dict) and isinstance(item[key], dict):
                for k, v in value.items():
                    if k not in item[key] or item[key][k] != v:
                        return False
            # 简单值的相等检查
            elif item[key] != value:
                return False
        
        return True
    
    def form_association(self, source_id: str, target_id: str, relation_type: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        形成记忆关联
        
        Args:
            source_id: 源记忆ID
            target_id: 目标记忆ID
            relation_type: 关系类型
            strength: 关联强度
            
        Returns:
            关联结果
        """
        self.logger.info(f"形成记忆关联: {source_id} --{relation_type}--> {target_id}, 强度: {strength}")
        
        # 创建关系
        relation = {
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "strength": strength,
            "created_at": time.time()
        }
        
        # 添加到知识图谱
        self.knowledge_graph["relations"].append(relation)
        
        # 更新关联的记忆项目
        self._update_related_memories(source_id, target_id, relation)
        
        return {
            "status": "success",
            "relation": relation
        }
    
    def _update_related_memories(self, source_id: str, target_id: str, relation: Dict[str, Any]) -> None:
        """更新关联的记忆项目"""
        # 更新语义记忆
        if source_id in self.semantic_memory:
            if "relations" not in self.semantic_memory[source_id]:
                self.semantic_memory[source_id]["relations"] = []
            self.semantic_memory[source_id]["relations"].append(relation)
        
        if target_id in self.semantic_memory:
            if "relations" not in self.semantic_memory[target_id]:
                self.semantic_memory[target_id]["relations"] = []
            self.semantic_memory[target_id]["relations"].append(relation)
    
    def integrate_knowledge(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        整合知识
        
        Args:
            memory_ids: 要整合的记忆ID列表
            
        Returns:
            整合结果
        """
        self.logger.info(f"整合 {len(memory_ids)} 个记忆项目的知识")
        
        items = []
        
        # 收集要整合的记忆项目
        for memory_id in memory_ids:
            item = self._find_memory_by_id(memory_id)
            if item:
                items.append(item)
        
        if not items:
            return {"status": "error", "message": "未找到要整合的记忆项目"}
        
        # 抽取共同概念和实体
        common_concepts = self._extract_common_concepts(items)
        common_entities = self._extract_common_entities(items)
        
        # 抽取上下文关系
        contextual_relations = self._extract_contextual_relations(items)
        
        # 创建新的整合知识
        integrated_knowledge = {
            "id": f"ik_{int(time.time())}",
            "type": "integrated_knowledge",
            "source_memories": memory_ids,
            "concepts": common_concepts,
            "entities": common_entities,
            "relations": contextual_relations,
            "creation_time": time.time(),
            "abstraction_level": self._determine_abstraction_level(items)
        }
        
        # 添加到语义记忆
        concept_id = f"concept_{integrated_knowledge['id']}"
        self.semantic_memory[concept_id] = {
            "id": concept_id,
            "name": f"Integrated Concept {len(common_concepts)} entities",
            "type": "integrated_concept",
            "attributes": {},
            "instance_count": len(items),
            "first_encountered": time.time(),
            "last_updated": time.time(),
            "abstraction_level": integrated_knowledge["abstraction_level"],
            "related_concepts": [c["id"] for c in common_concepts if "id" in c],
            "examples": memory_ids,
            "integrated_knowledge": integrated_knowledge
        }
        
        self.memory_stats["integrations"] += 1
        self.memory_stats["total_semantic"] += 1
        
        return {
            "status": "success",
            "concept_id": concept_id,
            "integrated_knowledge": integrated_knowledge
        }
    
    def _find_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据ID查找记忆项目"""
        # 查找工作记忆
        for item in self.working_memory:
            if item["id"] == memory_id:
                return item
        
        # 查找情景记忆
        for item in self.episodic_memory:
            if item["id"] == memory_id:
                return item
        
        # 查找语义记忆
        if memory_id in self.semantic_memory:
            return self.semantic_memory[memory_id]
        
        return None
    
    def _extract_common_concepts(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取共同概念"""
        # 简化实现
        concept_count = defaultdict(int)
        concept_data = {}
        
        for item in items:
            if "concepts" in item:
                for concept in item["concepts"]:
                    concept_id = concept.get("id", concept.get("name", ""))
                    if concept_id:
                        concept_count[concept_id] += 1
                        concept_data[concept_id] = concept
        
        # 选择出现在至少一半项目中的概念
        threshold = len(items) / 2
        common_concepts = [concept_data[cid] for cid, count in concept_count.items() if count >= threshold]
        
        return common_concepts
    
    def _extract_common_entities(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取共同实体"""
        # 类似概念的提取逻辑
        entity_count = defaultdict(int)
        entity_data = {}
        
        for item in items:
            if "entities" in item:
                for entity in item["entities"]:
                    entity_id = entity.get("id", entity.get("name", ""))
                    if entity_id:
                        entity_count[entity_id] += 1
                        entity_data[entity_id] = entity
        
        threshold = len(items) / 2
        common_entities = [entity_data[eid] for eid, count in entity_count.items() if count >= threshold]
        
        return common_entities
    
    def _extract_contextual_relations(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取上下文关系"""
        # 收集所有关系
        all_relations = []
        
        for item in items:
            if "relations" in item:
                all_relations.extend(item["relations"])
        
        # 简化实现：返回所有关系
        return all_relations
    
    def _determine_abstraction_level(self, items: List[Dict[str, Any]]) -> int:
        """确定抽象层次"""
        # 计算所有项目的平均抽象层次并加1
        total_level = 0
        count = 0
        
        for item in items:
            if "abstraction_level" in item:
                total_level += item["abstraction_level"]
                count += 1
        
        if count > 0:
            return min(self.abstraction_levels, int(total_level / count) + 1)
        return 1
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆系统统计
        
        Returns:
            统计信息
        """
        return {
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "semantic_memory_size": len(self.semantic_memory),
            "knowledge_graph_entities": len(self.knowledge_graph["entities"]),
            "knowledge_graph_relations": len(self.knowledge_graph["relations"]),
            "total_retrievals": self.memory_stats["retrievals"],
            "total_integrations": self.memory_stats["integrations"],
            "last_cleanup": self.memory_stats["last_cleanup"]
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
                "memory_parameters": {
                    "episodic_capacity": self.episodic_capacity,
                    "semantic_capacity": self.semantic_capacity,
                    "working_memory_size": self.working_memory_size,
                    "integration_threshold": self.integration_threshold,
                    "abstraction_levels": self.abstraction_levels
                },
                "memory_stats": self.memory_stats,
                "saved_at": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"记忆系统状态已保存到: {file_path}")
            
            return {"success": True, "file_path": file_path}
        
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return {"success": False, "error": str(e)} 