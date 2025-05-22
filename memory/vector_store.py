# -*- coding: utf-8 -*-
"""
向量存储 (Vector Store)

负责管理和检索向量化的知识表示，支持语义相似性搜索
提供高效的近似最近邻搜索和向量索引
增强知识图谱集成和概念关联，支持零样本学习
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Set
import json
import os
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import scipy.spatial.distance as distance
from collections import defaultdict
import itertools
import math

# 尝试导入FAISS，如果不可用则使用内置备选方案
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，将使用基础向量存储。安装FAISS可提高性能: pip install faiss-cpu")

# 尝试导入社区检测算法
try:
    from networkx.algorithms import community
    COMMUNITY_DETECTION_AVAILABLE = True
except ImportError:
    COMMUNITY_DETECTION_AVAILABLE = False
    print("警告: 社区检测功能不可用，某些模式提取能力将受限")

class VectorStore:
    def __init__(self, dimension: int = 768, distance_metric: str = "cosine", index_type: str = "flat"):
        """
        初始化向量存储
        
        Args:
            dimension (int): 向量维度，默认768（适用于某些预训练模型）
            distance_metric (str): 距离度量方式 ('cosine', 'l2', 'ip')
            index_type (str): 索引类型 ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.index_type = index_type
        self.vectors = []  # 向量数据
        self.metadata = {}  # 向量元数据
        self.id_to_index = {}  # ID到索引的映射
        self.index_to_id = {}  # 索引到ID的映射
        self.last_modified = time.time()
        self.executor = ThreadPoolExecutor(max_workers=2)  # 用于异步操作
        
        # 知识图谱相关
        self.knowledge_graph = nx.DiGraph()  # 有向图表示概念关系
        self.concept_vectors = {}  # 概念嵌入
        self.concept_instances = defaultdict(set)  # 概念实例映射
        self.concept_hierarchy = defaultdict(set)  # 概念层次结构
        self.relation_types = set()  # 支持的关系类型
        
        # 语义簇
        self.semantic_clusters = {}  # 语义聚类
        self.cluster_centers = {}  # 聚类中心
        
        # 模式缓存
        self.pattern_cache = {}  # 缓存已经发现的模式
        self.pattern_confidences = {}  # 模式的置信度
        
        # 初始化基本关系类型
        self._initialize_relation_types()
        
        # 初始化索引
        self._initialize_index()
        
    def _initialize_relation_types(self):
        """初始化基本关系类型"""
        base_relations = {
            "is_a",  # 类型关系
            "part_of",  # 部分关系
            "has_property",  # 属性关系
            "related_to",  # 一般关联
            "causes",  # 因果关系
            "precedes",  # 时序关系
            "similar_to",  # 相似性关系
            "opposite_of",  # 对立关系
            "instance_of",  # 实例关系
            "derives_from"  # 派生关系
        }
        self.relation_types.update(base_relations)
        
    def _initialize_index(self):
        """
        初始化向量索引
        """
        # 根据距离度量方式选择索引类型
        if FAISS_AVAILABLE:
            if self.distance_metric == "cosine":
                # 余弦相似度
                self.index = faiss.IndexFlatIP(self.dimension)  # 内积，需要先归一化向量
                self.normalize = True
            elif self.distance_metric == "l2":
                # 欧氏距离
                self.index = faiss.IndexFlatL2(self.dimension)
                self.normalize = False
            elif self.distance_metric == "ip":
                # 内积
                self.index = faiss.IndexFlatIP(self.dimension)
                self.normalize = False
            else:
                # 默认使用L2
                self.index = faiss.IndexFlatL2(self.dimension)
                self.normalize = False
                
            # 如果不是flat索引，可以进一步配置
            if self.index_type != "flat" and self.vectors:
                # 这里只是示例，实际应用中需要更复杂的配置
                if self.index_type == "ivf" and len(self.vectors) > 100:
                    n_clusters = min(int(len(self.vectors) / 10), 100)
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
                    # 需要训练
                    vectors_array = np.array(self.vectors).astype('float32')
                    self.index.train(vectors_array)
        else:
            # 如果没有FAISS，使用内置的基础索引
            self.index = None
            self.normalize = self.distance_metric == "cosine"
            
    def add_item(self, item_id: Optional[str] = None, vector: Optional[List[float]] = None, 
                metadata: Optional[Dict[str, Any]] = None, text: Optional[str] = None) -> str:
        """
        添加向量项到存储
        
        Args:
            item_id (str, optional): 项目ID，如果不提供则生成新ID
            vector (List[float], optional): 向量数据
            metadata (Dict[str, Any], optional): 项目元数据
            text (str, optional): 文本内容，如果提供且未提供向量，将生成嵌入
            
        Returns:
            str: 项目ID
        """
        # 生成或使用提供的ID
        if item_id is None:
            item_id = str(uuid.uuid4())
            
        # 检查向量
        if vector is None:
            if text:
                # 从文本生成向量嵌入
                vector = self._text_to_embedding(text)
            else:
                # 如果既没有向量也没有文本，返回错误
                return ""
                
        # 确保向量是numpy数组并且是正确的维度
        vector = np.array(vector).astype('float32')
        if vector.shape[0] != self.dimension:
            # 维度不匹配
            return ""
            
        # 如果需要归一化
        if self.normalize:
            vector = self._normalize_vector(vector)
            
        # 添加到向量列表
        index = len(self.vectors)
        self.vectors.append(vector)
        self.id_to_index[item_id] = index
        self.index_to_id[index] = item_id
        
        # 保存元数据
        self.metadata[item_id] = metadata or {}
        if text:
            self.metadata[item_id]["text"] = text
        self.metadata[item_id]["timestamp"] = time.time()
        
        # 更新索引
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vector.reshape(1, -1))
            
        self.last_modified = time.time()
        return item_id
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        批量添加文本
        
        Args:
            texts (List[str]): 文本列表
            metadatas (List[Dict[str, Any]], optional): 元数据列表
            
        Returns:
            List[str]: 项目ID列表
        """
        item_ids = []
        
        for i, text in enumerate(texts):
            metadata = None
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
                
            item_id = self.add_item(text=text, metadata=metadata)
            item_ids.append(item_id)
            
        return item_ids
        
    def search(self, query: Union[str, List[float]], k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索最相似的向量
        
        Args:
            query (str or List[float]): 查询文本或向量
            k (int): 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        if not self.vectors:
            return []
            
        # 确保k不超过向量数量
        k = min(k, len(self.vectors))
        
        # 如果查询是文本，先转换为向量
        if isinstance(query, str):
            query_vector = self._text_to_embedding(query)
        else:
            query_vector = np.array(query).astype('float32')
            
        # 确保向量维度正确
        if query_vector.shape[0] != self.dimension:
            return []
            
        # 如果需要归一化
        if self.normalize:
            query_vector = self._normalize_vector(query_vector)
            
        # 使用索引搜索
        if FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)
            results = []
            
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx == -1:  # FAISS可能返回-1表示无结果
                    continue
                    
                item_id = self.index_to_id.get(idx)
                if not item_id:
                    continue
                    
                results.append({
                    "id": item_id,
                    "score": 1.0 - float(distances[0][i]) if self.distance_metric in ["l2", "cosine"] else float(distances[0][i]),
                    "metadata": self.metadata.get(item_id, {})
                })
        else:
            # 使用内置的基础搜索
            results = self._basic_search(query_vector, k)
            
        return results
        
    def _basic_search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """
        基础向量搜索（当FAISS不可用时使用）
        
        Args:
            query_vector (np.ndarray): 查询向量
            k (int): 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        similarities = []
        
        for idx, vector in enumerate(self.vectors):
            if self.distance_metric == "cosine":
                # 计算余弦相似度
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            elif self.distance_metric == "l2":
                # 计算欧氏距离，并转换为相似度分数
                dist = np.linalg.norm(query_vector - vector)
                similarity = 1.0 / (1.0 + dist)  # 转换为0-1之间的分数
            elif self.distance_metric == "ip":
                # 内积
                similarity = np.dot(query_vector, vector)
            else:
                # 默认使用余弦相似度
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                
            similarities.append((idx, float(similarity)))
            
        # 排序，取top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            item_id = self.index_to_id.get(idx)
            if not item_id:
                continue
                
            results.append({
                "id": item_id,
                "score": score,
                "metadata": self.metadata.get(item_id, {})
            })
            
        return results
        
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定项目
        
        Args:
            item_id (str): 项目ID
            
        Returns:
            Dict[str, Any]: 项目信息
        """
        if item_id not in self.id_to_index:
            return None
            
        idx = self.id_to_index[item_id]
        
        return {
            "id": item_id,
            "vector": self.vectors[idx].tolist() if idx < len(self.vectors) else None,
            "metadata": self.metadata.get(item_id, {})
        }
        
    def delete_item(self, item_id: str) -> bool:
        """
        删除项目
        
        Args:
            item_id (str): 项目ID
            
        Returns:
            bool: 是否成功删除
        """
        if item_id not in self.id_to_index:
            return False
            
        # 获取索引
        idx = self.id_to_index[item_id]
        
        # 删除元数据和索引映射
        del self.metadata[item_id]
        del self.id_to_index[item_id]
        del self.index_to_id[idx]
        
        # 在FAISS中，我们通常不能直接删除向量
        # 最安全的方法是重建索引，但这里我们标记为删除
        # 在vectors中保留占位，但在将来的搜索中忽略此索引
        
        # 如果是最后一个向量，可以直接移除
        if idx == len(self.vectors) - 1:
            self.vectors.pop()
        else:
            # 否则标记为删除（用零向量替代）
            self.vectors[idx] = np.zeros(self.dimension, dtype='float32')
            
        # 重建索引（在实际应用中，应该批量进行以提高效率）
        if len(self.vectors) == 0:
            self._initialize_index()
        elif FAISS_AVAILABLE and self.index is not None:
            # 完全重建索引
            self._rebuild_index()
            
        self.last_modified = time.time()
        return True
        
    def _rebuild_index(self):
        """
        重建索引
        """
        if not FAISS_AVAILABLE:
            return
            
        # 重新初始化索引
        self._initialize_index()
        
        # 如果有向量，重新添加到索引
        if self.vectors:
            vectors_array = np.array(self.vectors).astype('float32')
            
            # 如果需要训练（例如IVF索引）
            if hasattr(self.index, 'train') and self.index_type != "flat":
                self.index.train(vectors_array)
                
            # 添加向量
            self.index.add(vectors_array)
            
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        将文本转换为向量嵌入
        
        Args:
            text (str): 输入文本
            
        Returns:
            np.ndarray: 向量嵌入
        """
        # 在实际应用中，应该使用预训练模型（如BERT）生成嵌入
        # 这里提供一个简单的实现作为占位符
        
        # 简单的哈希函数，保证相同文本生成相同向量
        # 注意：这不是一个好的嵌入方法，仅用于演示
        import hashlib
        
        np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
        vector = np.random.rand(self.dimension).astype('float32')
        
        # 恢复随机数生成器状态
        np.random.seed(None)
        
        return vector
        
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            vector (np.ndarray): 输入向量
            
        Returns:
            np.ndarray: 归一化后的向量
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
        
    def save(self, path: str) -> bool:
        """
        保存向量存储到文件
        
        Args:
            path (str): 保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存向量和元数据
            data = {
                "dimension": self.dimension,
                "distance_metric": self.distance_metric,
                "index_type": self.index_type,
                "vectors": [v.tolist() for v in self.vectors],
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()},  # 将整数键转换为字符串
                "last_modified": self.last_modified
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
                
            return True
        except Exception as e:
            print(f"保存向量存储错误: {str(e)}")
            return False
            
    def load(self, path: str) -> bool:
        """
        从文件加载向量存储
        
        Args:
            path (str): 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 加载基本属性
            self.dimension = data.get("dimension", 768)
            self.distance_metric = data.get("distance_metric", "cosine")
            self.index_type = data.get("index_type", "flat")
            
            # 加载向量和元数据
            self.vectors = [np.array(v, dtype='float32') for v in data.get("vectors", [])]
            self.metadata = data.get("metadata", {})
            self.id_to_index = data.get("id_to_index", {})
            
            # 加载索引到ID的映射，将字符串键转换回整数
            self.index_to_id = {int(k): v for k, v in data.get("index_to_id", {}).items()}
            
            self.last_modified = data.get("last_modified", time.time())
            
            # 重建索引
            self._rebuild_index()
            
            return True
        except Exception as e:
            print(f"加载向量存储错误: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "item_count": len(self.vectors),
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "index_type": self.index_type,
            "faiss_available": FAISS_AVAILABLE,
            "last_modified": self.last_modified
        }
        
    def similarity(self, item_id1: str, item_id2: str) -> float:
        """
        计算两个项目的相似度
        
        Args:
            item_id1 (str): 第一个项目ID
            item_id2 (str): 第二个项目ID
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if item_id1 not in self.id_to_index or item_id2 not in self.id_to_index:
            return 0.0
            
        idx1 = self.id_to_index[item_id1]
        idx2 = self.id_to_index[item_id2]
        
        vec1 = self.vectors[idx1]
        vec2 = self.vectors[idx2]
        
        if self.distance_metric == "cosine":
            # 计算余弦相似度
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif self.distance_metric == "l2":
            # 计算欧氏距离，并转换为相似度分数
            dist = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + dist))
        elif self.distance_metric == "ip":
            # 内积
            return float(np.dot(vec1, vec2))
        else:
            # 默认余弦相似度
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def add_concept(self, concept_name: str, vector: Optional[List[float]] = None, 
                  properties: Dict[str, Any] = None) -> str:
        """
        添加概念到知识体系
        
        Args:
            concept_name (str): 概念名称
            vector (List[float], optional): 概念的向量表示
            properties (Dict[str, Any], optional): 概念属性
            
        Returns:
            str: 概念ID
        """
        concept_id = f"concept:{concept_name.lower().replace(' ', '_')}"
        
        # 如果没有向量，生成一个随机向量或使用名称生成
        if vector is None:
            vector = self._text_to_embedding(concept_name)
            
        # 存储概念向量
        self.concept_vectors[concept_id] = np.array(vector).astype('float32')
        
        # 将概念添加到知识图谱
        self.knowledge_graph.add_node(concept_id, 
                                       type="concept", 
                                       name=concept_name, 
                                       properties=properties or {},
                                       vector=vector)
                                       
        return concept_id
        
    def add_relation(self, source_id: str, target_id: str, relation_type: str, 
                   properties: Dict[str, Any] = None) -> bool:
        """
        添加概念间的关系
        
        Args:
            source_id (str): 源概念ID
            target_id (str): 目标概念ID
            relation_type (str): 关系类型
            properties (Dict[str, Any], optional): 关系属性
            
        Returns:
            bool: 是否成功添加
        """
        # 确保关系类型有效
        if relation_type not in self.relation_types:
            self.relation_types.add(relation_type)
            
        # 添加关系到知识图谱
        self.knowledge_graph.add_edge(source_id, target_id, 
                                      type=relation_type, 
                                      properties=properties or {})
        
        # 如果是层次关系，更新概念层次结构
        if relation_type == "is_a":
            self.concept_hierarchy[target_id].add(source_id)
            
        # 如果是实例关系，更新实例映射
        elif relation_type == "instance_of":
            self.concept_instances[target_id].add(source_id)
            
        return True
        
    def find_related_concepts(self, concept_id: str, relation_types: List[str] = None, 
                            max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        查找与给定概念相关的概念
        
        Args:
            concept_id (str): 概念ID
            relation_types (List[str], optional): 要考虑的关系类型
            max_depth (int): 最大搜索深度
            
        Returns:
            List[Dict[str, Any]]: 相关概念列表
        """
        if concept_id not in self.knowledge_graph:
            return []
            
        related = []
        visited = set()
        
        def dfs(node_id, depth=0):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            # 获取所有出边
            for _, target, data in self.knowledge_graph.out_edges(node_id, data=True):
                edge_type = data.get("type", "")
                
                # 如果未指定关系类型或当前关系在指定类型中
                if relation_types is None or edge_type in relation_types:
                    target_data = self.knowledge_graph.nodes[target]
                    related.append({
                        "id": target,
                        "name": target_data.get("name", target),
                        "relation": edge_type,
                        "properties": target_data.get("properties", {})
                    })
                    
                    # 继续深度搜索
                    dfs(target, depth + 1)
                    
            # 获取所有入边
            for source, _, data in self.knowledge_graph.in_edges(node_id, data=True):
                edge_type = data.get("type", "")
                
                # 如果未指定关系类型或当前关系在指定类型中
                if relation_types is None or edge_type in relation_types:
                    source_data = self.knowledge_graph.nodes[source]
                    related.append({
                        "id": source,
                        "name": source_data.get("name", source),
                        "relation": f"inverse_{edge_type}",
                        "properties": source_data.get("properties", {})
                    })
                    
                    # 继续深度搜索
                    dfs(source, depth + 1)
        
        # 从目标概念开始搜索
        dfs(concept_id)
        
        return related
        
    def infer_relations(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """
        推理两个概念间可能的关系
        
        Args:
            source_id (str): 源概念ID
            target_id (str): 目标概念ID
            
        Returns:
            List[Dict[str, Any]]: 可能的关系列表，包含置信度
        """
        # 如果存在直接关系，返回
        if self.knowledge_graph.has_edge(source_id, target_id):
            data = self.knowledge_graph.get_edge_data(source_id, target_id)
            return [{
                "relation": data.get("type", "related_to"),
                "confidence": 1.0,
                "properties": data.get("properties", {})
            }]
            
        # 计算向量相似度
        if source_id in self.concept_vectors and target_id in self.concept_vectors:
            source_vec = self.concept_vectors[source_id]
            target_vec = self.concept_vectors[target_id]
            
            if self.normalize:
                source_vec = self._normalize_vector(source_vec)
                target_vec = self._normalize_vector(target_vec)
                
            sim = 1 - distance.cosine(source_vec, target_vec)
            
            # 如果相似度很高，可能是相似概念
            if sim > 0.85:
                return [{
                    "relation": "similar_to",
                    "confidence": sim,
                    "inferred": True
                }]
                
        # 查找共同上位概念
        source_ancestors = self._get_ancestors(source_id)
        target_ancestors = self._get_ancestors(target_id)
        
        common_ancestors = source_ancestors.intersection(target_ancestors)
        
        results = []
        
        if common_ancestors:
            # 找到最近的共同祖先
            nearest_common_ancestor = None
            min_distance = float('inf')
            
            for ancestor in common_ancestors:
                source_dist = self._get_path_length(source_id, ancestor)
                target_dist = self._get_path_length(target_id, ancestor)
                total_dist = source_dist + target_dist
                
                if total_dist < min_distance:
                    min_distance = total_dist
                    nearest_common_ancestor = ancestor
                    
            if nearest_common_ancestor:
                confidence = 1.0 / (1 + min_distance)  # 距离越短置信度越高
                
                results.append({
                    "relation": "common_ancestor",
                    "ancestor": nearest_common_ancestor,
                    "confidence": confidence,
                    "inferred": True
                })
                
        # 寻找两跳关系
        for mid_node in self.knowledge_graph.nodes():
            if mid_node == source_id or mid_node == target_id:
                continue
                
            # 检查source->mid->target路径
            if (self.knowledge_graph.has_edge(source_id, mid_node) and 
                self.knowledge_graph.has_edge(mid_node, target_id)):
                rel1 = self.knowledge_graph.get_edge_data(source_id, mid_node).get("type", "")
                rel2 = self.knowledge_graph.get_edge_data(mid_node, target_id).get("type", "")
                
                results.append({
                    "relation": f"via_{rel1}_{rel2}",
                    "intermediate": mid_node,
                    "confidence": 0.7,  # 两跳关系置信度较低
                    "inferred": True
                })
                
        return results
        
    def _get_ancestors(self, concept_id: str) -> Set[str]:
        """获取概念的所有祖先节点"""
        if concept_id not in self.knowledge_graph:
            return set()
            
        ancestors = set()
        visited = set()
        
        def collect_ancestors(node):
            if node in visited:
                return
                
            visited.add(node)
            
            # 查找所有"is_a"关系的目标节点
            for _, target, data in self.knowledge_graph.out_edges(node, data=True):
                if data.get("type") == "is_a":
                    ancestors.add(target)
                    collect_ancestors(target)
                    
        collect_ancestors(concept_id)
        return ancestors
        
    def _get_path_length(self, source: str, target: str) -> int:
        """计算两个节点间的最短路径长度"""
        try:
            return nx.shortest_path_length(self.knowledge_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')
            
    def find_analogies(self, a: str, b: str, c: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        寻找类比关系: a:b::c:?
        
        Args:
            a (str): 概念A的ID
            b (str): 概念B的ID
            c (str): 概念C的ID
            n (int): 返回的类比结果数量
            
        Returns:
            List[Dict[str, Any]]: 可能的类比概念列表
        """
        if not all(x in self.concept_vectors for x in [a, b, c]):
            return []
            
        # 获取向量
        vec_a = self.concept_vectors[a]
        vec_b = self.concept_vectors[b]
        vec_c = self.concept_vectors[c]
        
        # 归一化
        if self.normalize:
            vec_a = self._normalize_vector(vec_a)
            vec_b = self._normalize_vector(vec_b)
            vec_c = self._normalize_vector(vec_c)
            
        # 计算类比向量: vec_d ≈ vec_c + (vec_b - vec_a)
        target_vector = vec_c + (vec_b - vec_a)
        
        # 归一化目标向量
        if self.normalize:
            target_vector = self._normalize_vector(target_vector)
            
        # 在概念空间中寻找最相似的向量
        results = []
        
        # 排除已知的概念
        exclude = {a, b, c}
        
        for concept_id, vector in self.concept_vectors.items():
            if concept_id in exclude:
                continue
                
            # 归一化
            if self.normalize:
                vector = self._normalize_vector(vector)
                
            # 计算相似度
            similarity = 1 - distance.cosine(target_vector, vector)
            
            # 添加到结果
            results.append({
                "id": concept_id,
                "name": self.knowledge_graph.nodes[concept_id]["name"] if concept_id in self.knowledge_graph.nodes else concept_id,
                "similarity": similarity
            })
            
        # 按相似度排序并返回前n个
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:n]
        
        return results
        
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        从知识图谱中提取模式和规律
        
        Returns:
            List[Dict[str, Any]]: 提取的模式列表
        """
        if not self.knowledge_graph or len(self.knowledge_graph.nodes) < 5:
            return []  # 图太小，无法提取有意义的模式
            
        patterns = []
        
        # 检查是否有缓存的模式且图结构未改变
        cache_key = f"patterns_{len(self.knowledge_graph.nodes)}_{len(self.knowledge_graph.edges)}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # 1. 发现频繁关系路径 (三元组模式)
        relation_paths = self._find_frequent_relation_paths()
        if relation_paths:
            patterns.extend(relation_paths)
            
        # 2. 识别常见的树形结构模式
        tree_patterns = self._find_tree_patterns()
        if tree_patterns:
            patterns.extend(tree_patterns)
            
        # 3. 发现概念簇和社区
        if COMMUNITY_DETECTION_AVAILABLE and len(self.knowledge_graph.nodes) >= 10:
            community_patterns = self._find_concept_communities()
            if community_patterns:
                patterns.extend(community_patterns)
                
        # 4. 识别概念分组模式（具有相似属性的概念）
        property_patterns = self._find_property_patterns()
        if property_patterns:
            patterns.extend(property_patterns)
            
        # 5. 发现层次结构模式
        hierarchy_patterns = self._find_hierarchy_patterns()
        if hierarchy_patterns:
            patterns.extend(hierarchy_patterns)
            
        # 按照置信度/重要性对模式进行排序
        patterns.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        # 缓存提取的模式
        self.pattern_cache[cache_key] = patterns[:100]  # 只保留最重要的100个模式
        
        return patterns
        
    def _find_frequent_relation_paths(self) -> List[Dict[str, Any]]:
        """
        发现频繁出现的关系路径模式
        """
        path_counts = defaultdict(int)
        path_examples = defaultdict(list)
        
        # 计算图的平均度
        avg_degree = sum(d for _, d in self.knowledge_graph.degree()) / max(1, len(self.knowledge_graph))
        min_support = max(2, int(len(self.knowledge_graph.nodes) / 20))  # 根据图大小调整支持度
        
        # 搜索所有3跳内的路径
        for node in self.knowledge_graph.nodes():
            for target in self.knowledge_graph.nodes():
                if node == target:
                    continue
                
                # 使用NetworkX的all_simple_paths查找简单路径
                try:
                    for path in nx.all_simple_paths(self.knowledge_graph, node, target, cutoff=3):
                        if len(path) < 2:
                            continue
                            
                        # 构建关系路径签名
                        path_sig = []
                        for i in range(len(path) - 1):
                            source = path[i]
                            dest = path[i+1]
                            if self.knowledge_graph.has_edge(source, dest):
                                rel_type = self.knowledge_graph[source][dest].get("type", "related_to")
                                path_sig.append(rel_type)
                                
                        if path_sig:
                            path_key = "->".join(path_sig)
                            path_counts[path_key] += 1
                            
                            # 保存最多5个例子
                            if len(path_examples[path_key]) < 5:
                                path_examples[path_key].append({
                                    "path": [n for n in path],
                                    "nodes": [self.knowledge_graph.nodes[n].get("name", n) for n in path]
                                })
                except nx.NetworkXNoPath:
                    continue
        
        # 过滤频繁路径
        frequent_paths = []
        for path_key, count in path_counts.items():
            if count >= min_support:
                examples = path_examples[path_key]
                frequent_paths.append({
                    "pattern": "关系路径",
                    "path_signature": path_key,
                    "frequency": count,
                    "importance": count * len(path_key.split("->")),  # 频率 * 路径长度
                    "examples": examples[:3],  # 最多3个例子
                    "description": f"发现频繁关系路径: {path_key}，出现{count}次"
                })
                
        return frequent_paths
        
    def _find_tree_patterns(self) -> List[Dict[str, Any]]:
        """
        识别常见的树形结构模式
        """
        tree_patterns = []
        
        # 查找具有多个子节点的节点
        potential_roots = []
        for node in self.knowledge_graph.nodes():
            out_degree = self.knowledge_graph.out_degree(node)
            if out_degree >= 3:  # 至少有3个子节点
                potential_roots.append((node, out_degree))
                
        # 按出度排序
        potential_roots.sort(key=lambda x: x[1], reverse=True)
        
        # 分析前10个潜在根节点
        for root, degree in potential_roots[:10]:
            successors = list(self.knowledge_graph.successors(root))
            if not successors:
                continue
                
            # 收集子节点关系类型
            relation_types = {}
            for child in successors:
                if self.knowledge_graph.has_edge(root, child):
                    rel_type = self.knowledge_graph[root][child].get("type", "related_to")
                    relation_types[child] = rel_type
            
            # 按关系类型分组
            relation_groups = defaultdict(list)
            for child, rel_type in relation_types.items():
                relation_groups[rel_type].append(child)
                
            # 查找具有相同关系类型的大组
            for rel_type, children in relation_groups.items():
                if len(children) >= 3:  # 至少有3个同类型关系的子节点
                    root_name = self.knowledge_graph.nodes[root].get("name", root)
                    
                    tree_patterns.append({
                        "pattern": "树形结构",
                        "root": root,
                        "root_name": root_name,
                        "relation_type": rel_type,
                        "children_count": len(children),
                        "importance": len(children) * 1.5,  # 子节点数 * 权重
                        "examples": [self.knowledge_graph.nodes[c].get("name", c) for c in children[:5]],
                        "description": f"发现树形结构: {root_name} 通过 '{rel_type}' 关系连接到 {len(children)} 个子节点"
                    })
                    
        return tree_patterns
                    
    def _find_concept_communities(self) -> List[Dict[str, Any]]:
        """
        发现概念社区和聚类
        """
        if not COMMUNITY_DETECTION_AVAILABLE:
            return []
            
        # 创建无向图用于社区检测
        undirected_graph = self.knowledge_graph.to_undirected()
        
        # 使用Louvain方法检测社区
        try:
            communities = community.louvain_communities(undirected_graph)
        except:
            # 如果Louvain方法失败，回退到标签传播
            try:
                communities = community.label_propagation_communities(undirected_graph)
            except:
                return []
                
        community_patterns = []
        
        # 分析每个社区
        for i, comm in enumerate(communities):
            if len(comm) < 3:  # 忽略太小的社区
                continue
                
            # 提取社区内节点名称
            node_names = []
            for node in comm:
                node_names.append(self.knowledge_graph.nodes[node].get("name", node))
                
            # 分析社区内最常见的关系类型
            relation_counts = defaultdict(int)
            for u, v, data in undirected_graph.edges(comm, data=True):
                if u in comm and v in comm:  # 只计算社区内部边
                    rel_type = data.get("type", "related_to")
                    relation_counts[rel_type] += 1
                    
            # 找出主导关系类型
            dominant_relation = max(relation_counts.items(), key=lambda x: x[1]) if relation_counts else ("unknown", 0)
            
            community_patterns.append({
                "pattern": "概念社区",
                "community_id": i,
                "size": len(comm),
                "members": node_names[:10],  # 最多显示10个成员
                "dominant_relation": dominant_relation[0],
                "relation_strength": dominant_relation[1],
                "importance": len(comm) * (dominant_relation[1] / max(1, len(comm))),
                "description": f"发现概念社区 #{i}: 包含 {len(comm)} 个节点，主导关系为 '{dominant_relation[0]}'"
            })
            
        return community_patterns
        
    def _find_property_patterns(self) -> List[Dict[str, Any]]:
        """
        发现具有相似属性的概念分组
        """
        # 收集所有节点的属性
        node_properties = {}
        for node, data in self.knowledge_graph.nodes(data=True):
            props = data.get("properties", {})
            if props:
                node_properties[node] = props
                
        if not node_properties:
            return []
            
        # 查找具有相同属性键的节点组
        property_groups = defaultdict(list)
        for node, props in node_properties.items():
            prop_keys = frozenset(props.keys())
            if prop_keys:  # 忽略没有属性的节点
                key_str = ",".join(sorted(prop_keys))
                property_groups[key_str].append(node)
                
        # 过滤大小足够的组
        property_patterns = []
        for prop_key, nodes in property_groups.items():
            if len(nodes) >= 3:  # 至少3个具有相同属性集的节点
                # 提取节点名称
                node_names = [self.knowledge_graph.nodes[n].get("name", n) for n in nodes]
                
                property_patterns.append({
                    "pattern": "属性模式",
                    "property_set": prop_key.split(","),
                    "node_count": len(nodes),
                    "nodes": node_names[:5],  # 最多显示5个节点
                    "importance": len(nodes) * len(prop_key.split(",")),  # 节点数 * 属性数
                    "description": f"发现属性模式: {len(nodes)} 个概念共享属性集 [{prop_key}]"
                })
                
        return property_patterns
        
    def _find_hierarchy_patterns(self) -> List[Dict[str, Any]]:
        """
        发现层次结构模式
        """
        # 专注于is_a关系寻找层次结构
        hierarchy_subgraph = nx.DiGraph()
        for u, v, data in self.knowledge_graph.edges(data=True):
            if data.get("type") == "is_a":
                hierarchy_subgraph.add_edge(u, v)
                
        if not hierarchy_subgraph.edges:
            return []
            
        # 查找层次深度较深的路径
        max_depth = 0
        deep_paths = []
        
        for node in hierarchy_subgraph.nodes():
            if hierarchy_subgraph.out_degree(node) == 0:  # 叶节点
                # 尝试找到最长路径
                current_path = [node]
                current = node
                depth = 0
                
                while hierarchy_subgraph.in_degree(current) > 0:
                    parents = list(hierarchy_subgraph.predecessors(current))
                    if not parents:
                        break
                        
                    current = parents[0]  # 取第一个父节点
                    current_path.append(current)
                    depth += 1
                    
                if depth > max_depth:
                    max_depth = depth
                    
                if depth >= 2:  # 至少3层的层次结构
                    deep_paths.append(current_path)
                    
        # 只保留最深的几条路径
        deep_paths.sort(key=len, reverse=True)
        deep_paths = deep_paths[:5]  # 最多5条路径
        
        hierarchy_patterns = []
        for i, path in enumerate(deep_paths):
            node_names = [self.knowledge_graph.nodes[n].get("name", n) for n in path]
            
            hierarchy_patterns.append({
                "pattern": "层次结构",
                "path_id": i,
                "depth": len(path) - 1,
                "path": node_names,
                "importance": (len(path) - 1) * 2,  # 深度权重更高
                "description": f"发现层次结构路径 #{i}: 深度 {len(path)-1}, 从 {node_names[0]} 到 {node_names[-1]}"
            })
            
        return hierarchy_patterns

    def save_knowledge_graph(self, path: str) -> bool:
        """
        保存知识图谱到文件
        
        Args:
            path (str): 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 将NetworkX图转换为可序列化字典
            graph_data = {
                "nodes": [],
                "edges": []
            }
            
            # 添加节点
            for node, data in self.knowledge_graph.nodes(data=True):
                node_data = {"id": node}
                
                # 复制节点属性，排除向量（向量单独保存）
                for key, value in data.items():
                    if key != "vector":  # 不保存向量，太大
                        node_data[key] = value
                        
                graph_data["nodes"].append(node_data)
                
            # 添加边
            for source, target, data in self.knowledge_graph.edges(data=True):
                edge_data = {
                    "source": source,
                    "target": target
                }
                
                for key, value in data.items():
                    edge_data[key] = value
                    
                graph_data["edges"].append(edge_data)
                
            # 保存图结构
            with open(path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
                
            # 保存概念向量
            vectors_path = f"{os.path.splitext(path)[0]}_vectors.npy"
            
            # 构建向量矩阵和ID映射
            concept_ids = list(self.concept_vectors.keys())
            vectors = [self.concept_vectors[cid] for cid in concept_ids]
            
            if vectors:
                vectors_array = np.array(vectors)
                np.save(vectors_path, vectors_array)
                
                # 保存ID映射
                id_map_path = f"{os.path.splitext(path)[0]}_id_map.json"
                with open(id_map_path, "w", encoding="utf-8") as f:
                    json.dump(concept_ids, f, ensure_ascii=False)
                    
            return True
        except Exception as e:
            print(f"保存知识图谱失败: {str(e)}")
            return False
            
    def load_knowledge_graph(self, path: str) -> bool:
        """
        从文件加载知识图谱
        
        Args:
            path (str): 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 加载图结构
            with open(path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
                
            # 创建新图
            new_graph = nx.DiGraph()
            
            # 添加节点
            for node_data in graph_data["nodes"]:
                node_id = node_data.pop("id")
                new_graph.add_node(node_id, **node_data)
                
            # 添加边
            for edge_data in graph_data["edges"]:
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                new_graph.add_edge(source, target, **edge_data)
                
            # 加载概念向量
            vectors_path = f"{os.path.splitext(path)[0]}_vectors.npy"
            id_map_path = f"{os.path.splitext(path)[0]}_id_map.json"
            
            if os.path.exists(vectors_path) and os.path.exists(id_map_path):
                vectors_array = np.load(vectors_path)
                
                with open(id_map_path, "r", encoding="utf-8") as f:
                    concept_ids = json.load(f)
                    
                # 重建概念向量字典
                concept_vectors = {}
                for i, concept_id in enumerate(concept_ids):
                    concept_vectors[concept_id] = vectors_array[i]
                    
                self.concept_vectors = concept_vectors
                
            # 更新图和关系
            self.knowledge_graph = new_graph
            
            # 重建概念层次结构和实例映射
            self.concept_hierarchy = defaultdict(set)
            self.concept_instances = defaultdict(set)
            
            for source, target, data in new_graph.edges(data=True):
                rel_type = data.get("type", "")
                
                if rel_type == "is_a":
                    self.concept_hierarchy[target].add(source)
                elif rel_type == "instance_of":
                    self.concept_instances[target].add(source)
                    
                # 更新关系类型集合
                self.relation_types.add(rel_type)
                
            return True
        except Exception as e:
            print(f"加载知识图谱失败: {str(e)}")
            return False