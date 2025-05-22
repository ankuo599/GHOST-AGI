# -*- coding: utf-8 -*-
"""
记忆系统 (Memory System)

负责管理短期和长期记忆，基于图数据库和向量数据库实现
支持知识存储、检索和关联分析
"""

import time
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import networkx as nx
from queue import Queue
from collections import defaultdict
import threading

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，将使用基础向量存储。安装FAISS可提高性能: pip install faiss-cpu")

class MemorySystem:
    def __init__(self, vector_store=None, vector_dim=1536, event_system=None):
        """
        初始化记忆系统
        
        Args:
            vector_store (VectorStore, optional): 向量存储实例
            vector_dim (int): 向量维度，默认1536（适用于OpenAI嵌入模型）
            event_system: 事件系统实例
        """
        self.short_term_memory = []
        self.long_term_memory = {}
        self.vector_store = vector_store
        self.memory_graph = nx.DiGraph()  # 使用NetworkX图数据结构
        self.max_short_term_size = 100
        
        # 向量存储初始化
        self.vector_dim = vector_dim
        self.vectors = []
        self.vector_ids = []
        self.vector_metadata = {}
        
        # 如果FAISS可用，初始化FAISS索引
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(vector_dim)  # 使用L2距离的平面索引
            self.is_faiss_index = True
        else:
            self.index = None
            self.is_faiss_index = False
        
        # 记忆转换阈值
        self.importance_threshold = 0.6  # 重要性阈值
        self.repetition_threshold = 3  # 重复次数阈值
        
        # 记忆处理队列
        self.memory_queue = Queue()
        self.processing_thread = None
        self.processing_active = False
        
        # 初始化处理线程
        self.start_memory_processing()
        
        self.event_system = event_system
        
    def add_to_short_term(self, memory_item):
        """
        添加记忆到短期记忆
        
        Args:
            memory_item (dict): 记忆项，包含内容、时间戳和类型
            
        Returns:
            bool: 是否成功添加
        """
        if not isinstance(memory_item, dict) or 'content' not in memory_item:
            return False
            
        # 确保记忆项有时间戳和类型
        if 'timestamp' not in memory_item:
            import time
            memory_item['timestamp'] = time.time()
        if 'type' not in memory_item:
            memory_item['type'] = 'general'
            
        # 添加到短期记忆
        self.short_term_memory.append(memory_item)
        
        # 如果短期记忆超过最大容量，移除最旧的记忆
        if len(self.short_term_memory) > self.max_short_term_size:
            self._consolidate_memory()
            
        # 将记忆放入处理队列
        self.memory_queue.put(memory_item)
        
        # 发布记忆创建事件
        if self.event_system:
            self.event_system.publish("memory.new", {
                "id": memory_item.get('id', str(uuid.uuid4())),
                "type": memory_item.get('type', 'unknown'),
                "timestamp": time.time()
            })
            
        return True
    
    def _consolidate_memory(self, items_to_move=10):
        """
        将短期记忆中的部分内容整合到长期记忆
        
        Args:
            items_to_move (int): 要移动的记忆项数量
        """
        if len(self.short_term_memory) <= items_to_move:
            return
            
        # 获取最旧的记忆项
        oldest_items = sorted(self.short_term_memory, key=lambda x: x.get('timestamp', 0))[:items_to_move]
        
        # 移动到长期记忆
        for item in oldest_items:
            # 根据类型分类存储
            memory_type = item.get('type', 'general')
            if memory_type not in self.long_term_memory:
                self.long_term_memory[memory_type] = []
                
            self.long_term_memory[memory_type].append(item)
            
            # 如果有向量表示，添加到向量存储
            if 'vector' in item or 'content' in item:
                self._add_to_vector_store(item)
                
            # 添加到图存储
            self._add_to_graph(item)
            
            self.short_term_memory.remove(item)
    
    def _add_to_long_term(self, memory_item):
        """
        添加记忆到长期记忆
        
        Args:
            memory_item (dict): 记忆项
        """
        memory_type = memory_item.get('type', 'general')
        
        # 确保该类型的记忆容器存在
        if memory_type not in self.long_term_memory:
            self.long_term_memory[memory_type] = []
            
        # 添加到长期记忆
        self.long_term_memory[memory_type].append(memory_item)
        
        # 如果有向量存储，也添加到向量存储
        if self.vector_store and hasattr(self.vector_store, 'add_item'):
            self.vector_store.add_item(memory_item)
            
        # 更新记忆图
        self._update_memory_graph(memory_item)
    
    def _update_memory_graph(self, memory_item):
        """
        更新记忆图，建立记忆项之间的关联
        
        Args:
            memory_item (dict): 记忆项
        """
        # 简单的图更新逻辑，实际应用中可能需要更复杂的图结构
        item_id = memory_item.get('id', str(hash(str(memory_item))))
        
        # 将记忆项添加到图中
        if item_id not in self.memory_graph:
            self.memory_graph[item_id] = {
                'data': memory_item,
                'connections': []
            }
            
        # 建立与相关记忆的连接
        for existing_id, node in self.memory_graph.items():
            if existing_id != item_id:
                # 这里可以实现更复杂的相关性判断逻辑
                similarity = self._calculate_similarity(memory_item, node['data'])
                if similarity > 0.5:  # 相似度阈值
                    self.memory_graph[item_id]['connections'].append(existing_id)
                    self.memory_graph[existing_id]['connections'].append(item_id)
    
    def _calculate_similarity(self, item1, item2):
        """
        计算两个记忆项之间的相似度
        
        Args:
            item1 (dict): 第一个记忆项
            item2 (dict): 第二个记忆项
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 简单的相似度计算，实际应用中可能使用向量相似度或其他方法
        if item1.get('type') == item2.get('type'):
            return 0.7  # 同类型记忆有一定相似度
        return 0.3  # 不同类型记忆相似度较低
    
    def query_memory(self, query, memory_type=None, use_vector=True):
        """
        查询记忆
        
        Args:
            query (str/dict): 查询内容
            memory_type (str, optional): 限定查询的记忆类型
            use_vector (bool): 是否使用向量搜索
            
        Returns:
            list: 相关记忆列表
        """
        results = []
        
        # 首先在短期记忆中查找
        for item in self.short_term_memory:
            if memory_type and item.get('type') != memory_type:
                continue
                
            # 简单的文本匹配，实际应用中可能使用更复杂的匹配逻辑
            if isinstance(query, str) and query.lower() in str(item.get('content', '')).lower():
                results.append(item)
        
        # 然后在长期记忆中查找
        if memory_type and memory_type in self.long_term_memory:
            for item in self.long_term_memory[memory_type]:
                if isinstance(query, str) and query.lower() in str(item.get('content', '')).lower():
                    results.append(item)
        elif not memory_type:
            for memory_list in self.long_term_memory.values():
                for item in memory_list:
                    if isinstance(query, str) and query.lower() in str(item.get('content', '')).lower():
                        results.append(item)
        
        # 如果启用向量搜索且有向量存储
        if use_vector and self.vector_store and hasattr(self.vector_store, 'search'):
            vector_results = self.vector_store.search(query, limit=10)
            results.extend([r for r in vector_results if r not in results])
            
        return results
        
    def retrieve_from_short_term(self, query=None, memory_type=None, limit=10):
        """
        从短期记忆中检索信息
        
        Args:
            query (str, optional): 查询字符串
            memory_type (str, optional): 记忆类型
            limit (int, optional): 返回结果数量限制
            
        Returns:
            list: 检索结果
        """
        results = []
        
        for item in self.short_term_memory:
            # 类型过滤
            if memory_type and item.get('type') != memory_type:
                continue
                
            # 内容匹配
            if query and query.lower() not in item.get('content', '').lower():
                continue
                
            results.append(item)
            
            if len(results) >= limit:
                break
                
        return results
        
    def _add_to_vector_store(self, memory_item):
        """
        将记忆项添加到向量存储
        
        Args:
            memory_item (dict): 记忆项
            
        Returns:
            bool: 是否成功添加
        """
        # 获取或生成向量
        vector = memory_item.get('vector')
        if vector is None and 'content' in memory_item:
            # 实际应用中，这里应该调用嵌入模型生成向量
            # 这里使用随机向量作为示例
            vector = np.random.rand(self.vector_dim).astype('float32')
            memory_item['vector'] = vector.tolist()  # 存储向量到记忆项
        
        if vector is None:
            return False
            
        # 确保向量是numpy数组且类型正确
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype='float32')
        if vector.dtype != np.float32:
            vector = vector.astype('float32')
            
        # 确保向量维度正确
        if vector.shape[0] != self.vector_dim:
            return False
            
        # 生成ID
        memory_id = memory_item.get('id', str(uuid.uuid4()))
        
        # 添加到向量存储
        if self.is_faiss_index:
            self.index.add(vector.reshape(1, -1))  # FAISS需要2D数组
            self.vector_ids.append(memory_id)
            self.vector_metadata[memory_id] = {
                'content': memory_item.get('content', ''),
                'type': memory_item.get('type', 'general'),
                'timestamp': memory_item.get('timestamp', time.time())
            }
        else:
            # 基础向量存储
            self.vectors.append(vector)
            self.vector_ids.append(memory_id)
            self.vector_metadata[memory_id] = {
                'content': memory_item.get('content', ''),
                'type': memory_item.get('type', 'general'),
                'timestamp': memory_item.get('timestamp', time.time())
            }
            
        return True
        
    def _add_to_graph(self, memory_item):
        """
        将记忆项添加到图存储
        
        Args:
            memory_item (dict): 记忆项
            
        Returns:
            bool: 是否成功添加
        """
        # 获取记忆ID
        memory_id = memory_item.get('id', str(uuid.uuid4()))
        
        # 添加节点
        self.memory_graph.add_node(
            memory_id,
            content=memory_item.get('content', ''),
            type=memory_item.get('type', 'general'),
            timestamp=memory_item.get('timestamp', time.time())
        )
        
        # 处理关系
        relations = memory_item.get('relations', [])
        for relation in relations:
            target_id = relation.get('target_id')
            rel_type = relation.get('type', 'related_to')
            
            if target_id and target_id in self.memory_graph:
                self.memory_graph.add_edge(
                    memory_id,
                    target_id,
                    type=rel_type,
                    weight=relation.get('weight', 1.0),
                    timestamp=time.time()
                )
                
        return True
        
    def search_by_vector(self, query_vector, limit=5):
        """
        通过向量相似度搜索记忆
        
        Args:
            query_vector (np.ndarray): 查询向量
            limit (int): 返回结果数量限制
            
        Returns:
            list: 检索结果
        """
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype='float32')
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype('float32')
            
        # 确保向量维度正确
        if query_vector.shape[0] != self.vector_dim:
            return []
            
        # 使用FAISS搜索
        if self.is_faiss_index and self.index.ntotal > 0:
            distances, indices = self.index.search(query_vector.reshape(1, -1), min(limit, self.index.ntotal))
            results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.vector_ids):
                    memory_id = self.vector_ids[idx]
                    metadata = self.vector_metadata.get(memory_id, {})
                    results.append({
                        'id': memory_id,
                        'content': metadata.get('content', ''),
                        'type': metadata.get('type', 'general'),
                        'timestamp': metadata.get('timestamp', 0),
                        'distance': float(distances[0][i])
                    })
                    
            return results
        elif not self.is_faiss_index and self.vectors:
            # 基础向量搜索
            distances = []
            for i, vector in enumerate(self.vectors):
                # 计算欧几里得距离
                dist = np.linalg.norm(query_vector - vector)
                distances.append((dist, i))
                
            # 排序并获取最近的结果
            distances.sort()
            results = []
            
            for dist, idx in distances[:limit]:
                memory_id = self.vector_ids[idx]
                metadata = self.vector_metadata.get(memory_id, {})
                results.append({
                    'id': memory_id,
                    'content': metadata.get('content', ''),
                    'type': metadata.get('type', 'general'),
                    'timestamp': metadata.get('timestamp', 0),
                    'distance': float(dist)
                })
                
            return results
        
        return []
        
    def search_by_content(self, query_text, limit=5):
        """
        通过文本内容搜索记忆
        
        Args:
            query_text (str): 查询文本
            limit (int): 返回结果数量限制
            
        Returns:
            list: 检索结果
        """
        # 实际应用中，这里应该调用嵌入模型将文本转换为向量
        # 这里使用随机向量作为示例
        query_vector = np.random.rand(self.vector_dim).astype('float32')
        
        return self.search_by_vector(query_vector, limit)
    
    def get_related_memories(self, memory_id):
        """
        获取与指定记忆相关的其他记忆
        
        Args:
            memory_id (str): 记忆ID
            
        Returns:
            list: 相关记忆列表
        """
        if memory_id not in self.memory_graph:
            return []
            
        related_ids = self.memory_graph[memory_id]['connections']
        related_memories = []
        
        for related_id in related_ids:
            if related_id in self.memory_graph:
                related_memories.append(self.memory_graph[related_id]['data'])
                
        return related_memories
    
    def clear_short_term(self):
        """
        清空短期记忆
        """
        self.short_term_memory = []
        
    def get_memory_stats(self):
        """
        获取记忆系统统计信息
        
        Returns:
            dict: 统计信息
        """
        long_term_count = sum(len(items) for items in self.long_term_memory.values())
        
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": long_term_count,
            "memory_types": list(self.long_term_memory.keys()),
            "graph_nodes": len(self.memory_graph),
            "graph_connections": sum(len(node['connections']) for node in self.memory_graph.values()) // 2
        }

    def start_memory_processing(self) -> bool:
        """
        启动记忆处理线程
        
        Returns:
            bool: 是否成功启动
        """
        if self.processing_active:
            return False
            
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._memory_processing_loop, daemon=True)
        self.processing_thread.start()
        
        return True
        
    def stop_memory_processing(self) -> bool:
        """
        停止记忆处理线程
        
        Returns:
            bool: 是否成功停止
        """
        if not self.processing_active:
            return False
            
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        return True
        
    def _memory_processing_loop(self) -> None:
        """
        记忆处理循环
        """
        while self.processing_active:
            try:
                # 从队列获取记忆项目
                try:
                    memory = self.memory_queue.get(timeout=1.0)
                except:
                    # 定期整合记忆
                    if time.time() % 60 < 1:  # 大约每分钟
                        self.consolidate_memories()
                    continue
                    
                # 处理记忆
                self._process_memory(memory)
                
                # 标记任务完成
                self.memory_queue.task_done()
                
            except Exception as e:
                print(f"记忆处理错误: {str(e)}")
                time.sleep(1)  # 出错后暂停一下
                
    def _process_memory(self, memory: Dict[str, Any]) -> None:
        """
        处理单个记忆项目
        
        Args:
            memory: 记忆项目
        """
        # 计算记忆重要性
        importance = self._calculate_memory_importance(memory)
        
        # 提取和分配标签
        tags = memory.get("tags", [])
        if not tags:
            tags = self._extract_tags(memory)
            memory["tags"] = tags
            
        # 检查是否应该立即转移到长期记忆
        if importance >= self.importance_threshold:
            self._transfer_to_long_term(memory, importance)
            
        # 查找相关记忆并建立关联
        if self.vector_store:
            content = self._extract_memory_content(memory)
            similar_memories = self.vector_store.search(content, k=3)
            
            for similar in similar_memories:
                similarity = similar.get("score", 0)
                if similarity > 0.7:  # 相似度阈值
                    self.add_memory_relation(memory["id"], similar["id"], "content_similarity")
                    
    def _extract_memory_content(self, memory: Dict[str, Any]) -> str:
        """
        提取记忆内容用于向量化
        
        Args:
            memory: 记忆项目
            
        Returns:
            str: 记忆内容文本
        """
        content_parts = []
        
        # 提取不同类型的内容
        if "content" in memory:
            content_parts.append(str(memory["content"]))
            
        if "message" in memory:
            content_parts.append(str(memory["message"]))
            
        if "description" in memory:
            content_parts.append(str(memory["description"]))
            
        if "text" in memory:
            content_parts.append(str(memory["text"]))
            
        # 如果有标签，也添加到内容中
        if "tags" in memory and memory["tags"]:
            content_parts.append("标签: " + " ".join(memory["tags"]))
            
        # 如果有类型，添加到内容
        if "type" in memory:
            content_parts.append(f"类型: {memory['type']}")
            
        # 如果内容仍为空，使用整个记忆的字符串表示
        if not content_parts:
            # 过滤掉特定字段
            filtered_memory = {k: v for k, v in memory.items() if k not in ["id", "created_at", "access_count", "last_accessed"]}
            content_parts.append(str(filtered_memory))
            
        return " ".join(content_parts)
        
    def _extract_tags(self, memory: Dict[str, Any]) -> List[str]:
        """
        从记忆中提取标签
        
        Args:
            memory: 记忆项目
            
        Returns:
            List[str]: 标签列表
        """
        tags = []
        
        # 添加记忆类型作为标签
        if "type" in memory:
            tags.append(memory["type"])
            
        # 提取关键词作为标签（简化版）
        content = self._extract_memory_content(memory)
        words = content.split()
        
        # 计算词频
        word_freq = defaultdict(int)
        for word in words:
            # 忽略短词和常见词
            if len(word) >= 4 and word.lower() not in ["this", "that", "with", "from", "have", "were", "they", "what"]:
                word_freq[word.lower()] += 1
                
        # 提取频率最高的几个词作为标签
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, _ in top_words:
            tags.append(word)
            
        return tags
        
    def _calculate_memory_importance(self, memory: Dict[str, Any]) -> float:
        """
        计算记忆重要性
        
        Args:
            memory: 记忆项目
            
        Returns:
            float: 重要性分数 (0-1)
        """
        importance = 0.5  # 默认中等重要性
        
        # 基于记忆类型调整
        memory_type = memory.get("type", "").lower()
        
        if "error" in memory_type:
            importance += 0.2  # 错误更重要
        elif "user" in memory_type:
            importance += 0.15  # 用户相关更重要
        elif "decision" in memory_type:
            importance += 0.1  # 决策相关更重要
            
        # 基于访问计数调整
        access_count = memory.get("access_count", 0)
        importance += min(0.1 * (access_count / 5), 0.2)  # 最多+0.2
        
        # 基于内容长度调整
        content = self._extract_memory_content(memory)
        content_length = len(content)
        if content_length > 500:
            importance += 0.1  # 较长内容可能更重要
            
        # 如果明确指定了重要性，使用指定值
        if "importance" in memory:
            return float(memory["importance"])
            
        # 确保在有效范围内
        return max(0.1, min(importance, 1.0))
        
    def _should_transfer_to_long_term(self, memory: Dict[str, Any]) -> bool:
        """
        判断记忆是否应该转移到长期记忆
        
        Args:
            memory: 记忆项目
            
        Returns:
            bool: 是否应该转移
        """
        # 计算重要性
        importance = self._calculate_memory_importance(memory)
        
        # 基于重要性判断
        if importance >= self.importance_threshold:
            return True
            
        # 基于访问频率判断
        access_count = memory.get("access_count", 0)
        if access_count >= self.repetition_threshold:
            return True
            
        # 基于年龄判断
        age = time.time() - memory.get("created_at", time.time())
        recent_access = time.time() - memory.get("last_accessed", time.time())
        
        # 如果是较旧但最近被访问的记忆
        if age > 3600 and recent_access < 300 and access_count > 0:
            return True
            
        return False
        
    def _transfer_to_long_term(self, memory: Dict[str, Any], importance: float) -> str:
        """
        将记忆从短期转移到长期记忆
        
        Args:
            memory: 记忆项目
            importance: 重要性分数
            
        Returns:
            str: 长期记忆ID
        """
        # 创建长期记忆副本
        memory_copy = dict(memory)
        
        # 确保记忆有标签
        if "tags" not in memory_copy or not memory_copy["tags"]:
            memory_copy["tags"] = self._extract_tags(memory)
            
        # 添加到长期记忆
        memory_id = memory_copy["id"]
        self.long_term_memory[memory_id] = memory_copy
        
        # 记录重要性
        self.memory_importance[memory_id] = importance
        
        # 添加标签索引
        for tag in memory_copy.get("tags", []):
            self.tags_index[tag].add(memory_id)
            
        # 如果有向量存储，添加向量表示
        if self.vector_store:
            content = self._extract_memory_content(memory_copy)
            self.vector_store.add_item(
                item_id=memory_id,
                text=content,
                metadata={"type": memory_copy.get("type", "unknown"), "importance": importance}
            )
            
        # 发布长期记忆创建事件
        if self.event_system:
            self.event_system.publish("memory.transfer_to_long_term", {
                "id": memory_id,
                "type": memory_copy.get("type", "unknown"),
                "importance": importance,
                "timestamp": time.time()
            })
            
        return memory_id
        
    def _evaluate_for_long_term(self, memory: Dict[str, Any]) -> bool:
        """
        评估记忆是否应该保留到长期记忆
        
        Args:
            memory: 记忆项目
            
        Returns:
            bool: 是否已转移到长期记忆
        """
        if self._should_transfer_to_long_term(memory):
            importance = self._calculate_memory_importance(memory)
            self._transfer_to_long_term(memory, importance)
            return True
            
        return False
        
    def forget_short_term(self, memory_id: str) -> bool:
        """
        从短期记忆中删除特定记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 是否成功删除
        """
        for i, memory in enumerate(self.short_term_memory):
            if memory["id"] == memory_id:
                # 先评估是否应该保留到长期记忆
                self._evaluate_for_long_term(memory)
                
                # 从短期记忆中删除
                self.short_term_memory.pop(i)
                return True
                
        return False
        
    def forget_long_term(self, memory_id: str) -> bool:
        """
        从长期记忆中删除特定记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 是否成功删除
        """
        if memory_id not in self.long_term_memory:
            return False
            
        # 获取记忆的标签
        memory = self.long_term_memory[memory_id]
        tags = memory.get("tags", [])
        
        # 从长期记忆中删除
        del self.long_term_memory[memory_id]
        
        # 从标签索引中删除
        for tag in tags:
            if memory_id in self.tags_index[tag]:
                self.tags_index[tag].remove(memory_id)
                
        # 从重要性映射中删除
        if memory_id in self.memory_importance:
            del self.memory_importance[memory_id]
            
        # 从关联图中删除
        if memory_id in self.memory_graph:
            # 删除指向该记忆的关联
            for source_id, targets in self.memory_graph.items():
                if memory_id in targets:
                    targets.remove(memory_id)
                    
            # 删除该记忆的关联
            del self.memory_graph[memory_id]
            
        # 从向量存储中删除
        if self.vector_store:
            try:
                self.vector_store.delete_item(memory_id)
            except:
                pass
                
        # 发布记忆删除事件
        if self.event_system:
            self.event_system.publish("memory.deleted", {
                "id": memory_id,
                "timestamp": time.time()
            })
            
        return True
        
    def clear_short_term(self) -> int:
        """
        清除短期记忆
        
        Returns:
            int: 清除的记忆数量
        """
        # 评估所有短期记忆
        for memory in self.short_term_memory:
            self._evaluate_for_long_term(memory)
            
        count = len(self.short_term_memory)
        self.short_term_memory = []
        
        # 发布短期记忆清除事件
        if self.event_system:
            self.event_system.publish("memory.short_term_cleared", {
                "count": count,
                "timestamp": time.time()
            })
            
        return count
        
    def query_memory(self, query_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        按类型查询记忆
        
        Args:
            query_type: 记忆类型
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 记忆列表
        """
        results = []
        
        # 先从长期记忆中查找
        for memory_id, memory in self.long_term_memory.items():
            if memory.get("type") == query_type:
                results.append(memory)
                
        # 再从短期记忆中查找
        for memory in self.short_term_memory:
            if memory.get("type") == query_type:
                results.append(memory)
                
        # 按时间倒序排序
        results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return results[:limit]
        
    def query_by_time_range(self, start_time: float, end_time: float,
                          memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        按时间范围查询记忆
        
        Args:
            start_time: 起始时间戳
            end_time: 结束时间戳
            memory_type: 记忆类型过滤
            
        Returns:
            List[Dict]: 记忆列表
        """
        results = []
        
        # 从长期记忆中查找
        for memory_id, memory in self.long_term_memory.items():
            created_at = memory.get("created_at", 0)
            
            if start_time <= created_at <= end_time:
                if memory_type is None or memory.get("type") == memory_type:
                    results.append(memory)
                    
        # 从短期记忆中查找
        for memory in self.short_term_memory:
            created_at = memory.get("created_at", 0)
            
            if start_time <= created_at <= end_time:
                if memory_type is None or memory.get("type") == memory_type:
                    results.append(memory)
                    
        # 按时间排序
        results.sort(key=lambda x: x.get("created_at", 0))
        
        return results
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 短期记忆统计
        short_term_count = len(self.short_term_memory)
        short_term_types = defaultdict(int)
        for memory in self.short_term_memory:
            memory_type = memory.get("type", "unknown")
            short_term_types[memory_type] += 1
            
        # 长期记忆统计
        long_term_count = len(self.long_term_memory)
        long_term_types = defaultdict(int)
        for memory in self.long_term_memory.values():
            memory_type = memory.get("type", "unknown")
            long_term_types[memory_type] += 1
            
        # 标签统计
        tag_counts = {tag: len(memories) for tag, memories in self.tags_index.items()}
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 关联统计
        relation_count = sum(len(targets) for targets in self.memory_graph.values()) // 2  # 除以2因为是双向的
        
        return {
            "short_term": {
                "count": short_term_count,
                "capacity": self.max_short_term_size,
                "usage": short_term_count / self.max_short_term_size if self.max_short_term_size > 0 else 0,
                "types": dict(short_term_types)
            },
            "long_term": {
                "count": long_term_count,
                "types": dict(long_term_types)
            },
            "tags": {
                "count": len(self.tags_index),
                "top_tags": dict(top_tags)
            },
            "relations": {
                "count": relation_count
            },
            "vector_store": {
                "available": self.vector_store is not None
            }
        }
        
    def query_by_content(self, content: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        根据内容查询记忆
        
        Args:
            content: 查询内容
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 记忆列表
        """
        results = []
        
        # 如果有向量存储，使用语义搜索
        if self.vector_store:
            vector_results = self.vector_store.search(content, k=limit)
            
            for result in vector_results:
                memory_id = result["id"]
                memory = self.get_from_long_term(memory_id)
                if memory:
                    results.append(memory)
        else:
            # 回退到简单的文本匹配
            content_lower = content.lower()
            
            # 搜索长期记忆
            for memory_id, memory in self.long_term_memory.items():
                memory_text = self._extract_memory_content(memory).lower()
                if content_lower in memory_text:
                    results.append(memory)
                    
                    if len(results) >= limit:
                        break
                        
            # 如果结果不足，搜索短期记忆
            if len(results) < limit:
                needed = limit - len(results)
                
                for memory in self.short_term_memory:
                    memory_text = self._extract_memory_content(memory).lower()
                    if content_lower in memory_text:
                        results.append(memory)
                        needed -= 1
                        
                        if needed <= 0:
                            break
                            
        return results

    def add_memory_relation(self, source_id: str, target_id: str, 
                          relation_type: str = "association") -> bool:
        """
        添加记忆关联
        
        Args:
            source_id: 源记忆ID
            target_id: 目标记忆ID
            relation_type: 关联类型
            
        Returns:
            bool: 是否成功添加
        """
        if source_id == target_id:
            return False  # 不能关联自身
            
        # 确保两个记忆存在
        source_exists = source_id in self.long_term_memory or any(m["id"] == source_id for m in self.short_term_memory)
        target_exists = target_id in self.long_term_memory or any(m["id"] == target_id for m in self.short_term_memory)
        
        if not source_exists or not target_exists:
            return False
            
        # 添加双向关联
        self.memory_graph[source_id].add(target_id)
        self.memory_graph[target_id].add(source_id)
        
        # 发布关联创建事件
        if self.event_system:
            self.event_system.publish("memory.relation_added", {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "timestamp": time.time()
            })
            
        return True
        
    def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """
        更新记忆重要性
        
        Args:
            memory_id: 记忆ID
            importance: 重要性分数 (0-1)
            
        Returns:
            bool: 是否成功更新
        """
        # 确保记忆存在
        memory = self.get_from_long_term(memory_id)
        if not memory:
            short_term_memory = self.get_from_short_term(memory_id)
            if short_term_memory:
                # 如果在短期记忆中且重要性高，考虑转移到长期记忆
                if importance >= self.importance_threshold:
                    self._transfer_to_long_term(short_term_memory, importance)
                return True
            return False
            
        # 更新重要性
        self.memory_importance[memory_id] = importance
        
        # 如果有向量存储，更新元数据
        if self.vector_store:
            item = self.vector_store.get_item(memory_id)
            if item:
                metadata = item.get("metadata", {})
                metadata["importance"] = importance
                
                # 更新项目（这里假设向量存储有更新元数据的能力）
                try:
                    self.vector_store.add_item(
                        item_id=memory_id,
                        vector=item.get("vector"),
                        metadata=metadata
                    )
                except:
                    pass
                    
        return True
        
    def consolidate_memories(self) -> int:
        """
        整合和优化记忆
        
        Returns:
            int: 处理的记忆数量
        """
        # 评估短期记忆，将重要的转移到长期记忆
        transfer_count = 0
        
        for memory in list(self.short_term_memory):
            if self._should_transfer_to_long_term(memory):
                importance = self._calculate_memory_importance(memory)
                self._transfer_to_long_term(memory, importance)
                transfer_count += 1
                
        # 发布记忆整合事件
        if self.event_system and transfer_count > 0:
            self.event_system.publish("memory.consolidated", {
                "transfer_count": transfer_count,
                "timestamp": time.time()
            })
            
        return transfer_count

    # 添加与学习和进化模块兼容的接口方法
    def add_to_long_term(self, memory_item):
        """
        直接添加记忆到长期记忆 (兼容接口)
        
        Args:
            memory_item (dict): 记忆项
            
        Returns:
            bool: 是否成功添加
        """
        if not isinstance(memory_item, dict):
            return False
            
        # 确保记忆项有时间戳和ID
        if 'timestamp' not in memory_item:
            memory_item['timestamp'] = time.time()
        
        if 'id' not in memory_item:
            memory_item['id'] = str(uuid.uuid4())
        
        # 确保记忆项有类型
        memory_type = memory_item.get('type', 'general')
        
        # 确保该类型的记忆容器存在
        if memory_type not in self.long_term_memory:
            self.long_term_memory[memory_type] = []
            
        # 添加到长期记忆
        self.long_term_memory[memory_type].append(memory_item)
        
        # 如果有向量存储，也添加到向量存储
        if self.vector_store and hasattr(self.vector_store, 'add_item'):
            self.vector_store.add_item(memory_item)
        else:
            # 添加到内部向量存储
            self._add_to_vector_store(memory_item)
            
        # 添加到图存储
        self._add_to_graph(memory_item)
        
        # 发布记忆添加事件
        if self.event_system:
            self.event_system.publish("memory.added_to_long_term", {
                "id": memory_item['id'],
                "type": memory_type,
                "timestamp": time.time()
            })
            
        return True
    
    def search(self, query_criteria):
        """
        搜索记忆 (兼容接口)
        
        Args:
            query_criteria (dict): 查询条件
            
        Returns:
            List[Dict]: 匹配的记忆项列表
        """
        results = []
        
        # 处理不同类型的查询
        if isinstance(query_criteria, dict):
            query_type = query_criteria.get("type")
            domain = query_criteria.get("domain")
            
            # 针对特定领域知识的查询
            if query_type == "domain_knowledge" and domain:
                # 从长期记忆中检索特定领域的知识
                domain_knowledge = []
                
                # 遍历所有类型的长期记忆
                for memory_type, memories in self.long_term_memory.items():
                    for item in memories:
                        item_domain = item.get("domain")
                        if item_domain == domain:
                            domain_knowledge.append(item)
                
                return domain_knowledge
                
            # 处理其他类型的查询
            for memory_type, memories in self.long_term_memory.items():
                if query_type and memory_type != query_type:
                    continue
                    
                for item in memories:
                    # 按照条件过滤
                    match = True
                    for key, value in query_criteria.items():
                        if key not in item or item[key] != value:
                            match = False
                            break
                            
                    if match:
                        results.append(item)
        
        # 如果是文本查询，使用内容搜索
        elif isinstance(query_criteria, str):
            content_results = self.search_by_content(query_criteria)
            results.extend(content_results)
            
        return results