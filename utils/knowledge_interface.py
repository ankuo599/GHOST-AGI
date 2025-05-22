# -*- coding: utf-8 -*-
"""
知识注入与迁移接口 (Knowledge Interface)

负责加载结构化知识图谱、外部文献或开源模型作为背景知识
提供知识导入、导出和转换功能
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

class KnowledgeInterface:
    def __init__(self, memory_system=None):
        """
        初始化知识接口
        
        Args:
            memory_system: 记忆系统实例
        """
        self.memory_system = memory_system
        self.logger = logging.getLogger("KnowledgeInterface")
        self.knowledge_sources = {}  # 已注册的知识源
        self.imported_knowledge = {}  # 已导入的知识
        self.knowledge_formats = ["json", "csv", "txt", "kg"]  # 支持的知识格式
        
    def register_knowledge_source(self, source_id, source_type, source_path, metadata=None):
        """
        注册知识源
        
        Args:
            source_id: 知识源ID
            source_type: 知识源类型（文件、API、模型等）
            source_path: 知识源路径或URL
            metadata: 元数据
            
        Returns:
            bool: 是否成功注册
        """
        if source_id in self.knowledge_sources:
            self.logger.warning(f"知识源 {source_id} 已存在，将被覆盖")
            
        self.knowledge_sources[source_id] = {
            "type": source_type,
            "path": source_path,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "last_imported": None,
            "import_count": 0
        }
        
        self.logger.info(f"知识源 {source_id} 已注册，类型: {source_type}")
        return True
    
    def import_knowledge(self, source_id, format_type=None, filters=None):
        """
        从知识源导入知识
        
        Args:
            source_id: 知识源ID
            format_type: 知识格式类型，如果为None则自动检测
            filters: 导入过滤条件
            
        Returns:
            dict: 导入结果
        """
        if source_id not in self.knowledge_sources:
            self.logger.error(f"知识源 {source_id} 不存在")
            return {"success": False, "error": "知识源不存在"}
            
        source = self.knowledge_sources[source_id]
        source_type = source["type"]
        source_path = source["path"]
        
        # 自动检测格式类型
        if format_type is None and isinstance(source_path, str):
            if source_path.endswith(".json"):
                format_type = "json"
            elif source_path.endswith(".csv"):
                format_type = "csv"
            elif source_path.endswith(".txt"):
                format_type = "txt"
            else:
                format_type = "unknown"
                
        # 根据源类型和格式类型导入知识
        try:
            if source_type == "file":
                knowledge_data = self._import_from_file(source_path, format_type, filters)
            elif source_type == "api":
                knowledge_data = self._import_from_api(source_path, format_type, filters)
            elif source_type == "model":
                knowledge_data = self._import_from_model(source_path, format_type, filters)
            else:
                self.logger.error(f"不支持的知识源类型: {source_type}")
                return {"success": False, "error": f"不支持的知识源类型: {source_type}"}
                
            # 更新知识源信息
            source["last_imported"] = time.time()
            source["import_count"] += 1
            
            # 生成知识ID
            knowledge_id = f"{source_id}_{source['import_count']}"
            
            # 存储导入的知识
            self.imported_knowledge[knowledge_id] = {
                "source_id": source_id,
                "data": knowledge_data,
                "format": format_type,
                "imported_at": time.time(),
                "metadata": {
                    "filters": filters,
                    "size": self._get_knowledge_size(knowledge_data)
                }
            }
            
            # 如果有记忆系统，将知识添加到长期记忆
            if self.memory_system:
                self.memory_system.add_to_long_term({
                    "type": "imported_knowledge",
                    "id": knowledge_id,
                    "source": source_id,
                    "content": knowledge_data,
                    "timestamp": time.time()
                })
                
            self.logger.info(f"从知识源 {source_id} 成功导入知识，ID: {knowledge_id}")
            return {
                "success": True,
                "knowledge_id": knowledge_id,
                "size": self._get_knowledge_size(knowledge_data)
            }
            
        except Exception as e:
            self.logger.error(f"从知识源 {source_id} 导入知识失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _import_from_file(self, file_path, format_type, filters=None):
        """
        从文件导入知识
        
        Args:
            file_path: 文件路径
            format_type: 文件格式
            filters: 过滤条件
            
        Returns:
            dict: 知识数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        if format_type == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 应用过滤条件
            if filters and isinstance(data, dict):
                filtered_data = {}
                for key, value in data.items():
                    if key in filters.get("include_keys", [key]):
                        filtered_data[key] = value
                return filtered_data
            elif filters and isinstance(data, list):
                # 对列表类型的过滤逻辑
                if "max_items" in filters:
                    data = data[:filters["max_items"]]
                return data
            else:
                return data
                
        elif format_type == "csv":
            # 这里应该实现CSV文件的导入逻辑
            # 简化实现，返回空字典
            return {}
            
        elif format_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 简单处理文本内容
            lines = content.split("\n")
            
            # 应用过滤条件
            if filters and "max_lines" in filters:
                lines = lines[:filters["max_lines"]]
                
            return {"text": "\n".join(lines)}
            
        else:
            raise ValueError(f"不支持的文件格式: {format_type}")
    
    def _import_from_api(self, api_url, format_type, filters=None):
        """
        从API导入知识
        
        Args:
            api_url: API URL
            format_type: 数据格式
            filters: 过滤条件
            
        Returns:
            dict: 知识数据
        """
        # 这里应该实现从API获取知识的逻辑
        # 简化实现，返回空字典
        return {}
    
    def _import_from_model(self, model_path, format_type, filters=None):
        """
        从模型导入知识
        
        Args:
            model_path: 模型路径
            format_type: 数据格式
            filters: 过滤条件
            
        Returns:
            dict: 知识数据
        """
        # 这里应该实现从模型提取知识的逻辑
        # 简化实现，返回空字典
        return {}
    
    def _get_knowledge_size(self, knowledge_data):
        """
        获取知识数据大小
        
        Args:
            knowledge_data: 知识数据
            
        Returns:
            int: 数据大小（字节）
        """
        if isinstance(knowledge_data, dict):
            return len(json.dumps(knowledge_data))
        elif isinstance(knowledge_data, list):
            return len(json.dumps(knowledge_data))
        elif isinstance(knowledge_data, str):
            return len(knowledge_data)
        else:
            return 0
    
    def get_knowledge(self, knowledge_id):
        """
        获取已导入的知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            dict: 知识数据，如果不存在则返回None
        """
        if knowledge_id not in self.imported_knowledge:
            return None
            
        return self.imported_knowledge[knowledge_id]["data"]
    
    def export_knowledge(self, knowledge_id, export_path, format_type="json"):
        """
        导出知识
        
        Args:
            knowledge_id: 知识ID
            export_path: 导出路径
            format_type: 导出格式
            
        Returns:
            bool: 是否成功导出
        """
        if knowledge_id not in self.imported_knowledge:
            self.logger.error(f"知识 {knowledge_id} 不存在")
            return False
            
        knowledge = self.imported_knowledge[knowledge_id]
        knowledge_data = knowledge["data"]
        
        try:
            if format_type == "json":
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
                    
            elif format_type == "txt":
                # 将知识转换为文本格式
                if isinstance(knowledge_data, dict):
                    text_content = json.dumps(knowledge_data, ensure_ascii=False, indent=2)
                elif isinstance(knowledge_data, list):
                    text_content = "\n".join([str(item) for item in knowledge_data])
                else:
                    text_content = str(knowledge_data)
                    
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                    
            else:
                self.logger.error(f"不支持的导出格式: {format_type}")
                return False
                
            self.logger.info(f"知识 {knowledge_id} 已导出到 {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出知识 {knowledge_id} 失败: {str(e)}")
            return False
    
    def convert_knowledge_format(self, knowledge_id, target_format):
        """
        转换知识格式
        
        Args:
            knowledge_id: 知识ID
            target_format: 目标格式
            
        Returns:
            dict: 转换后的知识数据
        """
        if knowledge_id not in self.imported_knowledge:
            self.logger.error(f"知识 {knowledge_id} 不存在")
            return None
            
        if target_format not in self.knowledge_formats:
            self.logger.error(f"不支持的目标格式: {target_format}")
            return None
            
        knowledge = self.imported_knowledge[knowledge_id]
        knowledge_data = knowledge["data"]
        current_format = knowledge["format"]
        
        # 如果当前格式与目标格式相同，直接返回
        if current_format == target_format:
            return knowledge_data
            
        # 转换格式
        try:
            if target_format == "json":
                # 转换为JSON格式
                if isinstance(knowledge_data, dict) or isinstance(knowledge_data, list):
                    return knowledge_data
                else:
                    return {"content": str(knowledge_data)}
                    
            elif target_format == "txt":
                # 转换为文本格式
                if isinstance(knowledge_data, dict):
                    return json.dumps(knowledge_data, ensure_ascii=False, indent=2)
                elif isinstance(knowledge_data, list):
                    return "\n".join([str(item) for item in knowledge_data])
                else:
                    return str(knowledge_data)
                    
            elif target_format == "kg":
                # 转换为知识图谱格式
                # 这里应该实现知识图谱转换逻辑
                # 简化实现，返回原始数据
                return knowledge_data
                
            else:
                self.logger.error(f"不支持从 {current_format} 转换到 {target_format}")
                return None
                
        except Exception as e:
            self.logger.error(f"转换知识 {knowledge_id} 格式失败: {str(e)}")
            return None
    
    def merge_knowledge(self, knowledge_ids, merge_strategy="union"):
        """
        合并多个知识
        
        Args:
            knowledge_ids: 知识ID列表
            merge_strategy: 合并策略（union、intersection等）
            
        Returns:
            dict: 合并后的知识数据
        """
        if not knowledge_ids or len(knowledge_ids) < 2:
            self.logger.error("合并知识需要至少两个知识ID")
            return None
            
        # 检查所有知识是否存在
        for knowledge_id in knowledge_ids:
            if knowledge_id not in self.imported_knowledge:
                self.logger.error(f"知识 {knowledge_id} 不存在")
                return None
                
        # 获取所有知识数据
        knowledge_data_list = [self.imported_knowledge[knowledge_id]["data"] for knowledge_id in knowledge_ids]
        
        # 根据合并策略合并知识
        try:
            if merge_strategy == "union":
                # 合并字典或列表
                if all(isinstance(data, dict) for data in knowledge_data_list):
                    merged_data = {}
                    for data in knowledge_data_list:
                        merged_data.update(data)
                    return merged_data
                elif all(isinstance(data, list) for data in knowledge_data_list):
                    merged_data = []
                    for data in knowledge_data_list:
                        merged_data.extend(data)
                    return merged_data
                else:
                    self.logger.error("无法合并不同类型的知识数据")
                    return None
                    
            elif merge_strategy == "intersection":
                # 取交集
                if all(isinstance(data, dict) for data in knowledge_data_list):
                    # 取字典键的交集
                    common_keys = set(knowledge_data_list[0].keys())
                    for data in knowledge_data_list[1:]:
                        common_keys &= set(data.keys())
                        
                    merged_data = {}
                    for key in common_keys:
                        merged_data[key] = knowledge_data_list[0][key]
                    return merged_data
                elif all(isinstance(data, list) for data in knowledge_data_list):
                    # 取列表元素的交集
                    merged_data = list(set(knowledge_data_list[0]).intersection(*[set(data) for data in knowledge_data_list[1:]]))
                    return merged_data
                else:
                    self.logger.error("无法合并不同类型的知识数据")
                    return None
                    
            else:
                self.logger.error(f"不支持的合并策略: {merge_strategy}")
                return None
                
        except Exception as e:
            self.logger.error(f"合并知识失败: {str(e)}")
            return None
    
    def get_all_knowledge_sources(self):
        """
        获取所有知识源信息
        
        Returns:
            dict: 知识源信息
        """
        return self.knowledge_sources
    
    def get_all_imported_knowledge(self):
        """
        获取所有已导入的知识信息
        
        Returns:
            dict: 已导入的知识信息
        """
        # 返回知识信息，但不包括实际数据（可能很大）
        result = {}
        for knowledge_id, knowledge in self.imported_knowledge.items():
            result[knowledge_id] = {
                "source_id": knowledge["source_id"],
                "format": knowledge["format"],
                "imported_at": knowledge["imported_at"],
                "metadata": knowledge["metadata"]
            }
        return result