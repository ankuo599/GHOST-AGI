"""
跨模态整合模块 (Cross-Modal Integrator)

该模块负责集成和关联来自不同感知模态的信息，创建统一的多模态表示。
支持跨模态搜索、关联发现和模态间信息转换。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import json
import os
import logging
from collections import defaultdict

class CrossModalIntegrator:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.modality_encoders = {}
        self.joint_representations = {}
        self.modality_weights = {
            "text": 1.0,
            "image": 0.8,
            "audio": 0.7,
            "video": 0.6
        }
        self.cross_modal_patterns = {}
        self.modality_transformers = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 缓存
        self.similarity_cache = {}
        self.association_cache = {}
        
        # 初始化默认模态转换器
        self._initialize_default_transformers()
        
    def _initialize_default_transformers(self):
        """初始化默认的模态转换器"""
        # 文本到图像描述转换器 (占位函数，实际应用需要实现)
        self.modality_transformers["text_to_image_description"] = lambda text: {
            "type": "image_description",
            "content": f"基于文本'{text[:20]}...'生成的图像描述"
        }
        
        # 图像到文本描述转换器 (占位函数，实际应用需要实现)
        self.modality_transformers["image_to_text"] = lambda image_data: {
            "type": "text",
            "content": "这是一个图像的文字描述"
        }
        
        # 音频到文本转换器 (占位函数，实际应用需要实现)
        self.modality_transformers["audio_to_text"] = lambda audio_data: {
            "type": "text",
            "content": "这是音频内容的转录文本"
        }
        
    def register_modality_encoder(self, modality_name: str, encoder_function: callable, weight: float = None):
        """
        注册特定模态的编码器
        
        Args:
            modality_name: 模态名称（如"text", "image", "audio"）
            encoder_function: 将该模态数据转换为向量的函数
            weight: 该模态在联合表示中的权重（可选）
        """
        self.modality_encoders[modality_name] = encoder_function
        if weight is not None:
            self.modality_weights[modality_name] = weight
        
        self.logger.info(f"已注册{modality_name}模态编码器")
        
    def register_modality_transformer(self, source_modality: str, target_modality: str, transformer_function: callable):
        """
        注册模态转换器，用于在模态间转换
        
        Args:
            source_modality: 源模态名称
            target_modality: 目标模态名称
            transformer_function: 转换函数
        """
        transformer_key = f"{source_modality}_to_{target_modality}"
        self.modality_transformers[transformer_key] = transformer_function
        
        self.logger.info(f"已注册{source_modality}到{target_modality}的转换器")
        
    def create_multimodal_embedding(self, data_dict: Dict[str, Any], entity_id: Optional[str] = None) -> Dict[str, Any]:
        """
        创建跨模态联合表示
        
        Args:
            data_dict: 包含不同模态数据的字典，键为模态名称
            entity_id: 可选的实体ID，如果未提供则自动生成
            
        Returns:
            包含联合表示和各模态嵌入的字典
        """
        # 生成或使用提供的实体ID
        if not entity_id:
            entity_id = f"multimodal:{int(time.time())}_{hash(str(data_dict))[:8]}"
        
        modality_embeddings = {}
        valid_modality_count = 0
        original_data = {}
        
        # 并行生成各模态的嵌入
        futures = {}
        for modality, data in data_dict.items():
            if modality in self.modality_encoders and data:
                encoder = self.modality_encoders[modality]
                futures[modality] = self.executor.submit(encoder, data)
                original_data[modality] = data  # 保存原始数据
        
        # 收集嵌入结果
        for modality, future in futures.items():
            try:
                embedding = future.result(timeout=10)  # 10秒超时
                if isinstance(embedding, np.ndarray):
                    modality_embeddings[modality] = embedding
                    valid_modality_count += 1
            except Exception as e:
                self.logger.error(f"为{modality}模态生成嵌入时出错: {str(e)}")
        
        if valid_modality_count == 0:
            return {"status": "error", "message": "没有成功生成任何模态嵌入"}
        
        # 规范化每个嵌入
        normalized_embeddings = {}
        for modality, embedding in modality_embeddings.items():
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embeddings[modality] = embedding / norm
            else:
                normalized_embeddings[modality] = embedding
        
        # 生成联合表示（加权平均）
        total_weight = 0
        joint_embedding = np.zeros(list(normalized_embeddings.values())[0].shape)
        
        for modality, embedding in normalized_embeddings.items():
            weight = self.modality_weights.get(modality, 0.5)
            joint_embedding += embedding * weight
            total_weight += weight
        
        if total_weight > 0:
            joint_embedding /= total_weight
        
        # 规范化联合嵌入
        joint_norm = np.linalg.norm(joint_embedding)
        if joint_norm > 0:
            joint_embedding = joint_embedding / joint_norm
        
        # 存储联合表示
        self.joint_representations[entity_id] = {
            "joint_embedding": joint_embedding,
            "modality_embeddings": modality_embeddings,
            "timestamp": time.time(),
            "modalities": list(modality_embeddings.keys()),
            "original_data": original_data
        }
        
        # 添加到向量存储（如果可用）
        if self.vector_store:
            try:
                metadata = {
                    "entity_id": entity_id,
                    "modalities": list(modality_embeddings.keys()),
                    "creation_time": time.time(),
                    "is_multimodal": True
                }
                
                self.vector_store.add_item(
                    item_id=entity_id,
                    vector=joint_embedding.tolist(),
                    metadata=metadata
                )
                
                # 为每个独立模态也添加向量
                for modality, embedding in modality_embeddings.items():
                    modality_item_id = f"{entity_id}_{modality}"
                    
                    self.vector_store.add_item(
                        item_id=modality_item_id,
                        vector=embedding.tolist(),
                        metadata={
                            "parent_id": entity_id,
                            "modality": modality,
                            "creation_time": time.time()
                        }
                    )
            except Exception as e:
                self.logger.error(f"将多模态嵌入添加到向量存储时出错: {str(e)}")
        
        return {
            "status": "success",
            "entity_id": entity_id,
            "modalities": list(modality_embeddings.keys()),
            "embedding_dimension": len(joint_embedding),
            "representation": {
                "joint": joint_embedding.tolist(),
                "modalities": {m: e.tolist() for m, e in modality_embeddings.items()}
            }
        }
    
    def retrieve_cross_modal_associations(self, query: Union[str, np.ndarray], 
                                       source_modality: str,
                                       target_modality: Optional[str] = None, 
                                       k: int = 5,
                                       threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        检索跨模态关联
        
        Args:
            query: 查询内容（字符串或向量）
            source_modality: 查询的模态类型
            target_modality: 目标模态类型，如果为None则搜索所有模态
            k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            关联列表，每个关联包含相关模态数据和相似度
        """
        # 将查询转换为向量
        query_vector = None
        
        if isinstance(query, str) and source_modality == "text" and "text" in self.modality_encoders:
            # 文本转向量
            query_vector = self.modality_encoders["text"](query)
        elif isinstance(query, dict) and source_modality in self.modality_encoders:
            # 其他模态数据转向量
            query_vector = self.modality_encoders[source_modality](query)
        elif isinstance(query, np.ndarray):
            # 已经是向量
            query_vector = query
        else:
            return {"status": "error", "message": "无法处理查询输入"}
        
        # 生成缓存键
        cache_key = f"{hash(str(query))}_{source_modality}_{target_modality}_{k}_{threshold}"
        
        # 检查缓存
        if cache_key in self.association_cache:
            # 检查缓存是否过期（10分钟）
            cached_result = self.association_cache[cache_key]
            if time.time() - cached_result.get("timestamp", 0) < 600:
                return cached_result.get("results", [])
        
        # 使用向量存储搜索相似项
        if self.vector_store:
            try:
                # 搜索相似项，获取更多结果以便过滤
                results = self.vector_store.search(query_vector.tolist(), k=k*2)
                
                # 过滤结果
                filtered_results = []
                
                for item in results:
                    metadata = item.get("metadata", {})
                    
                    # 检查是否为多模态项
                    if metadata.get("is_multimodal"):
                        modalities = metadata.get("modalities", [])
                        
                        # 如果指定了目标模态，检查该项是否包含目标模态
                        if target_modality and target_modality not in modalities:
                            continue
                        
                        # 相似度检查
                        if item.get("score", 0) < threshold:
                            continue
                            
                        filtered_results.append(item)
                    
                    # 检查单模态项
                    elif metadata.get("modality") == target_modality:
                        parent_id = metadata.get("parent_id")
                        if parent_id:
                            parent_item = self.vector_store.get_item(parent_id)
                            if parent_item and parent_item.get("score", 0) >= threshold:
                                filtered_results.append(parent_item)
                
                # 只保留前k个结果
                top_results = filtered_results[:k]
                
                # 生成模态关联
                cross_modal_associations = []
                
                for item in top_results:
                    entity_id = item.get("id")
                    metadata = item.get("metadata", {})
                    
                    # 获取完整的表示
                    if entity_id in self.joint_representations:
                        joint_rep = self.joint_representations[entity_id]
                        modalities = joint_rep.get("modalities", [])
                        
                        # 如果存在目标模态，提取该模态的表示
                        modality_data = {}
                        
                        for mod in modalities:
                            if target_modality is None or mod == target_modality:
                                # 为每个模态获取或生成描述
                                modality_data[mod] = self._generate_modality_description(entity_id, mod)
                        
                        if modality_data:
                            cross_modal_associations.append({
                                "entity_id": entity_id,
                                "similarity": item.get("score", 0),
                                "modalities": modalities,
                                "modality_data": modality_data
                            })
                
                # 缓存结果
                self.association_cache[cache_key] = {
                    "results": cross_modal_associations,
                    "timestamp": time.time()
                }
                
                return cross_modal_associations
            except Exception as e:
                self.logger.error(f"检索跨模态关联时出错: {str(e)}")
                return []
        
        # 如果没有向量存储，使用内部表示
        else:
            # 计算联合表示向量与查询向量的相似度
            similarities = []
            
            for entity_id, representation in self.joint_representations.items():
                joint_embedding = representation.get("joint_embedding")
                modalities = representation.get("modalities", [])
                
                # 如果指定了目标模态，检查该项是否包含目标模态
                if target_modality and target_modality not in modalities:
                    continue
                
                # 计算相似度
                try:
                    similarity = np.dot(query_vector, joint_embedding) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(joint_embedding))
                    
                    if similarity >= threshold:
                        similarities.append((entity_id, similarity, modalities))
                except:
                    continue
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:k]
            
            # 生成关联
            cross_modal_associations = []
            
            for entity_id, similarity, modalities in top_similarities:
                modality_data = {}
                
                for mod in modalities:
                    if target_modality is None or mod == target_modality:
                        modality_data[mod] = self._generate_modality_description(entity_id, mod)
                
                cross_modal_associations.append({
                    "entity_id": entity_id,
                    "similarity": float(similarity),
                    "modalities": modalities,
                    "modality_data": modality_data
                })
            
            # 缓存结果
            self.association_cache[cache_key] = {
                "results": cross_modal_associations,
                "timestamp": time.time()
            }
            
            return cross_modal_associations
    
    def _generate_modality_description(self, entity_id: str, modality: str) -> Dict[str, Any]:
        """
        为特定模态生成描述
        
        Args:
            entity_id: 实体ID
            modality: 模态类型
            
        Returns:
            模态描述字典
        """
        if entity_id in self.joint_representations:
            representation = self.joint_representations[entity_id]
            original_data = representation.get("original_data", {})
            
            # 优先使用原始数据
            if modality in original_data:
                if modality == "text":
                    text_content = original_data[modality]
                    if isinstance(text_content, str):
                        return {
                            "type": "text",
                            "content": text_content
                        }
                    else:
                        return {
                            "type": "text",
                            "content": str(text_content)
                        }
                
                elif modality == "image":
                    # 对于图像，可能需要返回URL或描述
                    image_data = original_data[modality]
                    if isinstance(image_data, dict) and "description" in image_data:
                        return {
                            "type": "image",
                            "description": image_data["description"],
                            "metadata": image_data.get("metadata", {})
                        }
                    else:
                        return {
                            "type": "image",
                            "reference": str(image_data),
                            "description": "图像内容" 
                        }
                
                elif modality == "audio":
                    audio_data = original_data[modality]
                    if isinstance(audio_data, dict) and "transcript" in audio_data:
                        return {
                            "type": "audio",
                            "transcript": audio_data["transcript"],
                            "metadata": audio_data.get("metadata", {})
                        }
                    else:
                        return {
                            "type": "audio",
                            "reference": str(audio_data),
                            "description": "音频内容"
                        }
                
                else:
                    # 其他模态
                    return {
                        "type": modality,
                        "data": str(original_data[modality])
                    }
            
            # 如果没有原始数据，尝试生成描述
            if modality == "text" and "text" in representation.get("modality_embeddings", {}):
                return {
                    "type": "text",
                    "content": "[文本内容不可用]"
                }
            
            elif modality == "image" and "image" in representation.get("modality_embeddings", {}):
                return {
                    "type": "image",
                    "description": "图像内容描述不可用",
                    "features": []
                }
            
            elif modality == "audio" and "audio" in representation.get("modality_embeddings", {}):
                return {
                    "type": "audio",
                    "description": "音频内容描述不可用",
                    "duration": "未知"
                }
            
            else:
                return {
                    "type": modality,
                    "description": f"未知{modality}模态内容"
                }
        
        return {"type": "unknown", "description": "无法找到模态表示"}
    
    def transform_modality(self, data: Any, source_modality: str, target_modality: str) -> Dict[str, Any]:
        """
        将一种模态的数据转换为另一种模态
        
        Args:
            data: 源模态数据
            source_modality: 源模态类型
            target_modality: 目标模态类型
            
        Returns:
            转换结果字典
        """
        transformer_key = f"{source_modality}_to_{target_modality}"
        
        if transformer_key not in self.modality_transformers:
            return {
                "status": "error", 
                "message": f"没有可用的从{source_modality}到{target_modality}的转换器"
            }
        
        try:
            transformer = self.modality_transformers[transformer_key]
            result = transformer(data)
            
            return {
                "status": "success",
                "source_modality": source_modality,
                "target_modality": target_modality,
                "result": result
            }
        except Exception as e:
            self.logger.error(f"转换模态时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"转换失败: {str(e)}",
                "source_modality": source_modality,
                "target_modality": target_modality
            }
    
    def extract_cross_modal_patterns(self) -> List[Dict[str, Any]]:
        """
        提取跨模态模式
        
        Returns:
            发现的模式列表
        """
        if len(self.joint_representations) < 5:
            return []
        
        patterns = []
        
        # 获取所有实体
        entities = list(self.joint_representations.keys())
        
        # 仅分析包含多个模态的实体
        multi_modal_entities = []
        for entity_id in entities:
            modalities = self.joint_representations[entity_id].get("modalities", [])
            if len(modalities) > 1:
                multi_modal_entities.append((entity_id, modalities))
        
        if len(multi_modal_entities) < 3:
            return []
        
        # 分析模态组合模式
        modality_combinations = {}
        for entity_id, modalities in multi_modal_entities:
            modality_key = "-".join(sorted(modalities))
            if modality_key not in modality_combinations:
                modality_combinations[modality_key] = []
            modality_combinations[modality_key].append(entity_id)
        
        # 提取常见模态组合
        for combo, entities in modality_combinations.items():
            if len(entities) >= 2:
                modalities = combo.split("-")
                patterns.append({
                    "type": "modality_combination",
                    "modalities": modalities,
                    "frequency": len(entities),
                    "entities": entities[:5]  # 只包含前5个示例
                })
        
        # 分析跨模态相关性
        for entity_id, modalities in multi_modal_entities:
            if len(modalities) < 2:
                continue
                
            representation = self.joint_representations[entity_id]
            modality_embeddings = representation.get("modality_embeddings", {})
            
            # 计算模态间相似度
            for i, mod1 in enumerate(modalities):
                if mod1 not in modality_embeddings:
                    continue
                    
                for mod2 in modalities[i+1:]:
                    if mod2 not in modality_embeddings:
                        continue
                        
                    emb1 = modality_embeddings[mod1]
                    emb2 = modality_embeddings[mod2]
                    
                    # 计算相似度
                    try:
                        similarity = np.dot(emb1, emb2) / (
                            np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        pattern_key = f"{mod1}-{mod2}"
                        if pattern_key not in self.cross_modal_patterns:
                            self.cross_modal_patterns[pattern_key] = []
                        
                        self.cross_modal_patterns[pattern_key].append({
                            "entity_id": entity_id,
                            "similarity": float(similarity)
                        })
                    except:
                        continue
        
        # 提取显著的模态间相关性
        for pattern_key, similarities in self.cross_modal_patterns.items():
            if len(similarities) < 3:
                continue
                
            avg_similarity = sum(item["similarity"] for item in similarities) / len(similarities)
            
            if avg_similarity > 0.5:  # 仅报告较强的相关性
                mod1, mod2 = pattern_key.split("-")
                patterns.append({
                    "type": "cross_modal_correlation",
                    "modality_pair": [mod1, mod2],
                    "average_similarity": avg_similarity,
                    "sample_count": len(similarities),
                    "examples": similarities[:3]  # 只包含前3个示例
                })
        
        return patterns
    
    def generate_cross_modal_description(self, entity_id: str) -> Dict[str, Any]:
        """
        生成跨模态内容的综合描述
        
        Args:
            entity_id: 实体ID
            
        Returns:
            综合描述字典
        """
        if entity_id not in self.joint_representations:
            return {"status": "error", "message": "实体不存在"}
        
        representation = self.joint_representations[entity_id]
        modalities = representation.get("modalities", [])
        
        if len(modalities) < 2:
            return {"status": "error", "message": "不是跨模态实体"}
        
        # 为每个模态获取描述
        modality_descriptions = {}
        for modality in modalities:
            modality_descriptions[modality] = self._generate_modality_description(entity_id, modality)
        
        # 生成模态间关系描述
        modal_relationships = []
        
        # 基于已知的跨模态模式生成关系描述
        for pattern_key, similarities in self.cross_modal_patterns.items():
            mod1, mod2 = pattern_key.split("-")
            
            if mod1 in modalities and mod2 in modalities:
                # 查找包含此实体的相似度记录
                for item in similarities:
                    if item["entity_id"] == entity_id:
                        similarity = item["similarity"]
                        
                        relationship = {
                            "modalities": [mod1, mod2],
                            "similarity": similarity,
                            "description": self._generate_relationship_description(mod1, mod2, similarity)
                        }
                        
                        modal_relationships.append(relationship)
                        break
        
        # 生成综合描述
        summary = self._generate_multimodal_summary(entity_id, modality_descriptions, modal_relationships)
        
        return {
            "status": "success",
            "entity_id": entity_id,
            "modalities": modalities,
            "modality_descriptions": modality_descriptions,
            "modal_relationships": modal_relationships,
            "summary": summary
        }
    
    def _generate_relationship_description(self, mod1: str, mod2: str, similarity: float) -> str:
        """
        根据模态和相似性生成关系描述
        
        Args:
            mod1: 第一个模态
            mod2: 第二个模态
            similarity: 相似度
            
        Returns:
            描述字符串
        """
        if similarity > 0.8:
            strength = "高度"
        elif similarity > 0.5:
            strength = "中度"
        else:
            strength = "轻微"
            
        if mod1 == "text" and mod2 == "image":
            return f"文字内容与图像内容{strength}一致"
        elif mod1 == "text" and mod2 == "audio":
            return f"文字内容与音频内容{strength}一致"
        elif mod1 == "image" and mod2 == "audio":
            return f"图像内容与音频内容{strength}相关"
        else:
            return f"{mod1}模态与{mod2}模态{strength}相关"
    
    def _generate_multimodal_summary(self, entity_id: str, 
                                  modality_descriptions: Dict[str, Dict[str, Any]],
                                  modal_relationships: List[Dict[str, Any]]) -> str:
        """
        生成多模态内容摘要
        
        Args:
            entity_id: 实体ID
            modality_descriptions: 各模态描述
            modal_relationships: 模态间关系
            
        Returns:
            摘要字符串
        """
        summary_parts = []
        
        # 添加文本内容（如果有）
        if "text" in modality_descriptions:
            text_desc = modality_descriptions["text"]
            if "content" in text_desc:
                content = text_desc["content"]
                # 限制长度
                if len(content) > 100:
                    content = content[:97] + "..."
                summary_parts.append(f"文本: {content}")
        
        # 添加图像描述（如果有）
        if "image" in modality_descriptions:
            image_desc = modality_descriptions["image"]
            if "description" in image_desc:
                summary_parts.append(f"图像: {image_desc['description']}")
                
        # 添加音频描述（如果有）
        if "audio" in modality_descriptions:
            audio_desc = modality_descriptions["audio"]
            if "transcript" in audio_desc:
                transcript = audio_desc["transcript"]
                if len(transcript) > 100:
                    transcript = transcript[:97] + "..."
                summary_parts.append(f"音频: {transcript}")
            elif "description" in audio_desc:
                summary_parts.append(f"音频: {audio_desc['description']}")
        
        # 添加其他模态描述
        for modality, desc in modality_descriptions.items():
            if modality not in ["text", "image", "audio"]:
                summary_parts.append(f"{modality}: {desc.get('description', '未知内容')}")
        
        # 添加模态关系描述
        if modal_relationships:
            relationship_descs = []
            for rel in modal_relationships:
                if "description" in rel:
                    relationship_descs.append(rel["description"])
            
            if relationship_descs:
                summary_parts.append("模态关系: " + "; ".join(relationship_descs))
        
        # 组合摘要
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return f"实体 {entity_id} 包含多种模态信息，但无法生成详细摘要。"
    
    def get_modality_statistics(self) -> Dict[str, Any]:
        """
        获取模态统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_entities": len(self.joint_representations),
            "modality_counts": defaultdict(int),
            "modality_combinations": defaultdict(int),
            "average_similarities": {}
        }
        
        # 计算各模态数量
        for entity_id, representation in self.joint_representations.items():
            modalities = representation.get("modalities", [])
            
            # 统计单个模态
            for modality in modalities:
                stats["modality_counts"][modality] += 1
            
            # 统计模态组合
            if len(modalities) > 1:
                combo_key = "-".join(sorted(modalities))
                stats["modality_combinations"][combo_key] += 1
        
        # 计算模态间平均相似度
        for pattern_key, similarities in self.cross_modal_patterns.items():
            if similarities:
                avg_similarity = sum(item["similarity"] for item in similarities) / len(similarities)
                stats["average_similarities"][pattern_key] = avg_similarity
        
        return stats
    
    def save_representations(self, filepath: str) -> Dict[str, Any]:
        """
        保存多模态表示到文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            操作结果
        """
        try:
            # 准备要保存的数据
            save_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "entity_count": len(self.joint_representations),
                    "version": "1.0"
                },
                "representations": {}
            }
            
            # 复制表示数据，但不包括大型numpy数组
            for entity_id, representation in self.joint_representations.items():
                # 转换numpy数组为列表
                joint_emb = representation.get("joint_embedding")
                modality_embs = representation.get("modality_embeddings", {})
                
                save_data["representations"][entity_id] = {
                    "modalities": representation.get("modalities", []),
                    "timestamp": representation.get("timestamp", 0),
                    # 只保存元数据，不保存原始数据（可能很大）
                    "has_original_data": "original_data" in representation
                }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 保存为JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            return {
                "status": "success",
                "message": f"已保存{len(self.joint_representations)}个多模态表示",
                "filepath": filepath
            }
        except Exception as e:
            self.logger.error(f"保存多模态表示时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"保存失败: {str(e)}"
            }
    
    def load_representations(self, filepath: str) -> Dict[str, Any]:
        """
        加载多模态表示
        
        Args:
            filepath: 文件路径
            
        Returns:
            操作结果
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            metadata = load_data.get("metadata", {})
            representations = load_data.get("representations", {})
            
            # 只恢复元数据（不包括向量，因为没有保存）
            loaded_count = 0
            for entity_id, rep_data in representations.items():
                if entity_id not in self.joint_representations:
                    self.joint_representations[entity_id] = {
                        "modalities": rep_data.get("modalities", []),
                        "timestamp": rep_data.get("timestamp", 0),
                        "needs_reembedding": True  # 标记需要重新嵌入
                    }
                    loaded_count += 1
            
            return {
                "status": "success",
                "message": f"已加载{loaded_count}个多模态表示元数据",
                "version": metadata.get("version", "unknown"),
                "original_count": metadata.get("entity_count", 0)
            }
        except Exception as e:
            self.logger.error(f"加载多模态表示时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"加载失败: {str(e)}"
            }
    
    def perform_cross_modal_search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行跨模态搜索
        
        Args:
            query: 包含搜索参数的字典，支持多模态查询
            
        Returns:
            搜索结果字典
        """
        results = {
            "status": "success",
            "query_time": time.time(),
            "results": []
        }
        
        try:
            # 解析查询参数
            source_modality = query.get("source_modality")
            target_modality = query.get("target_modality")
            query_content = query.get("content")
            limit = query.get("limit", 5)
            threshold = query.get("threshold", 0.6)
            
            if not source_modality or not query_content:
                return {
                    "status": "error",
                    "message": "查询必须包含source_modality和content字段"
                }
            
            # 执行搜索
            associations = self.retrieve_cross_modal_associations(
                query_content, 
                source_modality, 
                target_modality, 
                k=limit,
                threshold=threshold
            )
            
            if not associations:
                results["status"] = "no_results"
                results["message"] = "未找到匹配内容"
                return results
                
            # 处理结果
            results["count"] = len(associations)
            results["results"] = associations
            
            return results
        except Exception as e:
            self.logger.error(f"执行跨模态搜索时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"搜索失败: {str(e)}"
            } 