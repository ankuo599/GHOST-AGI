"""
零知识核心模块 (Zero Knowledge Core)

提供系统从零开始学习的核心功能，实现知识的自主发现和构建。
支持基础概念抽取、原子知识推断和自主知识网络构建。
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import re
import json
from collections import defaultdict
import threading

class ZeroKnowledgeCore:
    """零知识核心，提供从零开始学习的基础能力"""
    
    def __init__(self, knowledge_system=None, memory_system=None, logger=None, central_coordinator=None):
        """
        初始化零知识核心
        
        Args:
            knowledge_system: 知识系统
            memory_system: 记忆系统
            logger: 日志记录器
            central_coordinator: 中央协调器
        """
        # 设置日志
        self.logger = logger or self._setup_logger()
        
        # 系统组件
        self.knowledge_system = knowledge_system
        self.memory_system = memory_system
        self.central_coordinator = central_coordinator
        self.meta_learning_module = None
        
        # 原子知识存储
        self.atomic_concepts = {}  # {concept_id: concept_data}
        self.observed_patterns = []  # 观察到的模式列表
        
        # 学习状态
        self.learning_state = {
            "phase": "initialization",
            "observed_entities": set(),
            "observed_relations": set(),
            "confidence_thresholds": {
                "concept_formation": 0.6,
                "relation_inference": 0.7,
                "pattern_recognition": 0.75
            }
        }
        
        # 推理统计
        self.inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "inference_accuracy": 0.0
        }
        
        # 配置参数
        self.config = {
            "min_observations": 3,  # 形成概念的最小观察次数
            "max_inference_depth": 5,  # 最大推理深度
            "pattern_similarity_threshold": 0.8,  # 模式相似度阈值
            "enable_autonomous_learning": True,  # 是否启用自主学习
            "knowledge_verification_required": True,  # 是否需要验证生成的知识
            "max_concurrent_learning_tasks": 5,  # 最大并发学习任务数
            "learning_rate": 0.1,  # 学习速率
            "confidence_decay": 0.95,  # 概念置信度衰减
            "memory_retention": 0.9  # 记忆保留率
        }
        
        # 自适应学习参数
        self.adaptive_params = {
            "learning_rate": self.config["learning_rate"],
            "current_performance": 0.0,
            "adjustment_factor": 1.0,
            "last_adjustment_time": time.time()
        }
        
        # 核心原子知识库（系统的起始知识）
        self.seed_knowledge = {}
        
        # 初始化核心原子知识
        self._initialize_seed_knowledge()
        
        # 学习任务队列
        self.learning_tasks = []
        self.active_tasks = set()
        self.task_lock = threading.RLock()
        
        # 学习进度跟踪
        self.learning_progress = {
            "started_at": time.time(),
            "observations_count": 0,
            "concepts_learned": 0,
            "relations_inferred": 0,
            "knowledge_updates": []
        }
        
        self.logger.info("零知识核心初始化完成")
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ZeroKnowledgeCore")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("zero_knowledge.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_seed_knowledge(self):
        """初始化种子知识"""
        self.logger.info("初始化核心原子知识...")
        
        # 基本逻辑概念
        self.seed_knowledge["logic"] = {
            "concepts": {
                "true": {"description": "逻辑真值，表示命题为真"},
                "false": {"description": "逻辑假值，表示命题为假"},
                "and": {"description": "逻辑与操作，只有当所有条件都为真时结果才为真"},
                "or": {"description": "逻辑或操作，只要有一个条件为真结果就为真"},
                "not": {"description": "逻辑非操作，取反一个命题的真假值"}
            },
            "relations": [
                {"source": "true", "target": "false", "type": "opposite"},
                {"source": "and", "target": "or", "type": "complement"}
            ]
        }
        
        # 基本数学概念
        self.seed_knowledge["math"] = {
            "concepts": {
                "number": {"description": "表示计数或测量的抽象概念"},
                "addition": {"description": "将两个或多个数合并为一个数的操作"},
                "subtraction": {"description": "从一个数中减去另一个数的操作"},
                "equal": {"description": "两个值相同的关系"}
            },
            "relations": [
                {"source": "addition", "target": "subtraction", "type": "inverse_operation"}
            ]
        }
        
        # 基本物理概念
        self.seed_knowledge["physics"] = {
            "concepts": {
                "object": {"description": "物理世界中存在的实体"},
                "property": {"description": "对象的特性或属性"},
                "change": {"description": "对象状态的转变或修改"}
            },
            "relations": [
                {"source": "object", "target": "property", "type": "has"}
            ]
        }
        
        # 基本认知概念
        self.seed_knowledge["cognition"] = {
            "concepts": {
                "pattern": {"description": "重复出现的结构或关系"},
                "similarity": {"description": "对象间共享特性的程度"},
                "difference": {"description": "对象间不同特性的程度"}
            },
            "relations": [
                {"source": "similarity", "target": "difference", "type": "opposite"}
            ]
        }
        
        # 注册所有种子知识
        self._register_seed_knowledge()
        
    def _register_seed_knowledge(self):
        """将种子知识注册到系统中"""
        # 注册所有概念
        for domain, domain_knowledge in self.seed_knowledge.items():
            for concept_name, concept_data in domain_knowledge["concepts"].items():
                # 创建完整的概念数据
                full_concept = {
                    "id": f"seed_{domain}_{concept_name}",
                    "name": concept_name,
                    "domain": domain,
                    "source": "seed_knowledge",
                    "confidence": 1.0,
                    "creation_time": time.time(),
                    "is_atomic": True,
                    "verification_status": "verified",
                    "description": concept_data.get("description", "")
                }
                
                # 添加到原子概念
                self.atomic_concepts[full_concept["id"]] = full_concept
                
                # 添加到知识系统(如果可用)
                if self.knowledge_system:
                    try:
                        self.knowledge_system.add_concept(full_concept)
                    except Exception as e:
                        self.logger.warning(f"无法添加种子概念到知识系统: {str(e)}")
        
        # 注册所有关系
        for domain, domain_knowledge in self.seed_knowledge.items():
            for relation in domain_knowledge.get("relations", []):
                source_id = f"seed_{domain}_{relation['source']}"
                target_id = f"seed_{domain}_{relation['target']}"
                
                # 如果知识系统可用，添加关系
                if self.knowledge_system:
                    try:
                        self.knowledge_system.add_relation(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=relation["type"],
                            confidence=1.0
                        )
                    except Exception as e:
                        self.logger.warning(f"无法添加种子关系到知识系统: {str(e)}")
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理观察数据，从中提取概念和关系
        
        Args:
            observation: 观察数据
            
        Returns:
            Dict: 处理结果
        """
        self.logger.info("处理观察数据...")
        
        # 检查观察数据格式
        if not isinstance(observation, dict) or "content" not in observation:
            return {
                "status": "error",
                "message": "观察数据格式无效，缺少content字段"
            }
            
        # 提取观察内容
        content = observation["content"]
        source = observation.get("source", "unknown")
        timestamp = observation.get("timestamp", time.time())
        
        # 创建观察记录
        observation_id = str(uuid.uuid4())
        observation_record = {
            "id": observation_id,
            "content": content,
            "source": source,
            "timestamp": timestamp,
            "processed": False,
            "extracted_concepts": [],
            "extracted_relations": []
        }
        
        # 更新学习进度
        self.learning_progress["observations_count"] += 1
        
        # 提取实体和概念
        try:
            extracted_concepts = self._extract_concepts(content)
            observation_record["extracted_concepts"] = extracted_concepts
            
            # 更新学习状态
            for concept in extracted_concepts:
                self.learning_state["observed_entities"].add(concept["name"])
                
            # 更新学习进度
            self.learning_progress["concepts_learned"] += len(extracted_concepts)
                
            self.logger.info(f"从观察中提取了 {len(extracted_concepts)} 个概念")
            
            # 如果有足够的概念，安排概念优化任务
            if extracted_concepts and self.config["enable_autonomous_learning"]:
                self._schedule_concept_optimization(extracted_concepts)
                
        except Exception as e:
            self.logger.error(f"提取概念时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        # 提取关系
        try:
            extracted_relations = self._extract_relations(content, extracted_concepts)
            observation_record["extracted_relations"] = extracted_relations
            
            # 更新学习状态
            for relation in extracted_relations:
                relation_key = f"{relation['source']}-{relation['type']}-{relation['target']}"
                self.learning_state["observed_relations"].add(relation_key)
                
            self.logger.info(f"从观察中提取了 {len(extracted_relations)} 个关系")
            
            # 如果有足够的关系，安排关系推理任务
            if extracted_relations and self.config["enable_autonomous_learning"]:
                self._schedule_relation_inference(extracted_relations)
                
        except Exception as e:
            self.logger.error(f"提取关系时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        # 识别模式
        try:
            self._identify_patterns(observation_record)
            
            # 安排模式识别任务
            if self.config["enable_autonomous_learning"]:
                self._schedule_pattern_recognition([observation_record])
                
        except Exception as e:
            self.logger.error(f"识别模式时出错: {str(e)}")
            
        # 更新观察记录状态
        observation_record["processed"] = True
        
        # 存储观察记录(如果记忆系统可用)
        if self.memory_system:
            try:
                self.memory_system.store_memory({
                    "type": "observation",
                    "content": observation_record,
                    "tags": ["zero_knowledge", "observation", source]
                })
            except Exception as e:
                self.logger.warning(f"存储观察记录到记忆系统失败: {str(e)}")
                
        # 触发知识推理
        if self.config["enable_autonomous_learning"]:
            self._trigger_knowledge_inference(observation_record)
            
        # 记录到学习进度
        self.learning_progress["knowledge_updates"].append({
            "timestamp": time.time(),
            "type": "observation_processed",
            "source": source,
            "concepts_count": len(extracted_concepts),
            "relations_count": len(extracted_relations)
        })
        
        # 检查是否应该持久化知识
        if (self.learning_progress["concepts_learned"] % 10 == 0 and 
            extracted_concepts and self.memory_system):
            self._schedule_knowledge_persistence(extracted_concepts, extracted_relations)
            
        # 通知中央协调器（如果可用）
        if self.central_coordinator and hasattr(self.central_coordinator, "publish_event"):
            try:
                event_data = {
                    "module": "zero_knowledge",
                    "event": "observation_processed",
                    "observation_id": observation_id,
                    "timestamp": time.time(),
                    "concepts_count": len(extracted_concepts),
                    "relations_count": len(extracted_relations),
                    "learning_phase": self.learning_state["phase"]
                }
                self.central_coordinator.publish_event("knowledge_update", event_data)
            except Exception as e:
                self.logger.warning(f"发布知识更新事件失败: {str(e)}")
                
        # 通知元认知模块（如果可用）
        if self.meta_learning_module and hasattr(self.meta_learning_module, "notify_observation_processed"):
            try:
                self.meta_learning_module.notify_observation_processed("zero_knowledge", observation_id, 
                                                                     len(extracted_concepts),
                                                                     len(extracted_relations))
            except Exception as e:
                self.logger.warning(f"通知元认知模块失败: {str(e)}")
            
        return {
            "status": "success",
            "observation_id": observation_id,
            "concepts_extracted": len(observation_record["extracted_concepts"]),
            "relations_extracted": len(observation_record["extracted_relations"])
        }
    
    def _extract_concepts(self, content: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从内容中提取概念
        
        Args:
            content: 观察内容
            
        Returns:
            List[Dict]: 提取的概念列表
        """
        extracted_concepts = []
        
        # 如果内容是字符串
        if isinstance(content, str):
            # 简单的概念提取(实际系统中应使用NLP技术)
            # 这里仅提供简化示例
            
            # 1. 分割为句子
            sentences = re.split(r'[.!?]', content)
            
            # 2. 提取名词短语作为概念候选
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # 提取名词短语(简化处理)
                words = sentence.strip().split()
                
                for i, word in enumerate(words):
                    # 过滤常见的停用词
                    if word.lower() in {"a", "an", "the", "this", "that", "these", "those", "it", "they"}:
                        continue
                        
                    # 创建概念候选
                    concept = {
                        "name": word,
                        "source_text": sentence,
                        "confidence": 0.6,  # 默认置信度
                        "attributes": {}
                    }
                    
                    # 检查是否有修饰词
                    if i > 0 and words[i-1].lower() not in {"a", "an", "the"}:
                        concept["attributes"]["modifier"] = words[i-1]
                        
                    extracted_concepts.append(concept)
                    
        # 如果内容是结构化数据
        elif isinstance(content, dict):
            # 处理结构化数据
            for key, value in content.items():
                # 键作为概念
                concept_key = {
                    "name": key,
                    "source_text": f"{key}: {value}",
                    "confidence": 0.7,  # 结构化数据的键通常是明确概念
                    "attributes": {}
                }
                extracted_concepts.append(concept_key)
                
                # 如果值是字符串且足够长，也将其作为概念
                if isinstance(value, str) and len(value) > 3:
                    concept_value = {
                        "name": value,
                        "source_text": f"{key}: {value}",
                        "confidence": 0.5,  # 值作为概念的置信度较低
                        "attributes": {
                            "related_to": key
                        }
                    }
                    extracted_concepts.append(concept_value)
                    
                # 如果值是列表，处理列表项
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str) and len(item) > 2:
                            concept_item = {
                                "name": item,
                                "source_text": f"{key}[{i}]: {item}",
                                "confidence": 0.6,
                                "attributes": {
                                    "part_of": key,
                                    "index": i
                                }
                            }
                            extracted_concepts.append(concept_item)
                            
        # 为概念生成唯一ID
        for concept in extracted_concepts:
            concept["id"] = f"concept_{str(uuid.uuid4())[:8]}"
            
        return extracted_concepts
    
    def _extract_relations(self, content: Union[str, Dict[str, Any]], 
                         concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从内容中提取关系
        
        Args:
            content: 观察内容
            concepts: 已提取的概念列表
            
        Returns:
            List[Dict]: 提取的关系列表
        """
        extracted_relations = []
        
        # 如果内容是字符串
        if isinstance(content, str):
            # 简单的关系提取(实际系统中应使用NLP技术)
            # 这里仅提供简化示例
            
            # 常见关系模式
            relation_patterns = [
                (r'(\w+)\s+is\s+a\s+(\w+)', "is_a"),
                (r'(\w+)\s+has\s+(\w+)', "has"),
                (r'(\w+)\s+contains\s+(\w+)', "contains"),
                (r'(\w+)\s+part\s+of\s+(\w+)', "part_of"),
                (r'(\w+)\s+belongs\s+to\s+(\w+)', "belongs_to"),
                (r'(\w+)\s+similar\s+to\s+(\w+)', "similar_to"),
                (r'(\w+)\s+causes\s+(\w+)', "causes"),
                (r'(\w+)\s+affects\s+(\w+)', "affects")
            ]
            
            # 应用关系匹配模式
            for pattern, relation_type in relation_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    source_term, target_term = match
                    
                    # 查找对应的概念
                    source_id = None
                    target_id = None
                    
                    for concept in concepts:
                        if concept["name"].lower() == source_term.lower():
                            source_id = concept["id"]
                        elif concept["name"].lower() == target_term.lower():
                            target_id = concept["id"]
                            
                    # 如果找到了概念对应，创建关系
                    if source_id and target_id:
                        relation = {
                            "id": f"relation_{str(uuid.uuid4())[:8]}",
                            "source": source_id,
                            "target": target_id,
                            "type": relation_type,
                            "confidence": 0.7,  # 默认置信度
                            "source_text": content
                        }
                        extracted_relations.append(relation)
                        
        # 如果内容是结构化数据
        elif isinstance(content, dict):
            # 处理结构化数据中的关系
            concept_map = {c["name"]: c["id"] for c in concepts}
            
            for key, value in content.items():
                if key in concept_map:
                    # 键-值关系
                    if isinstance(value, str) and value in concept_map:
                        relation = {
                            "id": f"relation_{str(uuid.uuid4())[:8]}",
                            "source": concept_map[key],
                            "target": concept_map[value],
                            "type": "related_to",
                            "confidence": 0.8,
                            "source_text": f"{key}: {value}"
                        }
                        extracted_relations.append(relation)
                        
                    # 键-列表关系
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item in concept_map:
                                relation = {
                                    "id": f"relation_{str(uuid.uuid4())[:8]}",
                                    "source": concept_map[key],
                                    "target": concept_map[item],
                                    "type": "contains",
                                    "confidence": 0.8,
                                    "source_text": f"{key}: {value}"
                                }
                                extracted_relations.append(relation)
                                
        return extracted_relations
    
    def _identify_patterns(self, observation: Dict[str, Any]):
        """
        从观察中识别模式
        
        Args:
            observation: 观察记录
        """
        # 简化实现：查找重复出现的概念和关系组合
        # TODO: 实现更复杂的模式识别
        pass
    
    def _trigger_knowledge_inference(self, observation: Dict[str, Any]):
        """
        触发知识推理，基于观察生成新知识
        
        Args:
            observation: 观察记录
        """
        # 启动异步推理任务
        if self.config["enable_autonomous_learning"]:
            threading.Thread(target=self._run_inference_process, 
                           args=(observation,), daemon=True).start()
    
    def _run_inference_process(self, observation: Dict[str, Any]):
        """
        运行推理过程
        
        Args:
            observation: 观察记录
        """
        self.logger.info(f"开始推理过程，基于观察 {observation['id']}")
        
        # 执行一系列推理步骤
        try:
            # 1. 概念泛化
            inferred_concepts = self._infer_generalizations(observation["extracted_concepts"])
            
            # 2. 关系推理
            inferred_relations = self._infer_relations(observation["extracted_relations"])
            
            # 3. 矛盾检测
            contradictions = self._detect_contradictions(inferred_concepts, inferred_relations)
            
            # 4. 知识整合
            if self.knowledge_system:
                for concept in inferred_concepts:
                    if concept["confidence"] >= self.learning_state["confidence_thresholds"]["concept_formation"]:
                        # 准备知识系统格式
                        concept_data = {
                            "id": concept["id"],
                            "name": concept["name"],
                            "description": concept.get("description", ""),
                            "domain": concept.get("domain", "general"),
                            "confidence": concept["confidence"],
                            "source": "zero_knowledge_inference",
                            "attributes": concept.get("attributes", {})
                        }
                        
                        # 如果需要验证
                        if self.config["knowledge_verification_required"]:
                            concept_data["verification_status"] = "pending"
                        else:
                            concept_data["verification_status"] = "accepted"
                            
                        # 添加到知识系统
                        try:
                            self.knowledge_system.add_concept(concept_data)
                        except Exception as e:
                            self.logger.warning(f"添加推理概念到知识系统失败: {str(e)}")
                            
                # 添加推理的关系
                for relation in inferred_relations:
                    if relation["confidence"] >= self.learning_state["confidence_thresholds"]["relation_inference"]:
                        try:
                            self.knowledge_system.add_relation(
                                source_id=relation["source"],
                                target_id=relation["target"],
                                relation_type=relation["type"],
                                confidence=relation["confidence"]
                            )
                        except Exception as e:
                            self.logger.warning(f"添加推理关系到知识系统失败: {str(e)}")
            
            # 更新推理统计
            self.inference_stats["total_inferences"] += 1
            self.inference_stats["successful_inferences"] += 1
            
            self.logger.info(f"推理过程完成，生成了 {len(inferred_concepts)} 个概念和 {len(inferred_relations)} 个关系")
            
        except Exception as e:
            self.logger.error(f"推理过程失败: {str(e)}")
            self.inference_stats["total_inferences"] += 1
            self.inference_stats["failed_inferences"] += 1
            
        finally:
            # 更新推理准确率
            if self.inference_stats["total_inferences"] > 0:
                self.inference_stats["inference_accuracy"] = (
                    self.inference_stats["successful_inferences"] / 
                    self.inference_stats["total_inferences"]
                )
    
    def _infer_generalizations(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推断概念的泛化
        
        Args:
            concepts: 概念列表
            
        Returns:
            List[Dict]: 推断的概念列表
        """
        inferred = []
        
        # 概念聚类和泛化
        if not concepts:
            return inferred
            
        # 按属性相似度对概念进行分组
        concept_groups = defaultdict(list)
        for concept in concepts:
            # 提取关键特征作为分组依据
            key_attributes = []
            if "attributes" in concept:
                for attr_name, attr_value in concept["attributes"].items():
                    if isinstance(attr_value, (str, int, float, bool)):
                        key_attributes.append(f"{attr_name}:{attr_value}")
            
            # 使用属性组合作为分组键
            group_key = "|".join(sorted(key_attributes)) if key_attributes else "no_attributes"
            concept_groups[group_key].append(concept)
        
        # 从每组中推断通用概念
        for group_key, group_concepts in concept_groups.items():
            if len(group_concepts) >= self.config["min_observations"]:
                # 提取共同名称部分
                concept_names = [c["name"] for c in group_concepts]
                
                # 找到共同词根或前缀(简化实现)
                common_prefix = self._find_common_prefix(concept_names)
                
                if common_prefix and len(common_prefix) >= 3:  # 要求前缀至少有3个字符
                    # 创建泛化概念
                    generalized_concept = {
                        "id": f"gen_{str(uuid.uuid4())[:8]}",
                        "name": common_prefix + "*",  # 使用星号表示泛化概念
                        "description": f"从 {len(group_concepts)} 个相似概念中泛化的概念",
                        "source": "generalization",
                        "confidence": min(0.5 + 0.1 * len(group_concepts), 0.9),  # 置信度随样本数增加
                        "attributes": self._extract_common_attributes(group_concepts),
                        "instances": [c["id"] for c in group_concepts],
                        "is_generalization": True
                    }
                    
                    inferred.append(generalized_concept)
                    
                    # 记录日志
                    self.logger.info(f"推断出泛化概念: {generalized_concept['name']} (基于 {len(group_concepts)} 个概念)")
        
        # 基于领域知识的泛化
        # 检查是否有可应用的泛化规则
        common_generalizations = {
            "颜色": ["红", "蓝", "绿", "黄", "黑", "白"],
            "形状": ["圆形", "方形", "三角形", "球形", "立方体"],
            "大小": ["大", "小", "中等"],
            "数量": ["单个", "多个", "几个", "许多"]
        }
        
        for concept in concepts:
            for category, instances in common_generalizations.items():
                for instance in instances:
                    if instance in concept["name"]:
                        # 创建类别泛化
                        category_concept = {
                            "id": f"cat_{str(uuid.uuid4())[:8]}",
                            "name": category,
                            "description": f"表示{category}的泛化概念",
                            "source": "category_generalization",
                            "confidence": 0.7,
                            "instances": [concept["id"]],
                            "is_generalization": True
                        }
                        
                        # 检查是否已经存在类似泛化
                        if not any(c["name"] == category for c in inferred):
                            inferred.append(category_concept)
                            self.logger.info(f"推断出类别泛化概念: {category_concept['name']}")
        
        return inferred
        
    def _find_common_prefix(self, strings: List[str]) -> str:
        """
        查找字符串列表中的共同前缀
        
        Args:
            strings: 字符串列表
            
        Returns:
            str: 共同前缀
        """
        if not strings:
            return ""
            
        # 获取最短字符串的长度
        min_length = min(len(s) for s in strings)
        
        # 找到共同前缀
        prefix = ""
        for i in range(min_length):
            char = strings[0][i]
            if all(s[i] == char for s in strings):
                prefix += char
            else:
                break
                
        return prefix
        
    def _extract_common_attributes(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        提取概念列表中的共同属性
        
        Args:
            concepts: 概念列表
            
        Returns:
            Dict: 共同属性
        """
        if not concepts:
            return {}
            
        # 获取第一个概念的属性
        common_attrs = concepts[0].get("attributes", {}).copy()
        
        # 与其他概念的属性求交集
        for concept in concepts[1:]:
            concept_attrs = concept.get("attributes", {})
            
            # 删除不一致的属性
            keys_to_remove = []
            for key, value in common_attrs.items():
                if key not in concept_attrs or concept_attrs[key] != value:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                common_attrs.pop(key, None)
                
        return common_attrs
    
    def _infer_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推断新的关系
        
        Args:
            relations: 关系列表
            
        Returns:
            List[Dict]: 推断的关系列表
        """
        inferred = []
        
        if not relations or len(relations) < 2:
            return inferred
            
        # 创建关系索引
        relation_index = defaultdict(list)
        for relation in relations:
            relation_index[relation["source"]].append(relation)
            relation_index[relation["target"]].append(relation)
            
        # 关系传递推理
        for relation in relations:
            source_id = relation["source"]
            target_id = relation["target"]
            relation_type = relation["type"]
            
            # 传递关系规则
            transitive_rules = {
                "is_a": "is_a",           # A是B，B是C => A是C
                "part_of": "part_of",     # A是B的一部分，B是C的一部分 => A是C的一部分
                "contains": "contains",   # A包含B，B包含C => A包含C
                "precedes": "precedes",   # A在B之前，B在C之前 => A在C之前
                "causes": "causes"        # A导致B，B导致C => A导致C
            }
            
            # 逆关系映射
            inverse_relations = {
                "contains": "part_of",
                "part_of": "contains",
                "precedes": "follows",
                "follows": "precedes",
                "parent_of": "child_of",
                "child_of": "parent_of",
                "larger_than": "smaller_than",
                "smaller_than": "larger_than"
            }
            
            # 传递关系推理
            if relation_type in transitive_rules:
                # 查找以target为source的关系
                for second_relation in relation_index[target_id]:
                    if second_relation["source"] == target_id and second_relation["type"] == relation_type:
                        # 创建传递关系
                        inferred_relation = {
                            "id": f"inferred_{str(uuid.uuid4())[:8]}",
                            "source": source_id,
                            "target": second_relation["target"],
                            "type": transitive_rules[relation_type],
                            "confidence": min(relation["confidence"], second_relation["confidence"]) * 0.9,
                            "inferred_from": [relation["id"], second_relation["id"]],
                            "source_text": "通过传递关系推理得出"
                        }
                        
                        # 验证新关系是否已存在
                        if not self._relation_exists(inferred_relation, relations + inferred):
                            inferred.append(inferred_relation)
                            self.logger.info(f"推断出传递关系: {inferred_relation['type']} (源:{source_id}, 目标:{inferred_relation['target']})")
            
            # 逆关系推理
            if relation_type in inverse_relations:
                inverse_type = inverse_relations[relation_type]
                inferred_relation = {
                    "id": f"inferred_{str(uuid.uuid4())[:8]}",
                    "source": target_id,
                    "target": source_id,
                    "type": inverse_type,
                    "confidence": relation["confidence"] * 0.95,
                    "inferred_from": [relation["id"]],
                    "source_text": "通过逆关系推理得出"
                }
                
                # 验证新关系是否已存在
                if not self._relation_exists(inferred_relation, relations + inferred):
                    inferred.append(inferred_relation)
                    self.logger.info(f"推断出逆关系: {inferred_relation['type']} (源:{target_id}, 目标:{source_id})")
                    
        # 对称关系推理
        symmetric_relations = {"similar_to", "related_to", "connected_to", "equals"}
        for relation in relations:
            if relation["type"] in symmetric_relations:
                inferred_relation = {
                    "id": f"inferred_{str(uuid.uuid4())[:8]}",
                    "source": relation["target"],
                    "target": relation["source"],
                    "type": relation["type"],
                    "confidence": relation["confidence"] * 0.98,
                    "inferred_from": [relation["id"]],
                    "source_text": "通过对称关系推理得出"
                }
                
                # 验证新关系是否已存在
                if not self._relation_exists(inferred_relation, relations + inferred):
                    inferred.append(inferred_relation)
                    self.logger.info(f"推断出对称关系: {inferred_relation['type']} (源:{relation['target']}, 目标:{relation['source']})")
        
        # 组合关系推理
        # 例如: A是B的一部分，B有属性C => A也可能有属性C (但置信度降低)
        for relation in relations:
            if relation["type"] == "part_of":
                part_id = relation["source"]
                whole_id = relation["target"]
                
                # 查找整体的属性关系
                for attr_relation in relation_index[whole_id]:
                    if attr_relation["source"] == whole_id and attr_relation["type"] == "has_attribute":
                        # 创建部分到属性的关系
                        inferred_relation = {
                            "id": f"inferred_{str(uuid.uuid4())[:8]}",
                            "source": part_id,
                            "target": attr_relation["target"],
                            "type": "has_attribute",
                            "confidence": min(relation["confidence"], attr_relation["confidence"]) * 0.7,
                            "inferred_from": [relation["id"], attr_relation["id"]],
                            "source_text": "通过部分-整体关系推理得出的属性"
                        }
                        
                        # 验证新关系是否已存在
                        if not self._relation_exists(inferred_relation, relations + inferred):
                            inferred.append(inferred_relation)
                            self.logger.info(f"推断出部分-整体属性关系: {inferred_relation['type']} (源:{part_id})")
        
        return inferred
        
    def _relation_exists(self, relation: Dict[str, Any], relation_list: List[Dict[str, Any]]) -> bool:
        """
        检查关系是否已经存在于关系列表中
        
        Args:
            relation: 要检查的关系
            relation_list: 关系列表
            
        Returns:
            bool: 关系是否存在
        """
        for existing in relation_list:
            if (existing["source"] == relation["source"] and 
                existing["target"] == relation["target"] and 
                existing["type"] == relation["type"]):
                return True
        return False
    
    def _detect_contradictions(self, concepts: List[Dict[str, Any]], 
                             relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测概念和关系中的矛盾
        
        Args:
            concepts: 概念列表
            relations: 关系列表
            
        Returns:
            List[Dict]: 检测到的矛盾列表
        """
        contradictions = []
        
        # 1. 检测关系矛盾
        if relations:
            # 创建关系索引
            relation_map = {}
            for relation in relations:
                key = (relation["source"], relation["target"])
                if key not in relation_map:
                    relation_map[key] = []
                relation_map[key].append(relation)
            
            # 矛盾关系对
            contradictory_pairs = {
                ("is_a", "not_a"),
                ("contains", "not_contains"),
                ("equals", "not_equals"),
                ("larger_than", "smaller_than"),
                ("before", "after"),
                ("causes", "prevents")
            }
            
            # 互斥关系类型
            exclusive_relations = {
                "larger_than": "smaller_than",
                "smaller_than": "larger_than",
                "before": "after",
                "after": "before",
                "parent_of": "child_of",
                "child_of": "parent_of"
            }
            
            # 检查相同实体对之间的矛盾关系
            for key, rel_list in relation_map.items():
                if len(rel_list) > 1:
                    for i, rel1 in enumerate(rel_list):
                        for rel2 in rel_list[i+1:]:
                            # 检查是否为矛盾关系对
                            if (rel1["type"], rel2["type"]) in contradictory_pairs or \
                               (rel2["type"], rel1["type"]) in contradictory_pairs:
                                contradiction = {
                                    "id": f"contradiction_{str(uuid.uuid4())[:8]}",
                                    "type": "relation_contradiction",
                                    "elements": [rel1["id"], rel2["id"]],
                                    "description": f"关系矛盾: {rel1['type']} 与 {rel2['type']}",
                                    "confidence": min(rel1["confidence"], rel2["confidence"]),
                                    "detected_at": time.time()
                                }
                                contradictions.append(contradiction)
                                self.logger.warning(f"检测到关系矛盾: {contradiction['description']}")
                                
                            # 检查互斥关系
                            if rel1["type"] in exclusive_relations and \
                               rel2["type"] == exclusive_relations[rel1["type"]]:
                                # 如果关系方向相同则矛盾
                                if rel1["source"] == rel2["source"] and rel1["target"] == rel2["target"]:
                                    contradiction = {
                                        "id": f"contradiction_{str(uuid.uuid4())[:8]}",
                                        "type": "exclusive_relation",
                                        "elements": [rel1["id"], rel2["id"]],
                                        "description": f"互斥关系: {rel1['type']} 与 {rel2['type']}",
                                        "confidence": min(rel1["confidence"], rel2["confidence"]),
                                        "detected_at": time.time()
                                    }
                                    contradictions.append(contradiction)
                                    self.logger.warning(f"检测到互斥关系: {contradiction['description']}")
            
            # 检测循环依赖
            self._detect_circular_dependencies(relations, contradictions)
        
        # 2. 检测概念属性矛盾
        if concepts:
            # 检查属性值矛盾
            concept_map = {c["id"]: c for c in concepts}
            
            for concept_id, concept in concept_map.items():
                if "attributes" in concept:
                    # 检查数值属性的矛盾约束
                    self._check_numerical_constraints(concept, contradictions)
                    
                    # 检查互斥属性的矛盾
                    exclusive_attributes = {
                        "size": ["large", "small", "medium"],
                        "state": ["solid", "liquid", "gas"],
                        "status": ["active", "inactive", "pending"]
                    }
                    
                    for attr_category, exclusive_values in exclusive_attributes.items():
                        found_values = []
                        for attr_name, attr_value in concept["attributes"].items():
                            if attr_name == attr_category and attr_value in exclusive_values:
                                found_values.append(attr_value)
                        
                        if len(found_values) > 1:
                            contradiction = {
                                "id": f"contradiction_{str(uuid.uuid4())[:8]}",
                                "type": "exclusive_attribute",
                                "concept_id": concept_id,
                                "attribute": attr_category,
                                "values": found_values,
                                "description": f"互斥属性值: 概念具有多个互斥的{attr_category}值",
                                "confidence": 0.9,
                                "detected_at": time.time()
                            }
                            contradictions.append(contradiction)
                            self.logger.warning(f"检测到属性矛盾: {contradiction['description']}")
        
        return contradictions
    
    def _detect_circular_dependencies(self, relations: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]):
        """
        检测关系中的循环依赖
        
        Args:
            relations: 关系列表
            contradictions: 矛盾列表
        """
        # 构建关系图
        graph = defaultdict(list)
        for relation in relations:
            # 只考虑有向关系
            directed_relations = {"is_a", "part_of", "contains", "precedes", "follows", 
                               "causes", "depends_on", "parent_of", "child_of"}
            if relation["type"] in directed_relations:
                graph[relation["source"]].append((relation["target"], relation["id"], relation["type"]))
        
        # 对每个节点执行DFS以检测环
        visited = set()
        path = []
        path_relations = []
        
        def dfs(node):
            if node in path:
                # 找到环
                cycle_start = path.index(node)
                cycle_path = path[cycle_start:]
                cycle_relations = path_relations[cycle_start:]
                
                # 创建矛盾记录
                contradiction = {
                    "id": f"contradiction_{str(uuid.uuid4())[:8]}",
                    "type": "circular_dependency",
                    "elements": cycle_relations,
                    "path": cycle_path,
                    "description": f"循环依赖: {' -> '.join(cycle_path)}",
                    "confidence": 0.85,
                    "detected_at": time.time()
                }
                contradictions.append(contradiction)
                self.logger.warning(f"检测到循环依赖: {contradiction['description']}")
                return
            
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for neighbor, relation_id, relation_type in graph[node]:
                path_relations.append(relation_id)
                dfs(neighbor)
                if path_relations:
                    path_relations.pop()
            
            path.pop()
        
        # 对所有节点执行DFS
        for node in graph:
            if node not in visited:
                dfs(node)
    
    def _check_numerical_constraints(self, concept: Dict[str, Any], contradictions: List[Dict[str, Any]]):
        """
        检查概念中数值属性的矛盾约束
        
        Args:
            concept: 概念
            contradictions: 矛盾列表
        """
        attributes = concept.get("attributes", {})
        
        # 数值约束对
        numerical_constraints = [
            ("min_value", "max_value", lambda min_val, max_val: min_val <= max_val),
            ("start", "end", lambda start, end: start <= end),
            ("width", "area", lambda width, area: width * width <= area * 1.01),  # 允许1%的误差
            ("value", "min_value", lambda val, min_val: val >= min_val),
            ("value", "max_value", lambda val, max_val: val <= max_val)
        ]
        
        for attr1, attr2, constraint_func in numerical_constraints:
            if attr1 in attributes and attr2 in attributes:
                try:
                    val1 = float(attributes[attr1])
                    val2 = float(attributes[attr2])
                    
                    if not constraint_func(val1, val2):
                        contradiction = {
                            "id": f"contradiction_{str(uuid.uuid4())[:8]}",
                            "type": "numerical_constraint",
                            "concept_id": concept["id"],
                            "attributes": [attr1, attr2],
                            "values": [val1, val2],
                            "description": f"数值约束违反: {attr1}={val1}, {attr2}={val2}",
                            "confidence": 0.95,
                            "detected_at": time.time()
                        }
                        contradictions.append(contradiction)
                        self.logger.warning(f"检测到数值约束矛盾: {contradiction['description']}")
                except (ValueError, TypeError):
                    # 忽略非数值的属性
                    pass
    
    def get_learning_state(self) -> Dict[str, Any]:
        """
        获取学习状态
        
        Returns:
            Dict: 学习状态信息
        """
        # 更新学习阶段
        self._update_learning_phase()
        
        # 构建状态报告
        state_report = {
            "phase": self.learning_state["phase"],
            "observed_entities_count": len(self.learning_state["observed_entities"]),
            "observed_relations_count": len(self.learning_state["observed_relations"]),
            "confidence_thresholds": self.learning_state["confidence_thresholds"],
            "atomic_concepts_count": len(self.atomic_concepts),
            "inference_stats": self.inference_stats,
            "configuration": self.config
        }
        
        return state_report
    
    def _update_learning_phase(self):
        """更新学习阶段"""
        entity_count = len(self.learning_state["observed_entities"])
        relation_count = len(self.learning_state["observed_relations"])
        
        if entity_count == 0:
            self.learning_state["phase"] = "initialization"
        elif entity_count < 10:
            self.learning_state["phase"] = "early_acquisition"
        elif entity_count < 50:
            self.learning_state["phase"] = "basic_learning"
        elif relation_count < 100:
            self.learning_state["phase"] = "relation_building"
        else:
            self.learning_state["phase"] = "knowledge_expansion" 
    
    def set_coordinator(self, coordinator):
        """
        设置中央协调器
        
        Args:
            coordinator: 中央协调器实例
        """
        self.central_coordinator = coordinator
        self.logger.info("已设置中央协调器")
        
        # 如果启用自主学习，订阅中央协调器事件
        if self.config["enable_autonomous_learning"] and hasattr(coordinator, "subscribe_event"):
            try:
                # 订阅相关事件
                coordinator.subscribe_event("new_observation", self._handle_observation_event)
                coordinator.subscribe_event("knowledge_update", self._handle_knowledge_update)
                self.logger.info("已订阅中央协调器事件")
            except Exception as e:
                self.logger.warning(f"订阅中央协调器事件失败: {str(e)}")
    
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        配置零知识核心参数
        
        Args:
            config: 配置参数
            
        Returns:
            Dict: 配置结果
        """
        # 更新配置
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
                
        # 同步更新自适应参数
        self.adaptive_params["learning_rate"] = self.config["learning_rate"]
        
        self.logger.info("已更新零知识核心配置")
        
        return {
            "status": "success",
            "message": "配置已更新",
            "config": self.config
        }
    
    def start(self) -> Dict[str, Any]:
        """
        启动零知识核心
        
        Returns:
            Dict: 启动结果
        """
        # 启动自适应参数调整
        self._start_adaptive_adjustment()
        
        # 启动学习任务处理
        if self.config["enable_autonomous_learning"]:
            self._start_learning_task_processor()
            
        # 通知元认知监督模块（如果可用）
        if self.meta_learning_module and hasattr(self.meta_learning_module, "notify_module_started"):
            try:
                self.meta_learning_module.notify_module_started("zero_knowledge")
            except Exception as e:
                self.logger.warning(f"通知元认知模块失败: {str(e)}")
                
        # 通知中央协调器（如果可用）
        if self.central_coordinator and hasattr(self.central_coordinator, "publish_event"):
            try:
                event_data = {
                    "module": "zero_knowledge",
                    "status": "started",
                    "timestamp": time.time(),
                    "learning_state": self.learning_state["phase"]
                }
                self.central_coordinator.publish_event("module_started", event_data)
            except Exception as e:
                self.logger.warning(f"发布启动事件失败: {str(e)}")
                
        return {
            "status": "success",
            "message": "零知识核心已启动",
            "learning_state": self.learning_state
        }
    
    def stop(self) -> Dict[str, Any]:
        """
        停止零知识核心
        
        Returns:
            Dict: 停止结果
        """
        # 停止自适应参数调整
        self._stop_adaptive_adjustment()
        
        # 停止学习任务处理
        self._stop_learning_task_processor()
        
        # 通知元认知监督模块（如果可用）
        if self.meta_learning_module and hasattr(self.meta_learning_module, "notify_module_stopped"):
            try:
                self.meta_learning_module.notify_module_stopped("zero_knowledge")
            except Exception as e:
                self.logger.warning(f"通知元认知模块失败: {str(e)}")
                
        # 通知中央协调器（如果可用）
        if self.central_coordinator and hasattr(self.central_coordinator, "publish_event"):
            try:
                event_data = {
                    "module": "zero_knowledge",
                    "status": "stopped",
                    "timestamp": time.time(),
                    "learning_summary": {
                        "observations": self.learning_progress["observations_count"],
                        "concepts": self.learning_progress["concepts_learned"],
                        "relations": self.learning_progress["relations_inferred"]
                    }
                }
                self.central_coordinator.publish_event("module_stopped", event_data)
            except Exception as e:
                self.logger.warning(f"发布停止事件失败: {str(e)}")
                
        return {
            "status": "success",
            "message": "零知识核心已停止",
            "learning_summary": {
                "observations": self.learning_progress["observations_count"],
                "concepts": self.learning_progress["concepts_learned"],
                "relations": self.learning_progress["relations_inferred"],
                "run_time": time.time() - self.learning_progress["started_at"]
            }
        }
        
    def _handle_observation_event(self, event_data: Dict[str, Any]):
        """
        处理观察事件
        
        Args:
            event_data: 事件数据
        """
        # 从事件数据中提取观察
        if "observation" in event_data:
            try:
                self.process_observation(event_data["observation"])
            except Exception as e:
                self.logger.error(f"处理观察事件失败: {str(e)}")
                
    def _handle_knowledge_update(self, event_data: Dict[str, Any]):
        """
        处理知识更新事件
        
        Args:
            event_data: 事件数据
        """
        # 更新相关的知识状态
        if "concept" in event_data:
            concept = event_data["concept"]
            
            # 如果是原子概念，更新本地存储
            if concept.get("is_atomic", False):
                self.atomic_concepts[concept["id"]] = concept
                self.logger.info(f"从知识更新事件更新了原子概念: {concept['name']}")
                
    def _start_adaptive_adjustment(self):
        """启动自适应参数调整"""
        self.adaptive_adjustment_active = True
        self.adaptive_adjustment_thread = threading.Thread(target=self._adaptive_adjustment_loop)
        self.adaptive_adjustment_thread.daemon = True
        self.adaptive_adjustment_thread.start()
        
    def _stop_adaptive_adjustment(self):
        """停止自适应参数调整"""
        self.adaptive_adjustment_active = False
        if hasattr(self, 'adaptive_adjustment_thread') and self.adaptive_adjustment_thread.is_alive():
            self.adaptive_adjustment_thread.join(timeout=2.0)
            
    def _adaptive_adjustment_loop(self):
        """自适应参数调整循环"""
        while self.adaptive_adjustment_active:
            # 每60秒调整一次
            time.sleep(60)
            
            try:
                # 更新性能评估
                current_performance = self._evaluate_learning_performance()
                
                # 根据性能调整学习率
                self._adjust_learning_rate(current_performance)
                
                # 调整置信度阈值
                self._adjust_confidence_thresholds()
                
                # 记录调整时间
                self.adaptive_params["last_adjustment_time"] = time.time()
                
                # 通知元认知模块（如果可用）
                if self.meta_learning_module and hasattr(self.meta_learning_module, "notify_parameter_adjustment"):
                    try:
                        adjustment_data = {
                            "module": "zero_knowledge",
                            "learning_rate": self.adaptive_params["learning_rate"],
                            "confidence_thresholds": self.learning_state["confidence_thresholds"],
                            "performance": current_performance
                        }
                        self.meta_learning_module.notify_parameter_adjustment(adjustment_data)
                    except Exception as e:
                        self.logger.warning(f"通知元认知模块参数调整失败: {str(e)}")
                        
            except Exception as e:
                self.logger.error(f"自适应参数调整失败: {str(e)}")
                
    def _evaluate_learning_performance(self) -> float:
        """
        评估学习性能
        
        Returns:
            float: 性能得分(0.0-1.0)
        """
        # 计算相关指标
        observations = self.learning_progress["observations_count"]
        if observations == 0:
            return 0.0
            
        # 计算概念生成效率
        concept_efficiency = min(1.0, self.learning_progress["concepts_learned"] / max(1, observations))
        
        # 计算关系推理效率
        relation_efficiency = min(1.0, self.learning_progress["relations_inferred"] / max(1, self.learning_progress["concepts_learned"]))
        
        # 计算推理准确率
        inference_accuracy = self.inference_stats["inference_accuracy"]
        
        # 加权计算总体性能
        performance = (0.3 * concept_efficiency + 
                      0.3 * relation_efficiency + 
                      0.4 * inference_accuracy)
                      
        self.adaptive_params["current_performance"] = performance
        
        return performance
        
    def _adjust_learning_rate(self, current_performance: float):
        """
        根据性能调整学习率
        
        Args:
            current_performance: 当前性能
        """
        previous_rate = self.adaptive_params["learning_rate"]
        
        # 如果性能很低，增加学习率
        if current_performance < 0.3:
            new_rate = previous_rate * 1.2
        # 如果性能一般，小幅调整学习率
        elif current_performance < 0.7:
            # 根据与0.5的差距来决定调整方向
            adjustment = 1.0 + (0.5 - current_performance) * 0.2
            new_rate = previous_rate * adjustment
        # 如果性能很好，减小学习率以稳定
        else:
            new_rate = previous_rate * 0.95
            
        # 限制学习率范围
        new_rate = max(0.01, min(0.5, new_rate))
        
        # 应用新学习率
        self.adaptive_params["learning_rate"] = new_rate
        self.config["learning_rate"] = new_rate
        
        self.logger.info(f"调整学习率: {previous_rate:.4f} -> {new_rate:.4f} (性能: {current_performance:.4f})")
        
    def _adjust_confidence_thresholds(self):
        """调整置信度阈值"""
        # 根据学习阶段调整阈值
        phase = self.learning_state["phase"]
        
        if phase == "early_acquisition":
            # 初始阶段宽松阈值，鼓励探索
            self.learning_state["confidence_thresholds"]["concept_formation"] = 0.5
            self.learning_state["confidence_thresholds"]["relation_inference"] = 0.6
            self.learning_state["confidence_thresholds"]["pattern_recognition"] = 0.65
        elif phase == "basic_learning":
            # 基础学习阶段，适中阈值
            self.learning_state["confidence_thresholds"]["concept_formation"] = 0.6
            self.learning_state["confidence_thresholds"]["relation_inference"] = 0.7
            self.learning_state["confidence_thresholds"]["pattern_recognition"] = 0.75
        elif phase in ["relation_building", "knowledge_expansion"]:
            # 后期阶段严格阈值，确保质量
            self.learning_state["confidence_thresholds"]["concept_formation"] = 0.7
            self.learning_state["confidence_thresholds"]["relation_inference"] = 0.75
            self.learning_state["confidence_thresholds"]["pattern_recognition"] = 0.8
            
    def _start_learning_task_processor(self):
        """启动学习任务处理器"""
        self.learning_processor_active = True
        self.learning_processor_thread = threading.Thread(target=self._learning_task_processor_loop)
        self.learning_processor_thread.daemon = True
        self.learning_processor_thread.start()
        
    def _stop_learning_task_processor(self):
        """停止学习任务处理器"""
        self.learning_processor_active = False
        if hasattr(self, 'learning_processor_thread') and self.learning_processor_thread.is_alive():
            self.learning_processor_thread.join(timeout=2.0)
            
    def _learning_task_processor_loop(self):
        """学习任务处理循环"""
        while self.learning_processor_active:
            with self.task_lock:
                # 检查是否有任务需要处理
                available_slots = self.config["max_concurrent_learning_tasks"] - len(self.active_tasks)
                tasks_to_process = []
                
                for _ in range(min(available_slots, len(self.learning_tasks))):
                    if self.learning_tasks:
                        task = self.learning_tasks.pop(0)
                        tasks_to_process.append(task)
                        self.active_tasks.add(task["id"])
                        
            # 处理任务
            for task in tasks_to_process:
                # 启动单独的线程处理每个任务
                task_thread = threading.Thread(target=self._process_learning_task, args=(task,))
                task_thread.daemon = True
                task_thread.start()
                
            # 休眠一段时间
            time.sleep(0.5)
            
    def _process_learning_task(self, task: Dict[str, Any]):
        """
        处理学习任务
        
        Args:
            task: 学习任务
        """
        try:
            task_type = task["type"]
            
            # 根据任务类型执行不同处理
            if task_type == "concept_optimization":
                self._optimize_concept_representation(task["concept_id"])
            elif task_type == "relation_inference":
                self._execute_relation_inference(task["relations"])
            elif task_type == "pattern_recognition":
                self._execute_pattern_recognition(task["observations"])
            elif task_type == "knowledge_persistence":
                self._persist_knowledge_to_memory(task["concepts"], task["relations"])
                
            # 更新任务状态
            task["status"] = "completed"
            task["completed_at"] = time.time()
            
        except Exception as e:
            self.logger.error(f"处理学习任务失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 更新任务状态
            task["status"] = "failed"
            task["error"] = str(e)
            
        finally:
            # 从活动任务中移除
            with self.task_lock:
                self.active_tasks.discard(task["id"])
                
    def _optimize_concept_representation(self, concept_id: str):
        """
        优化概念表示
        
        Args:
            concept_id: 概念ID
        """
        # 实现概念表示优化...
        pass
        
    def _execute_relation_inference(self, relations: List[Dict[str, Any]]):
        """
        执行关系推理任务
        
        Args:
            relations: 关系列表
        """
        inferred_relations = self._infer_relations(relations)
        
        # 更新学习进度
        self.learning_progress["relations_inferred"] += len(inferred_relations)
        
        # 添加到知识系统
        self._add_inferred_relations_to_knowledge(inferred_relations)
        
    def _execute_pattern_recognition(self, observations: List[Dict[str, Any]]):
        """
        执行模式识别任务
        
        Args:
            observations: 观察列表
        """
        # 实现模式识别...
        pass
        
    def _persist_knowledge_to_memory(self, concepts: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """
        将知识持久化到记忆系统
        
        Args:
            concepts: 概念列表
            relations: 关系列表
        """
        if not self.memory_system:
            self.logger.warning("无法持久化知识：记忆系统不可用")
            return
            
        try:
            # 创建知识包
            knowledge_package = {
                "id": f"knowledge_package_{str(uuid.uuid4())[:8]}",
                "timestamp": time.time(),
                "concepts": concepts,
                "relations": relations,
                "tags": ["zero_knowledge", "learned_knowledge"]
            }
            
            # 存储到记忆系统
            self.memory_system.store_memory({
                "type": "knowledge_package",
                "content": knowledge_package,
                "tags": ["knowledge", "zero_knowledge"]
            })
            
            self.logger.info(f"知识持久化完成: {len(concepts)}个概念, {len(relations)}个关系")
            
        except Exception as e:
            self.logger.error(f"知识持久化失败: {str(e)}")
            
    def _add_inferred_relations_to_knowledge(self, relations: List[Dict[str, Any]]):
        """
        将推理的关系添加到知识系统
        
        Args:
            relations: 关系列表
        """
        if not self.knowledge_system:
            return
            
        for relation in relations:
            if relation["confidence"] >= self.learning_state["confidence_thresholds"]["relation_inference"]:
                try:
                    self.knowledge_system.add_relation(
                        source_id=relation["source"],
                        target_id=relation["target"],
                        relation_type=relation["type"],
                        confidence=relation["confidence"]
                    )
                except Exception as e:
                    self.logger.warning(f"添加推理关系到知识系统失败: {str(e)}")
                    
    def get_status(self) -> Dict[str, Any]:
        """
        获取零知识核心状态
        
        Returns:
            Dict: 状态信息
        """
        # 更新学习阶段
        self._update_learning_phase()
        
        return {
            "learning_state": self.learning_state,
            "inference_stats": self.inference_stats,
            "atomic_concepts_count": len(self.atomic_concepts),
            "active_learning_tasks": len(self.active_tasks),
            "queued_learning_tasks": len(self.learning_tasks),
            "adaptive_params": self.adaptive_params,
            "learning_progress": self.learning_progress
        }

    def _schedule_concept_optimization(self, concepts: List[Dict[str, Any]]):
        """
        安排概念优化任务
        
        Args:
            concepts: 概念列表
        """
        # 仅对较复杂的概念进行优化
        for concept in concepts:
            if "attributes" in concept and len(concept["attributes"]) >= 2:
                # 创建概念优化任务
                task = {
                    "id": f"task_{str(uuid.uuid4())[:8]}",
                    "type": "concept_optimization",
                    "concept_id": concept["id"],
                    "concept_name": concept["name"],
                    "created_at": time.time(),
                    "status": "queued"
                }
                
                # 添加到任务队列
                with self.task_lock:
                    self.learning_tasks.append(task)
                    
                self.logger.info(f"安排概念优化任务: {concept['name']}")
    
    def _schedule_relation_inference(self, relations: List[Dict[str, Any]]):
        """
        安排关系推理任务
        
        Args:
            relations: 关系列表
        """
        # 只有当有足够的关系时才安排任务
        if len(relations) >= 2:
            # 创建关系推理任务
            task = {
                "id": f"task_{str(uuid.uuid4())[:8]}",
                "type": "relation_inference",
                "relations": relations,
                "created_at": time.time(),
                "status": "queued"
            }
            
            # 添加到任务队列
            with self.task_lock:
                self.learning_tasks.append(task)
                
            self.logger.info(f"安排关系推理任务，包含 {len(relations)} 个关系")
    
    def _schedule_pattern_recognition(self, observations: List[Dict[str, Any]]):
        """
        安排模式识别任务
        
        Args:
            observations: 观察列表
        """
        # 创建模式识别任务
        task = {
            "id": f"task_{str(uuid.uuid4())[:8]}",
            "type": "pattern_recognition",
            "observations": observations,
            "created_at": time.time(),
            "status": "queued"
        }
        
        # 添加到任务队列
        with self.task_lock:
            self.learning_tasks.append(task)
            
        self.logger.info(f"安排模式识别任务，包含 {len(observations)} 个观察")
    
    def _schedule_knowledge_persistence(self, concepts: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """
        安排知识持久化任务
        
        Args:
            concepts: 概念列表
            relations: 关系列表
        """
        if not self.memory_system:
            return
            
        # 创建知识持久化任务
        task = {
            "id": f"task_{str(uuid.uuid4())[:8]}",
            "type": "knowledge_persistence",
            "concepts": concepts,
            "relations": relations,
            "created_at": time.time(),
            "status": "queued"
        }
        
        # 添加到任务队列
        with self.task_lock:
            self.learning_tasks.append(task)
            
        self.logger.info(f"安排知识持久化任务：{len(concepts)}个概念, {len(relations)}个关系") 