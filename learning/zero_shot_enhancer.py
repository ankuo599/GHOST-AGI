"""
零知识学习增强器 (Zero-Shot Learning Enhancer)

提供系统零知识学习能力的增强功能，改进系统在面对未见过的任务时的表现。
支持知识迁移、概念泛化和快速适应新领域的能力。
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from collections import defaultdict, deque
import random
import os
import json

class ZeroShotEnhancer:
    """零知识学习增强器，提升系统面对新任务的泛化能力"""
    
    def __init__(self, knowledge_system=None, memory_system=None, logger=None):
        """
        初始化零知识学习增强器
        
        Args:
            knowledge_system: 知识系统
            memory_system: 记忆系统
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 系统组件
        self.knowledge_system = knowledge_system
        self.memory_system = memory_system
        
        # 迁移学习模型
        self.transfer_models = {}  # {domain_pair: model_info}
        
        # 泛化模式库
        self.generalization_patterns = {}  # {pattern_id: pattern_info}
        
        # 零样本性能跟踪
        self.performance_tracking = {
            "attempts": 0,
            "successes": 0,
            "domains": defaultdict(lambda: {"attempts": 0, "successes": 0}),
            "history": []
        }
        
        # 适应性缓存
        self.adaptation_cache = {}  # {task_signature: adaptation_info}
        
        # 配置
        self.config = {
            "similarity_threshold": 0.75,
            "min_examples_for_transfer": 3,
            "max_pattern_complexity": 5,
            "adaptation_learning_rate": 0.1,
            "enable_meta_learning": True
        }
        
        self.logger.info("零知识学习增强器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ZeroShotEnhancer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("zero_shot_enhancer.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def enhance_zero_shot_task(self, task: Dict[str, Any], 
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        增强零样本任务处理能力
        
        Args:
            task: 任务描述
            context: 上下文信息
            
        Returns:
            Dict: 增强结果
        """
        if context is None:
            context = {}
            
        task_id = task.get("id", str(uuid.uuid4()))
        task_domain = task.get("domain", "general")
        task_description = task.get("description", "")
        
        self.logger.info(f"处理零样本任务: {task_id} ({task_domain})")
        
        # 1. 任务签名计算
        task_signature = self._compute_task_signature(task, context)
        
        # 2. 检查缓存
        if task_signature in self.adaptation_cache:
            cache_info = self.adaptation_cache[task_signature]
            self.logger.info(f"找到任务缓存匹配: {task_signature[:8]}")
            return {
                "status": "success",
                "task_id": task_id,
                "enhanced": True,
                "method": "cache_retrieval",
                "adaptations": cache_info["adaptations"],
                "confidence": cache_info["confidence"],
                "similarity": 1.0
            }
        
        # 3. 搜索相似任务
        similar_tasks = self._find_similar_tasks(task, context)
        
        # 4. 应用领域迁移
        transfer_result = self._apply_domain_transfer(task, similar_tasks, context)
        
        # 5. 泛化模式识别与应用
        pattern_result = self._apply_generalization_patterns(task, context)
        
        # 6. 组合增强
        adaptations = []
        confidence = 0.0
        
        if transfer_result["applicable"]:
            adaptations.extend(transfer_result["adaptations"])
            confidence = max(confidence, transfer_result["confidence"])
        
        if pattern_result["applicable"]:
            adaptations.extend(pattern_result["adaptations"])
            confidence = max(confidence, pattern_result["confidence"])
        
        # 如果没有可用适应，尝试快速适应生成
        if not adaptations:
            rapid_adaptation = self._generate_rapid_adaptation(task, context)
            adaptations.append(rapid_adaptation)
            confidence = rapid_adaptation.get("confidence", 0.5)
        
        # 7. 更新缓存
        if confidence > 0.6:
            self.adaptation_cache[task_signature] = {
                "adaptations": adaptations,
                "confidence": confidence,
                "created_at": time.time()
            }
        
        # 8. 更新性能跟踪
        self.performance_tracking["attempts"] += 1
        self.performance_tracking["domains"][task_domain]["attempts"] += 1
        
        # 记录历史
        self.performance_tracking["history"].append({
            "task_id": task_id,
            "domain": task_domain,
            "timestamp": time.time(),
            "adaptations_count": len(adaptations),
            "confidence": confidence
        })
        
        return {
            "status": "success",
            "task_id": task_id,
            "enhanced": len(adaptations) > 0,
            "adaptations": adaptations,
            "confidence": confidence,
            "method": "combined_enhancement",
            "similar_tasks_found": len(similar_tasks)
        }
    
    def register_success(self, task_id: str, task_domain: str) -> Dict[str, Any]:
        """
        注册任务成功处理，用于跟踪和改进
        
        Args:
            task_id: 任务ID
            task_domain: 任务领域
            
        Returns:
            Dict: 注册结果
        """
        # 更新性能跟踪
        self.performance_tracking["successes"] += 1
        self.performance_tracking["domains"][task_domain]["successes"] += 1
        
        # 更新历史记录
        for entry in self.performance_tracking["history"]:
            if entry["task_id"] == task_id:
                entry["success"] = True
                break
                
        # 计算当前成功率
        attempts = self.performance_tracking["attempts"]
        successes = self.performance_tracking["successes"]
        success_rate = successes / max(1, attempts)
        
        domain_attempts = self.performance_tracking["domains"][task_domain]["attempts"]
        domain_successes = self.performance_tracking["domains"][task_domain]["successes"]
        domain_success_rate = domain_successes / max(1, domain_attempts)
        
        self.logger.info(f"任务成功 {task_id}，当前成功率: {success_rate:.2f}，领域成功率: {domain_success_rate:.2f}")
        
        return {
            "status": "success",
            "task_id": task_id,
            "overall_success_rate": success_rate,
            "domain_success_rate": domain_success_rate
        }
    
    def extract_generalization_pattern(self, successful_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从成功的任务中提取泛化模式
        
        Args:
            successful_tasks: 成功完成的任务列表
            
        Returns:
            Dict: 提取结果
        """
        if len(successful_tasks) < self.config["min_examples_for_transfer"]:
            return {
                "status": "error",
                "message": f"样本数量不足，需要至少 {self.config['min_examples_for_transfer']} 个"
            }
            
        # 简单实现: 基于任务描述和解决方案的共同特征提取模式
        common_features = self._extract_common_features(successful_tasks)
        
        if not common_features:
            return {
                "status": "error",
                "message": "无法提取有意义的共同特征"
            }
            
        # 创建泛化模式
        pattern_id = str(uuid.uuid4())
        
        pattern = {
            "id": pattern_id,
            "name": f"Pattern-{pattern_id[:8]}",
            "features": common_features,
            "source_tasks": [task["id"] for task in successful_tasks],
            "domains": list(set(task.get("domain", "general") for task in successful_tasks)),
            "complexity": len(common_features),
            "created_at": time.time(),
            "usage_count": 0,
            "success_rate": 0.0
        }
        
        # 保存模式
        self.generalization_patterns[pattern_id] = pattern
        
        self.logger.info(f"已提取泛化模式: {pattern['name']} (复杂度: {pattern['complexity']})")
        
        return {
            "status": "success",
            "pattern_id": pattern_id,
            "pattern": pattern
        }
    
    def create_domain_transfer(self, source_domain: str, target_domain: str, 
                             examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        创建领域间迁移模型
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            examples: 示例数据
            
        Returns:
            Dict: 创建结果
        """
        if len(examples) < self.config["min_examples_for_transfer"]:
            return {
                "status": "error",
                "message": f"示例数量不足，需要至少 {self.config['min_examples_for_transfer']} 个"
            }
            
        # 领域对标识
        domain_pair = f"{source_domain}→{target_domain}"
        
        # 创建简单迁移模型
        transfer_model = {
            "id": str(uuid.uuid4()),
            "source_domain": source_domain,
            "target_domain": target_domain,
            "examples": examples,
            "mapping_rules": self._extract_mapping_rules(examples, source_domain, target_domain),
            "created_at": time.time(),
            "updated_at": time.time(),
            "usage_count": 0,
            "success_count": 0
        }
        
        # 保存模型
        self.transfer_models[domain_pair] = transfer_model
        
        self.logger.info(f"已创建领域迁移模型: {source_domain} → {target_domain}")
        
        return {
            "status": "success",
            "domain_pair": domain_pair,
            "transfer_model": transfer_model
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取零样本任务性能统计
        
        Returns:
            Dict: 性能统计
        """
        # 计算整体成功率
        attempts = self.performance_tracking["attempts"]
        successes = self.performance_tracking["successes"]
        overall_success_rate = successes / max(1, attempts)
        
        # 领域成功率
        domain_stats = {}
        for domain, stats in self.performance_tracking["domains"].items():
            domain_attempts = stats["attempts"]
            domain_successes = stats["successes"]
            domain_stats[domain] = {
                "attempts": domain_attempts,
                "successes": domain_successes,
                "success_rate": domain_successes / max(1, domain_attempts)
            }
            
        # 计算泛化模式统计
        pattern_stats = []
        for pattern_id, pattern in self.generalization_patterns.items():
            if pattern["usage_count"] > 0:
                pattern_stats.append({
                    "id": pattern_id,
                    "name": pattern["name"],
                    "usage_count": pattern["usage_count"],
                    "success_rate": pattern["success_rate"],
                    "complexity": pattern["complexity"]
                })
                
        # 按使用量排序
        pattern_stats.sort(key=lambda x: x["usage_count"], reverse=True)
        
        # 领域迁移统计
        transfer_stats = []
        for domain_pair, model in self.transfer_models.items():
            if model["usage_count"] > 0:
                success_rate = model["success_count"] / max(1, model["usage_count"])
                transfer_stats.append({
                    "domain_pair": domain_pair,
                    "usage_count": model["usage_count"],
                    "success_count": model["success_count"],
                    "success_rate": success_rate
                })
                
        # 按成功率排序
        transfer_stats.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "overall": {
                "attempts": attempts,
                "successes": successes,
                "success_rate": overall_success_rate
            },
            "domains": domain_stats,
            "patterns": pattern_stats,
            "transfers": transfer_stats,
            "adaptation_cache_size": len(self.adaptation_cache)
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新配置
        
        Args:
            config_updates: 配置更新
            
        Returns:
            Dict: 更新后的配置
        """
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
                
        return dict(self.config)
    
    def _compute_task_signature(self, task: Dict[str, Any], context: Dict[str, Any]) -> str:
        """计算任务签名"""
        # 简化实现：使用任务描述和关键属性计算签名
        signature_parts = [
            task.get("description", ""),
            task.get("domain", ""),
            json.dumps(task.get("requirements", {}), sort_keys=True)
        ]
        
        signature_str = "||".join(signature_parts)
        return str(hash(signature_str))
    
    def _find_similar_tasks(self, task: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相似任务"""
        # 实际实现应从记忆系统中检索相似任务
        # 简化实现：返回空列表
        return []
    
    def _apply_domain_transfer(self, task: Dict[str, Any], 
                             similar_tasks: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """应用领域迁移"""
        task_domain = task.get("domain", "general")
        
        # 查找适用的迁移模型
        applicable_models = []
        
        for domain_pair, model in self.transfer_models.items():
            target_domain = model["target_domain"]
            
            if target_domain == task_domain:
                # 可能适用的模型
                applicable_models.append(model)
        
        if not applicable_models:
            return {
                "applicable": False,
                "message": f"没有找到适用于领域 {task_domain} 的迁移模型"
            }
        
        # 选择最适合的模型
        # 简单实现：选择使用次数最多的模型
        applicable_models.sort(key=lambda x: x["usage_count"], reverse=True)
        best_model = applicable_models[0]
        
        # 应用迁移规则
        adaptations = []
        
        for rule in best_model["mapping_rules"]:
            adaptation = {
                "type": "domain_transfer",
                "source": best_model["source_domain"],
                "target": best_model["target_domain"],
                "rule": rule["name"],
                "transformation": rule["transformation"],
                "confidence": rule["confidence"]
            }
            adaptations.append(adaptation)
        
        # 更新使用计数
        best_model["usage_count"] += 1
        
        return {
            "applicable": True,
            "adaptations": adaptations,
            "confidence": 0.7,  # 简化: 固定置信度
            "model_id": best_model["id"]
        }
    
    def _apply_generalization_patterns(self, task: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """应用泛化模式"""
        task_description = task.get("description", "")
        
        # 查找适用的泛化模式
        applicable_patterns = []
        
        for pattern_id, pattern in self.generalization_patterns.items():
            # 简单匹配：检查特征是否出现在任务描述中
            match_count = 0
            
            for feature in pattern["features"]:
                if feature["type"] == "keyword" and feature["value"] in task_description:
                    match_count += 1
            
            match_ratio = match_count / max(1, len(pattern["features"]))
            
            if match_ratio >= self.config["similarity_threshold"]:
                applicable_patterns.append({
                    "pattern": pattern,
                    "match_ratio": match_ratio
                })
        
        if not applicable_patterns:
            return {
                "applicable": False,
                "message": "没有找到适用的泛化模式"
            }
            
        # 按匹配度排序
        applicable_patterns.sort(key=lambda x: x["match_ratio"], reverse=True)
        best_match = applicable_patterns[0]
        
        # 应用泛化模式
        adaptations = []
        pattern = best_match["pattern"]
        
        adaptation = {
            "type": "pattern_application",
            "pattern_id": pattern["id"],
            "pattern_name": pattern["name"],
            "match_ratio": best_match["match_ratio"],
            "features": pattern["features"],
            "confidence": best_match["match_ratio"] * 0.8  # 简化: 基于匹配比例计算置信度
        }
        adaptations.append(adaptation)
        
        # 更新使用计数
        pattern["usage_count"] += 1
        
        return {
            "applicable": True,
            "adaptations": adaptations,
            "confidence": best_match["match_ratio"] * 0.8,
            "pattern_id": pattern["id"]
        }
    
    def _generate_rapid_adaptation(self, task: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """生成快速适应方案"""
        # 简化实现：创建基本的适应方案
        return {
            "type": "rapid_adaptation",
            "approach": "decomposition",
            "steps": [
                "分解任务为更小的子任务",
                "寻找任务中的关键概念",
                "应用一般性问题解决策略"
            ],
            "confidence": 0.5
        }
    
    def _extract_common_features(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取任务的共同特征"""
        # 简化实现：提取关键词
        all_descriptions = [task.get("description", "") for task in tasks]
        
        # 识别共同关键词
        common_keywords = []
        
        # 实际实现应使用NLP技术提取关键词和特征
        # 此处仅用简单逻辑代替
        keywords = ["分析", "优化", "预测", "分类", "生成", "理解", "改进"]
        
        for keyword in keywords:
            if all(keyword in desc for desc in all_descriptions):
                common_keywords.append({
                    "type": "keyword",
                    "value": keyword,
                    "importance": 0.8
                })
        
        return common_keywords
    
    def _extract_mapping_rules(self, examples: List[Dict[str, Any]], 
                            source_domain: str, target_domain: str) -> List[Dict[str, Any]]:
        """提取领域映射规则"""
        # 简化实现：创建一些基本规则
        rules = [
            {
                "name": f"{source_domain}术语映射",
                "transformation": "替换领域特定术语",
                "confidence": 0.8
            },
            {
                "name": f"{source_domain}方法适应",
                "transformation": "调整方法论以适应目标领域",
                "confidence": 0.7
            },
            {
                "name": "通用概念映射",
                "transformation": "保留通用概念和原则",
                "confidence": 0.9
            }
        ]
        
        return rules 