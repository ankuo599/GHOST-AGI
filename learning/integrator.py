# -*- coding: utf-8 -*-
"""
学习系统集成模块 (Learning System Integrator)

整合多种学习模块，协调不同学习策略，提供统一的学习接口
"""

import time
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

class LearningIntegrator:
    def __init__(self, memory_system=None, vector_store=None, event_system=None):
        """
        初始化学习系统集成器
        
        Args:
            memory_system: 记忆系统实例
            vector_store: 向量存储实例
            event_system: 事件系统实例
        """
        self.memory_system = memory_system
        self.vector_store = vector_store
        self.event_system = event_system
        self.logger = logging.getLogger("LearningIntegrator")
        
        # 学习模块列表
        self.learning_modules = {}
        
        # 学习策略映射
        self.strategy_to_module = {}
        
        # 学习统计
        self.learning_stats = {
            "total_queries": 0,
            "module_usage": {},
            "strategy_usage": {},
            "success_rate": {}
        }
        
        # 加载学习模块
        self._load_learning_modules()
        
    def _load_learning_modules(self):
        """加载和初始化所有学习模块"""
        try:
            # 导入标准学习引擎
            from learning.learning_engine import LearningEngine
            self.learning_modules["standard"] = LearningEngine(
                memory_system=self.memory_system,
                event_system=self.event_system
            )
            self.logger.info("标准学习引擎加载成功")
            
            # 导入零样本学习模块
            try:
                from learning.zero_shot_learning import ZeroShotLearningModule
                self.learning_modules["zero_shot"] = ZeroShotLearningModule(
                    memory_system=self.memory_system,
                    vector_store=self.vector_store,
                    event_system=self.event_system
                )
                self.logger.info("零样本学习模块加载成功")
            except ImportError as e:
                self.logger.warning(f"零样本学习模块加载失败: {str(e)}")
                
            # 导入强化学习模块
            try:
                from learning.reinforcement_learning import ReinforcementLearning
                self.learning_modules["reinforcement"] = ReinforcementLearning(
                    memory_system=self.memory_system,
                    event_system=self.event_system
                )
                self.logger.info("强化学习模块加载成功")
            except ImportError as e:
                self.logger.warning(f"强化学习模块加载失败: {str(e)}")
                
            # 导入增强型学习引擎（如果存在）
            try:
                from learning.enhanced_learning_engine import EnhancedLearningEngine
                self.learning_modules["enhanced"] = EnhancedLearningEngine(
                    memory_system=self.memory_system,
                    vector_store=self.vector_store,
                    event_system=self.event_system
                )
                self.logger.info("增强型学习引擎加载成功")
            except ImportError as e:
                self.logger.warning(f"增强型学习引擎加载失败: {str(e)}")
                
            # 导入进化学习模块（如果存在）
            try:
                from learning.evolution_engine import EvolutionEngine
                self.learning_modules["evolution"] = EvolutionEngine(
                    memory_system=self.memory_system,
                    event_system=self.event_system
                )
                self.logger.info("进化学习模块加载成功")
            except ImportError as e:
                self.logger.warning(f"进化学习模块加载失败: {str(e)}")
                
            # 映射学习策略到模块
            self._map_strategies_to_modules()
            
            self.logger.info(f"学习模块加载完成，共加载 {len(self.learning_modules)} 个模块")
            
        except Exception as e:
            self.logger.error(f"加载学习模块失败: {str(e)}")
            
    def _map_strategies_to_modules(self):
        """将学习策略映射到对应的学习模块"""
        # 标准学习策略
        if "standard" in self.learning_modules:
            standard_strategies = [
                "supervised_learning", 
                "self_supervised_learning",
                "pattern_recognition"
            ]
            for strategy in standard_strategies:
                self.strategy_to_module[strategy] = "standard"
                
        # 零样本学习策略
        if "zero_shot" in self.learning_modules:
            zero_shot_strategies = [
                "zero_shot_classification",
                "zero_shot_generation",
                "zero_shot_relation",
                "zero_shot_analogy",
                "conceptual_similarity",
                "hierarchical_inference"
            ]
            for strategy in zero_shot_strategies:
                self.strategy_to_module[strategy] = "zero_shot"
                
        # 强化学习策略
        if "reinforcement" in self.learning_modules:
            rl_strategies = [
                "reinforcement_learning",
                "q_learning",
                "policy_optimization"
            ]
            for strategy in rl_strategies:
                self.strategy_to_module[strategy] = "reinforcement"
                
        # 进化学习策略
        if "evolution" in self.learning_modules:
            evolution_strategies = [
                "evolutionary_learning",
                "genetic_algorithm",
                "mutation_based_learning"
            ]
            for strategy in evolution_strategies:
                self.strategy_to_module[strategy] = "evolution"
                
    def learn(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一学习接口，根据学习数据选择合适的学习模块
        
        Args:
            learning_data: 学习数据，包含任务类型、策略和输入数据
            
        Returns:
            Dict: 学习结果
        """
        # 更新统计
        self.learning_stats["total_queries"] += 1
        
        # 提取学习任务类型和策略
        task_type = learning_data.get("task_type", "general")
        strategy = learning_data.get("strategy", "auto")
        
        # 如果指定了特定策略，使用对应的模块
        if strategy != "auto" and strategy in self.strategy_to_module:
            module_name = self.strategy_to_module[strategy]
            result = self._apply_module(module_name, learning_data)
            
            # 更新使用统计
            self._update_usage_stats(module_name, strategy, result)
            
            return result
            
        # 根据任务类型自动选择模块
        if task_type == "zero_shot":
            # 零样本任务
            if "zero_shot" in self.learning_modules:
                result = self._apply_module("zero_shot", learning_data)
                self._update_usage_stats("zero_shot", "auto_zero_shot", result)
                return result
                
        elif task_type == "interaction":
            # 处理交互学习
            if "standard" in self.learning_modules:
                result = self._apply_module("standard", learning_data)
                self._update_usage_stats("standard", "auto_interaction", result)
                return result
                
        elif task_type == "reinforcement":
            # 强化学习任务
            if "reinforcement" in self.learning_modules:
                result = self._apply_module("reinforcement", learning_data)
                self._update_usage_stats("reinforcement", "auto_reinforcement", result)
                return result
                
        elif task_type == "evolution":
            # 进化学习任务
            if "evolution" in self.learning_modules:
                result = self._apply_module("evolution", learning_data)
                self._update_usage_stats("evolution", "auto_evolution", result)
                return result
                
        # 如果没有匹配的模块或策略，尝试按优先级使用可用模块
        priority_order = ["enhanced", "zero_shot", "standard", "reinforcement", "evolution"]
        
        for module_name in priority_order:
            if module_name in self.learning_modules:
                result = self._apply_module(module_name, learning_data)
                self._update_usage_stats(module_name, "fallback", result)
                return result
                
        # 如果没有可用模块，返回错误
        return {
            "status": "error",
            "message": "没有可用的学习模块处理该请求",
            "data": learning_data
        }
        
    def _apply_module(self, module_name: str, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用特定学习模块处理学习数据"""
        if module_name not in self.learning_modules:
            return {
                "status": "error",
                "message": f"学习模块 '{module_name}' 不可用",
                "data": learning_data
            }
            
        module = self.learning_modules[module_name]
        
        try:
            # 根据模块类型和任务调用不同方法
            if module_name == "standard":
                if "interaction_data" in learning_data:
                    result = module.learn_from_interaction(learning_data["interaction_data"])
                else:
                    result = {"status": "error", "message": "标准学习模块需要互动数据"}
                    
            elif module_name == "zero_shot":
                if "query" in learning_data:
                    result = module.zero_shot_inference(learning_data["query"])
                else:
                    result = {"status": "error", "message": "零样本学习模块需要查询数据"}
                    
            elif module_name == "reinforcement":
                if all(k in learning_data for k in ["state", "action", "reward"]):
                    result = module.update(
                        learning_data["state"],
                        learning_data["action"],
                        learning_data["reward"],
                        learning_data.get("next_state", {})
                    )
                else:
                    result = {"status": "error", "message": "强化学习模块需要状态-动作-奖励数据"}
                    
            elif module_name == "evolution":
                # 假设进化模块有learn方法
                if hasattr(module, "learn"):
                    result = module.learn(learning_data)
                else:
                    result = {"status": "error", "message": "进化学习模块缺少learn方法"}
                    
            elif module_name == "enhanced":
                # 假设增强型模块有process_learning方法
                if hasattr(module, "process_learning"):
                    result = module.process_learning(learning_data)
                else:
                    result = {"status": "error", "message": "增强型学习模块缺少process_learning方法"}
                    
            else:
                result = {"status": "error", "message": f"未知的学习模块类型: {module_name}"}
                
            # 添加模块信息到结果
            if isinstance(result, dict):
                result["module"] = module_name
                
            return result
            
        except Exception as e:
            self.logger.error(f"应用学习模块 '{module_name}' 失败: {str(e)}")
            return {
                "status": "error",
                "message": f"学习模块执行错误: {str(e)}",
                "module": module_name
            }
            
    def _update_usage_stats(self, module_name: str, strategy: str, result: Dict[str, Any]):
        """更新模块和策略使用统计"""
        # 更新模块使用统计
        if module_name not in self.learning_stats["module_usage"]:
            self.learning_stats["module_usage"][module_name] = {
                "total": 0,
                "success": 0
            }
            
        self.learning_stats["module_usage"][module_name]["total"] += 1
        
        if result.get("status") == "success":
            self.learning_stats["module_usage"][module_name]["success"] += 1
            
        # 更新策略使用统计
        if strategy not in self.learning_stats["strategy_usage"]:
            self.learning_stats["strategy_usage"][strategy] = {
                "total": 0,
                "success": 0
            }
            
        self.learning_stats["strategy_usage"][strategy]["total"] += 1
        
        if result.get("status") == "success":
            self.learning_stats["strategy_usage"][strategy]["success"] += 1
            
        # 更新总体成功率
        successful = result.get("status") == "success"
        
        if "total" not in self.learning_stats["success_rate"]:
            self.learning_stats["success_rate"] = {"total": 0, "success": 0}
            
        self.learning_stats["success_rate"]["total"] += 1
        if successful:
            self.learning_stats["success_rate"]["success"] += 1
            
        # 发布学习事件
        if self.event_system:
            self.event_system.publish("learning.module_used", {
                "module": module_name,
                "strategy": strategy,
                "success": successful,
                "timestamp": time.time()
            })
            
    def meta_optimize(self) -> Dict[str, Any]:
        """
        执行元优化，分析和改进各模块的学习性能
        
        Returns:
            Dict: 优化结果
        """
        results = {}
        improvements = []
        
        # 1. 对每个学习模块执行自优化
        for module_name, module in self.learning_modules.items():
            try:
                # 检查模块是否支持元学习/自优化
                if hasattr(module, "meta_learn"):
                    module_result = module.meta_learn()
                    results[module_name] = module_result
                    
                    # 收集改进
                    if module_result.get("improvements"):
                        for imp in module_result["improvements"]:
                            imp["module"] = module_name
                            improvements.append(imp)
            except Exception as e:
                self.logger.warning(f"模块 {module_name} 元优化失败: {str(e)}")
                results[module_name] = {"status": "error", "message": str(e)}
        
        # 2. 分析模块使用统计，调整路由策略
        if self.learning_stats["total_queries"] > 20:  # 确保有足够数据
            # 分析模块成功率
            success_rates = {}
            for module_name, usage in self.learning_stats["module_usage"].items():
                if usage["total"] > 0:
                    success_rate = usage["success"] / usage["total"]
                    success_rates[module_name] = success_rate
            
            # 分析任务类型与模块匹配
            task_module_performance = defaultdict(lambda: defaultdict(list))
            
            # 收集每个任务类型在每个模块上的表现
            for module_name, module in self.learning_modules.items():
                if hasattr(module, "task_performances"):
                    for task_type, records in module.task_performances.items():
                        # 计算该任务类型的成功率
                        if records:
                            success_count = sum(1 for r in records if r.get("success", False))
                            success_rate = success_count / len(records)
                            task_module_performance[task_type][module_name] = success_rate
            
            # 为每种任务类型选择最佳模块
            best_modules_for_tasks = {}
            for task_type, module_rates in task_module_performance.items():
                if module_rates:
                    best_module = max(module_rates.items(), key=lambda x: x[1])
                    if best_module[1] > 0.6:  # 至少60%成功率
                        best_modules_for_tasks[task_type] = best_module[0]
                        
                        # 如果不是当前的默认模块，创建改进记录
                        current_default = None
                        for strategy, module in self.strategy_to_module.items():
                            if strategy.startswith(task_type) or strategy.endswith(task_type):
                                current_default = module
                                break
                                
                        if current_default and current_default != best_module[0]:
                            improvements.append({
                                "type": "task_routing",
                                "task_type": task_type,
                                "old_module": current_default,
                                "new_module": best_module[0],
                                "success_rate": best_module[1],
                                "reason": f"模块 {best_module[0]} 对任务 {task_type} 的成功率更高"
                            })
                            
                            # 更新路由映射
                            for strategy in list(self.strategy_to_module.keys()):
                                if strategy.startswith(task_type) or strategy.endswith(task_type):
                                    self.strategy_to_module[strategy] = best_module[0]
        
        # 3. 执行知识共享和跨模块学习
        if len(self.learning_modules) >= 2:
            knowledge_transfers = self._share_knowledge_between_modules()
            if knowledge_transfers:
                for transfer in knowledge_transfers:
                    improvements.append({
                        "type": "knowledge_transfer",
                        "source": transfer["source"],
                        "target": transfer["target"],
                        "concepts_transferred": transfer["concepts_count"],
                        "reason": "共享学习经验和概念表示"
                    })
        
        # 4. 优化学习策略参数
        strategy_optimizations = self._optimize_learning_strategies()
        if strategy_optimizations:
            improvements.extend(strategy_optimizations)
            
        # 5. 添加新学习策略（如果发现有效的新模式）
        new_strategies = self._discover_new_strategies()
        if new_strategies:
            for strategy in new_strategies:
                improvements.append({
                    "type": "new_strategy",
                    "strategy_name": strategy["name"],
                    "description": strategy["description"],
                    "reason": "基于成功模式创建新策略"
                })
                
                # 添加到策略列表
                self.learning_strategies.append(strategy)
        
        return {
            "status": "success",
            "module_results": results,
            "improvements": improvements,
            "total_improvements": len(improvements),
            "timestamp": time.time()
        }
        
    def _share_knowledge_between_modules(self) -> List[Dict[str, Any]]:
        """
        在不同学习模块间共享知识和学习成果
        
        Returns:
            List: 知识转移记录
        """
        transfers = []
        
        # 确定哪些模块可以共享知识
        knowledge_shareable_modules = []
        
        for module_name, module in self.learning_modules.items():
            # 检查模块是否有存储知识的能力
            if (hasattr(module, "vector_store") or 
                hasattr(module, "concept_vectors") or 
                hasattr(module, "knowledge_graph")):
                knowledge_shareable_modules.append(module_name)
        
        if len(knowledge_shareable_modules) < 2:
            return []  # 需要至少两个可以共享知识的模块
            
        # 对每对模块执行知识共享
        for i, source_name in enumerate(knowledge_shareable_modules):
            source_module = self.learning_modules[source_name]
            
            for j in range(i+1, len(knowledge_shareable_modules)):
                target_name = knowledge_shareable_modules[j]
                target_module = self.learning_modules[target_name]
                
                # 确定哪个模块是更好的知识源（基于使用统计）
                source_success = self.learning_stats["module_usage"].get(source_name, {}).get("success", 0)
                target_success = self.learning_stats["module_usage"].get(target_name, {}).get("success", 0)
                
                # 始终从更成功的模块向不太成功的模块传输
                if source_success < target_success:
                    source_module, target_module = target_module, source_module
                    source_name, target_name = target_name, source_name
                
                # 执行知识转移
                transferred = 0
                
                # 转移向量表示
                if hasattr(source_module, "vector_store") and hasattr(target_module, "vector_store"):
                    try:
                        # 转移概念向量
                        if hasattr(source_module.vector_store, "concept_vectors") and hasattr(target_module.vector_store, "concept_vectors"):
                            source_concepts = source_module.vector_store.concept_vectors
                            target_concepts = target_module.vector_store.concept_vectors
                            
                            # 找出目标模块没有的概念
                            new_concepts = set(source_concepts.keys()) - set(target_concepts.keys())
                            
                            # 转移概念向量
                            for concept_id in new_concepts:
                                if concept_id.startswith("concept:"):
                                    # 提取概念名称
                                    concept_name = concept_id.split(":", 1)[1].replace("_", " ")
                                    vector = source_concepts[concept_id]
                                    
                                    # 获取概念属性（如果可用）
                                    properties = {}
                                    if hasattr(source_module.vector_store, "knowledge_graph") and concept_id in source_module.vector_store.knowledge_graph.nodes:
                                        node_data = source_module.vector_store.knowledge_graph.nodes[concept_id]
                                        properties = node_data.get("properties", {})
                                    
                                    # 添加到目标模块
                                    if hasattr(target_module.vector_store, "add_concept"):
                                        target_module.vector_store.add_concept(concept_name, vector=vector, properties=properties)
                                        transferred += 1
                        
                        # 转移关系
                        if hasattr(source_module.vector_store, "knowledge_graph") and hasattr(target_module.vector_store, "knowledge_graph"):
                            source_graph = source_module.vector_store.knowledge_graph
                            target_graph = target_module.vector_store.knowledge_graph
                            
                            # 遍历源图的所有边
                            for u, v, data in source_graph.edges(data=True):
                                # 如果两个节点都在目标图中但关系不存在
                                if (u in target_graph.nodes and v in target_graph.nodes and 
                                    not target_graph.has_edge(u, v)):
                                    rel_type = data.get("type", "related_to")
                                    if hasattr(target_module.vector_store, "add_relation"):
                                        target_module.vector_store.add_relation(u, v, rel_type, data.get("properties", {}))
                                        transferred += 1
                    except Exception as e:
                        self.logger.warning(f"知识共享错误: {str(e)}")
                
                # 只有实际发生转移时才记录
                if transferred > 0:
                    transfers.append({
                        "source": source_name,
                        "target": target_name,
                        "concepts_count": transferred
                    })
        
        return transfers
        
    def _optimize_learning_strategies(self) -> List[Dict[str, Any]]:
        """
        优化学习策略参数
        
        Returns:
            List: 策略优化记录
        """
        optimizations = []
        
        # 分析每个策略的使用和成功率
        strategy_stats = self.learning_stats.get("strategy_usage", {})
        
        for strategy_name, stats in strategy_stats.items():
            if stats.get("total", 0) < 5:
                continue  # 忽略使用次数少的策略
                
            # 计算成功率
            success_rate = stats.get("success", 0) / stats.get("total", 1)
            
            # 低成功率策略需要优化
            if success_rate < 0.7:
                # 找到负责该策略的模块
                responsible_module = self.strategy_to_module.get(strategy_name)
                if not responsible_module or responsible_module not in self.learning_modules:
                    continue
                    
                module = self.learning_modules[responsible_module]
                
                # 查找该策略在模块中的定义
                strategy_def = None
                if hasattr(module, "learning_strategies"):
                    for s in module.learning_strategies:
                        if s.get("name") == strategy_name:
                            strategy_def = s
                            break
                
                if not strategy_def:
                    continue
                    
                # 根据策略类型优化参数
                params = strategy_def.get("params", {})
                optimized = False
                
                if "similarity_threshold" in params:
                    # 降低相似度阈值以增加匹配率
                    old_threshold = params["similarity_threshold"]
                    new_threshold = max(0.5, old_threshold - 0.1)
                    params["similarity_threshold"] = new_threshold
                    optimized = True
                    
                    optimizations.append({
                        "type": "strategy_param",
                        "strategy": strategy_name,
                        "module": responsible_module,
                        "param": "similarity_threshold",
                        "old_value": old_threshold,
                        "new_value": new_threshold,
                        "reason": "降低相似度阈值以提高匹配率"
                    })
                
                if "min_confidence" in params:
                    # 降低最小置信度要求
                    old_confidence = params["min_confidence"]
                    new_confidence = max(0.4, old_confidence - 0.1)
                    params["min_confidence"] = new_confidence
                    optimized = True
                    
                    optimizations.append({
                        "type": "strategy_param",
                        "strategy": strategy_name,
                        "module": responsible_module,
                        "param": "min_confidence",
                        "old_value": old_confidence,
                        "new_value": new_confidence,
                        "reason": "降低最小置信度要求以提高匹配率"
                    })
                
                # 更新策略定义
                if optimized:
                    strategy_def["params"] = params
                    
                    # 更新模块中的策略
                    if hasattr(module, "learning_strategies"):
                        for i, s in enumerate(module.learning_strategies):
                            if s.get("name") == strategy_name:
                                module.learning_strategies[i] = strategy_def
                                break
        
        return optimizations
        
    def _discover_new_strategies(self) -> List[Dict[str, Any]]:
        """
        发现新的有效学习策略
        
        Returns:
            List: 新策略列表
        """
        new_strategies = []
        
        # 检查零样本模块是否有提取的类比模式
        if "zero_shot" in self.learning_modules:
            zero_shot_module = self.learning_modules["zero_shot"]
            if hasattr(zero_shot_module, "analogy_patterns") and zero_shot_module.analogy_patterns:
                # 从类比模式创建新策略
                analogy_strategies = []
                
                for pattern in zero_shot_module.analogy_patterns:
                    pattern_type = pattern.get("type")
                    
                    if pattern_type == "relation_pattern":
                        # 创建基于常见关系的策略
                        common_relations = pattern.get("common_relations", [])
                        if common_relations:
                            analogy_strategies.append({
                                "name": f"relation_based_analogy_{len(new_strategies)}",
                                "description": "使用常见概念关系执行类比推理",
                                "params": {
                                    "target_relations": [rel for rel, _ in common_relations],
                                    "confidence_boost": 0.2
                                },
                                "pattern_source": pattern
                            })
                    
                    elif pattern_type == "vector_offset_pattern":
                        # 创建基于向量偏移的策略
                        avg_sim = pattern.get("average_offset_similarity", 0)
                        if avg_sim > 0.6:
                            analogy_strategies.append({
                                "name": f"vector_offset_analogy_{len(new_strategies)}",
                                "description": "使用向量空间偏移执行类比推理",
                                "params": {
                                    "offset_similarity_threshold": avg_sim - 0.1,
                                    "vector_combination": True
                                },
                                "pattern_source": pattern
                            })
                    
                    elif pattern_type == "string_transformation":
                        # 创建基于字符串变换的策略
                        str_pattern = pattern.get("pattern")
                        if str_pattern:
                            analogy_strategies.append({
                                "name": f"string_transform_analogy_{len(new_strategies)}",
                                "description": f"使用字符串变换规则({str_pattern})执行类比推理",
                                "params": {
                                    "transformation_type": str_pattern,
                                    "min_confidence": 0.7
                                },
                                "pattern_source": pattern
                            })
                
                # 只返回置信度较高的新策略
                for strategy in analogy_strategies:
                    source_pattern = strategy.pop("pattern_source", {})
                    if source_pattern.get("confidence", 0) > 0.7:
                        new_strategies.append(strategy)
                        
                        # 将策略映射到零样本模块
                        self.strategy_to_module[strategy["name"]] = "zero_shot"
        
        # 检查其他模块是否有可用于创建新策略的模式
        # (可以添加更多模块特定的策略发现逻辑)
        
        return new_strategies
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习系统统计信息"""
        stats = self.learning_stats.copy()
        
        # 计算成功率
        for module, usage in stats["module_usage"].items():
            if usage["total"] > 0:
                usage["success_rate"] = usage["success"] / usage["total"]
                
        for strategy, usage in stats["strategy_usage"].items():
            if usage["total"] > 0:
                usage["success_rate"] = usage["success"] / usage["total"]
                
        if stats["success_rate"]["total"] > 0:
            stats["success_rate"]["rate"] = (
                stats["success_rate"]["success"] / stats["success_rate"]["total"]
            )
            
        # 获取各模块的详细统计
        module_specific_stats = {}
        for module_name, module in self.learning_modules.items():
            if hasattr(module, "get_learning_stats"):
                try:
                    module_stats = module.get_learning_stats()
                    module_specific_stats[module_name] = module_stats
                except Exception as e:
                    self.logger.error(f"获取 {module_name} 统计信息失败: {str(e)}")
                    
        stats["module_specific_stats"] = module_specific_stats
        
        return stats
        
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """获取所有可用的学习策略"""
        strategies = []
        
        for strategy, module_name in self.strategy_to_module.items():
            strategies.append({
                "name": strategy,
                "module": module_name,
                "description": self._get_strategy_description(strategy)
            })
            
        return strategies
        
    def _get_strategy_description(self, strategy: str) -> str:
        """获取学习策略的描述"""
        descriptions = {
            # 标准学习策略
            "supervised_learning": "基于标记数据的监督学习",
            "self_supervised_learning": "从未标记数据中自我学习",
            "pattern_recognition": "识别数据中的模式",
            
            # 零样本学习策略
            "zero_shot_classification": "无示例分类",
            "zero_shot_generation": "无示例生成",
            "zero_shot_relation": "无示例关系预测",
            "zero_shot_analogy": "无示例类比推理",
            "conceptual_similarity": "基于概念相似性的推理",
            "hierarchical_inference": "基于层次结构的推理",
            
            # 强化学习策略
            "reinforcement_learning": "基于奖励信号的强化学习",
            "q_learning": "Q-学习算法",
            "policy_optimization": "策略优化",
            
            # 进化学习策略
            "evolutionary_learning": "基于进化算法的学习",
            "genetic_algorithm": "遗传算法",
            "mutation_based_learning": "基于变异的学习"
        }
        
        return descriptions.get(strategy, "未知学习策略")
        
    def get_module_info(self) -> Dict[str, Any]:
        """获取学习模块信息"""
        module_info = {}
        
        for module_name, module in self.learning_modules.items():
            module_info[module_name] = {
                "available": True,
                "strategies": [s for s, m in self.strategy_to_module.items() if m == module_name],
                "supports_meta_learning": hasattr(module, "meta_learn")
            }
            
        return module_info 