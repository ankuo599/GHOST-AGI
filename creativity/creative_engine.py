"""
创新能力引擎 (Creative Engine)

提供创新思维、想法生成和概念组合功能，增强系统的创造性思考能力。
支持多种创新策略、灵活的知识重组和创新评估。
"""

import time
import logging
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict, deque

class CreativeEngine:
    """创新能力引擎，负责创造性思维和想法生成"""
    
    def __init__(self, knowledge_system=None, memory_system=None, logger=None):
        """
        初始化创新能力引擎
        
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
        
        # 创新数据
        self.ideas = {}  # {idea_id: idea_info}
        self.concept_combinations = {}  # {combination_id: combination_info}
        self.innovation_strategies = {}  # {strategy_id: strategy_info}
        
        # 创新历史
        self.innovation_history = []
        self.evaluation_history = {}  # {idea_id: [evaluations]}
        
        # 创新统计
        self.stats = {
            "ideas_generated": 0,
            "ideas_implemented": 0,
            "combinations_created": 0,
            "successful_innovations": 0
        }
        
        # 创新策略
        self._register_default_strategies()
        
        # 配置
        self.config = {
            "random_exploration_rate": 0.3,
            "combination_depth": 3,
            "novelty_threshold": 0.6,
            "utility_threshold": 0.5,
            "implementation_threshold": 0.7
        }
        
        self.logger.info("创新能力引擎初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("CreativeEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("creativity.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_default_strategies(self):
        """注册默认创新策略"""
        # 随机组合策略
        self.register_strategy({
            "name": "random_combination",
            "description": "随机组合两个或多个概念",
            "method": self._random_combination_strategy,
            "parameters": {
                "combination_count": 2,
                "min_similarity": 0.0
            }
        })
        
        # 类比推理策略
        self.register_strategy({
            "name": "analogical_reasoning",
            "description": "从一个领域映射知识到另一个领域",
            "method": self._analogical_reasoning_strategy,
            "parameters": {
                "source_domain": None,
                "target_domain": None,
                "mapping_strength": 0.7
            }
        })
        
        # 概念变异策略
        self.register_strategy({
            "name": "concept_mutation",
            "description": "修改现有概念的属性生成新概念",
            "method": self._concept_mutation_strategy,
            "parameters": {
                "mutation_rate": 0.3,
                "preserve_core": True
            }
        })
        
        # 远距离联想策略
        self.register_strategy({
            "name": "distant_association",
            "description": "连接表面上不相关的概念",
            "method": self._distant_association_strategy,
            "parameters": {
                "distance_threshold": 0.8,
                "context_relevance": 0.5
            }
        })
    
    def register_strategy(self, strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        注册创新策略
        
        Args:
            strategy_info: 策略信息
            
        Returns:
            Dict: 注册结果
        """
        strategy_id = strategy_info.get("id", str(uuid.uuid4()))
        
        # 补充策略信息
        strategy_info["id"] = strategy_id
        strategy_info["registered_at"] = time.time()
        strategy_info["usage_count"] = 0
        strategy_info["success_rate"] = 0.0
        
        # 保存策略
        self.innovation_strategies[strategy_id] = strategy_info
        
        self.logger.info(f"已注册创新策略: {strategy_info['name']} (ID: {strategy_id})")
        
        return {
            "status": "success",
            "strategy_id": strategy_id
        }
    
    def generate_idea(self, 
                     strategy_id: Optional[str] = None, 
                     context: Optional[Dict[str, Any]] = None, 
                     parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成创新想法
        
        Args:
            strategy_id: 策略ID，如果为None则自动选择
            context: 生成上下文
            parameters: 附加参数
            
        Returns:
            Dict: 生成结果
        """
        # 默认参数
        if context is None:
            context = {}
            
        if parameters is None:
            parameters = {}
            
        # 自动选择策略
        if strategy_id is None:
            strategy_id = self._select_strategy(context)
            
        if strategy_id not in self.innovation_strategies:
            return {
                "status": "error",
                "message": f"未知的创新策略: {strategy_id}"
            }
            
        strategy = self.innovation_strategies[strategy_id]
        
        # 更新使用计数
        strategy["usage_count"] += 1
        
        # 合并策略参数和传入参数
        merged_parameters = dict(strategy.get("parameters", {}))
        merged_parameters.update(parameters)
        
        # 生成想法
        try:
            # 调用策略方法
            if callable(strategy.get("method")):
                idea_result = strategy["method"](context, merged_parameters)
            else:
                # 如果没有有效方法，使用默认生成方法
                idea_result = self._default_idea_generation(context, merged_parameters)
                
            # 检查结果有效性
            if not idea_result or not isinstance(idea_result, dict):
                raise ValueError("策略返回了无效结果")
                
            # 生成ID并完善想法信息
            idea_id = str(uuid.uuid4())
            
            idea = {
                "id": idea_id,
                "title": idea_result.get("title", f"Idea-{idea_id[:8]}"),
                "description": idea_result.get("description", ""),
                "components": idea_result.get("components", []),
                "strategy": strategy_id,
                "context": context,
                "parameters": merged_parameters,
                "created_at": time.time(),
                "novelty": idea_result.get("novelty", self._estimate_novelty(idea_result)),
                "utility": idea_result.get("utility", self._estimate_utility(idea_result, context)),
                "implementation_score": idea_result.get("implementation_score", self._estimate_implementation_score(idea_result)),
                "tags": idea_result.get("tags", [])
            }
            
            # 保存想法
            self.ideas[idea_id] = idea
            
            # 更新统计
            self.stats["ideas_generated"] += 1
            
            # 添加到历史
            self.innovation_history.append({
                "type": "idea_generation",
                "idea_id": idea_id,
                "strategy_id": strategy_id,
                "timestamp": time.time()
            })
            
            self.logger.info(f"已生成创新想法: {idea['title']} (ID: {idea_id})")
            
            return {
                "status": "success",
                "idea": idea
            }
        except Exception as e:
            self.logger.error(f"生成想法失败: {str(e)}")
            
            return {
                "status": "error",
                "message": f"生成想法失败: {str(e)}",
                "strategy_id": strategy_id
            }
    
    def combine_concepts(self, concept_ids: List[str], 
                        method: str = "blend", 
                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        组合多个概念
        
        Args:
            concept_ids: 概念ID列表
            method: 组合方法(blend, intersection, network)
            parameters: 组合参数
            
        Returns:
            Dict: 组合结果
        """
        if not concept_ids or len(concept_ids) < 2:
            return {
                "status": "error",
                "message": "需要至少两个概念进行组合"
            }
            
        if not self.knowledge_system:
            return {
                "status": "error",
                "message": "知识系统不可用，无法获取概念"
            }
            
        # 默认参数
        if parameters is None:
            parameters = {}
            
        # 获取概念信息
        concepts = []
        for concept_id in concept_ids:
            concept_result = self.knowledge_system.get_concept(concept_id)
            
            if concept_result.get("status") == "success":
                concepts.append(concept_result["concept"])
            else:
                return {
                    "status": "error",
                    "message": f"无法获取概念 {concept_id}: {concept_result.get('message', '未知错误')}"
                }
        
        # 根据方法选择组合函数
        if method == "blend":
            combination = self._blend_concepts(concepts, parameters)
        elif method == "intersection":
            combination = self._intersect_concepts(concepts, parameters)
        elif method == "network":
            combination = self._network_concepts(concepts, parameters)
        else:
            return {
                "status": "error",
                "message": f"不支持的组合方法: {method}"
            }
            
        # 生成组合ID
        combination_id = str(uuid.uuid4())
        
        # 完善组合信息
        combination.update({
            "id": combination_id,
            "source_concepts": concept_ids,
            "method": method,
            "parameters": parameters,
            "created_at": time.time()
        })
        
        # 保存组合
        self.concept_combinations[combination_id] = combination
        
        # 更新统计
        self.stats["combinations_created"] += 1
        
        self.logger.info(f"已组合概念: {combination['name']} (ID: {combination_id})")
        
        return {
            "status": "success",
            "combination": combination
        }
    
    def evaluate_idea(self, idea_id: str, 
                     evaluation_criteria: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        评估创新想法
        
        Args:
            idea_id: 想法ID
            evaluation_criteria: 评估标准和权重
            
        Returns:
            Dict: 评估结果
        """
        if idea_id not in self.ideas:
            return {
                "status": "error",
                "message": f"未知的想法: {idea_id}"
            }
            
        idea = self.ideas[idea_id]
        
        # 默认评估标准
        if evaluation_criteria is None:
            evaluation_criteria = {
                "novelty": 0.3,
                "utility": 0.3,
                "feasibility": 0.2,
                "impact": 0.2
            }
            
        # 计算各维度评分
        novelty = idea.get("novelty", self._estimate_novelty(idea))
        utility = idea.get("utility", self._estimate_utility(idea, idea.get("context", {})))
        feasibility = self._estimate_feasibility(idea)
        impact = self._estimate_impact(idea)
        
        # 总体评分
        total_score = (
            novelty * evaluation_criteria.get("novelty", 0) +
            utility * evaluation_criteria.get("utility", 0) +
            feasibility * evaluation_criteria.get("feasibility", 0) +
            impact * evaluation_criteria.get("impact", 0)
        )
        
        # 创建评估记录
        evaluation = {
            "id": str(uuid.uuid4()),
            "idea_id": idea_id,
            "criteria": evaluation_criteria,
            "scores": {
                "novelty": novelty,
                "utility": utility,
                "feasibility": feasibility,
                "impact": impact,
                "total": total_score
            },
            "timestamp": time.time()
        }
        
        # 保存评估
        if idea_id not in self.evaluation_history:
            self.evaluation_history[idea_id] = []
            
        self.evaluation_history[idea_id].append(evaluation)
        
        # 更新想法的评分
        idea["evaluation"] = evaluation["scores"]
        idea["last_evaluated_at"] = time.time()
        
        self.logger.info(f"已评估想法 {idea_id}，总分: {total_score:.2f}")
        
        return {
            "status": "success",
            "evaluation": evaluation
        }
    
    def implement_idea(self, idea_id: str, 
                      implementation_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        实现创新想法
        
        Args:
            idea_id: 想法ID
            implementation_plan: 实现计划
            
        Returns:
            Dict: 实现结果
        """
        if idea_id not in self.ideas:
            return {
                "status": "error",
                "message": f"未知的想法: {idea_id}"
            }
            
        idea = self.ideas[idea_id]
        
        # 默认实现计划
        if implementation_plan is None:
            implementation_plan = {
                "steps": [],
                "resources": {},
                "timeline": {}
            }
            
        # 检查想法是否已评估
        if "evaluation" not in idea:
            self.evaluate_idea(idea_id)
            
        # 检查实现分数是否达到阈值
        implementation_score = idea.get("implementation_score", 0)
        if implementation_score < self.config["implementation_threshold"]:
            return {
                "status": "rejected",
                "message": f"想法实现分数过低: {implementation_score:.2f}",
                "threshold": self.config["implementation_threshold"]
            }
            
        # 创建实现记录
        implementation = {
            "id": str(uuid.uuid4()),
            "idea_id": idea_id,
            "plan": implementation_plan,
            "status": "started",
            "started_at": time.time(),
            "progress": 0.0
        }
        
        # 更新想法状态
        idea["implementation"] = implementation
        idea["status"] = "implementing"
        
        # 更新统计
        self.stats["ideas_implemented"] += 1
        
        # 更新创新策略成功率
        strategy_id = idea.get("strategy")
        if strategy_id in self.innovation_strategies:
            strategy = self.innovation_strategies[strategy_id]
            total_ideas = strategy["usage_count"]
            successful_ideas = strategy.get("successful_ideas", 0) + 1
            strategy["successful_ideas"] = successful_ideas
            if total_ideas > 0:
                strategy["success_rate"] = successful_ideas / total_ideas
        
        self.logger.info(f"开始实现想法: {idea['title']} (ID: {idea_id})")
        
        return {
            "status": "success",
            "implementation": implementation
        }
    
    def get_idea(self, idea_id: str) -> Dict[str, Any]:
        """
        获取想法信息
        
        Args:
            idea_id: 想法ID
            
        Returns:
            Dict: 想法信息
        """
        if idea_id not in self.ideas:
            return {
                "status": "error",
                "message": f"未知的想法: {idea_id}"
            }
            
        idea = self.ideas[idea_id]
        
        # 添加评估历史
        idea["evaluation_history"] = self.evaluation_history.get(idea_id, [])
        
        return {
            "status": "success",
            "idea": idea
        }
    
    def search_ideas(self, query: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """
        搜索想法
        
        Args:
            query: 查询参数
            limit: 返回结果数量限制
            
        Returns:
            Dict: 搜索结果
        """
        results = []
        
        # 按标题搜索
        if "title" in query:
            title_query = query["title"].lower()
            for idea_id, idea in self.ideas.items():
                if title_query in idea["title"].lower():
                    results.append({
                        "id": idea_id,
                        "title": idea["title"],
                        "description": idea["description"],
                        "novelty": idea.get("novelty", 0),
                        "utility": idea.get("utility", 0),
                        "created_at": idea["created_at"],
                        "relevance": 1.0 if idea["title"].lower() == title_query else 0.8
                    })
        
        # 按标签搜索
        elif "tags" in query:
            tag_queries = [t.lower() for t in query["tags"]]
            for idea_id, idea in self.ideas.items():
                idea_tags = [t.lower() for t in idea.get("tags", [])]
                matches = sum(1 for t in tag_queries if t in idea_tags)
                
                if matches > 0:
                    relevance = matches / len(tag_queries)
                    results.append({
                        "id": idea_id,
                        "title": idea["title"],
                        "description": idea["description"],
                        "novelty": idea.get("novelty", 0),
                        "utility": idea.get("utility", 0),
                        "created_at": idea["created_at"],
                        "relevance": relevance
                    })
        
        # 按评分搜索
        elif "min_score" in query:
            min_score = query["min_score"]
            score_type = query.get("score_type", "total")
            
            for idea_id, idea in self.ideas.items():
                if "evaluation" in idea:
                    score = idea["evaluation"].get("scores", {}).get(score_type, 0)
                    if score >= min_score:
                        results.append({
                            "id": idea_id,
                            "title": idea["title"],
                            "description": idea["description"],
                            "novelty": idea.get("novelty", 0),
                            "utility": idea.get("utility", 0),
                            "created_at": idea["created_at"],
                            "score": score,
                            "relevance": score / 1.0
                        })
        
        # 排序并限制结果数量
        results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:limit]
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def get_innovation_statistics(self) -> Dict[str, Any]:
        """
        获取创新统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 统计策略使用情况
        strategies_stats = []
        for strategy_id, strategy in self.innovation_strategies.items():
            strategies_stats.append({
                "id": strategy_id,
                "name": strategy["name"],
                "usage_count": strategy["usage_count"],
                "success_rate": strategy.get("success_rate", 0.0)
            })
            
        # 按使用量排序
        strategies_stats.sort(key=lambda x: x["usage_count"], reverse=True)
        
        # 创新效率统计
        efficiency = {
            "implementation_ratio": self.stats["ideas_implemented"] / max(1, self.stats["ideas_generated"]),
            "success_ratio": self.stats["successful_innovations"] / max(1, self.stats["ideas_implemented"])
        }
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "idea_count": len(self.ideas),
            "combination_count": len(self.concept_combinations),
            "strategies": strategies_stats,
            "generation_stats": dict(self.stats),
            "efficiency": efficiency
        }
    
    def _select_strategy(self, context: Dict[str, Any]) -> str:
        """选择合适的创新策略"""
        # 简化实现：根据使用历史选择最佳策略或随机探索
        # 探索策略：有一定概率完全随机选择
        if random.random() < self.config["random_exploration_rate"]:
            return random.choice(list(self.innovation_strategies.keys()))
            
        # 选择历史成功率最高的策略
        strategies = [(s_id, s.get("success_rate", 0)) 
                      for s_id, s in self.innovation_strategies.items()]
        
        # 按成功率排序
        strategies.sort(key=lambda x: x[1], reverse=True)
        
        return strategies[0][0] if strategies else list(self.innovation_strategies.keys())[0]
    
    def _estimate_novelty(self, idea: Dict[str, Any]) -> float:
        """估计想法的新颖性"""
        # 简化实现：随机生成分数
        # 真实实现应比较已有想法和知识
        return random.uniform(0.5, 1.0)
    
    def _estimate_utility(self, idea: Dict[str, Any], context: Dict[str, Any]) -> float:
        """估计想法的实用性"""
        # 简化实现：随机生成分数
        # 真实实现应评估想法对特定问题的解决能力
        return random.uniform(0.4, 0.9)
    
    def _estimate_implementation_score(self, idea: Dict[str, Any]) -> float:
        """估计想法的实现可行性"""
        # 简化实现：随机生成分数
        # 真实实现应考虑资源、时间和技术限制
        return random.uniform(0.3, 0.8)
    
    def _estimate_feasibility(self, idea: Dict[str, Any]) -> float:
        """估计想法的可行性"""
        # 与实现分数相关，但更关注技术可行性
        return idea.get("implementation_score", 0) * random.uniform(0.8, 1.2)
    
    def _estimate_impact(self, idea: Dict[str, Any]) -> float:
        """估计想法的影响力"""
        # 简化实现：结合新颖性和实用性
        novelty = idea.get("novelty", 0)
        utility = idea.get("utility", 0)
        
        # 高新颖性和高实用性的想法有更高的影响力
        return (novelty * 0.6) + (utility * 0.4)
    
    def _default_idea_generation(self, context: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """默认的想法生成方法"""
        # 生成随机想法
        return {
            "title": f"创新想法 {random.randint(1000, 9999)}",
            "description": "自动生成的创新想法",
            "components": [],
            "novelty": random.uniform(0.5, 0.9),
            "utility": random.uniform(0.4, 0.8),
            "tags": ["自动生成"]
        }
    
    # 创新策略实现
    def _random_combination_strategy(self, context: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """随机组合策略"""
        if not self.knowledge_system:
            return self._default_idea_generation(context, parameters)
            
        # 获取一些随机概念
        combination_count = parameters.get("combination_count", 2)
        # 实际实现应查询知识系统获取概念
        
        # 模拟概念组合
        components = [f"概念{i}" for i in range(combination_count)]
        
        return {
            "title": f"概念组合: {' + '.join(components)}",
            "description": f"将{components}组合产生的新想法",
            "components": components,
            "novelty": random.uniform(0.6, 0.9),
            "utility": random.uniform(0.5, 0.8),
            "tags": ["概念组合", "随机探索"]
        }
    
    def _analogical_reasoning_strategy(self, context: Dict[str, Any], 
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """类比推理策略"""
        source_domain = parameters.get("source_domain")
        target_domain = parameters.get("target_domain")
        
        # 实际实现应找到源领域和目标领域的对应关系
        
        return {
            "title": f"领域类比: {source_domain or '未知领域'} → {target_domain or '新领域'}",
            "description": f"将{source_domain or '源领域'}的原理应用到{target_domain or '目标领域'}",
            "components": [source_domain, target_domain] if source_domain and target_domain else [],
            "novelty": random.uniform(0.7, 0.95),
            "utility": random.uniform(0.6, 0.9),
            "tags": ["类比推理", "跨领域迁移"]
        }
    
    def _concept_mutation_strategy(self, context: Dict[str, Any], 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """概念变异策略"""
        # 实际实现应随机修改概念的属性
        
        mutation_rate = parameters.get("mutation_rate", 0.3)
        
        return {
            "title": f"概念变异 (变异率: {mutation_rate})",
            "description": "通过修改现有概念属性生成的新想法",
            "components": [],
            "novelty": random.uniform(0.5, 0.8),
            "utility": random.uniform(0.4, 0.7),
            "tags": ["概念变异", "属性修改"]
        }
    
    def _distant_association_strategy(self, context: Dict[str, Any], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """远距离联想策略"""
        # 实际实现应找到表面上不相关的概念并建立联系
        
        distance_threshold = parameters.get("distance_threshold", 0.8)
        
        return {
            "title": "远距离概念联想",
            "description": "连接表面上不相关概念形成的新想法",
            "components": [],
            "novelty": random.uniform(0.8, 1.0),
            "utility": random.uniform(0.3, 0.7),
            "tags": ["远距离联想", "意外连接"]
        }
    
    # 概念组合方法
    def _blend_concepts(self, concepts: List[Dict[str, Any]], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """混合多个概念"""
        # 获取概念名称
        concept_names = [c.get("name", f"Concept-{i}") for i, c in enumerate(concepts)]
        
        # 混合属性（简化实现）
        blended_attributes = {}
        for concept in concepts:
            if "attributes" in concept:
                blended_attributes.update(concept["attributes"])
        
        # 创建混合概念
        return {
            "name": " × ".join(concept_names),
            "description": f"通过混合{len(concepts)}个概念创建的新概念",
            "attributes": blended_attributes,
            "source_type": "blend",
            "blending_level": parameters.get("blending_level", 0.7)
        }
    
    def _intersect_concepts(self, concepts: List[Dict[str, Any]], 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """取多个概念的交集"""
        # 获取概念名称
        concept_names = [c.get("name", f"Concept-{i}") for i, c in enumerate(concepts)]
        
        # 寻找共同属性（简化实现）
        common_attributes = {}
        if all("attributes" in c for c in concepts):
            # 获取第一个概念的属性
            first_attrs = concepts[0]["attributes"]
            
            # 查找在所有概念中都出现的属性
            for key, value in first_attrs.items():
                if all(c.get("attributes", {}).get(key) == value for c in concepts[1:]):
                    common_attributes[key] = value
        
        # 创建交集概念
        return {
            "name": " ∩ ".join(concept_names),
            "description": f"提取{len(concepts)}个概念的共同特征",
            "attributes": common_attributes,
            "source_type": "intersection",
            "common_attribute_count": len(common_attributes)
        }
    
    def _network_concepts(self, concepts: List[Dict[str, Any]], 
                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """将多个概念构建为网络"""
        # 获取概念名称
        concept_names = [c.get("name", f"Concept-{i}") for i, c in enumerate(concepts)]
        
        # 创建关系网络（简化实现）
        network = {
            "nodes": [{"id": i, "name": name} for i, name in enumerate(concept_names)],
            "edges": []
        }
        
        # 添加一些连接
        edge_density = parameters.get("edge_density", 0.5)
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                if random.random() < edge_density:
                    network["edges"].append({
                        "source": i,
                        "target": j,
                        "type": "related"
                    })
        
        # 创建网络概念
        return {
            "name": f"概念网络({len(concept_names)}个节点)",
            "description": f"将{len(concepts)}个概念构建为关联网络",
            "attributes": {
                "node_count": len(network["nodes"]),
                "edge_count": len(network["edges"]),
                "density": edge_density
            },
            "source_type": "network",
            "network_structure": network
        } 