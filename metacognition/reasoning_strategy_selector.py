"""
推理策略选择器模块 (Reasoning Strategy Selector)

该模块负责选择适合当前问题和上下文的最优推理策略。
"""

import time
import json
import logging
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

class ReasoningStrategySelector:
    def __init__(self, learning_integrator=None, cognitive_monitor=None):
        self.learning_integrator = learning_integrator
        self.cognitive_monitor = cognitive_monitor
        self.strategy_performance = {}
        self.context_similarity = {}
        self.strategy_registry = self._initialize_strategies()
        self.strategy_history = []
        self.problem_features_cache = {}
        
        # 日志记录
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """初始化推理策略库"""
        return {
            "deductive": {
                "name": "演绎推理",
                "description": "从一般原则推导出特定结论",
                "suitable_for": ["分类问题", "逻辑谜题", "规则应用"],
                "example": "所有人都会死，苏格拉底是人，所以苏格拉底会死",
                "features": ["规则明确", "逻辑严密", "确定性高"],
                "implementation": self._apply_deductive_reasoning
            },
            "inductive": {
                "name": "归纳推理",
                "description": "从特定观察归纳出一般规律",
                "suitable_for": ["模式识别", "规律发现", "预测问题"],
                "example": "观察到所有乌鸦都是黑色的，推断乌鸦这一物种是黑色的",
                "features": ["基于观察", "概率性结论", "可能存在例外"],
                "implementation": self._apply_inductive_reasoning
            },
            "abductive": {
                "name": "溯因推理",
                "description": "从观察结果推测最可能的解释",
                "suitable_for": ["诊断问题", "故障排查", "现象解释"],
                "example": "看到地面湿了，推测可能下雨了",
                "features": ["多种可能解释", "寻找最佳解释", "创造性"],
                "implementation": self._apply_abductive_reasoning
            },
            "analogical": {
                "name": "类比推理",
                "description": "基于相似情况进行推理",
                "suitable_for": ["新颖问题", "跨领域迁移", "创新思考"],
                "example": "太阳系像原子结构，行星围绕太阳就像电子围绕原子核",
                "features": ["寻找相似性", "知识迁移", "启发式"],
                "implementation": self._apply_analogical_reasoning
            },
            "causal": {
                "name": "因果推理",
                "description": "分析事物间的因果关系",
                "suitable_for": ["预测结果", "解释现象", "干预设计"],
                "example": "吸烟导致肺癌几率增加",
                "features": ["时序关系", "机制分析", "控制变量"],
                "implementation": self._apply_causal_reasoning
            },
            "probabilistic": {
                "name": "概率推理",
                "description": "基于不确定性证据的推理",
                "suitable_for": ["风险评估", "不确定决策", "预测分析"],
                "example": "基于症状判断疾病概率",
                "features": ["处理不确定性", "概率更新", "多种可能性"],
                "implementation": self._apply_probabilistic_reasoning
            },
            "spatial": {
                "name": "空间推理",
                "description": "处理空间关系和视觉问题",
                "suitable_for": ["几何问题", "路径规划", "视觉分析"],
                "example": "根据地图规划最短路径",
                "features": ["空间关系", "视觉思考", "几何直觉"],
                "implementation": self._apply_spatial_reasoning
            },
            "counterfactual": {
                "name": "反事实推理",
                "description": "假设性地考虑'如果...会怎样'",
                "suitable_for": ["决策评估", "历史分析", "策略规划"],
                "example": "如果没有发明互联网，世界会怎样",
                "features": ["假设情景", "多重可能性", "创造性思考"],
                "implementation": self._apply_counterfactual_reasoning
            }
        }
    
    def select_reasoning_strategy(self, problem: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        选择最适合的推理策略
        
        Args:
            problem: 问题信息
            context: 上下文信息
            
        Returns:
            Dict: 选择的策略信息
        """
        # 生成唯一标识
        selection_id = str(uuid.uuid4())
        
        # 提取问题特征
        problem_features = self._extract_problem_features(problem)
        
        # 缓存特征以便后续学习
        self.problem_features_cache[selection_id] = problem_features
        
        # 通过特征匹配获取策略候选
        candidates = self._match_strategies_by_features(problem_features)
        
        # 考虑历史表现
        if self.strategy_performance:
            candidates = self._adjust_by_performance(candidates, problem_features)
        
        # 考虑上下文因素（如可用资源、时间限制等）
        candidates = self._adjust_by_context(candidates, context)
        
        # 选择最佳策略
        if not candidates:
            # 如果没有找到合适的策略，使用默认策略（归纳推理）
            selected_strategy = "inductive"
            confidence = 0.5
        else:
            # 选择得分最高的策略
            selected_strategy = max(candidates.items(), key=lambda x: x[1])[0]
            confidence = candidates[selected_strategy]
        
        # 获取策略详情
        strategy_details = self.strategy_registry.get(selected_strategy, {}).copy()
        
        # 记录策略选择
        selection_record = {
            "selection_id": selection_id,
            "timestamp": time.time(),
            "problem_type": problem.get("type", "unknown"),
            "problem_features": problem_features,
            "selected_strategy": selected_strategy,
            "confidence": confidence,
            "context_factors": {k: v for k, v in context.items() if k in ["time_constraint", "resources", "complexity"]}
        }
        
        self.strategy_history.append(selection_record)
        
        # 添加选择信息到返回结果
        strategy_details.update({
            "strategy_id": selected_strategy,
            "selection_id": selection_id,
            "confidence": confidence,
            "alternative_strategies": {k: v for k, v in candidates.items() if k != selected_strategy}
        })
        
        return strategy_details
    
    def _extract_problem_features(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        提取问题特征
        
        Args:
            problem: 问题信息
            
        Returns:
            Dict: 特征向量
        """
        features = {
            "complexity": 0.5,        # 复杂度: 0低 - 1高
            "uncertainty": 0.5,       # 不确定性: 0低 - 1高
            "structuredness": 0.5,    # 结构化程度: 0低 - 1高
            "prior_knowledge": 0.5,   # 先验知识要求: 0低 - 1高
            "creativity": 0.5,        # 创造性要求: 0低 - 1高
            "time_sensitivity": 0.5,  # 时间敏感度: 0低 - 1高
            "causality": 0.5,         # 因果关系程度: 0低 - 1高
            "spatial": 0.5,           # 空间推理要求: 0低 - 1高
            "social": 0.5,            # 社会性推理要求: 0低 - 1高
            "logic": 0.5              # 逻辑性要求: 0低 - 1高
        }
        
        # 分析问题类型
        problem_type = problem.get("type", "").lower()
        problem_description = problem.get("description", "").lower()
        
        # 基于问题类型调整特征
        if problem_type == "classification" or "分类" in problem_description:
            features["structuredness"] = 0.8
            features["logic"] = 0.7
            features["prior_knowledge"] = 0.7
            features["creativity"] = 0.3
            
        elif problem_type == "prediction" or "预测" in problem_description:
            features["uncertainty"] = 0.8
            features["causality"] = 0.7
            features["prior_knowledge"] = 0.6
            
        elif problem_type == "diagnosis" or "诊断" in problem_description or "故障" in problem_description:
            features["causality"] = 0.9
            features["uncertainty"] = 0.7
            features["logic"] = 0.7
            
        elif problem_type == "planning" or "规划" in problem_description:
            features["structuredness"] = 0.7
            features["time_sensitivity"] = 0.8
            features["complexity"] = 0.7
            
        elif problem_type == "design" or "设计" in problem_description:
            features["creativity"] = 0.9
            features["structuredness"] = 0.4
            features["complexity"] = 0.8
            
        elif problem_type == "spatial" or "空间" in problem_description or "几何" in problem_description:
            features["spatial"] = 0.9
            features["structuredness"] = 0.7
            
        elif problem_type == "social" or "社会" in problem_description or "人际" in problem_description:
            features["social"] = 0.9
            features["uncertainty"] = 0.7
            features["creativity"] = 0.6
            
        elif problem_type == "causal" or "因果" in problem_description:
            features["causality"] = 0.9
            features["logic"] = 0.7
            
        # 分析问题描述中的关键词
        keywords = {
            "uncertainty": ["不确定", "可能", "概率", "风险", "随机", "模糊"],
            "logic": ["逻辑", "推导", "证明", "规则", "定理", "公理"],
            "creativity": ["创新", "创造", "新颖", "独特", "想象", "突破"],
            "causality": ["因果", "导致", "引起", "影响", "效应"],
            "complexity": ["复杂", "难题", "挑战", "困难"],
            "spatial": ["空间", "位置", "方向", "距离", "几何"],
            "social": ["社会", "人际", "群体", "文化", "互动"],
            "time_sensitivity": ["时间", "紧急", "截止", "快速", "即时"],
        }
        
        # 根据关键词调整特征
        for feature, words in keywords.items():
            for word in words:
                if word in problem_description:
                    features[feature] = min(1.0, features.get(feature, 0.5) + 0.1)
        
        # 考虑问题其他属性
        complexity_level = problem.get("complexity", "medium").lower()
        if complexity_level == "high":
            features["complexity"] = 0.8
        elif complexity_level == "low":
            features["complexity"] = 0.3
            
        time_constraint = problem.get("time_constraint", "medium").lower()
        if time_constraint == "urgent":
            features["time_sensitivity"] = 0.9
        elif time_constraint == "relaxed":
            features["time_sensitivity"] = 0.3
            
        return features
    
    def _match_strategies_by_features(self, problem_features: Dict[str, float]) -> Dict[str, float]:
        """
        通过特征匹配推理策略
        
        Args:
            problem_features: 问题特征
            
        Returns:
            Dict: 策略匹配度
        """
        strategy_matches = {}
        
        # 策略特征表 - 每种策略的特征适配性
        strategy_feature_map = {
            # 演绎推理
            "deductive": {
                "structuredness": 0.9,  # 高度结构化
                "logic": 0.9,          # 高度逻辑性
                "uncertainty": 0.1,    # 低不确定性
                "creativity": 0.2,     # 低创造性
                "prior_knowledge": 0.7  # 较高先验知识
            },
            # 归纳推理
            "inductive": {
                "structuredness": 0.6,  # 中等结构化
                "logic": 0.7,          # 较高逻辑性
                "uncertainty": 0.5,    # 中等不确定性
                "pattern_recognition": 0.8,  # 高模式识别
                "prior_knowledge": 0.5  # 中等先验知识
            },
            # 溯因推理
            "abductive": {
                "structuredness": 0.4,  # 较低结构化
                "creativity": 0.7,     # 较高创造性
                "uncertainty": 0.7,    # 较高不确定性
                "causality": 0.8,      # 高因果关系
                "complexity": 0.7      # 较高复杂度
            },
            # 类比推理
            "analogical": {
                "creativity": 0.8,     # 高创造性
                "prior_knowledge": 0.8,  # 高先验知识
                "structuredness": 0.3,  # 低结构化
                "complexity": 0.6      # 中等复杂度
            },
            # 因果推理
            "causal": {
                "causality": 0.9,      # 高因果关系
                "logic": 0.7,          # 较高逻辑性
                "time_sensitivity": 0.6,  # 中等时间敏感
                "uncertainty": 0.6     # 中等不确定性
            },
            # 概率推理
            "probabilistic": {
                "uncertainty": 0.9,    # 高不确定性
                "logic": 0.6,          # 中等逻辑性
                "complexity": 0.7,     # 较高复杂度
                "structuredness": 0.5  # 中等结构化
            },
            # 空间推理
            "spatial": {
                "spatial": 0.9,        # 高空间推理
                "structuredness": 0.6,  # 中等结构化
                "complexity": 0.6      # 中等复杂度
            },
            # 反事实推理
            "counterfactual": {
                "creativity": 0.8,     # 高创造性
                "causality": 0.8,      # 高因果关系
                "complexity": 0.8,     # 高复杂度
                "uncertainty": 0.7     # 较高不确定性
            }
        }
        
        # 计算每个策略的匹配度
        for strategy, feature_map in strategy_feature_map.items():
            match_score = 0
            feature_count = 0
            
            for feature, ideal_value in feature_map.items():
                if feature in problem_features:
                    # 计算特征值与理想值的接近程度(1 - 差异)
                    similarity = 1 - abs(problem_features[feature] - ideal_value)
                    match_score += similarity
                    feature_count += 1
            
            # 计算平均匹配度
            if feature_count > 0:
                avg_match = match_score / feature_count
                strategy_matches[strategy] = avg_match
        
        return strategy_matches
    
    def _adjust_by_performance(self, candidates: Dict[str, float], problem_features: Dict[str, float]) -> Dict[str, float]:
        """
        根据历史表现调整策略优先级
        
        Args:
            candidates: 候选策略及其匹配度
            problem_features: 问题特征
            
        Returns:
            Dict: 调整后的策略匹配度
        """
        adjusted_candidates = candidates.copy()
        
        # 查找相似问题的历史表现
        similar_problems = self._find_similar_problems(problem_features)
        
        if not similar_problems:
            return adjusted_candidates
            
        # 计算各策略在相似问题上的平均表现
        strategy_avg_performance = defaultdict(list)
        
        for problem_id, similarity in similar_problems:
            for strategy_id, performance in self.strategy_performance.items():
                if problem_id in performance:
                    # 根据问题相似度加权的表现分数
                    weighted_score = performance[problem_id] * similarity
                    strategy_avg_performance[strategy_id].append(weighted_score)
        
        # 根据历史表现调整候选分数
        for strategy_id, scores in strategy_avg_performance.items():
            if strategy_id in adjusted_candidates and scores:
                avg_score = sum(scores) / len(scores)
                
                # 历史表现权重为0.3，原始匹配度权重为0.7
                adjusted_candidates[strategy_id] = (
                    0.7 * adjusted_candidates[strategy_id] + 
                    0.3 * avg_score
                )
        
        return adjusted_candidates
    
    def _adjust_by_context(self, candidates: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
        """
        根据上下文因素调整策略优先级
        
        Args:
            candidates: 候选策略及其匹配度
            context: 上下文信息
            
        Returns:
            Dict: 调整后的策略匹配度
        """
        adjusted_candidates = candidates.copy()
        
        # 时间限制调整
        time_constraint = context.get("time_constraint", "medium").lower()
        if time_constraint == "urgent":
            # 紧急情况下，降低复杂策略的优先级
            complex_strategies = ["abductive", "counterfactual"]
            for strategy in complex_strategies:
                if strategy in adjusted_candidates:
                    adjusted_candidates[strategy] *= 0.8
                    
            # 提高简单快速策略的优先级
            fast_strategies = ["deductive", "inductive"]
            for strategy in fast_strategies:
                if strategy in adjusted_candidates:
                    adjusted_candidates[strategy] *= 1.2
        
        # 资源限制调整
        resource_constraint = context.get("resource_constraint", "medium").lower()
        if resource_constraint == "limited":
            # 资源有限时，降低资源密集型策略优先级
            resource_heavy = ["probabilistic", "causal"]
            for strategy in resource_heavy:
                if strategy in adjusted_candidates:
                    adjusted_candidates[strategy] *= 0.8
        
        # 问题重要性调整
        importance = context.get("importance", "medium").lower()
        if importance == "high":
            # 高重要性问题，提高高精确度策略的优先级
            precision_strategies = ["deductive", "causal"]
            for strategy in precision_strategies:
                if strategy in adjusted_candidates:
                    adjusted_candidates[strategy] *= 1.1
        
        # 确保所有值在0-1范围内
        for strategy in adjusted_candidates:
            adjusted_candidates[strategy] = max(0, min(1, adjusted_candidates[strategy]))
        
        return adjusted_candidates
    
    def _find_similar_problems(self, target_features: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        查找特征相似的历史问题
        
        Args:
            target_features: 目标问题特征
            
        Returns:
            List: (问题ID, 相似度)元组列表
        """
        similarities = []
        
        for problem_id, features in self.problem_features_cache.items():
            # 计算特征向量相似度
            similarity = self._calculate_feature_similarity(target_features, features)
            
            # 只保留相似度大于0.7的问题
            if similarity > 0.7:
                similarities.append((problem_id, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回最相似的5个问题
        return similarities[:5]
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        计算两个特征向量的相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            float: 相似度得分(0-1)
        """
        # 获取共同特征
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0
        
        # 计算特征距离
        squared_diff_sum = 0
        for feature in common_features:
            squared_diff_sum += (features1[feature] - features2[feature]) ** 2
            
        # 欧氏距离转换为相似度(0-1)
        distance = np.sqrt(squared_diff_sum / len(common_features))
        similarity = 1 - min(1, distance)
        
        return similarity
    
    def evaluate_strategy_effectiveness(self, strategy: str, problem: Dict[str, Any], 
                                      result: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估策略效果，更新策略选择参数
        
        Args:
            strategy: 使用的策略
            problem: 问题信息
            result: 结果信息
            
        Returns:
            Dict: 评估结果
        """
        # 获取策略选择ID
        selection_id = result.get("selection_id", problem.get("selection_id"))
        
        if not selection_id:
            return {"status": "error", "message": "缺少策略选择ID"}
        
        # 从结果中提取效果评估指标
        accuracy = result.get("accuracy", 0)
        efficiency = result.get("efficiency", 0)
        completeness = result.get("completeness", 0)
        
        # 如果没有明确的评估指标，尝试从结果状态推断
        if "status" in result:
            status = result["status"].lower()
            if status == "success":
                accuracy = 1.0
                efficiency = 0.8
                completeness = 0.9
            elif status == "partial_success":
                accuracy = 0.7
                efficiency = 0.6
                completeness = 0.7
            elif status == "failure":
                accuracy = 0.3
                efficiency = 0.4
                completeness = 0.4
        
        # 计算整体效果分数
        effectiveness_score = (accuracy * 0.5) + (efficiency * 0.3) + (completeness * 0.2)
        
        # 更新策略表现记录
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {}
            
        self.strategy_performance[strategy][selection_id] = effectiveness_score
        
        # 更新策略历史
        for record in self.strategy_history:
            if record.get("selection_id") == selection_id:
                record["effectiveness"] = effectiveness_score
                record["evaluation_time"] = time.time()
                break
        
        # 返回评估结果
        return {
            "status": "success",
            "strategy": strategy,
            "selection_id": selection_id,
            "effectiveness": effectiveness_score,
            "metrics": {
                "accuracy": accuracy,
                "efficiency": efficiency,
                "completeness": completeness
            }
        }
    
    def get_strategy_recommendations(self, problem_description: str) -> List[Dict[str, Any]]:
        """
        根据问题描述推荐推理策略
        
        Args:
            problem_description: 问题描述
            
        Returns:
            List: 推荐策略列表
        """
        # 从问题描述中构建简化问题对象
        problem = {
            "description": problem_description,
            "type": self._infer_problem_type(problem_description)
        }
        
        # 提取问题特征
        features = self._extract_problem_features(problem)
        
        # 匹配策略
        matches = self._match_strategies_by_features(features)
        
        # 按匹配度排序
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        # 生成推荐列表
        recommendations = []
        for strategy_id, score in sorted_matches[:3]:  # 前3名策略
            strategy_info = self.strategy_registry.get(strategy_id, {})
            
            recommendations.append({
                "strategy_id": strategy_id,
                "name": strategy_info.get("name", strategy_id),
                "description": strategy_info.get("description", ""),
                "match_score": score,
                "suitable_for": strategy_info.get("suitable_for", []),
                "example": strategy_info.get("example", "")
            })
        
        return recommendations
    
    def _infer_problem_type(self, description: str) -> str:
        """
        从问题描述推断问题类型
        
        Args:
            description: 问题描述
            
        Returns:
            str: 推断的问题类型
        """
        description = description.lower()
        
        type_keywords = {
            "classification": ["分类", "归类", "识别", "区分", "属于"],
            "prediction": ["预测", "预估", "预期", "未来", "推测"],
            "diagnosis": ["诊断", "故障", "问题", "原因", "排查"],
            "planning": ["规划", "计划", "安排", "策略", "路线"],
            "design": ["设计", "创建", "构建", "开发", "制作"],
            "causal": ["因果", "导致", "引起", "影响", "效应"],
            "spatial": ["空间", "位置", "方向", "距离", "几何"],
            "social": ["社会", "人际", "群体", "互动", "关系"]
        }
        
        # 找出最匹配的类型
        max_matches = 0
        inferred_type = "general"
        
        for type_name, keywords in type_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in description)
            if matches > max_matches:
                max_matches = matches
                inferred_type = type_name
        
        return inferred_type
    
    def _apply_deductive_reasoning(self, problem, context):
        """演绎推理的实现方法"""
        # 这里只是占位，实际实现应该与具体应用场景结合
        return {"strategy": "deductive", "status": "implemented"}
    
    def _apply_inductive_reasoning(self, problem, context):
        """归纳推理的实现方法"""
        return {"strategy": "inductive", "status": "implemented"}
    
    def _apply_abductive_reasoning(self, problem, context):
        """溯因推理的实现方法"""
        return {"strategy": "abductive", "status": "implemented"}
    
    def _apply_analogical_reasoning(self, problem, context):
        """类比推理的实现方法"""
        return {"strategy": "analogical", "status": "implemented"}
    
    def _apply_causal_reasoning(self, problem, context):
        """因果推理的实现方法"""
        return {"strategy": "causal", "status": "implemented"}
    
    def _apply_probabilistic_reasoning(self, problem, context):
        """概率推理的实现方法"""
        return {"strategy": "probabilistic", "status": "implemented"}
    
    def _apply_spatial_reasoning(self, problem, context):
        """空间推理的实现方法"""
        return {"strategy": "spatial", "status": "implemented"}
    
    def _apply_counterfactual_reasoning(self, problem, context):
        """反事实推理的实现方法"""
        return {"strategy": "counterfactual", "status": "implemented"}
    
    def get_strategy_performance_stats(self) -> Dict[str, Any]:
        """
        获取策略表现统计信息
        
        Returns:
            Dict: 策略表现统计
        """
        stats = {
            "strategies": {},
            "overall_average": 0,
            "best_strategy": None,
            "worst_strategy": None,
            "strategy_count": len(self.strategy_performance),
            "total_evaluations": 0
        }
        
        if not self.strategy_performance:
            return stats
        
        # 计算每个策略的平均表现
        overall_sum = 0
        overall_count = 0
        
        for strategy, performances in self.strategy_performance.items():
            if not performances:
                continue
                
            avg_performance = sum(performances.values()) / len(performances)
            count = len(performances)
            
            stats["strategies"][strategy] = {
                "average_score": avg_performance,
                "evaluation_count": count,
                "min_score": min(performances.values()),
                "max_score": max(performances.values())
            }
            
            overall_sum += avg_performance * count
            overall_count += count
        
        # 计算总体平均分
        if overall_count > 0:
            stats["overall_average"] = overall_sum / overall_count
            stats["total_evaluations"] = overall_count
        
        # 找出最佳和最差策略
        if stats["strategies"]:
            best_strategy = max(stats["strategies"].items(), key=lambda x: x[1]["average_score"])
            worst_strategy = min(stats["strategies"].items(), key=lambda x: x[1]["average_score"])
            
            stats["best_strategy"] = {
                "name": best_strategy[0],
                "score": best_strategy[1]["average_score"]
            }
            
            stats["worst_strategy"] = {
                "name": worst_strategy[0],
                "score": worst_strategy[1]["average_score"]
            }
        
        return stats
    
    def get_strategies_by_problem_type(self) -> Dict[str, List[str]]:
        """
        获取每种问题类型的最适合策略
        
        Returns:
            Dict: 问题类型到策略的映射
        """
        problem_type_strategies = {
            "classification": [],
            "prediction": [],
            "diagnosis": [],
            "planning": [],
            "design": [],
            "causal": [],
            "spatial": [],
            "social": [],
            "general": []
        }
        
        # 如果没有足够的历史数据，使用预设映射
        if len(self.strategy_history) < 10:
            problem_type_strategies.update({
                "classification": ["deductive", "inductive"],
                "prediction": ["inductive", "probabilistic", "causal"],
                "diagnosis": ["abductive", "causal"],
                "planning": ["deductive", "causal", "counterfactual"],
                "design": ["analogical", "abductive"],
                "causal": ["causal", "counterfactual"],
                "spatial": ["spatial", "deductive"],
                "social": ["analogical", "inductive"],
                "general": ["inductive", "deductive"]
            })
            return problem_type_strategies
        
        # 分析历史数据中的问题类型和策略效果
        type_strategy_scores = defaultdict(lambda: defaultdict(list))
        
        for record in self.strategy_history:
            if "effectiveness" not in record:
                continue
                
            problem_type = record.get("problem_type", "general")
            strategy = record.get("selected_strategy")
            effectiveness = record.get("effectiveness")
            
            if strategy and effectiveness is not None:
                type_strategy_scores[problem_type][strategy].append(effectiveness)
        
        # 对每种问题类型，选择平均效果最好的策略
        for problem_type, strategy_scores in type_strategy_scores.items():
            # 计算每个策略的平均分
            avg_scores = {}
            for strategy, scores in strategy_scores.items():
                if scores:
                    avg_scores[strategy] = sum(scores) / len(scores)
            
            # 按平均分排序
            sorted_strategies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 选择前2个最有效的策略
            problem_type_strategies[problem_type] = [s[0] for s in sorted_strategies[:2]]
        
        return problem_type_strategies 