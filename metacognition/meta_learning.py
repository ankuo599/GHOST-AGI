"""
元学习模块 (Meta-Learning Module)

该模块负责管理学习过程，优化学习策略，评估和改进学习效果。
"""

import time
import json
import logging
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

class MetaLearningModule:
    def __init__(self, learning_system=None, cognitive_monitor=None):
        """
        初始化元学习模块
        
        Args:
            learning_system: 学习系统接口
            cognitive_monitor: 认知监控模块
        """
        self.learning_system = learning_system
        self.cognitive_monitor = cognitive_monitor
        self.learning_strategies = self._initialize_learning_strategies()
        self.strategy_performance = {}
        self.learning_history = []
        self.hyperparameters = self._initialize_hyperparameters()
        self.task_type_configs = {}
        
        # 日志记录
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_hyperparameters(self) -> Dict[str, Any]:
        """初始化学习超参数"""
        return {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 5,
            "regularization": 0.001,
            "dropout": 0.2,
            "early_stopping_patience": 3,
            "optimizer": "adam"
        }
    
    def _initialize_learning_strategies(self) -> Dict[str, Dict[str, Any]]:
        """初始化学习策略库"""
        return {
            "supervised": {
                "name": "监督学习",
                "description": "使用标记数据进行学习",
                "suitable_for": ["分类", "回归", "序列标注"],
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 64,
                    "epochs": 10
                }
            },
            "unsupervised": {
                "name": "无监督学习",
                "description": "从未标记数据中发现模式",
                "suitable_for": ["聚类", "降维", "异常检测"],
                "hyperparameters": {
                    "learning_rate": 0.005,
                    "batch_size": 128,
                    "clusters": 10
                }
            },
            "reinforcement": {
                "name": "强化学习",
                "description": "通过环境反馈学习策略",
                "suitable_for": ["决策", "控制", "游戏"],
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "discount_factor": 0.95,
                    "exploration_rate": 0.1
                }
            },
            "transfer": {
                "name": "迁移学习",
                "description": "利用已有知识学习新任务",
                "suitable_for": ["小样本学习", "领域适应", "知识迁移"],
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "fine_tune_layers": 2,
                    "epochs": 5
                }
            },
            "meta": {
                "name": "元学习",
                "description": "学习如何学习新任务",
                "suitable_for": ["快速适应", "小样本学习", "持续学习"],
                "hyperparameters": {
                    "meta_learning_rate": 0.0001,
                    "task_learning_rate": 0.01,
                    "meta_batch_size": 16
                }
            }
        }
    
    def optimize_learning_strategy(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        为特定任务优化学习策略
        
        Args:
            task: 学习任务信息
            
        Returns:
            Dict: 优化后的学习策略
        """
        # 生成任务ID
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        # 提取任务特征
        task_type = task.get("type", "unknown")
        data_size = task.get("data_size", 1000)
        complexity = task.get("complexity", "medium")
        time_constraint = task.get("time_constraint", "medium")
        
        # 匹配最合适的基础学习策略
        strategy = self._match_strategy_to_task(task)
        
        # 调整超参数
        optimized_params = self._optimize_hyperparameters(strategy, task)
        
        # 根据任务特性进一步优化
        if data_size < 100:
            # 小数据集优化
            optimized_params["batch_size"] = min(16, optimized_params.get("batch_size", 32))
            optimized_params["regularization"] = optimized_params.get("regularization", 0.001) * 2
            
        if complexity == "high":
            # 复杂任务优化
            optimized_params["epochs"] = optimized_params.get("epochs", 5) * 1.5
            
        if time_constraint == "urgent":
            # 时间受限优化
            optimized_params["early_stopping_patience"] = 2
            optimized_params["epochs"] = max(3, optimized_params.get("epochs", 5) // 2)
        
        # 生成学习策略配置
        learning_config = {
            "task_id": task_id,
            "strategy": strategy["name"],
            "description": strategy["description"],
            "hyperparameters": optimized_params,
            "timestamp": time.time()
        }
        
        # 记录到历史
        self.learning_history.append({
            "task_id": task_id,
            "task_type": task_type,
            "strategy": strategy["name"],
            "hyperparameters": optimized_params,
            "timestamp": time.time(),
            "status": "configured"
        })
        
        return learning_config
    
    def _match_strategy_to_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        匹配任务与最合适的学习策略
        
        Args:
            task: 任务信息
            
        Returns:
            Dict: 选择的策略
        """
        task_type = task.get("type", "").lower()
        
        # 基于任务类型的策略匹配
        type_strategy_map = {
            "classification": "supervised",
            "regression": "supervised",
            "clustering": "unsupervised",
            "anomaly_detection": "unsupervised",
            "decision_making": "reinforcement",
            "control": "reinforcement",
            "few_shot": "meta",
            "domain_adaptation": "transfer"
        }
        
        # 尝试直接匹配
        if task_type in type_strategy_map:
            strategy_id = type_strategy_map[task_type]
            return self.learning_strategies.get(strategy_id, self.learning_strategies["supervised"])
        
        # 如果没有直接匹配，进行关键词匹配
        task_description = task.get("description", "").lower()
        
        if any(kw in task_description for kw in ["分类", "预测类别", "识别"]):
            return self.learning_strategies["supervised"]
        
        if any(kw in task_description for kw in ["聚类", "分组", "无标签", "模式发现"]):
            return self.learning_strategies["unsupervised"]
        
        if any(kw in task_description for kw in ["决策", "奖励", "策略", "行动"]):
            return self.learning_strategies["reinforcement"]
        
        if any(kw in task_description for kw in ["迁移", "预训练", "微调"]):
            return self.learning_strategies["transfer"]
        
        if any(kw in task_description for kw in ["小样本", "快速适应", "学会学习"]):
            return self.learning_strategies["meta"]
        
        # 默认使用监督学习
        return self.learning_strategies["supervised"]
    
    def _optimize_hyperparameters(self, strategy: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            strategy: 学习策略
            task: 任务信息
            
        Returns:
            Dict: 优化后的超参数
        """
        # 获取基础超参数
        base_params = strategy.get("hyperparameters", {}).copy()
        
        # 检查是否有历史任务相似
        task_type = task.get("type", "unknown")
        if task_type in self.task_type_configs:
            # 使用该类型任务的历史最佳参数
            historical_params = self.task_type_configs[task_type]
            # 混合历史参数和基础参数
            for key, value in historical_params.items():
                base_params[key] = value
        
        # 任务特定调整
        data_size = task.get("data_size", 1000)
        if data_size > 10000:
            # 大数据集调整
            base_params["batch_size"] = base_params.get("batch_size", 32) * 2
            base_params["learning_rate"] = base_params.get("learning_rate", 0.01) / 2
        
        # 返回优化后的参数
        return base_params
    
    def evaluate_learning_effectiveness(self, task_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估学习效果
        
        Args:
            task_id: 学习任务ID
            results: 学习结果
            
        Returns:
            Dict: 评估结果
        """
        # 查找任务历史记录
        task_record = None
        for record in self.learning_history:
            if record["task_id"] == task_id:
                task_record = record
                break
        
        if not task_record:
            return {"status": "error", "message": f"未找到ID为{task_id}的任务记录"}
        
        # 提取结果指标
        accuracy = results.get("accuracy", 0)
        loss = results.get("loss", 1.0)
        convergence_time = results.get("convergence_time", 0)
        generalization = results.get("generalization", 0)
        
        # 计算综合评分
        if "target_metric" in results:
            # 如果有指定的目标指标，使用它
            effectiveness = results[results["target_metric"]]
        else:
            # 否则计算综合分数
            effectiveness = (accuracy * 0.4) + (max(0, 1 - loss) * 0.3) + (generalization * 0.3)
        
        # 更新策略表现记录
        strategy = task_record["strategy"]
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
            
        self.strategy_performance[strategy].append({
            "task_id": task_id,
            "effectiveness": effectiveness,
            "metrics": {
                "accuracy": accuracy,
                "loss": loss,
                "convergence_time": convergence_time,
                "generalization": generalization
            },
            "hyperparameters": task_record["hyperparameters"],
            "evaluation_time": time.time()
        })
        
        # 更新任务记录
        task_record["status"] = "completed"
        task_record["results"] = {
            "effectiveness": effectiveness,
            "metrics": {
                "accuracy": accuracy,
                "loss": loss,
                "convergence_time": convergence_time,
                "generalization": generalization
            }
        }
        
        # 更新任务类型的最佳配置
        task_type = task_record["task_type"]
        if task_type not in self.task_type_configs or \
           self.task_type_configs[task_type].get("effectiveness", 0) < effectiveness:
            self.task_type_configs[task_type] = {
                "hyperparameters": task_record["hyperparameters"],
                "effectiveness": effectiveness,
                "task_id": task_id,
                "update_time": time.time()
            }
        
        return {
            "status": "success",
            "task_id": task_id,
            "effectiveness": effectiveness,
            "percentile": self._calculate_percentile(strategy, effectiveness),
            "improvement_suggestions": self._generate_improvement_suggestions(task_record, results)
        }
    
    def _calculate_percentile(self, strategy: str, effectiveness: float) -> float:
        """
        计算当前效果在历史表现中的百分位
        
        Args:
            strategy: 学习策略
            effectiveness: 效果评分
            
        Returns:
            float: 百分位数(0-100)
        """
        if strategy not in self.strategy_performance:
            return 50.0  # 无历史数据，默认为中位数
            
        scores = [record["effectiveness"] for record in self.strategy_performance[strategy]]
        if not scores:
            return 50.0
            
        # 计算百分位
        below_count = sum(1 for score in scores if score < effectiveness)
        percentile = (below_count / len(scores)) * 100
        
        return percentile
    
    def _generate_improvement_suggestions(self, task_record: Dict[str, Any], 
                                        results: Dict[str, Any]) -> List[str]:
        """
        生成学习改进建议
        
        Args:
            task_record: 任务记录
            results: 学习结果
            
        Returns:
            List: 改进建议列表
        """
        suggestions = []
        
        # 分析结果指标
        accuracy = results.get("accuracy", 0)
        loss = results.get("loss", 1.0)
        generalization = results.get("generalization", 0)
        
        # 根据指标生成建议
        if accuracy < 0.7:
            suggestions.append("提高模型复杂度以增强拟合能力")
            
        if loss > 0.5:
            suggestions.append("增加训练轮次或调整学习率")
            
        if generalization < 0.6:
            suggestions.append("增加正则化强度以改善泛化能力")
            
        # 基于超参数建议
        params = task_record["hyperparameters"]
        
        if "batch_size" in params and params["batch_size"] > 128 and accuracy < 0.8:
            suggestions.append("尝试减小批量大小以提高优化精度")
            
        if "learning_rate" in params and params["learning_rate"] > 0.1 and loss > 0.3:
            suggestions.append("降低学习率以避免优化不稳定")
            
        # 如果效果不佳，建议尝试不同策略
        if results.get("effectiveness", 0) < 0.5:
            suggestions.append("考虑切换学习策略或组合多种学习方法")
        
        return suggestions
    
    def adapt_hyperparameters(self, task_type: str, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        根据任务表现调整超参数
        
        Args:
            task_type: 任务类型
            performance_metrics: 性能指标
            
        Returns:
            Dict: 调整后的超参数
        """
        # 如果有该任务类型的配置，获取基础参数
        if task_type in self.task_type_configs:
            params = self.task_type_configs[task_type]["hyperparameters"].copy()
        else:
            # 否则使用默认参数
            params = self.hyperparameters.copy()
        
        # 根据性能指标调整
        accuracy = performance_metrics.get("accuracy", 0)
        loss = performance_metrics.get("loss", 1.0)
        convergence_speed = performance_metrics.get("convergence_speed", 0.5)
        
        # 调整学习率
        if loss > 0.7 or convergence_speed < 0.3:
            # 如果损失高或收敛慢，增加学习率
            params["learning_rate"] = min(0.1, params.get("learning_rate", 0.01) * 1.5)
        elif loss < 0.3 and accuracy > 0.9:
            # 如果表现很好，微调学习率
            params["learning_rate"] = max(0.0001, params.get("learning_rate", 0.01) * 0.8)
        
        # 调整批量大小
        if convergence_speed < 0.4:
            # 如果收敛慢，减小批量提高每步优化效果
            params["batch_size"] = max(16, params.get("batch_size", 32) // 2)
        
        # 调整正则化
        if accuracy > 0.9 and performance_metrics.get("generalization", 0) < 0.7:
            # 如果训练准确率高但泛化能力差，增加正则化
            params["regularization"] = min(0.01, params.get("regularization", 0.001) * 2)
            params["dropout"] = min(0.5, params.get("dropout", 0.2) + 0.1)
        
        # 调整训练轮次
        if loss > 0.5 and convergence_speed > 0.7:
            # 如果损失仍高但收敛快，增加训练轮次
            params["epochs"] = params.get("epochs", 5) + 2
        
        return params
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """
        分析学习模式和趋势
        
        Returns:
            Dict: 学习模式分析结果
        """
        if len(self.learning_history) < 3:
            return {
                "status": "insufficient_data",
                "message": "历史数据不足，无法进行有效分析"
            }
            
        # 统计策略使用情况
        strategy_usage = defaultdict(int)
        strategy_effectiveness = defaultdict(list)
        
        # 任务类型统计
        task_type_count = defaultdict(int)
        task_type_effectiveness = defaultdict(list)
        
        # 时间趋势分析
        time_series_data = []
        
        for record in self.learning_history:
            if "results" not in record:
                continue
                
            strategy = record.get("strategy", "unknown")
            task_type = record.get("task_type", "unknown")
            effectiveness = record["results"].get("effectiveness", 0)
            timestamp = record.get("timestamp", 0)
            
            # 更新策略统计
            strategy_usage[strategy] += 1
            strategy_effectiveness[strategy].append(effectiveness)
            
            # 更新任务类型统计
            task_type_count[task_type] += 1
            task_type_effectiveness[task_type].append(effectiveness)
            
            # 添加时间序列数据点
            time_series_data.append({
                "timestamp": timestamp,
                "effectiveness": effectiveness,
                "strategy": strategy,
                "task_type": task_type
            })
        
        # 计算每个策略的平均效果
        strategy_avg_effectiveness = {}
        for strategy, scores in strategy_effectiveness.items():
            if scores:
                strategy_avg_effectiveness[strategy] = sum(scores) / len(scores)
        
        # 计算每种任务类型的平均效果
        task_type_avg_effectiveness = {}
        for task_type, scores in task_type_effectiveness.items():
            if scores:
                task_type_avg_effectiveness[task_type] = sum(scores) / len(scores)
        
        # 分析超参数影响
        hyperparameter_impact = self._analyze_hyperparameter_impact()
        
        # 分析学习效果趋势
        effectiveness_trend = self._analyze_effectiveness_trend(time_series_data)
        
        # 识别学习瓶颈
        learning_bottlenecks = self._identify_learning_bottlenecks()
        
        return {
            "status": "success",
            "strategy_usage": dict(strategy_usage),
            "strategy_effectiveness": strategy_avg_effectiveness,
            "task_type_distribution": dict(task_type_count),
            "task_type_effectiveness": task_type_avg_effectiveness,
            "effectiveness_trend": effectiveness_trend,
            "hyperparameter_impact": hyperparameter_impact,
            "learning_bottlenecks": learning_bottlenecks,
            "total_tasks_analyzed": len(self.learning_history)
        }
    
    def _analyze_hyperparameter_impact(self) -> List[Dict[str, Any]]:
        """
        分析超参数对学习效果的影响
        
        Returns:
            List: 超参数影响分析结果
        """
        if not self.learning_history:
            return []
            
        # 收集所有使用过的超参数
        all_hyperparams = set()
        for record in self.learning_history:
            if "hyperparameters" in record:
                all_hyperparams.update(record["hyperparameters"].keys())
        
        # 分析每个超参数的影响
        impact_analysis = []
        
        for param in all_hyperparams:
            # 收集该参数的所有取值及对应效果
            param_values = []
            effectiveness_values = []
            
            for record in self.learning_history:
                if "hyperparameters" in record and param in record["hyperparameters"] and "results" in record:
                    param_values.append(record["hyperparameters"][param])
                    effectiveness_values.append(record["results"].get("effectiveness", 0))
            
            if len(param_values) < 3:
                continue  # 数据点太少，跳过
            
            # 计算相关性
            try:
                correlation = np.corrcoef(param_values, effectiveness_values)[0, 1]
                
                impact_analysis.append({
                    "parameter": param,
                    "correlation": correlation,
                    "sample_count": len(param_values),
                    "min_value": min(param_values),
                    "max_value": max(param_values),
                    "impact_direction": "positive" if correlation > 0 else "negative"
                })
            except:
                # 计算失败，可能是数据类型不兼容
                continue
        
        # 按相关性强度排序
        impact_analysis.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return impact_analysis
    
    def _analyze_effectiveness_trend(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析学习效果随时间的变化趋势
        
        Args:
            time_series_data: 时间序列数据
            
        Returns:
            Dict: 趋势分析结果
        """
        if not time_series_data:
            return {"trend": "unknown", "slope": 0}
            
        # 按时间排序
        sorted_data = sorted(time_series_data, key=lambda x: x["timestamp"])
        
        # 提取时间和效果数据
        times = [entry["timestamp"] for entry in sorted_data]
        effectiveness = [entry["effectiveness"] for entry in sorted_data]
        
        if len(times) < 2:
            return {"trend": "insufficient_data", "slope": 0}
        
        # 简化分析：比较前半部分和后半部分的平均效果
        mid_point = len(effectiveness) // 2
        first_half = effectiveness[:mid_point]
        second_half = effectiveness[mid_point:]
        
        first_half_avg = sum(first_half) / len(first_half) if first_half else 0
        second_half_avg = sum(second_half) / len(second_half) if second_half else 0
        
        diff = second_half_avg - first_half_avg
        trend = "improving" if diff > 0.05 else "declining" if diff < -0.05 else "stable"
        
        # 计算最近5个任务的平均效果
        recent_effectiveness = effectiveness[-5:] if len(effectiveness) >= 5 else effectiveness
        recent_avg = sum(recent_effectiveness) / len(recent_effectiveness)
        
        return {
            "trend": trend,
            "slope": diff,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
            "recent_avg": recent_avg
        }
    
    def _identify_learning_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        识别学习过程中的瓶颈
        
        Returns:
            List: 瓶颈列表
        """
        bottlenecks = []
        
        # 检查是否有效果特别差的任务
        poor_performance_threshold = 0.4
        poor_performance_tasks = []
        
        for record in self.learning_history:
            if "results" in record and record["results"].get("effectiveness", 0) < poor_performance_threshold:
                poor_performance_tasks.append({
                    "task_id": record.get("task_id"),
                    "task_type": record.get("task_type"),
                    "effectiveness": record["results"].get("effectiveness", 0)
                })
        
        if len(poor_performance_tasks) >= 3:
            bottlenecks.append({
                "type": "poor_performance",
                "description": "多个任务表现不佳",
                "affected_tasks": poor_performance_tasks[:3],  # 只展示前3个
                "suggestion": "考虑重新评估基础学习策略，或增加模型复杂度"
            })
        
        # 检查学习效果是否停滞
        if len(self.learning_history) >= 5:
            recent_records = sorted(self.learning_history, key=lambda x: x.get("timestamp", 0), reverse=True)[:5]
            recent_effectiveness = [r["results"].get("effectiveness", 0) for r in recent_records if "results" in r]
            
            if recent_effectiveness and max(recent_effectiveness) - min(recent_effectiveness) < 0.1:
                bottlenecks.append({
                    "type": "stagnation",
                    "description": "学习效果停滞",
                    "effectiveness_range": [min(recent_effectiveness), max(recent_effectiveness)],
                    "suggestion": "尝试引入新的学习策略或大幅调整超参数"
                })
        
        # 检查是否有任务类型表现差异大
        if len(self.task_type_configs) >= 2:
            effectiveness_values = [config.get("effectiveness", 0) for config in self.task_type_configs.values()]
            if max(effectiveness_values) - min(effectiveness_values) > 0.3:
                bottlenecks.append({
                    "type": "task_type_disparity",
                    "description": "不同任务类型表现差异大",
                    "min_effectiveness": min(effectiveness_values),
                    "max_effectiveness": max(effectiveness_values),
                    "suggestion": "为表现差的任务类型开发专门的学习策略"
                })
        
        return bottlenecks
    
    def recommend_learning_improvements(self) -> List[Dict[str, Any]]:
        """
        推荐学习改进措施
        
        Returns:
            List: 改进建议列表
        """
        # 先分析学习模式
        analysis = self.analyze_learning_patterns()
        
        if analysis.get("status") != "success":
            return [{
                "type": "data_collection",
                "priority": "high",
                "description": "收集更多学习数据以便进行有效分析",
                "implementation": "增加更多样化的学习任务并记录结果"
            }]
        
        recommendations = []
        
        # 根据效果趋势推荐
        trend = analysis.get("effectiveness_trend", {}).get("trend")
        if trend == "declining":
            recommendations.append({
                "type": "trend_reversal",
                "priority": "high",
                "description": "学习效果呈下降趋势",
                "implementation": "重新评估当前学习策略，考虑引入新的学习方法"
            })
        
        # 根据学习瓶颈推荐
        bottlenecks = analysis.get("learning_bottlenecks", [])
        for bottleneck in bottlenecks:
            recommendations.append({
                "type": "bottleneck_resolution",
                "priority": "high",
                "description": bottleneck.get("description", "学习瓶颈"),
                "implementation": bottleneck.get("suggestion", "")
            })
        
        # 根据超参数影响推荐
        param_impacts = analysis.get("hyperparameter_impact", [])
        for impact in param_impacts[:2]:  # 只取影响最大的两个参数
            if abs(impact.get("correlation", 0)) > 0.3:
                direction = "增大" if impact.get("impact_direction") == "positive" else "减小"
                recommendations.append({
                    "type": "hyperparameter_tuning",
                    "priority": "medium",
                    "description": f"调整{impact['parameter']}参数",
                    "implementation": f"{direction}{impact['parameter']}值以提高学习效果"
                })
        
        # 根据策略效果推荐
        strategy_effectiveness = analysis.get("strategy_effectiveness", {})
        if strategy_effectiveness:
            best_strategy = max(strategy_effectiveness.items(), key=lambda x: x[1])[0]
            worst_strategy = min(strategy_effectiveness.items(), key=lambda x: x[1])[0]
            
            if strategy_effectiveness[best_strategy] - strategy_effectiveness[worst_strategy] > 0.2:
                recommendations.append({
                    "type": "strategy_preference",
                    "priority": "medium",
                    "description": f"偏好使用{best_strategy}策略，减少{worst_strategy}策略",
                    "implementation": f"对适合的任务优先考虑{best_strategy}策略"
                })
        
        # 数据相关建议
        recommendations.append({
            "type": "data_quality",
            "priority": "medium",
            "description": "改善训练数据质量",
            "implementation": "增加数据多样性，清理异常值，平衡数据分布"
        })
        
        # 如果建议太少，添加一些通用建议
        if len(recommendations) < 3:
            recommendations.extend([
                {
                    "type": "ensemble_methods",
                    "priority": "medium",
                    "description": "应用集成学习方法",
                    "implementation": "组合多个学习模型以提高整体效果"
                },
                {
                    "type": "feature_engineering",
                    "priority": "medium",
                    "description": "加强特征工程",
                    "implementation": "设计更有效的特征表示，尝试自动特征选择"
                }
            ])
        
        # 按优先级排序
        priority_map = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: priority_map.get(x["priority"], 0), reverse=True)
        
        return recommendations
    
    def get_learning_performance_stats(self) -> Dict[str, Any]:
        """
        获取学习表现统计信息
        
        Returns:
            Dict: 学习表现统计
        """
        if not self.learning_history:
            return {
                "status": "no_data",
                "message": "没有学习历史数据"
            }
            
        # 只分析已完成的任务
        completed_tasks = [task for task in self.learning_history if task.get("status") == "completed" and "results" in task]
        
        if not completed_tasks:
            return {
                "status": "no_completed_tasks",
                "message": "没有已完成的学习任务"
            }
            
        # 计算整体指标
        effectiveness_scores = [task["results"].get("effectiveness", 0) for task in completed_tasks]
        average_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        # 计算各策略的表现
        strategy_performance = defaultdict(list)
        for task in completed_tasks:
            strategy = task.get("strategy")
            if strategy:
                strategy_performance[strategy].append(
                    task["results"].get("effectiveness", 0)
                )
                
        strategy_averages = {}
        for strategy, scores in strategy_performance.items():
            strategy_averages[strategy] = sum(scores) / len(scores)
            
        # 计算任务类型的表现
        task_type_performance = defaultdict(list)
        for task in completed_tasks:
            task_type = task.get("task_type")
            if task_type:
                task_type_performance[task_type].append(
                    task["results"].get("effectiveness", 0)
                )
                
        task_type_averages = {}
        for task_type, scores in task_type_performance.items():
            task_type_averages[task_type] = sum(scores) / len(scores)
            
        # 时间趋势分析
        sorted_tasks = sorted(completed_tasks, key=lambda x: x.get("timestamp", 0))
        if len(sorted_tasks) >= 2:
            first_half = sorted_tasks[:len(sorted_tasks)//2]
            second_half = sorted_tasks[len(sorted_tasks)//2:]
            
            first_half_avg = sum(task["results"].get("effectiveness", 0) for task in first_half) / len(first_half)
            second_half_avg = sum(task["results"].get("effectiveness", 0) for task in second_half) / len(second_half)
            
            trend = "improving" if second_half_avg > first_half_avg else "declining" if second_half_avg < first_half_avg else "stable"
        else:
            trend = "insufficient_data"
            first_half_avg = 0
            second_half_avg = 0
            
        return {
            "status": "success",
            "task_count": len(completed_tasks),
            "average_effectiveness": average_effectiveness,
            "min_effectiveness": min(effectiveness_scores),
            "max_effectiveness": max(effectiveness_scores),
            "strategy_performance": strategy_averages,
            "task_type_performance": task_type_averages,
            "trend": {
                "direction": trend,
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
                "change": second_half_avg - first_half_avg
            }
        }
    
    # =========================  系统集成功能  ========================= #
    
    def integrate_with_cognitive_monitor(self, cognitive_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        集成认知监控反馈，优化学习过程
        
        Args:
            cognitive_feedback: 认知监控提供的反馈
            
        Returns:
            Dict: 处理结果
        """
        task_id = cognitive_feedback.get("task_id")
        if not task_id:
            return {"status": "error", "message": "缺少任务ID"}
            
        # 查找对应的任务记录
        task_record = None
        for record in self.learning_history:
            if record.get("task_id") == task_id:
                task_record = record
                break
                
        if not task_record:
            return {"status": "error", "message": f"未找到ID为{task_id}的任务记录"}
            
        # 提取认知反馈信息
        reasoning_confidence = cognitive_feedback.get("confidence", 0.5)
        biases = cognitive_feedback.get("detected_biases", [])
        quality_score = cognitive_feedback.get("quality_score", 0.5)
        
        # 根据认知反馈调整学习参数
        if "hyperparameters" in task_record:
            params = task_record["hyperparameters"]
            
            # 如果存在认知偏差，增加正则化
            if biases:
                params["regularization"] = params.get("regularization", 0.001) * 1.5
                params["dropout"] = min(0.5, params.get("dropout", 0.2) + 0.1)
                
            # 如果推理质量低，调整批处理大小
            if quality_score < 0.6:
                params["batch_size"] = max(16, params.get("batch_size", 32) // 2)
                
            # 如果置信度低，增加训练轮次
            if reasoning_confidence < 0.7:
                params["epochs"] = params.get("epochs", 5) + 2
                
            # 保存调整后的参数
            task_record["hyperparameters"] = params
            task_record["cognitive_adjustment"] = {
                "timestamp": time.time(),
                "reasoning_confidence": reasoning_confidence,
                "biases": biases,
                "quality_score": quality_score
            }
            
            return {
                "status": "success",
                "task_id": task_id,
                "message": "已根据认知反馈调整学习参数",
                "adjusted_parameters": params
            }
        else:
            return {"status": "error", "message": "任务记录中没有超参数信息"}
    
    def process_learning_system_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理学习系统的反馈，更新内部状态
        
        Args:
            feedback: 学习系统反馈
            
        Returns:
            Dict: 处理结果
        """
        feedback_type = feedback.get("type")
        task_id = feedback.get("task_id")
        
        if not feedback_type or not task_id:
            return {"status": "error", "message": "缺少反馈类型或任务ID"}
            
        # 处理不同类型的反馈
        if feedback_type == "task_started":
            # 学习任务开始
            return self._handle_task_started(feedback)
            
        elif feedback_type == "intermediate_result":
            # 中间结果反馈
            return self._handle_intermediate_result(feedback)
            
        elif feedback_type == "task_completed":
            # 学习任务完成
            return self._handle_task_completed(feedback)
            
        elif feedback_type == "error":
            # 错误反馈
            return self._handle_error_feedback(feedback)
            
        else:
            return {"status": "error", "message": f"未知的反馈类型: {feedback_type}"}
    
    def _handle_task_started(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务开始反馈"""
        task_id = feedback.get("task_id")
        task_type = feedback.get("task_type", "unknown")
        task_config = feedback.get("configuration", {})
        
        # 查找或创建任务记录
        task_record = None
        for record in self.learning_history:
            if record.get("task_id") == task_id:
                task_record = record
                break
                
        if not task_record:
            # 创建新记录
            task_record = {
                "task_id": task_id,
                "task_type": task_type,
                "status": "in_progress",
                "start_time": time.time(),
                "configuration": task_config
            }
            self.learning_history.append(task_record)
            
        else:
            # 更新记录
            task_record["status"] = "in_progress"
            task_record["start_time"] = time.time()
            task_record["configuration"] = task_config
            
        return {
            "status": "success",
            "task_id": task_id,
            "message": "已记录任务开始"
        }
    
    def _handle_intermediate_result(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理中间结果反馈"""
        task_id = feedback.get("task_id")
        progress = feedback.get("progress", 0)
        metrics = feedback.get("metrics", {})
        
        # 查找任务记录
        task_record = None
        for record in self.learning_history:
            if record.get("task_id") == task_id:
                task_record = record
                break
                
        if not task_record:
            return {"status": "error", "message": f"未找到ID为{task_id}的任务记录"}
            
        # 更新任务进度和指标
        if "intermediate_results" not in task_record:
            task_record["intermediate_results"] = []
            
        task_record["intermediate_results"].append({
            "timestamp": time.time(),
            "progress": progress,
            "metrics": metrics
        })
        
        # 如果学习过程不顺利，尝试调整参数
        if metrics.get("loss", 0) > 1.0 and len(task_record["intermediate_results"]) >= 3:
            if "hyperparameters" in task_record:
                # 获取当前超参数
                params = task_record["hyperparameters"]
                
                # 检查是否需要调整
                prev_metrics = task_record["intermediate_results"][-2]["metrics"]
                if metrics.get("loss", 0) > prev_metrics.get("loss", 0) * 1.1:
                    # 损失增加，调整学习率
                    params["learning_rate"] = params.get("learning_rate", 0.01) * 0.5
                    
                    # 记录调整
                    if "parameter_adjustments" not in task_record:
                        task_record["parameter_adjustments"] = []
                        
                    task_record["parameter_adjustments"].append({
                        "timestamp": time.time(),
                        "parameter": "learning_rate",
                        "old_value": params.get("learning_rate", 0.01) * 2,
                        "new_value": params.get("learning_rate", 0.01),
                        "reason": "损失函数上升"
                    })
                    
                    return {
                        "status": "adjusted",
                        "task_id": task_id,
                        "message": "已根据中间结果调整学习参数",
                        "adjusted_parameters": {"learning_rate": params.get("learning_rate", 0.01)}
                    }
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "已记录中间结果"
        }
    
    def _handle_task_completed(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务完成反馈"""
        task_id = feedback.get("task_id")
        results = feedback.get("results", {})
        
        # 查找任务记录
        task_record = None
        for record in self.learning_history:
            if record.get("task_id") == task_id:
                task_record = record
                break
                
        if not task_record:
            return {"status": "error", "message": f"未找到ID为{task_id}的任务记录"}
            
        # 更新任务状态和结果
        task_record["status"] = "completed"
        task_record["end_time"] = time.time()
        task_record["results"] = results
        
        # 计算任务时长
        if "start_time" in task_record:
            duration = task_record["end_time"] - task_record["start_time"]
            task_record["duration"] = duration
            
        # 评估学习效果
        evaluation = self.evaluate_learning_effectiveness(task_id, results)
        
        # 更新任务类型的最佳配置
        task_type = task_record["task_type"]
        effectiveness = evaluation.get("effectiveness", 0)
        
        if effectiveness > 0.7:  # 只在效果较好时更新配置
            if task_type not in self.task_type_configs or \
               self.task_type_configs[task_type].get("effectiveness", 0) < effectiveness:
                self.task_type_configs[task_type] = {
                    "hyperparameters": task_record.get("hyperparameters", {}),
                    "effectiveness": effectiveness,
                    "task_id": task_id,
                    "update_time": time.time()
                }
        
        return {
            "status": "success",
            "task_id": task_id,
            "evaluation": evaluation,
            "message": "已记录任务完成并评估效果"
        }
    
    def _handle_error_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误反馈"""
        task_id = feedback.get("task_id")
        error_type = feedback.get("error_type", "unknown")
        error_message = feedback.get("error_message", "")
        
        # 查找任务记录
        task_record = None
        for record in self.learning_history:
            if record.get("task_id") == task_id:
                task_record = record
                break
                
        if not task_record:
            return {"status": "error", "message": f"未找到ID为{task_id}的任务记录"}
            
        # 更新任务状态
        task_record["status"] = "failed"
        task_record["error"] = {
            "type": error_type,
            "message": error_message,
            "timestamp": time.time()
        }
        
        # 根据错误类型提供建议
        suggestion = "无具体建议"
        
        if "memory" in error_type.lower():
            suggestion = "减小批量大小或模型复杂度以减少内存占用"
            
        elif "convergence" in error_type.lower():
            suggestion = "调整学习率或增加正则化以改善收敛性"
            
        elif "data" in error_type.lower():
            suggestion = "检查数据质量，确保格式正确且无缺失值"
            
        task_record["recovery_suggestion"] = suggestion
        
        return {
            "status": "acknowledged",
            "task_id": task_id,
            "message": "已记录错误信息",
            "suggestion": suggestion
        }
    
    def export_learning_knowledge(self) -> Dict[str, Any]:
        """
        导出学习知识库，便于其他系统组件使用
        
        Returns:
            Dict: 学习知识库
        """
        # 导出策略性能统计
        strategy_stats = {}
        for strategy, performances in self.strategy_performance.items():
            if not performances:
                continue
                
            scores = [p["effectiveness"] for p in performances]
            strategy_stats[strategy] = {
                "average_effectiveness": sum(scores) / len(scores) if scores else 0,
                "sample_count": len(scores),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            }
            
        # 导出任务类型配置
        type_configs = {}
        for task_type, config in self.task_type_configs.items():
            type_configs[task_type] = {
                "effectiveness": config.get("effectiveness", 0),
                "hyperparameters": config.get("hyperparameters", {}),
                "update_time": config.get("update_time", 0)
            }
            
        # 导出学习趋势
        learning_trends = self._get_learning_trends()
        
        # 导出最佳实践
        best_practices = self._extract_best_practices()
        
        return {
            "timestamp": time.time(),
            "strategy_performance": strategy_stats,
            "task_type_configurations": type_configs,
            "learning_trends": learning_trends,
            "best_practices": best_practices,
            "hyperparameter_importance": self._analyze_hyperparameter_impact()
        }
    
    def _get_learning_trends(self) -> Dict[str, Any]:
        """获取学习趋势"""
        if len(self.learning_history) < 3:
            return {"status": "insufficient_data"}
            
        # 只分析已完成的任务
        completed_tasks = [task for task in self.learning_history 
                           if task.get("status") == "completed" and "results" in task]
        
        if not completed_tasks:
            return {"status": "no_completed_tasks"}
            
        # 按时间排序
        sorted_tasks = sorted(completed_tasks, key=lambda x: x.get("timestamp", 0))
        
        # 提取时间和效果数据
        timestamps = [task.get("timestamp", 0) for task in sorted_tasks]
        effectiveness = [task["results"].get("effectiveness", 0) for task in sorted_tasks]
        
        # 分析整体趋势
        overall_trend = "stable"
        if len(effectiveness) >= 5:
            first_part = effectiveness[:len(effectiveness)//2]
            second_part = effectiveness[len(effectiveness)//2:]
            
            first_avg = sum(first_part) / len(first_part)
            second_avg = sum(second_part) / len(second_part)
            
            if second_avg > first_avg * 1.1:
                overall_trend = "improving"
            elif second_avg < first_avg * 0.9:
                overall_trend = "declining"
        
        # 最近趋势
        recent_tasks = sorted_tasks[-5:] if len(sorted_tasks) >= 5 else sorted_tasks
        recent_effectiveness = [task["results"].get("effectiveness", 0) for task in recent_tasks]
        recent_avg = sum(recent_effectiveness) / len(recent_effectiveness) if recent_effectiveness else 0
        
        return {
            "status": "success",
            "overall_trend": overall_trend,
            "total_tasks": len(completed_tasks),
            "recent_average": recent_avg,
            "first_task_time": timestamps[0] if timestamps else 0,
            "last_task_time": timestamps[-1] if timestamps else 0
        }
    
    def _extract_best_practices(self) -> List[Dict[str, Any]]:
        """提取学习最佳实践"""
        best_practices = []
        
        # 查找效果最好的任务
        top_tasks = []
        for record in self.learning_history:
            if "results" in record and "effectiveness" in record["results"]:
                top_tasks.append((record, record["results"]["effectiveness"]))
                
        # 按效果排序
        top_tasks.sort(key=lambda x: x[1], reverse=True)
        top_tasks = top_tasks[:3]  # 取前3个
        
        for task_record, effectiveness in top_tasks:
            if effectiveness < 0.7:
                continue  # 只考虑效果较好的任务
                
            practice = {
                "task_type": task_record.get("task_type", "unknown"),
                "effectiveness": effectiveness,
                "strategy": task_record.get("strategy", "unknown"),
                "hyperparameters": task_record.get("hyperparameters", {})
            }
            
            best_practices.append(practice)
            
        # 添加通用最佳实践
        if self.learning_history:
            # 寻找常见的超参数模式
            common_params = self._find_common_hyperparameters()
            if common_params:
                best_practices.append({
                    "task_type": "general",
                    "description": "常见有效超参数组合",
                    "hyperparameters": common_params
                })
                
        return best_practices
    
    def _find_common_hyperparameters(self) -> Dict[str, Any]:
        """寻找常见的超参数模式"""
        # 收集效果较好的任务的超参数
        good_params = []
        for record in self.learning_history:
            if "results" in record and record["results"].get("effectiveness", 0) >= 0.7:
                if "hyperparameters" in record:
                    good_params.append(record["hyperparameters"])
                    
        if not good_params:
            return {}
            
        # 寻找共同的参数
        common_params = {}
        param_counts = defaultdict(lambda: defaultdict(int))
        
        for params in good_params:
            for param, value in params.items():
                param_counts[param][value] += 1
                
        # 提取出现频率最高的参数值
        for param, value_counts in param_counts.items():
            if not value_counts:
                continue
                
            # 找出现次数最多的值
            max_count = 0
            max_value = None
            
            for value, count in value_counts.items():
                if count > max_count:
                    max_count = count
                    max_value = value
                    
            # 只保留至少出现在一半以上任务中的参数
            if max_count >= len(good_params) * 0.5:
                common_params[param] = max_value
                
        return common_params 