"""
GHOST-AGI 自我进化核心模块

该模块是系统自我进化能力的核心实现，使系统能够自主优化和改进自身的各个组件。
"""

import logging
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict

class SelfEvolution:
    """自我进化核心模块，提供系统自我优化和改进能力"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化自我进化模块
        
        Args:
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 记录系统组件
        self.system_components = {}
        
        # 记录每个组件的性能指标
        self.component_performance = {}
        
        # 进化历史
        self.evolution_history = []
        
        # 演化策略和优化器
        self.evolution_strategies = {
            "parameter_tuning": self._parameter_optimization,
            "structure_evolution": self._structure_evolution,
            "algorithm_selection": self._algorithm_selection,
            "module_integration": self._module_integration
        }
        
        # 初始化进化统计
        self.evolution_stats = {
            "iterations": 0,
            "improvements": 0,
            "last_evolution_time": None,
            "cumulative_improvement": 0.0
        }
        
        self.logger.info("自我进化核心模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("SelfEvolution")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("self_evolution.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def register_component(self, component_id: str, component: Any, metadata: Dict[str, Any]) -> None:
        """
        注册系统组件
        
        Args:
            component_id: 组件ID
            component: 组件对象
            metadata: 组件元数据
        """
        self.logger.info(f"注册组件: {component_id}")
        
        # 确保元数据包含必要信息
        if "type" not in metadata:
            metadata["type"] = "generic"
        
        metadata["registration_time"] = time.time()
        
        # 存储组件
        self.system_components[component_id] = {
            "component": component,
            "metadata": metadata,
            "version": 1.0
        }
        
        # 初始化性能指标
        self.component_performance[component_id] = {
            "metrics": {},
            "history": [],
            "last_evaluation": None
        }
    
    def update_component_metrics(self, component_id: str, metrics: Dict[str, float]) -> None:
        """
        更新组件性能指标
        
        Args:
            component_id: 组件ID
            metrics: 性能指标
        """
        if component_id not in self.component_performance:
            self.logger.warning(f"组件 {component_id} 未注册，无法更新指标")
            return
        
        current_time = time.time()
        
        # 更新当前指标
        self.component_performance[component_id]["metrics"] = metrics
        self.component_performance[component_id]["last_evaluation"] = current_time
        
        # 记录历史
        self.component_performance[component_id]["history"].append({
            "metrics": metrics.copy(),
            "timestamp": current_time
        })
        
        self.logger.debug(f"已更新组件 {component_id} 的性能指标")
    
    def evolve_system(self, target_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        对系统进行进化优化
        
        Args:
            target_components: 要优化的目标组件列表，如果为None则优化所有组件
            
        Returns:
            进化结果
        """
        start_time = time.time()
        self.logger.info("开始系统进化迭代")
        
        # 确定目标组件
        if target_components is None:
            target_components = list(self.system_components.keys())
        
        # 过滤掉未注册的组件
        target_components = [comp_id for comp_id in target_components if comp_id in self.system_components]
        
        if not target_components:
            self.logger.warning("没有有效的目标组件，终止进化过程")
            return {
                "status": "failed",
                "reason": "no_valid_components",
                "elapsed_time": time.time() - start_time
            }
        
        self.logger.info(f"将对 {len(target_components)} 个组件进行进化")
        
        # 优化结果
        evolution_results = {}
        overall_improvement = 0.0
        improvements_count = 0
        
        # 对每个组件应用进化策略
        for component_id in target_components:
            component_info = self.system_components[component_id]
            component_type = component_info["metadata"]["type"]
            
            # 选择适当的进化策略
            strategy = self._select_evolution_strategy(component_id, component_type)
            
            if strategy:
                # 应用进化策略
                try:
                    self.logger.info(f"对组件 {component_id} 应用进化策略: {strategy}")
                    result = self.evolution_strategies[strategy](component_id)
                    evolution_results[component_id] = result
                    
                    # 统计改进
                    if result["improved"]:
                        improvements_count += 1
                        overall_improvement += result["improvement_score"]
                except Exception as e:
                    self.logger.error(f"对组件 {component_id} 应用进化策略时出错: {str(e)}")
                    evolution_results[component_id] = {
                        "status": "error",
                        "error": str(e),
                        "improved": False
                    }
            else:
                self.logger.warning(f"未找到适合组件 {component_id} 的进化策略")
                evolution_results[component_id] = {
                    "status": "skipped",
                    "reason": "no_suitable_strategy",
                    "improved": False
                }
        
        # 更新统计信息
        self.evolution_stats["iterations"] += 1
        self.evolution_stats["improvements"] += improvements_count
        self.evolution_stats["last_evolution_time"] = time.time()
        self.evolution_stats["cumulative_improvement"] += overall_improvement
        
        # 记录进化历史
        evolution_record = {
            "timestamp": time.time(),
            "target_components": target_components,
            "results": evolution_results,
            "overall_improvement": overall_improvement,
            "improvements_count": improvements_count
        }
        self.evolution_history.append(evolution_record)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"系统进化迭代完成，耗时: {elapsed_time:.2f}秒，改进 {improvements_count} 个组件")
        
        return {
            "status": "success",
            "target_components": target_components,
            "results": evolution_results,
            "overall_improvement": overall_improvement,
            "improvements_count": improvements_count,
            "elapsed_time": elapsed_time,
            "stats": self.evolution_stats
        }
    
    def _select_evolution_strategy(self, component_id: str, component_type: str) -> Optional[str]:
        """选择合适的进化策略"""
        # 获取组件的性能指标
        performance = self.component_performance.get(component_id, {})
        metrics = performance.get("metrics", {})
        history = performance.get("history", [])
        
        # 检查是否有足够的性能历史数据
        if len(history) < 2:
            # 对于新组件，优先使用参数调优
            return "parameter_tuning"
        
        # 分析性能趋势
        recent_metrics = [entry["metrics"] for entry in history[-3:]]
        improvement_trend = self._analyze_performance_trend(recent_metrics)
        
        # 根据组件类型和性能趋势选择策略
        if component_type == "model":
            if improvement_trend < 0:
                # 性能下降，尝试算法选择
                return "algorithm_selection"
            elif improvement_trend > 0 and improvement_trend < 0.1:
                # 性能略有提升，继续参数调优
                return "parameter_tuning"
            else:
                # 性能提升明显，尝试结构进化
                return "structure_evolution"
        elif component_type == "processor":
            if improvement_trend < 0:
                # 性能下降，尝试结构进化
                return "structure_evolution"
            else:
                # 性能提升，继续参数调优
                return "parameter_tuning"
        elif component_type == "integrator":
            # 集成组件主要使用模块集成策略
            return "module_integration"
        else:
            # 默认使用参数调优
            return "parameter_tuning"
    
    def _analyze_performance_trend(self, metrics_history: List[Dict[str, float]]) -> float:
        """分析性能趋势"""
        if not metrics_history or len(metrics_history) < 2:
            return 0.0
        
        # 选择主要指标
        main_metrics = self._select_main_metrics(metrics_history[0])
        
        # 计算趋势
        trend_sum = 0.0
        count = 0
        
        for metric_name in main_metrics:
            # 检查该指标是否在所有历史记录中都存在
            if not all(metric_name in metrics for metrics in metrics_history):
                continue
            
            # 提取该指标的历史值
            values = [metrics[metric_name] for metrics in metrics_history]
            
            # 计算简单的线性趋势
            if len(values) >= 2:
                trend = (values[-1] - values[0]) / (len(values) - 1)
                
                # 归一化趋势
                max_val = max(values)
                if max_val > 0:
                    normalized_trend = trend / max_val
                else:
                    normalized_trend = 0.0
                
                trend_sum += normalized_trend
                count += 1
        
        # 平均趋势
        return trend_sum / count if count > 0 else 0.0
    
    def _select_main_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """选择主要性能指标"""
        # 优先选择常见的性能指标
        priority_metrics = ["accuracy", "precision", "recall", "f1_score", "efficiency", "latency", "throughput"]
        
        selected = [m for m in priority_metrics if m in metrics]
        
        # 如果没有常见指标，则选择所有可用指标
        if not selected:
            selected = list(metrics.keys())
        
        return selected
    
    def _parameter_optimization(self, component_id: str) -> Dict[str, Any]:
        """参数优化策略"""
        self.logger.info(f"对组件 {component_id} 进行参数优化")
        
        component_info = self.system_components[component_id]
        component = component_info["component"]
        current_version = component_info["version"]
        
        # 检查组件是否支持参数优化
        if not hasattr(component, "get_parameters") or not hasattr(component, "set_parameters"):
            self.logger.warning(f"组件 {component_id} 不支持参数优化")
            return {
                "status": "skipped",
                "reason": "component_not_support_parameter_optimization",
                "improved": False
            }
        
        # 获取当前参数
        try:
            current_params = component.get_parameters()
        except Exception as e:
            self.logger.error(f"获取组件 {component_id} 当前参数失败: {str(e)}")
            return {
                "status": "error",
                "error": f"获取参数失败: {str(e)}",
                "improved": False
            }
        
        # 获取当前性能指标
        current_metrics = self.component_performance[component_id]["metrics"]
        if not current_metrics:
            self.logger.warning(f"组件 {component_id} 缺少性能指标，无法进行参数优化")
            return {
                "status": "skipped",
                "reason": "missing_performance_metrics",
                "improved": False
            }
        
        # 选择要优化的参数
        tunable_params = self._select_tunable_parameters(current_params)
        if not tunable_params:
            self.logger.warning(f"组件 {component_id} 没有可调参数")
            return {
                "status": "skipped",
                "reason": "no_tunable_parameters",
                "improved": False
            }
        
        # 评估当前性能基准
        baseline_score = self._calculate_overall_score(current_metrics)
        
        # 使用简单的随机搜索进行参数优化 (实际系统中会使用更复杂的方法)
        best_params = current_params.copy()
        best_score = baseline_score
        
        # 简单的随机搜索
        max_trials = 10
        for trial in range(max_trials):
            # 生成候选参数
            candidate_params = self._generate_candidate_parameters(current_params, tunable_params)
            
            # 应用候选参数
            try:
                component.set_parameters(candidate_params)
            except Exception as e:
                self.logger.warning(f"设置组件 {component_id} 参数失败: {str(e)}")
                continue
            
            # 模拟评估 (在实际系统中，这应该是真实的评估)
            try:
                # 实际系统中应该调用组件的评估方法
                evaluation_result = self._simulate_component_evaluation(component_id, component, candidate_params)
                candidate_score = self._calculate_overall_score(evaluation_result)
                
                # 比较性能
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params.copy()
            except Exception as e:
                self.logger.warning(f"评估组件 {component_id} 候选参数失败: {str(e)}")
                continue
        
        # 判断是否有改进
        improvement = best_score - baseline_score
        improved = improvement > 0
        
        # 如果有改进，应用最佳参数
        if improved:
            try:
                component.set_parameters(best_params)
                # 更新组件版本
                self.system_components[component_id]["version"] = current_version + 0.01
                
                # 更新性能指标 (实际系统中应该进行实际评估)
                simulated_metrics = self._simulate_component_evaluation(component_id, component, best_params)
                self.update_component_metrics(component_id, simulated_metrics)
                
                self.logger.info(f"组件 {component_id} 参数优化成功，性能提升: {improvement:.4f}")
            except Exception as e:
                self.logger.error(f"应用最佳参数到组件 {component_id} 失败: {str(e)}")
                improved = False
                improvement = 0.0
        else:
            # 恢复原始参数
            try:
                component.set_parameters(current_params)
            except Exception as e:
                self.logger.error(f"恢复组件 {component_id} 原始参数失败: {str(e)}")
        
        return {
            "status": "success" if improved else "no_improvement",
            "improved": improved,
            "improvement_score": improvement,
            "parameter_changes": self._summarize_parameter_changes(current_params, best_params) if improved else [],
            "original_score": baseline_score,
            "new_score": best_score if improved else baseline_score
        }
    
    def _select_tunable_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """选择可调参数"""
        tunable = {}
        
        for param_name, param_value in parameters.items():
            # 忽略不可调参数 (如ID、名称等)
            if param_name in ["id", "name", "type", "version"]:
                continue
            
            # 根据参数类型设置调优范围
            if isinstance(param_value, (int, float)):
                # 数值参数
                tunable[param_name] = {
                    "type": "numeric",
                    "current": param_value,
                    "min": param_value * 0.5,
                    "max": param_value * 1.5
                }
            elif isinstance(param_value, bool):
                # 布尔参数
                tunable[param_name] = {
                    "type": "boolean",
                    "current": param_value
                }
            elif isinstance(param_value, str) and param_name.endswith(("_method", "_algorithm", "_strategy")):
                # 算法/策略选择参数
                tunable[param_name] = {
                    "type": "categorical",
                    "current": param_value,
                    "options": [param_value]  # 实际系统中，这应该包含更多选项
                }
        
        return tunable
    
    def _generate_candidate_parameters(self, current_params: Dict[str, Any], tunable_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成候选参数"""
        candidate = current_params.copy()
        
        # 随机选择要调整的参数数量
        num_params_to_tune = random.randint(1, max(1, len(tunable_params) // 2))
        params_to_tune = random.sample(list(tunable_params.keys()), min(num_params_to_tune, len(tunable_params)))
        
        for param_name in params_to_tune:
            param_info = tunable_params[param_name]
            
            if param_info["type"] == "numeric":
                # 数值参数：在范围内随机选择
                candidate[param_name] = random.uniform(param_info["min"], param_info["max"])
                
                # 如果原始参数是整数，保持整数类型
                if isinstance(current_params[param_name], int):
                    candidate[param_name] = int(candidate[param_name])
            
            elif param_info["type"] == "boolean":
                # 布尔参数：反转
                candidate[param_name] = not param_info["current"]
            
            elif param_info["type"] == "categorical":
                # 分类参数：从选项中随机选择
                if len(param_info["options"]) > 1:
                    options = [opt for opt in param_info["options"] if opt != param_info["current"]]
                    if options:
                        candidate[param_name] = random.choice(options)
        
        return candidate
    
    def _simulate_component_evaluation(self, component_id: str, component: Any, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        模拟组件评估
        注意：实际系统中，这应该调用组件的实际评估方法
        """
        # 获取当前指标作为基准
        current_metrics = self.component_performance[component_id]["metrics"].copy()
        
        # 模拟随机波动
        simulated_metrics = {}
        for metric_name, metric_value in current_metrics.items():
            # 添加-5%到+10%的随机波动
            random_factor = 1.0 + random.uniform(-0.05, 0.1)
            simulated_metrics[metric_name] = metric_value * random_factor
        
        return simulated_metrics
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算整体性能分数"""
        if not metrics:
            return 0.0
        
        # 选择要考虑的指标
        considered_metrics = self._select_main_metrics(metrics)
        
        # 计算加权平均分
        weights = {}
        for metric in considered_metrics:
            # 这里可以为不同指标分配不同权重
            if metric in ["accuracy", "precision", "recall", "f1_score"]:
                weights[metric] = 1.0
            elif metric in ["efficiency", "throughput"]:
                weights[metric] = 0.8
            elif metric in ["latency", "error_rate"]:
                # 对于这些指标，值越小越好，需要反转
                weights[metric] = -0.8
            else:
                weights[metric] = 0.5
        
        # 计算分数
        total_weight = sum(abs(w) for w in weights.values())
        weighted_sum = sum(metrics[m] * weights[m] for m in considered_metrics if m in metrics)
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _summarize_parameter_changes(self, original_params: Dict[str, Any], new_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """总结参数变更"""
        changes = []
        
        for param_name, original_value in original_params.items():
            if param_name in new_params and new_params[param_name] != original_value:
                change = {
                    "parameter": param_name,
                    "original": original_value,
                    "new": new_params[param_name]
                }
                
                # 计算变化百分比（如果适用）
                if isinstance(original_value, (int, float)) and isinstance(new_params[param_name], (int, float)) and original_value != 0:
                    change["percent_change"] = (new_params[param_name] - original_value) / original_value * 100
                
                changes.append(change)
        
        return changes
    
    def _structure_evolution(self, component_id: str) -> Dict[str, Any]:
        """结构进化策略"""
        self.logger.info(f"对组件 {component_id} 进行结构进化")
        
        # 简化实现：报告未实现
        # 在实际系统中，这应该实现结构变异、重组等操作
        return {
            "status": "not_implemented",
            "improved": False,
            "message": "结构进化策略尚未完全实现"
        }
    
    def _algorithm_selection(self, component_id: str) -> Dict[str, Any]:
        """算法选择策略"""
        self.logger.info(f"对组件 {component_id} 进行算法选择")
        
        # 简化实现：报告未实现
        # 在实际系统中，这应该实现算法库搜索和选择
        return {
            "status": "not_implemented",
            "improved": False,
            "message": "算法选择策略尚未完全实现"
        }
    
    def _module_integration(self, component_id: str) -> Dict[str, Any]:
        """模块集成策略"""
        self.logger.info(f"对组件 {component_id} 进行模块集成")
        
        # 简化实现：报告未实现
        # 在实际系统中，这应该实现模块组合和整合
        return {
            "status": "not_implemented",
            "improved": False,
            "message": "模块集成策略尚未完全实现"
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """
        获取进化状态
        
        Returns:
            状态信息
        """
        recent_improvements = []
        if self.evolution_history:
            recent_improvements = self.evolution_history[-min(5, len(self.evolution_history)):]
        
        return {
            "registered_components": len(self.system_components),
            "evolution_iterations": self.evolution_stats["iterations"],
            "total_improvements": self.evolution_stats["improvements"],
            "last_evolution_time": self.evolution_stats["last_evolution_time"],
            "cumulative_improvement": self.evolution_stats["cumulative_improvement"],
            "recent_improvements": recent_improvements
        } 