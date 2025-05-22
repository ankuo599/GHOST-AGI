"""
元认知监督模块 (Metacognition Supervisor)

提供系统的自评估和自监督能力，监控系统行为，评估性能，进行错误修正。
实现"思考自己的思考"的元认知能力，提高系统的可靠性和自主性。
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
import threading
from collections import defaultdict, deque
import json
import numpy as np
import os

class MetaSupervisor:
    """元认知监督模块，提供系统自评估和自监督能力"""
    
    def __init__(self, logger=None):
        """
        初始化元认知监督模块
        
        Args:
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 系统模块评估
        self.module_evaluations = {}  # {module_id: evaluation_metrics}
        
        # 决策记录
        self.decision_history = deque(maxlen=1000)  # 最多保存1000条决策记录
        
        # 错误记录
        self.error_records = []
        
        # 性能指标
        self.performance_metrics = {
            "accuracy": [],
            "consistency": [],
            "efficiency": [],
            "robustness": [],
            "adaptability": []
        }
        
        # 监控状态
        self.monitoring_state = {
            "is_active": False,
            "start_time": None,
            "monitored_modules": set()
        }
        
        # 自评估结果
        self.self_assessment = {
            "last_assessment_time": None,
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": []
        }
        
        # 元认知策略
        self.metacognitive_strategies = {}  # {strategy_id: strategy_info}
        
        # 系统组件引用
        self.component_references = {}  # {component_id: component_reference}
        
        # 监控线程
        self.monitor_thread = None
        
        # 配置
        self.config = {
            "monitoring_interval": 60,  # 监控间隔（秒）
            "assessment_interval": 3600,  # 自评估间隔（秒）
            "error_threshold": 0.1,  # 错误率阈值
            "confidence_threshold": 0.7,  # 置信度阈值
            "enable_auto_correction": True  # 启用自动修正
        }
        
        # 初始化默认策略
        self._initialize_default_strategies()
        
        self.logger.info("元认知监督模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("MetaSupervisor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("meta_supervisor.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_default_strategies(self):
        """初始化默认元认知策略"""
        self.metacognitive_strategies = {
            "contradiction_detection": {
                "id": "contradiction_detection",
                "name": "矛盾检测",
                "description": "检测系统内部逻辑矛盾",
                "activation_threshold": 0.6,
                "handler": self._handle_contradiction,
                "enabled": True
            },
            "uncertainty_monitoring": {
                "id": "uncertainty_monitoring",
                "name": "不确定性监控",
                "description": "监控系统的不确定性和置信度",
                "activation_threshold": 0.5,
                "handler": self._handle_uncertainty,
                "enabled": True
            },
            "error_correction": {
                "id": "error_correction",
                "name": "错误修正",
                "description": "检测和修正系统错误",
                "activation_threshold": 0.7,
                "handler": self._handle_error,
                "enabled": True
            },
            "reflection": {
                "id": "reflection",
                "name": "反思",
                "description": "定期反思系统决策过程",
                "activation_threshold": 0.4,
                "handler": self._handle_reflection,
                "enabled": True
            },
            "efficiency_optimization": {
                "id": "efficiency_optimization",
                "name": "效率优化",
                "description": "优化系统运行效率",
                "activation_threshold": 0.6,
                "handler": self._handle_efficiency,
                "enabled": True
            }
        }
    
    def initialize(self, central_coordinator=None, reasoning_engine=None, memory_system=None) -> Dict[str, Any]:
        """
        初始化元认知监督模块
        
        Args:
            central_coordinator: 中央协调引擎
            reasoning_engine: 推理引擎
            memory_system: 记忆系统
            
        Returns:
            Dict: 初始化结果
        """
        # 保存系统组件引用
        if central_coordinator:
            self.component_references["central_coordinator"] = central_coordinator
            
        if reasoning_engine:
            self.component_references["reasoning_engine"] = reasoning_engine
            
        if memory_system:
            self.component_references["memory_system"] = memory_system
        
        # 加载配置
        self._load_config()
        
        # 加载历史评估
        self._load_history()
        
        self.logger.info("元认知监督模块已初始化")
        
        return {
            "status": "success",
            "message": "元认知监督模块已初始化",
            "registered_components": list(self.component_references.keys())
        }
    
    def start_monitoring(self) -> Dict[str, Any]:
        """
        启动元认知监控
        
        Returns:
            Dict: 启动结果
        """
        if self.monitoring_state["is_active"]:
            return {
                "status": "warning",
                "message": "监控已在运行"
            }
            
        # 更新状态
        self.monitoring_state["is_active"] = True
        self.monitoring_state["start_time"] = time.time()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("元认知监控已启动")
        
        return {
            "status": "success",
            "message": "元认知监控已启动",
            "start_time": self.monitoring_state["start_time"]
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        停止元认知监控
        
        Returns:
            Dict: 停止结果
        """
        if not self.monitoring_state["is_active"]:
            return {
                "status": "warning",
                "message": "监控未在运行"
            }
            
        # 更新状态
        self.monitoring_state["is_active"] = False
        
        # 等待线程结束
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        self.logger.info("元认知监控已停止")
        
        return {
            "status": "success",
            "message": "元认知监控已停止",
            "duration": time.time() - self.monitoring_state["start_time"]
        }
    
    def register_module(self, module_id: str, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        注册模块以进行监控
        
        Args:
            module_id: 模块ID
            module_info: 模块信息
            
        Returns:
            Dict: 注册结果
        """
        # 初始化模块评估
        self.module_evaluations[module_id] = {
            "id": module_id,
            "name": module_info.get("name", module_id),
            "last_evaluation": None,
            "performance_score": 0.0,
            "reliability_score": 0.0,
            "efficiency_score": 0.0,
            "error_rate": 0.0,
            "evaluation_history": [],
            "registered_at": time.time()
        }
        
        # 添加到监控列表
        self.monitoring_state["monitored_modules"].add(module_id)
        
        self.logger.info(f"已注册模块: {module_id}")
        
        return {
            "status": "success",
            "module_id": module_id,
            "message": "模块已注册到元认知监控"
        }
    
    def evaluate_module(self, module_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估模块性能
        
        Args:
            module_id: 模块ID
            metrics: 性能指标
            
        Returns:
            Dict: 评估结果
        """
        if module_id not in self.module_evaluations:
            return {
                "status": "error",
                "message": f"未注册的模块: {module_id}"
            }
            
        # 获取当前评估
        evaluation = self.module_evaluations[module_id]
        
        # 计算新的评分
        performance_score = metrics.get("performance_score", evaluation["performance_score"])
        reliability_score = metrics.get("reliability_score", evaluation["reliability_score"])
        efficiency_score = metrics.get("efficiency_score", evaluation["efficiency_score"])
        error_rate = metrics.get("error_rate", evaluation["error_rate"])
        
        # 更新评估
        evaluation["last_evaluation"] = time.time()
        evaluation["performance_score"] = performance_score
        evaluation["reliability_score"] = reliability_score
        evaluation["efficiency_score"] = efficiency_score
        evaluation["error_rate"] = error_rate
        
        # 添加到历史
        evaluation["evaluation_history"].append({
            "timestamp": time.time(),
            "performance_score": performance_score,
            "reliability_score": reliability_score,
            "efficiency_score": efficiency_score,
            "error_rate": error_rate
        })
        
        # 检查错误率
        if error_rate > self.config["error_threshold"]:
            self.logger.warning(f"模块 {module_id} 错误率高: {error_rate}")
            
            # 记录错误
            self._record_error(module_id, "high_error_rate", {
                "error_rate": error_rate,
                "threshold": self.config["error_threshold"]
            })
            
            # 如果启用了自动修正，尝试修正
            if self.config["enable_auto_correction"]:
                correction_result = self._attempt_correction(module_id, "high_error_rate")
                
                if correction_result["status"] == "success":
                    self.logger.info(f"已自动修正模块 {module_id} 的高错误率问题")
        
        self.logger.info(f"已评估模块: {module_id}, 性能分数: {performance_score}")
        
        return {
            "status": "success",
            "module_id": module_id,
            "evaluation": {
                "performance_score": performance_score,
                "reliability_score": reliability_score,
                "efficiency_score": efficiency_score,
                "error_rate": error_rate,
                "timestamp": time.time()
            }
        }
    
    def record_decision(self, decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        记录系统决策
        
        Args:
            decision_info: 决策信息
            
        Returns:
            Dict: 记录结果
        """
        # 生成决策ID
        decision_id = decision_info.get("id", str(uuid.uuid4()))
        
        # 添加时间戳
        if "timestamp" not in decision_info:
            decision_info["timestamp"] = time.time()
            
        # 添加决策ID
        decision_info["id"] = decision_id
        
        # 添加到历史
        self.decision_history.append(decision_info)
        
        # 应用元认知策略
        self._apply_metacognitive_strategies(decision_info)
        
        return {
            "status": "success",
            "decision_id": decision_id,
            "recorded_at": time.time()
        }
    
    def conduct_self_assessment(self) -> Dict[str, Any]:
        """
        进行系统自评估
        
        Returns:
            Dict: 自评估结果
        """
        # 记录评估时间
        assessment_time = time.time()
        
        # 获取模块评估数据
        module_scores = []
        
        for module_id, evaluation in self.module_evaluations.items():
            module_scores.append({
                "module_id": module_id,
                "performance_score": evaluation["performance_score"],
                "reliability_score": evaluation["reliability_score"],
                "efficiency_score": evaluation["efficiency_score"],
                "error_rate": evaluation["error_rate"]
            })
        
        # 计算整体得分
        overall_score = self._calculate_overall_score(module_scores)
        
        # 分析优势和劣势
        strengths, weaknesses = self._analyze_strengths_weaknesses(module_scores)
        
        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(weaknesses)
        
        # 更新自评估结果
        self.self_assessment = {
            "last_assessment_time": assessment_time,
            "overall_score": overall_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_suggestions": improvement_suggestions
        }
        
        # 保存评估结果
        self._save_assessment()
        
        self.logger.info(f"已完成系统自评估, 总分: {overall_score:.2f}")
        
        return {
            "status": "success",
            "assessment_time": assessment_time,
            "overall_score": overall_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_suggestions": improvement_suggestions
        }
    
    def get_module_evaluation(self, module_id: str) -> Dict[str, Any]:
        """
        获取模块评估
        
        Args:
            module_id: 模块ID
            
        Returns:
            Dict: 模块评估
        """
        if module_id not in self.module_evaluations:
            return {
                "status": "error",
                "message": f"未找到模块: {module_id}"
            }
            
        evaluation = self.module_evaluations[module_id]
        
        return {
            "status": "success",
            "module_id": module_id,
            "evaluation": {
                "performance_score": evaluation["performance_score"],
                "reliability_score": evaluation["reliability_score"],
                "efficiency_score": evaluation["efficiency_score"],
                "error_rate": evaluation["error_rate"],
                "last_evaluation": evaluation["last_evaluation"]
            },
            "history": evaluation["evaluation_history"][-10:] if evaluation["evaluation_history"] else []
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """
        获取系统性能指标
        
        Returns:
            Dict: 系统性能指标
        """
        # 计算平均指标
        avg_metrics = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
            else:
                avg_metrics[metric_name] = 0.0
                
        # 获取最新的自评估
        last_assessment = {
            "time": self.self_assessment["last_assessment_time"],
            "overall_score": self.self_assessment["overall_score"]
        }
        
        # 计算模块评估统计
        module_stats = {
            "count": len(self.module_evaluations),
            "avg_performance": sum(m["performance_score"] for m in self.module_evaluations.values()) / max(1, len(self.module_evaluations)),
            "avg_reliability": sum(m["reliability_score"] for m in self.module_evaluations.values()) / max(1, len(self.module_evaluations)),
            "avg_efficiency": sum(m["efficiency_score"] for m in self.module_evaluations.values()) / max(1, len(self.module_evaluations)),
            "avg_error_rate": sum(m["error_rate"] for m in self.module_evaluations.values()) / max(1, len(self.module_evaluations))
        }
        
        return {
            "status": "success",
            "metrics": avg_metrics,
            "module_stats": module_stats,
            "last_assessment": last_assessment,
            "error_count": len(self.error_records),
            "monitoring_active": self.monitoring_state["is_active"],
            "monitoring_duration": time.time() - self.monitoring_state["start_time"] if self.monitoring_state["is_active"] else 0
        }
    
    def analyze_decisions(self, time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        分析决策历史
        
        Args:
            time_range: 时间范围(开始时间, 结束时间)
            
        Returns:
            Dict: 分析结果
        """
        # 过滤决策
        if time_range:
            start_time, end_time = time_range
            decisions = [d for d in self.decision_history if start_time <= d["timestamp"] <= end_time]
        else:
            decisions = list(self.decision_history)
            
        if not decisions:
            return {
                "status": "warning",
                "message": "没有决策记录",
                "analysis": {}
            }
            
        # 分析决策质量
        quality_scores = [d.get("quality_score", 0) for d in decisions if "quality_score" in d]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # 分析置信度
        confidence_scores = [d.get("confidence", 0) for d in decisions if "confidence" in d]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # 分析决策类型
        decision_types = {}
        for d in decisions:
            decision_type = d.get("type", "unknown")
            decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
            
        # 决策时间分析
        decisions.sort(key=lambda x: x["timestamp"])
        decision_times = [d["timestamp"] for d in decisions]
        
        if len(decision_times) > 1:
            decision_intervals = [decision_times[i] - decision_times[i-1] for i in range(1, len(decision_times))]
            avg_interval = sum(decision_intervals) / len(decision_intervals)
        else:
            avg_interval = 0
            
        return {
            "status": "success",
            "analysis": {
                "total_decisions": len(decisions),
                "average_quality": avg_quality,
                "average_confidence": avg_confidence,
                "decision_types": decision_types,
                "average_interval": avg_interval,
                "time_period": {
                    "start": decisions[0]["timestamp"] if decisions else None,
                    "end": decisions[-1]["timestamp"] if decisions else None,
                    "duration": decisions[-1]["timestamp"] - decisions[0]["timestamp"] if len(decisions) > 1 else 0
                }
            }
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新配置
        
        Args:
            config_updates: 配置更新
            
        Returns:
            Dict: 更新后的配置
        """
        # 更新配置
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
                
        # 保存配置
        self._save_config()
        
        self.logger.info("已更新配置")
        
        return {
            "status": "success",
            "config": dict(self.config)
        }
    
    def enable_strategy(self, strategy_id: str, enabled: bool = True) -> Dict[str, Any]:
        """
        启用或禁用元认知策略
        
        Args:
            strategy_id: 策略ID
            enabled: 是否启用
            
        Returns:
            Dict: 结果
        """
        if strategy_id not in self.metacognitive_strategies:
            return {
                "status": "error",
                "message": f"未找到策略: {strategy_id}"
            }
            
        self.metacognitive_strategies[strategy_id]["enabled"] = enabled
        
        status = "启用" if enabled else "禁用"
        self.logger.info(f"已{status}策略: {strategy_id}")
        
        return {
            "status": "success",
            "strategy_id": strategy_id,
            "enabled": enabled
        }
    
    def _monitoring_loop(self):
        """元认知监控循环"""
        last_assessment_time = time.time()
        
        while self.monitoring_state["is_active"]:
            try:
                # 检查所有已注册模块
                for module_id in list(self.monitoring_state["monitored_modules"]):
                    self._check_module_health(module_id)
                    
                # 定期进行自评估
                current_time = time.time()
                if current_time - last_assessment_time > self.config["assessment_interval"]:
                    self.conduct_self_assessment()
                    last_assessment_time = current_time
                    
                # 休眠一段时间
                time.sleep(self.config["monitoring_interval"])
            except Exception as e:
                self.logger.error(f"监控循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠
    
    def _check_module_health(self, module_id: str):
        """检查模块健康状态"""
        if module_id not in self.module_evaluations:
            return
            
        evaluation = self.module_evaluations[module_id]
        
        # 检查上次评估时间
        if evaluation["last_evaluation"] is None:
            # 未评估过，跳过
            return
            
        # 检查错误率
        if evaluation["error_rate"] > self.config["error_threshold"]:
            self.logger.warning(f"模块 {module_id} 持续高错误率: {evaluation['error_rate']}")
            
            # 记录错误
            self._record_error(module_id, "persistent_high_error_rate", {
                "error_rate": evaluation["error_rate"],
                "threshold": self.config["error_threshold"]
            })
    
    def _record_error(self, module_id: str, error_type: str, details: Dict[str, Any]):
        """记录错误"""
        error_record = {
            "id": str(uuid.uuid4()),
            "module_id": module_id,
            "error_type": error_type,
            "details": details,
            "timestamp": time.time(),
            "resolved": False
        }
        
        self.error_records.append(error_record)
        
        self.logger.info(f"已记录错误: {module_id}, 类型: {error_type}")
        
        return error_record
    
    def _attempt_correction(self, module_id: str, error_type: str) -> Dict[str, Any]:
        """尝试自动修正错误"""
        # 获取模块评估
        if module_id not in self.module_evaluations:
            return {
                "status": "error",
                "message": f"未找到模块: {module_id}"
            }
            
        evaluation = self.module_evaluations[module_id]
        
        # 根据错误类型选择修正策略
        if error_type == "high_error_rate":
            # 简单策略：降低模块评估中的错误率
            evaluation["error_rate"] *= 0.8  # 减少20%的错误率
            
            return {
                "status": "success",
                "message": f"已降低模块 {module_id} 的错误率",
                "correction_type": "error_rate_reduction"
            }
        else:
            return {
                "status": "error",
                "message": f"不支持的错误类型: {error_type}"
            }
    
    def _apply_metacognitive_strategies(self, decision: Dict[str, Any]):
        """应用元认知策略"""
        for strategy_id, strategy in self.metacognitive_strategies.items():
            if not strategy["enabled"]:
                continue
                
            # 检查是否应激活策略
            activation_value = self._calculate_strategy_activation(strategy, decision)
            
            if activation_value >= strategy["activation_threshold"]:
                try:
                    # 调用策略处理函数
                    handler = strategy["handler"]
                    handler(decision, activation_value)
                except Exception as e:
                    self.logger.error(f"应用策略 {strategy_id} 出错: {str(e)}")
    
    def _calculate_strategy_activation(self, strategy: Dict[str, Any], decision: Dict[str, Any]) -> float:
        """计算策略激活值"""
        # 简单实现：使用固定值
        if strategy["id"] == "contradiction_detection":
            return 0.7 if "contradiction_score" in decision else 0.0
        elif strategy["id"] == "uncertainty_monitoring":
            return 1.0 - decision.get("confidence", 0.5)
        elif strategy["id"] == "error_correction":
            return decision.get("error_probability", 0.0)
        elif strategy["id"] == "reflection":
            return 0.8 if len(self.decision_history) % 10 == 0 else 0.0  # 每10个决策反思一次
        elif strategy["id"] == "efficiency_optimization":
            return 0.5  # 默认中等可能性
        else:
            return 0.0
    
    # 策略处理函数
    def _handle_contradiction(self, decision: Dict[str, Any], activation_value: float):
        """处理矛盾"""
        self.logger.info(f"检测到可能的矛盾, 分数: {activation_value}")
        
        # 记录性能指标
        self.performance_metrics["consistency"].append(1.0 - activation_value)
    
    def _handle_uncertainty(self, decision: Dict[str, Any], activation_value: float):
        """处理不确定性"""
        confidence = decision.get("confidence", 0.5)
        
        if confidence < self.config["confidence_threshold"]:
            self.logger.info(f"检测到低置信度决策: {confidence}")
            
        # 记录性能指标
        self.performance_metrics["robustness"].append(confidence)
    
    def _handle_error(self, decision: Dict[str, Any], activation_value: float):
        """处理错误"""
        self.logger.info(f"检测到可能的错误, 概率: {activation_value}")
        
        # 记录性能指标
        self.performance_metrics["accuracy"].append(1.0 - activation_value)
    
    def _handle_reflection(self, decision: Dict[str, Any], activation_value: float):
        """处理反思"""
        self.logger.info("执行决策反思")
        
        # 分析最近的决策
        recent_decisions = list(self.decision_history)[-10:]
        
        if recent_decisions:
            # 计算平均置信度
            avg_confidence = sum(d.get("confidence", 0.5) for d in recent_decisions) / len(recent_decisions)
            
            # 记录性能指标
            self.performance_metrics["adaptability"].append(avg_confidence)
    
    def _handle_efficiency(self, decision: Dict[str, Any], activation_value: float):
        """处理效率优化"""
        self.logger.info("执行效率优化")
        
        # 记录性能指标
        execution_time = decision.get("execution_time", 1.0)
        normalized_efficiency = 1.0 / max(0.1, execution_time)
        self.performance_metrics["efficiency"].append(normalized_efficiency)
    
    def _calculate_overall_score(self, module_scores: List[Dict[str, Any]]) -> float:
        """计算整体得分"""
        if not module_scores:
            return 0.0
            
        # 计算平均得分
        avg_performance = sum(s["performance_score"] for s in module_scores) / len(module_scores)
        avg_reliability = sum(s["reliability_score"] for s in module_scores) / len(module_scores)
        avg_efficiency = sum(s["efficiency_score"] for s in module_scores) / len(module_scores)
        
        # 计算整体得分（简单加权平均）
        overall_score = avg_performance * 0.4 + avg_reliability * 0.4 + avg_efficiency * 0.2
        
        return overall_score
    
    def _analyze_strengths_weaknesses(self, module_scores: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """分析系统优势和劣势"""
        strengths = []
        weaknesses = []
        
        # 分析各模块得分
        for score in module_scores:
            module_id = score["module_id"]
            
            # 检查优势
            if score["performance_score"] > 0.8:
                strengths.append(f"模块 {module_id} 的性能表现优秀")
                
            if score["reliability_score"] > 0.8:
                strengths.append(f"模块 {module_id} 的可靠性很高")
                
            if score["efficiency_score"] > 0.8:
                strengths.append(f"模块 {module_id} 的效率很高")
                
            # 检查劣势
            if score["performance_score"] < 0.5:
                weaknesses.append(f"模块 {module_id} 的性能表现不佳")
                
            if score["reliability_score"] < 0.5:
                weaknesses.append(f"模块 {module_id} 的可靠性较低")
                
            if score["efficiency_score"] < 0.5:
                weaknesses.append(f"模块 {module_id} 的效率较低")
                
            if score["error_rate"] > self.config["error_threshold"]:
                weaknesses.append(f"模块 {module_id} 的错误率较高")
                
        return strengths, weaknesses
    
    def _generate_improvement_suggestions(self, weaknesses: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 根据劣势生成建议
        for weakness in weaknesses:
            if "性能表现不佳" in weakness:
                module_id = weakness.split(" ")[1]
                suggestions.append(f"为模块 {module_id} 优化算法，提高处理速度")
                
            elif "可靠性较低" in weakness:
                module_id = weakness.split(" ")[1]
                suggestions.append(f"为模块 {module_id} 增加错误处理和容错机制")
                
            elif "效率较低" in weakness:
                module_id = weakness.split(" ")[1]
                suggestions.append(f"优化模块 {module_id} 的资源使用效率")
                
            elif "错误率较高" in weakness:
                module_id = weakness.split(" ")[1]
                suggestions.append(f"调查模块 {module_id} 的错误原因并修复")
                
        # 添加一些通用建议
        if not suggestions:
            suggestions.append("继续监控系统性能，寻找进一步优化空间")
            
        if len(weaknesses) > 3:
            suggestions.append("考虑进行系统架构审核，识别潜在的设计问题")
            
        return suggestions
    
    def _save_config(self):
        """保存配置"""
        try:
            with open("meta_supervisor_config.json", "w") as f:
                json.dump(self.config, f)
        except Exception as e:
            self.logger.error(f"保存配置出错: {str(e)}")
    
    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists("meta_supervisor_config.json"):
                with open("meta_supervisor_config.json", "r") as f:
                    config = json.load(f)
                    
                for key, value in config.items():
                    if key in self.config:
                        self.config[key] = value
                        
                self.logger.info("已加载配置")
        except Exception as e:
            self.logger.error(f"加载配置出错: {str(e)}")
    
    def _save_assessment(self):
        """保存评估结果"""
        try:
            assessment_data = {
                "timestamp": time.time(),
                "self_assessment": self.self_assessment,
                "module_evaluations": {k: {f: v for f, v in m.items() if f != "evaluation_history"} 
                                      for k, m in self.module_evaluations.items()}
            }
            
            with open("meta_assessment.json", "w") as f:
                json.dump(assessment_data, f)
        except Exception as e:
            self.logger.error(f"保存评估结果出错: {str(e)}")
    
    def _load_history(self):
        """加载历史数据"""
        try:
            if os.path.exists("meta_assessment.json"):
                with open("meta_assessment.json", "r") as f:
                    data = json.load(f)
                    
                if "self_assessment" in data:
                    # 只加载关键字段
                    self.self_assessment["last_assessment_time"] = data["self_assessment"].get("last_assessment_time")
                    self.self_assessment["overall_score"] = data["self_assessment"].get("overall_score", 0.0)
                    self.self_assessment["strengths"] = data["self_assessment"].get("strengths", [])
                    self.self_assessment["weaknesses"] = data["self_assessment"].get("weaknesses", [])
                    
                self.logger.info("已加载历史评估数据")
        except Exception as e:
            self.logger.error(f"加载历史数据出错: {str(e)}")
    
    def shutdown(self) -> Dict[str, Any]:
        """
        关闭元认知监督模块
        
        Returns:
            Dict: 关闭结果
        """
        # 停止监控
        if self.monitoring_state["is_active"]:
            self.stop_monitoring()
            
        # 保存数据
        self._save_config()
        self._save_assessment()
        
        self.logger.info("元认知监督模块已关闭")
        
        return {
            "status": "success",
            "message": "元认知监督模块已关闭"
        } 