# -*- coding: utf-8 -*-
"""
元认知智能体 (Meta-Cognition Agent)

负责监控系统整体性能，进行自我评估和策略调整
提供目标管理、优先级调整和计划评估功能
"""

import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Union

class MetaCognitionAgent:
    def __init__(self, name="Meta", event_system=None, memory_system=None, core_agent=None):
        """
        初始化元认知智能体
        
        Args:
            name (str): 智能体名称
            event_system: 事件系统实例
            memory_system: 记忆系统实例
            core_agent: 核心智能体实例
        """
        self.name = name
        self.event_system = event_system
        self.memory_system = memory_system
        self.core_agent = core_agent
        
        self.state = "idle"
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitoring_interval = 60  # 监控间隔（秒）
        
        self.performance_history = []  # 性能历史
        self.feedback_history = []  # 反馈历史
        self.goal_assessments = {}  # 目标评估
        self.system_evaluations = []  # 系统评估
        
        self.last_evaluation_time = time.time()
        self.evaluation_interval = 300  # 评估间隔（秒）
        
        self.subscription_ids = []
        
        # 初始化
        if self.event_system:
            self._subscribe_to_events()
        
    def _subscribe_to_events(self):
        """订阅关键事件"""
        if not self.event_system:
            return
            
        # 订阅核心智能体事件
        self.subscription_ids.append(
            self.event_system.subscribe("core.task_completed", self._handle_task_completed)
        )
        
        # 订阅计划相关事件
        self.subscription_ids.append(
            self.event_system.subscribe("core.plan_completed", self._handle_plan_completed)
        )
        
        # 订阅用户反馈事件
        self.subscription_ids.append(
            self.event_system.subscribe("user.feedback", self._handle_user_feedback)
        )
        
        # 订阅系统错误事件
        self.subscription_ids.append(
            self.event_system.subscribe("system.error", self._handle_system_error)
        )
        
        # 发布初始化完成事件
        self.event_system.publish("agent.initialized", {
            "agent_id": self.name,
            "agent_type": "meta_cognition",
            "capabilities": ["self_monitoring", "goal_management", "performance_evaluation"],
            "timestamp": time.time()
        })
        
    def start_monitoring(self):
        """启动系统监控"""
        if self.monitoring_active:
            return False
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.event_system:
            self.event_system.publish("metacognition.monitoring_started", {
                "agent_id": self.name,
                "interval": self.monitoring_interval,
                "timestamp": time.time()
            })
            
        return True
        
    def stop_monitoring(self):
        """停止系统监控"""
        if not self.monitoring_active:
            return False
            
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        if self.event_system:
            self.event_system.publish("metacognition.monitoring_stopped", {
                "agent_id": self.name,
                "timestamp": time.time()
            })
            
        return True
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 执行监控检查
                self._check_system_health()
                
                # 检查是否需要进行评估
                current_time = time.time()
                if current_time - self.last_evaluation_time >= self.evaluation_interval:
                    self._perform_system_evaluation()
                    self.last_evaluation_time = current_time
                    
                # 等待下一个监控周期
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"监控错误: {str(e)}")
                time.sleep(5)  # 错误后短暂暂停
                
    def _check_system_health(self):
        """检查系统健康状态"""
        health_metrics = {
            "timestamp": time.time(),
            "components": {}
        }
        
        # 检查核心智能体
        if self.core_agent:
            health_metrics["components"]["core_agent"] = {
                "status": self.core_agent.state,
                "task_count": len(self.core_agent.task_history),
                "active_plans": len(self.core_agent.active_plans)
            }
            
        # 检查记忆系统
        if self.memory_system:
            try:
                memory_stats = self.memory_system.get_stats()
                health_metrics["components"]["memory_system"] = memory_stats
            except:
                health_metrics["components"]["memory_system"] = {"status": "error"}
                
        # 检查事件系统
        if self.event_system:
            try:
                # 获取事件数量
                event_count = len(self.event_system.get_history(limit=1000))
                health_metrics["components"]["event_system"] = {
                    "status": "active" if event_count > 0 else "idle",
                    "event_count": event_count
                }
            except:
                health_metrics["components"]["event_system"] = {"status": "error"}
                
        # 分析健康状态
        health_status = "healthy"
        issues = []
        
        for component, stats in health_metrics["components"].items():
            if stats.get("status") == "error":
                health_status = "unhealthy"
                issues.append(f"{component}出现错误")
            elif stats.get("status") == "overloaded":
                health_status = "stressed"
                issues.append(f"{component}负载过高")
                
        health_metrics["overall_status"] = health_status
        health_metrics["issues"] = issues
        
        # 记录健康检查结果
        if self.event_system:
            self.event_system.publish("metacognition.health_check", health_metrics)
            
        return health_metrics
        
    def _perform_system_evaluation(self):
        """执行系统评估"""
        # 收集评估指标
        evaluation = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        # 收集任务性能
        if self.core_agent and self.core_agent.task_history:
            recent_tasks = self.core_agent.task_history[-20:]  # 最近20个任务
            completion_times = []
            success_count = 0
            
            for task in recent_tasks:
                if task.get("status") == "completed":
                    success_count += 1
                    start_time = task.get("start_time", 0)
                    end_time = task.get("end_time", 0)
                    if end_time > start_time:
                        completion_times.append(end_time - start_time)
                        
            evaluation["metrics"]["task_performance"] = {
                "success_rate": success_count / len(recent_tasks) if recent_tasks else 0,
                "avg_completion_time": sum(completion_times) / len(completion_times) if completion_times else 0,
                "task_count": len(recent_tasks)
            }
            
        # 评估目标进展
        if self.core_agent:
            goals = self.core_agent.goals
            goal_assessments = {}
            
            for goal in goals:
                progress = self._assess_goal_progress(goal)
                goal_assessments[goal] = progress
                
            evaluation["metrics"]["goal_progress"] = goal_assessments
            
        # 用户反馈分析
        if self.feedback_history:
            recent_feedback = self.feedback_history[-10:]  # 最近10条反馈
            positive_count = sum(1 for f in recent_feedback if f.get("sentiment") == "positive")
            neutral_count = sum(1 for f in recent_feedback if f.get("sentiment") == "neutral")
            negative_count = sum(1 for f in recent_feedback if f.get("sentiment") == "negative")
            
            evaluation["metrics"]["user_feedback"] = {
                "positive_rate": positive_count / len(recent_feedback) if recent_feedback else 0,
                "neutral_rate": neutral_count / len(recent_feedback) if recent_feedback else 0,
                "negative_rate": negative_count / len(recent_feedback) if recent_feedback else 0,
                "feedback_count": len(recent_feedback)
            }
            
        # 系统错误率
        if self.event_system:
            try:
                recent_events = self.event_system.get_history(limit=100)
                error_events = [e for e in recent_events if "error" in e["type"]]
                
                evaluation["metrics"]["error_rate"] = len(error_events) / len(recent_events) if recent_events else 0
            except:
                evaluation["metrics"]["error_rate"] = 0
                
        # 记录评估结果
        self.system_evaluations.append(evaluation)
        
        # 发布评估结果
        if self.event_system:
            self.event_system.publish("metacognition.system_evaluation", evaluation)
            
        # 基于评估结果提供反馈
        self._provide_feedback_based_on_evaluation(evaluation)
        
        return evaluation
        
    def _assess_goal_progress(self, goal):
        """
        评估特定目标的进展
        
        Args:
            goal (str): 目标
            
        Returns:
            Dict: 进展评估
        """
        # 从记忆系统获取相关记忆
        relevant_memories = []
        if self.memory_system:
            try:
                # 查询与目标相关的记忆
                relevant_memories = self.memory_system.query_by_content(goal)
            except:
                pass
                
        # 计算目标相关活动数量
        activity_count = len(relevant_memories)
        
        # 从任务历史中找到与目标相关的任务
        relevant_tasks = []
        if self.core_agent and self.core_agent.task_history:
            for task in self.core_agent.task_history:
                desc = str(task.get("description", ""))
                if goal.lower() in desc.lower():
                    relevant_tasks.append(task)
                    
        # 计算成功率
        success_count = sum(1 for task in relevant_tasks if task.get("status") == "completed")
        success_rate = success_count / len(relevant_tasks) if relevant_tasks else 0
        
        # 计算进展分数
        if goal == "自我完善":
            # 自我完善目标的进展评估
            progress_score = min(0.2 + (activity_count * 0.01) + (success_rate * 0.3), 1.0)
        elif goal == "学习新知识":
            # 学习目标的进展评估
            knowledge_gain = activity_count * 0.02
            progress_score = min(0.15 + knowledge_gain + (success_rate * 0.25), 1.0)
        elif goal == "协助用户":
            # 用户协助目标的进展评估
            user_interactions = sum(1 for task in self.core_agent.task_history if "user" in str(task.get("description", "")))
            progress_score = min(0.1 + (user_interactions * 0.03) + (success_rate * 0.4), 1.0)
        else:
            # 通用目标进展评估
            progress_score = min(0.1 + (activity_count * 0.015) + (success_rate * 0.35), 1.0)
            
        # 记录目标评估
        assessment = {
            "goal": goal,
            "progress_score": progress_score,
            "activity_count": activity_count,
            "success_rate": success_rate,
            "relevant_tasks": len(relevant_tasks),
            "timestamp": time.time()
        }
        
        self.goal_assessments[goal] = assessment
        return assessment
        
    def _provide_feedback_based_on_evaluation(self, evaluation):
        """
        基于评估结果提供反馈
        
        Args:
            evaluation: 评估结果
        """
        feedback = []
        
        # 分析任务性能
        task_performance = evaluation["metrics"].get("task_performance", {})
        if task_performance:
            success_rate = task_performance.get("success_rate", 0)
            if success_rate < 0.6:
                feedback.append({
                    "type": "performance_evaluation",
                    "area": "task_completion",
                    "message": "任务完成率低于预期",
                    "score": success_rate,
                    "suggestion": "优化任务分配和执行策略"
                })
                
        # 分析目标进展
        goal_progress = evaluation["metrics"].get("goal_progress", {})
        for goal, progress in goal_progress.items():
            score = progress.get("progress_score", 0)
            if score < 0.4:
                feedback.append({
                    "type": "goal_alignment",
                    "goal": goal,
                    "message": f"目标'{goal}'进展缓慢",
                    "score": score,
                    "suggestion": "增加与该目标相关的活动和任务"
                })
                
        # 分析用户反馈
        user_feedback = evaluation["metrics"].get("user_feedback", {})
        if user_feedback:
            positive_rate = user_feedback.get("positive_rate", 0)
            negative_rate = user_feedback.get("negative_rate", 0)
            
            if negative_rate > 0.3:
                feedback.append({
                    "type": "user_satisfaction",
                    "message": "用户负面反馈率较高",
                    "score": 1.0 - negative_rate,
                    "suggestion": "提高响应质量和准确性"
                })
                
        # 分析错误率
        error_rate = evaluation["metrics"].get("error_rate", 0)
        if error_rate > 0.1:
            feedback.append({
                "type": "system_reliability",
                "message": "系统错误率超过阈值",
                "score": 1.0 - error_rate,
                "suggestion": "优先修复频繁出现的错误"
            })
            
        # 发送反馈给核心智能体
        for fb in feedback:
            if self.event_system:
                self.event_system.publish("metacognition.feedback", fb)
                
        return feedback
        
    def execute_task(self, task_description):
        """
        执行元认知任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 任务执行结果
        """
        task_type = task_description.get("type", "unknown")
        
        # 设置状态为工作中
        self.state = "working"
        
        result = None
        
        if task_type == "evaluate_system":
            result = self._perform_system_evaluation()
        elif task_type == "assess_goal":
            goal = task_description.get("goal")
            if goal:
                result = self._assess_goal_progress(goal)
            else:
                result = {"status": "error", "message": "未指定目标"}
        elif task_type == "adjust_strategy":
            strategy_params = task_description.get("parameters", {})
            result = self._adjust_system_strategy(strategy_params)
        elif task_type == "provide_feedback":
            target = task_description.get("target")
            message = task_description.get("message")
            feedback_type = task_description.get("feedback_type", "general")
            result = self._provide_explicit_feedback(target, message, feedback_type)
        else:
            # 默认处理
            result = {"status": "unknown_task", "message": f"未知任务类型: {task_type}"}
            
        # 恢复空闲状态
        self.state = "idle"
        
        return result
        
    def _handle_task_completed(self, event):
        """
        处理任务完成事件
        
        Args:
            event: 事件对象
        """
        task_id = event["data"]["task_id"]
        result = event["data"]["result"]
        execution_time = event["data"].get("execution_time", 0)
        
        # 记录性能指标
        performance_record = {
            "task_id": task_id,
            "execution_time": execution_time,
            "success": "error" not in result.get("status", "").lower(),
            "timestamp": event["timestamp"]
        }
        
        self.performance_history.append(performance_record)
        
        # 截断历史记录，保留最近的100条
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
    def _handle_plan_completed(self, event):
        """
        处理计划完成事件
        
        Args:
            event: 事件对象
        """
        plan_id = event["data"]["plan_id"]
        results = event["data"]["results"]
        execution_time = event["data"].get("execution_time", 0)
        
        # 评估计划执行效果
        success_steps = sum(1 for result in results if result and result.get("status") != "error")
        total_steps = len(results) if results else 1
        
        efficiency_score = min(10.0 / execution_time if execution_time > 0 else 0, 1.0)
        effectiveness_score = success_steps / total_steps if total_steps > 0 else 0
        
        evaluation = {
            "plan_id": plan_id,
            "efficiency": efficiency_score,
            "effectiveness": effectiveness_score,
            "overall_score": (efficiency_score * 0.4) + (effectiveness_score * 0.6),
            "timestamp": event["timestamp"]
        }
        
        # 发布计划评估事件
        if self.event_system:
            self.event_system.publish("metacognition.plan_evaluation", evaluation)
            
    def _handle_user_feedback(self, event):
        """
        处理用户反馈事件
        
        Args:
            event: 事件对象
        """
        feedback = event["data"]
        
        # 添加到反馈历史
        self.feedback_history.append({
            "feedback": feedback,
            "timestamp": event["timestamp"]
        })
        
        # 截断历史记录，保留最近的50条
        if len(self.feedback_history) > 50:
            self.feedback_history = self.feedback_history[-50:]
            
        # 分析情感
        sentiment = self._analyze_feedback_sentiment(feedback)
        
        # 发布反馈分析事件
        if self.event_system:
            self.event_system.publish("metacognition.feedback_analysis", {
                "feedback": feedback,
                "sentiment": sentiment,
                "timestamp": time.time()
            })
            
    def _handle_system_error(self, event):
        """
        处理系统错误事件
        
        Args:
            event: 事件对象
        """
        error = event["data"]
        
        # 分析错误严重性
        severity = "high" if "critical" in error.get("context", "").lower() else "medium"
        
        # 发布错误分析事件
        if self.event_system:
            self.event_system.publish("metacognition.error_analysis", {
                "error": error,
                "severity": severity,
                "timestamp": time.time()
            })
            
    def _analyze_feedback_sentiment(self, feedback):
        """
        分析反馈情感
        
        Args:
            feedback: 反馈数据
            
        Returns:
            str: 情感分类 ("positive", "neutral", "negative")
        """
        # 简单的基于关键词的情感分析
        text = str(feedback).lower()
        
        positive_words = ["好", "喜欢", "不错", "满意", "优秀", "感谢", "谢谢", "棒", "厉害"]
        negative_words = ["差", "不好", "不满意", "糟糕", "错误", "问题", "失败", "弱", "垃圾"]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
            
    def _adjust_system_strategy(self, parameters):
        """
        调整系统策略
        
        Args:
            parameters: 策略参数
            
        Returns:
            Dict: 调整结果
        """
        adjustments = []
        
        # 处理探索率调整
        if "exploration_rate" in parameters:
            new_rate = parameters["exploration_rate"]
            adjustments.append(f"探索率调整为{new_rate}")
            
        # 处理学习率调整
        if "learning_rate" in parameters:
            new_rate = parameters["learning_rate"]
            adjustments.append(f"学习率调整为{new_rate}")
            
        # 处理任务优先级调整
        if "task_priority" in parameters:
            priority_changes = parameters["task_priority"]
            adjustments.append(f"任务优先级调整: {priority_changes}")
            
        # 发布策略调整事件
        if self.event_system and adjustments:
            self.event_system.publish("metacognition.strategy_adjustment", {
                "adjustments": adjustments,
                "parameters": parameters,
                "timestamp": time.time()
            })
            
        return {
            "status": "adjusted" if adjustments else "no_change",
            "adjustments": adjustments,
            "timestamp": time.time()
        }
        
    def _provide_explicit_feedback(self, target, message, feedback_type):
        """
        提供明确的反馈
        
        Args:
            target: 反馈目标
            message: 反馈消息
            feedback_type: 反馈类型
            
        Returns:
            Dict: 反馈结果
        """
        if not target or not message:
            return {"status": "error", "message": "反馈目标和消息不能为空"}
            
        feedback = {
            "target": target,
            "message": message,
            "type": feedback_type,
            "timestamp": time.time()
        }
        
        # 发布反馈事件
        if self.event_system:
            event_type = f"metacognition.feedback.{feedback_type}"
            self.event_system.publish(event_type, feedback)
            
        return {
            "status": "feedback_sent",
            "feedback": feedback
        }
        
    def get_performance_stats(self):
        """
        获取性能统计信息
        
        Returns:
            Dict: 性能统计
        """
        if not self.performance_history:
            return {"status": "no_data"}
            
        # 计算整体成功率
        success_count = sum(1 for record in self.performance_history if record.get("success", False))
        total_count = len(self.performance_history)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # 计算平均执行时间
        execution_times = [record.get("execution_time", 0) for record in self.performance_history]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "total_tasks": total_count,
            "recent_success_rate": self._get_recent_success_rate(10)
        }
        
    def _get_recent_success_rate(self, count):
        """
        获取最近任务的成功率
        
        Args:
            count: 任务数量
            
        Returns:
            float: 成功率
        """
        if not self.performance_history or count <= 0:
            return 0
            
        recent = self.performance_history[-count:]
        success_count = sum(1 for record in recent if record.get("success", False))
        return success_count / len(recent) if recent else 0