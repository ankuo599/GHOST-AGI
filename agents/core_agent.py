# -*- coding: utf-8 -*-
"""
核心智能体 (Core Agent)

负责核心决策与任务执行的中央智能体，协调各个模块工作
整合规划、推理和执行能力
"""

import time
import uuid
from typing import Dict, List, Any, Tuple, Optional, Union

class CoreAgent:
    def __init__(self, name="Core", goals=None, event_system=None, 
                 memory_system=None, reasoning_engine=None, planning_engine=None,
                 agent_scheduler=None, evolution_engine=None, code_generator=None,
                 knowledge_transfer=None, autonomous_learning=None):
        """
        初始化核心智能体
        
        Args:
            name (str): 智能体名称
            goals (List[str], optional): 基础目标列表
            event_system: 事件系统实例
            memory_system: 记忆系统实例
            reasoning_engine: 推理引擎实例
            planning_engine: 规划引擎实例
            agent_scheduler: 智能体调度器实例
            evolution_engine: 进化引擎实例
            code_generator: 代码生成器实例
            knowledge_transfer: 知识迁移系统实例
            autonomous_learning: 自主学习系统实例
        """
        self.name = name
        self.goals = goals or ["自我完善", "学习新知识", "协助用户"]
        self.event_system = event_system
        self.memory_system = memory_system
        self.reasoning_engine = reasoning_engine
        self.planning_engine = planning_engine
        self.agent_scheduler = agent_scheduler
        self.evolution_engine = evolution_engine
        self.code_generator = code_generator
        self.knowledge_transfer = knowledge_transfer
        self.autonomous_learning = autonomous_learning
        
        self.state = "idle"
        self.current_task = None
        self.task_history = []
        self.active_plans = []
        self.initialization_time = time.time()
        self.subscription_ids = []
        
        # 初始化
        if self.event_system:
            self._subscribe_to_events()
            
    def _subscribe_to_events(self):
        """订阅关键事件"""
        if not self.event_system:
            return
            
        # 订阅用户输入事件
        self.subscription_ids.append(
            self.event_system.subscribe("user.input", self._handle_user_input)
        )
        
        # 订阅任务相关事件
        self.subscription_ids.append(
            self.event_system.subscribe("task.completed", self._handle_task_completed)
        )
        
        # 订阅记忆相关事件
        self.subscription_ids.append(
            self.event_system.subscribe("memory.new", self._evaluate_new_memory)
        )
        
        # 订阅元认知事件
        self.subscription_ids.append(
            self.event_system.subscribe("metacognition.feedback", self._handle_metacognition_feedback)
        )
        
        # 发布初始化完成事件
        self.event_system.publish("agent.initialized", {
            "agent_id": self.name,
            "agent_type": "core",
            "capabilities": ["decision_making", "planning", "coordination"],
            "goals": self.goals
        })
        
    def execute_task(self, task_description):
        """
        执行任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 任务执行结果
        """
        task_type = task_description.get("type", "unknown")
        
        # 记录任务
        self.current_task = {
            "id": str(uuid.uuid4()),
            "description": task_description,
            "start_time": time.time(),
            "status": "in_progress"
        }
        self.state = "working"
        
        # 更新记忆
        if self.memory_system:
            self.memory_system.add_to_short_term({
                "type": "task_start",
                "task": self.current_task,
                "timestamp": time.time()
            })
            
        # 发布任务开始事件
        if self.event_system:
            self.event_system.publish("core.task_started", {
                "task_id": self.current_task["id"],
                "task_type": task_type,
                "timestamp": time.time()
            })
            
        # 基于任务类型执行不同处理
        result = None
        
        if task_type == "user_query":
            result = self._process_user_query(task_description)
        elif task_type == "planning":
            result = self._create_execution_plan(task_description)
        elif task_type == "decision":
            result = self._make_decision(task_description)
        else:
            # 默认处理
            result = self._default_task_handler(task_description)
            
        # 更新任务状态
        self.current_task["end_time"] = time.time()
        self.current_task["result"] = result
        self.current_task["status"] = "completed"
        
        # 添加到历史
        self.task_history.append(self.current_task)
        
        # 更新智能体状态
        self.state = "idle"
        self.current_task = None
        
        # 发布任务完成事件
        if self.event_system:
            self.event_system.publish("core.task_completed", {
                "task_id": self.current_task["id"],
                "result": result,
                "execution_time": self.current_task["end_time"] - self.current_task["start_time"],
                "timestamp": time.time()
            })
            
        return result
        
    def _handle_user_input(self, event):
        """
        处理用户输入事件
        
        Args:
            event: 事件对象
        """
        user_input = event["data"]["content"]
        
        # 记录到短期记忆
        if self.memory_system:
            self.memory_system.add_to_short_term({
                "type": "user_input",
                "content": user_input,
                "timestamp": event["timestamp"]
            })
            
        # 使用推理引擎分析输入
        if self.reasoning_engine:
            reasoning_result = self.reasoning_engine.query(user_input)
            
            # 发布推理结果事件
            if self.event_system:
                self.event_system.publish("core.reasoning_result", {
                    "input": user_input,
                    "result": reasoning_result,
                    "timestamp": time.time()
                })
                
        # 创建响应任务
        if self.planning_engine:
            # 创建处理计划
            plan = self.planning_engine.create_plan(
                context={"user_input": user_input},
                goal="回应用户查询",
                constraints=["准确", "有帮助", "及时"]
            )
            
            # 添加到活动计划
            if plan:
                plan_id = str(uuid.uuid4())
                self.active_plans.append({
                    "id": plan_id,
                    "plan": plan,
                    "status": "created",
                    "created_at": time.time()
                })
                
                # 发布计划创建事件
                if self.event_system:
                    self.event_system.publish("core.plan_created", {
                        "plan_id": plan_id,
                        "context": {"user_input": user_input},
                        "timestamp": time.time()
                    })
                    
                # 执行计划
                self._execute_plan(plan_id)
                
    def _execute_plan(self, plan_id):
        """
        执行特定计划
        
        Args:
            plan_id: 计划ID
        """
        # 查找计划
        plan_data = None
        for plan in self.active_plans:
            if plan["id"] == plan_id:
                plan_data = plan
                break
                
        if not plan_data:
            return
            
        # 更新计划状态
        plan_data["status"] = "executing"
        
        # 获取计划步骤
        steps = plan_data["plan"].get("steps", [])
        
        # 执行每个步骤
        for i, step in enumerate(steps):
            # 更新当前步骤
            plan_data["current_step"] = i
            
            # 根据步骤类型执行不同操作
            if step["type"] == "task":
                # 通过任务调度器分配任务
                if self.agent_scheduler:
                    task_id = self.agent_scheduler.assign_task(
                        task_description=step["details"],
                        required_capabilities=step.get("required_capabilities", []),
                        priority=step.get("priority", 0),
                        callback=lambda task_result: self._handle_step_completion(plan_id, i, task_result)
                    )
                    
                    # 记录任务ID
                    step["task_id"] = task_id
                    
            elif step["type"] == "decision":
                # 内部决策
                decision_result = self._make_decision(step["details"])
                
                # 记录决策结果
                step["result"] = decision_result
                
                # 处理步骤完成
                self._handle_step_completion(plan_id, i, {"status": "completed", "result": decision_result})
                
        # 所有步骤已提交执行
        plan_data["all_steps_submitted"] = True
        
    def _handle_step_completion(self, plan_id, step_index, result):
        """
        处理计划步骤完成
        
        Args:
            plan_id: 计划ID
            step_index: 步骤索引
            result: 步骤执行结果
        """
        # 查找计划
        plan_data = None
        for plan in self.active_plans:
            if plan["id"] == plan_id:
                plan_data = plan
                break
                
        if not plan_data:
            return
            
        # 更新步骤结果
        steps = plan_data["plan"].get("steps", [])
        if 0 <= step_index < len(steps):
            steps[step_index]["status"] = "completed"
            steps[step_index]["result"] = result
            
        # 检查计划是否完成
        all_completed = all(step.get("status") == "completed" for step in steps)
        
        if all_completed:
            # 更新计划状态
            plan_data["status"] = "completed"
            plan_data["completed_at"] = time.time()
            
            # 发布计划完成事件
            if self.event_system:
                self.event_system.publish("core.plan_completed", {
                    "plan_id": plan_id,
                    "results": [step.get("result") for step in steps],
                    "execution_time": plan_data["completed_at"] - plan_data.get("created_at", plan_data["completed_at"]),
                    "timestamp": time.time()
                })
                
    def _handle_task_completed(self, event):
        """
        处理任务完成事件
        
        Args:
            event: 事件对象
        """
        task_id = event["data"]["task_id"]
        result = event["data"]["result"]
        
        # 更新记忆
        if self.memory_system:
            self.memory_system.add_to_short_term({
                "type": "task_completed",
                "task_id": task_id,
                "result": result,
                "timestamp": event["timestamp"]
            })
            
        # 检查是否有相关联的活动计划
        for plan in self.active_plans:
            steps = plan["plan"].get("steps", [])
            for step in steps:
                if step.get("task_id") == task_id:
                    # 更新步骤状态
                    step["status"] = "completed"
                    step["result"] = result
                    
                    # 检查计划是否完成
                    self._check_plan_completion(plan["id"])
                    
    def _check_plan_completion(self, plan_id):
        """
        检查计划是否完成
        
        Args:
            plan_id: 计划ID
        """
        # 查找计划
        plan_data = None
        for plan in self.active_plans:
            if plan["id"] == plan_id:
                plan_data = plan
                break
                
        if not plan_data:
            return
            
        # 检查所有步骤是否完成
        steps = plan_data["plan"].get("steps", [])
        all_completed = all(step.get("status") == "completed" for step in steps)
        
        if all_completed:
            # 更新计划状态
            plan_data["status"] = "completed"
            plan_data["completed_at"] = time.time()
            
            # 发布计划完成事件
            if self.event_system:
                self.event_system.publish("core.plan_completed", {
                    "plan_id": plan_id,
                    "results": [step.get("result") for step in steps],
                    "execution_time": plan_data["completed_at"] - plan_data.get("created_at", plan_data["completed_at"]),
                    "timestamp": time.time()
                })
                
    def _evaluate_new_memory(self, event):
        """
        评估新记忆事件
        
        Args:
            event: 事件对象
        """
        memory_data = event["data"]
        
        # 记录评估
        if self.event_system:
            self.event_system.publish("core.memory_evaluated", {
                "memory_id": memory_data.get("id"),
                "importance": self._assess_memory_importance(memory_data),
                "relevance": self._assess_memory_relevance(memory_data),
                "timestamp": time.time()
            })
            
    def _assess_memory_importance(self, memory_data):
        """
        评估记忆重要性
        
        Args:
            memory_data: 记忆数据
            
        Returns:
            float: 重要性分数 (0-1)
        """
        # 简单规则的重要性评估
        if "user_input" in memory_data.get("type", ""):
            return 0.8  # 用户输入非常重要
        elif "task" in memory_data.get("type", ""):
            return 0.6  # 任务相关记忆较重要
        elif "error" in memory_data.get("type", ""):
            return 0.7  # 错误相关记忆较重要
        return 0.4  # 默认中等重要性
        
    def _assess_memory_relevance(self, memory_data):
        """
        评估记忆相关性
        
        Args:
            memory_data: 记忆数据
            
        Returns:
            float: 相关性分数 (0-1)
        """
        # 简单评估记忆与当前目标和任务的相关性
        if self.current_task:
            # 检查记忆是否与当前任务相关
            task_desc_str = str(self.current_task["description"]).lower()
            memory_str = str(memory_data).lower()
            
            # 简单文本匹配
            common_words = set(task_desc_str.split()) & set(memory_str.split())
            if common_words:
                return min(0.5 + 0.1 * len(common_words), 0.9)
                
        return 0.3  # 默认低相关性
        
    def _handle_metacognition_feedback(self, event):
        """
        处理元认知反馈事件
        
        Args:
            event: 事件对象
        """
        feedback = event["data"]
        feedback_type = feedback.get("type")
        
        if feedback_type == "performance_evaluation":
            # 处理性能评估反馈
            if feedback.get("score", 0) < 0.6:
                # 低性能，尝试调整策略
                self._adjust_strategy(feedback)
                
        elif feedback_type == "goal_alignment":
            # 处理目标对齐反馈
            if not feedback.get("aligned", True):
                # 目标不对齐，更新目标优先级
                self._update_goal_priorities(feedback)
                
    def _adjust_strategy(self, feedback):
        """
        根据反馈调整策略
        
        Args:
            feedback: 反馈数据
        """
        # 发布策略调整事件
        if self.event_system:
            self.event_system.publish("core.strategy_adjusted", {
                "feedback": feedback,
                "adjustment": "增加决策时间和深度",
                "timestamp": time.time()
            })
            
    def _update_goal_priorities(self, feedback):
        """
        更新目标优先级
        
        Args:
            feedback: 反馈数据
        """
        # 发布目标更新事件
        if self.event_system:
            self.event_system.publish("core.goals_updated", {
                "feedback": feedback,
                "updated_goals": self.goals,
                "timestamp": time.time()
            })
            
    def _process_user_query(self, task_description):
        """
        处理用户查询
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 处理结果
        """
        query = task_description.get("query", "")
        
        # 使用推理引擎
        if self.reasoning_engine:
            result = self.reasoning_engine.query(query)
            return {
                "response": result,
                "source": "reasoning_engine",
                "confidence": result.get("confidence", 0.5)
            }
            
        # 如果没有推理引擎，返回基本响应
        return {
            "response": f"已接收查询: {query}",
            "source": "default",
            "confidence": 0.3
        }
        
    def _create_execution_plan(self, task_description):
        """
        创建执行计划
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 计划结果
        """
        if self.planning_engine:
            plan = self.planning_engine.create_plan(
                context=task_description.get("context", {}),
                goal=task_description.get("goal", "完成任务"),
                constraints=task_description.get("constraints", [])
            )
            return plan
            
        # 如果没有规划引擎，返回简单计划
        return {
            "steps": [
                {"type": "task", "details": {"action": "simple_action"}}
            ],
            "estimated_time": 10,
            "confidence": 0.4
        }
        
    def _make_decision(self, task_description):
        """
        做出决策
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 决策结果
        """
        options = task_description.get("options", [])
        criteria = task_description.get("criteria", [])
        
        if not options:
            return {"decision": None, "confidence": 0, "reason": "没有可选选项"}
            
        # 简单决策逻辑
        best_option = options[0]
        confidence = 0.5
        
        # 如果有推理引擎，使用它评估选项
        if self.reasoning_engine and criteria:
            option_scores = []
            
            for option in options:
                score = 0
                for criterion in criteria:
                    # 构建查询
                    query = f"{option}满足{criterion}吗?"
                    result = self.reasoning_engine.query(query)
                    if result["result"]:
                        score += result["confidence"]
                        
                option_scores.append((option, score))
                
            # 选择最高分选项
            if option_scores:
                option_scores.sort(key=lambda x: x[1], reverse=True)
                best_option = option_scores[0][0]
                confidence = min(option_scores[0][1] / len(criteria), 0.95)
                
        return {
            "decision": best_option,
            "confidence": confidence,
            "reason": "基于提供的标准评估"
        }
        
    def _default_task_handler(self, task_description):
        """
        默认任务处理器
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 处理结果
        """
        return {
            "status": "processed",
            "message": f"已处理任务: {task_description.get('type', 'unknown')}",
            "timestamp": time.time()
        }