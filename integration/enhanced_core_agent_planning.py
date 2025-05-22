# -*- coding: utf-8 -*-
"""
增强版核心智能体动态规划能力实现 (Enhanced Core Agent Dynamic Planning)

实现第四阶段计划中的核心智能体动态规划能力：
1. 增强 PlanningEngine 类，实现 A* 搜索算法
2. 在 CoreAgent 中实现目标分解机制
3. 将符号推理系统与规划引擎集成
4. 开发行动评估和优先级调整机制
5. 实现基于历史数据的行动效果预测
"""

import time
import copy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

# 导入相关模块
from agents.core_agent import CoreAgent
from reasoning.enhanced_planning import EnhancedPlanningEngine
from reasoning.symbolic import SymbolicReasoner

class EnhancedDynamicPlanningIntegrator:
    """
    增强版动态规划集成器：将高级规划能力集成到核心智能体中
    """
    def __init__(self, core_agent: CoreAgent):
        """
        初始化动态规划集成器
        
        Args:
            core_agent: 核心智能体实例
        """
        self.core_agent = core_agent
        self.planning_engine = EnhancedPlanningEngine()
        self.symbolic_reasoner = SymbolicReasoner()
        
        # 行动评估机制
        self.action_evaluations = {}
        self.action_success_history = {}
        self.action_effect_predictions = {}
        
        # 目标分解历史
        self.goal_decomposition_history = []
        
        # 初始化集成
        self._initialize_integration()
        
    def _initialize_integration(self):
        """
        初始化集成：增强核心智能体的规划能力
        """
        # 设置符号推理系统
        self.planning_engine.set_symbolic_reasoner(self.symbolic_reasoner)
        
        # 注册默认行动
        self._register_default_actions()
        
        # 设置目标优先级和依赖关系
        self._setup_goal_priorities()
        
        # 增强核心智能体的规划方法
        self._enhance_core_agent_planning()
        
    def _register_default_actions(self):
        """
        注册默认行动到规划引擎
        """
        # 信息检索行动
        self.planning_engine.register_action(
            name="search_knowledge",
            preconditions={"has_query": True},
            effects={"knowledge_retrieved": True},
            cost=1.0
        )
        
        # 回答生成行动
        self.planning_engine.register_action(
            name="formulate_answer",
            preconditions={"knowledge_retrieved": True},
            effects={"answer_formulated": True},
            cost=1.5
        )
        
        # 命令解析行动
        self.planning_engine.register_action(
            name="parse_command",
            preconditions={"has_input": True},
            effects={"command_parsed": True},
            cost=1.0
        )
        
        # 命令验证行动
        self.planning_engine.register_action(
            name="validate_command",
            preconditions={"command_parsed": True},
            effects={"command_validated": True},
            cost=1.2
        )
        
        # 命令执行行动
        self.planning_engine.register_action(
            name="execute_command",
            preconditions={"command_parsed": True, "command_validated": True},
            effects={"command_executed": True},
            cost=2.0
        )
        
        # 事实提取行动
        self.planning_engine.register_action(
            name="extract_facts",
            preconditions={"has_text": True},
            effects={"facts_extracted": True},
            cost=1.5
        )
        
        # 事实验证行动
        self.planning_engine.register_action(
            name="validate_facts",
            preconditions={"facts_extracted": True},
            effects={"facts_validated": True},
            cost=1.8
        )
        
        # 知识存储行动
        self.planning_engine.register_action(
            name="store_knowledge",
            preconditions={"facts_validated": True},
            effects={"knowledge_stored": True},
            cost=1.0
        )
        
    def _setup_goal_priorities(self):
        """
        设置目标优先级和依赖关系
        """
        # 设置一些基本目标的优先级
        self.planning_engine.goal_priorities = {
            "answer_formulated": 5.0,  # 高优先级
            "command_executed": 4.5,
            "knowledge_stored": 4.0,
            "knowledge_retrieved": 3.5,
            "facts_validated": 3.0,
            "command_validated": 2.5,
            "facts_extracted": 2.0,
            "command_parsed": 1.5,
            "has_query": 1.0,
            "has_text": 1.0,
            "has_input": 1.0
        }
        
        # 设置目标依赖关系
        self.planning_engine.goal_dependencies["answer_formulated"] = ["knowledge_retrieved"]
        self.planning_engine.goal_dependencies["command_executed"] = ["command_parsed", "command_validated"]
        self.planning_engine.goal_dependencies["knowledge_stored"] = ["facts_extracted", "facts_validated"]
        self.planning_engine.goal_dependencies["facts_validated"] = ["facts_extracted"]
        self.planning_engine.goal_dependencies["command_validated"] = ["command_parsed"]
    
    def _enhance_core_agent_planning(self):
        """
        增强核心智能体的规划方法
        """
        # 保存原始方法引用
        self.original_plan_actions = self.core_agent.plan_actions
        
        # 替换为增强版规划方法
        self.core_agent.plan_actions = self.enhanced_plan_actions
        
        # 添加目标分解方法
        self.core_agent.decompose_goal = self.decompose_goal
        
        # 添加行动评估方法
        self.core_agent.evaluate_action = self.evaluate_action
        
        # 添加行动效果预测方法
        self.core_agent.predict_action_effect = self.predict_action_effect
    
    def enhanced_plan_actions(self, goal=None):
        """
        增强版行动规划方法
        
        Args:
            goal: 目标（可选）
            
        Returns:
            list: 行动计划列表
        """
        target_goal = goal or (self.core_agent.goals[0] if self.core_agent.goals else None)
        if not target_goal:
            return []
        
        # 获取当前状态信息
        current_state = self.core_agent.current_state
        input_type = current_state.get("input_type", "unknown")
        input_content = current_state.get("last_input", "")
        perception = current_state.get("perception", {})
        
        # 根据输入类型和感知结果设置初始状态和目标状态
        initial_state = {}
        goal_state = {}
        
        if perception.get("type") == "question":
            # 问题类型的输入
            intent = perception.get("intent", "general")
            initial_state = {
                "has_query": True,
                "query_content": input_content,
                "query_intent": intent
            }
            goal_state = {
                "answer_formulated": True
            }
        elif perception.get("type") == "command":
            # 命令类型的输入
            initial_state = {
                "has_input": True,
                "input_content": input_content,
                "input_type": "command"
            }
            goal_state = {
                "command_executed": True
            }
        elif perception.get("type") == "statement":
            # 陈述类型的输入
            initial_state = {
                "has_text": True,
                "text_content": input_content,
                "text_type": "statement"
            }
            goal_state = {
                "knowledge_stored": True
            }
        else:
            # 默认情况
            initial_state = {
                "has_input": True,
                "input_content": input_content,
                "input_type": input_type
            }
            goal_state = {
                "input_processed": True
            }
            
        # 设置规划引擎的状态
        self.planning_engine.set_current_state(initial_state)
        self.planning_engine.set_goal_state(goal_state)
        
        # 使用增强版规划引擎生成计划
        action_names = self.planning_engine.plan(max_steps=10)
        
        # 将行动名称转换为行动计划格式
        plan = []
        for action_name in action_names:
            action_params = {}
            
            # 根据行动类型设置参数
            if action_name == "search_knowledge":
                action_params = {"query": input_content, "intent": perception.get("intent", "general")}
            elif action_name == "formulate_answer":
                action_params = {"style": "informative", "goal": target_goal}
            elif action_name == "parse_command":
                action_params = {"text": input_content}
            elif action_name == "validate_command":
                action_params = {"safety_check": True}
            elif action_name == "execute_command":
                action_params = {"mode": "safe"}
            elif action_name == "extract_facts":
                action_params = {"text": input_content}
            elif action_name == "validate_facts":
                action_params = {"confidence_threshold": 0.7}
            elif action_name == "store_knowledge":
                action_params = {"permanent": True}
            
            plan.append({"action": action_name, "params": action_params})
        
        # 如果规划失败，使用原始方法作为后备
        if not plan:
            return self.original_plan_actions(goal)
            
        return plan
    
    def decompose_goal(self, goal):
        """
        将复杂目标分解为子目标
        
        Args:
            goal: 目标描述
            
        Returns:
            list: 子目标列表
        """
        # 将目标转换为状态表示
        if isinstance(goal, str):
            goal_state = {"goal_description": goal, "goal_achieved": True}
        elif isinstance(goal, dict):
            goal_state = goal
        else:
            goal_state = {"goal_achieved": True}
        
        # 使用规划引擎的目标分解功能
        sub_goals = self.planning_engine.decompose_goal(goal_state)
        
        # 记录分解历史
        self.goal_decomposition_history.append({
            "original_goal": goal,
            "sub_goals": sub_goals,
            "timestamp": time.time()
        })
        
        return sub_goals
    
    def evaluate_action(self, action, result):
        """
        评估行动执行结果
        
        Args:
            action: 行动描述
            result: 执行结果
            
        Returns:
            float: 评估分数
        """
        action_name = action.get("action", "")
        if not action_name:
            return 0.0
        
        # 准备评估结果
        evaluation_result = {
            "success": result.get("success", False),
            "execution_time": result.get("execution_time", 0.0),
            "quality": result.get("quality", 0.0)
        }
        
        # 使用规划引擎的行动评估功能
        score = self.planning_engine.evaluate_action(action_name, evaluation_result)
        
        # 更新行动成功历史
        if action_name not in self.action_success_history:
            self.action_success_history[action_name] = []
        self.action_success_history[action_name].append(evaluation_result["success"])
        
        # 更新行动评估记录
        self.action_evaluations[action_name] = score
        
        return score
    
    def predict_action_effect(self, action, state=None):
        """
        预测行动在给定状态下的效果
        
        Args:
            action: 行动描述
            state: 当前状态（可选，默认使用核心智能体的当前状态）
            
        Returns:
            dict: 预测的新状态
        """
        action_name = action.get("action", "")
        if not action_name:
            return {}
        
        # 获取当前状态
        current_state = state or self.core_agent.current_state
        
        # 将状态转换为规划引擎可用的格式
        planning_state = {}
        for key, value in current_state.items():
            planning_state[key] = value
        
        # 使用规划引擎的行动效果预测功能
        predicted_state = self.planning_engine.predict_action_effect(action_name, planning_state)
        
        # 记录预测
        key = (action_name, str(current_state))
        self.action_effect_predictions[key] = predicted_state
        
        return predicted_state
    
    def update_effect_prediction(self, action, initial_state, actual_state):
        """
        根据实际结果更新行动效果预测
        
        Args:
            action: 行动描述
            initial_state: 执行前状态
            actual_state: 执行后实际状态
        """
        action_name = action.get("action", "")
        if not action_name:
            return
        
        # 将状态转换为规划引擎可用的格式
        planning_initial_state = {}
        for key, value in initial_state.items():
            planning_initial_state[key] = value
            
        planning_actual_state = {}
        for key, value in actual_state.items():
            planning_actual_state[key] = value
        
        # 使用规划引擎的效果预测更新功能
        self.planning_engine.update_effect_prediction(
            action_name, planning_initial_state, planning_actual_state)