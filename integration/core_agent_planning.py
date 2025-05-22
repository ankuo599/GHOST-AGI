# -*- coding: utf-8 -*-
"""
核心智能体动态规划能力实现 (Core Agent Dynamic Planning)

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
from reasoning.planning import PlanningEngine
from reasoning.advanced_planning import AdvancedPlanningEngine
from reasoning.symbolic import SymbolicReasoner

class DynamicPlanningIntegrator:
    """
    动态规划集成器：将高级规划能力集成到核心智能体中
    """
    def __init__(self, core_agent: CoreAgent):
        """
        初始化动态规划集成器
        
        Args:
            core_agent: 核心智能体实例
        """
        self.core_agent = core_agent
        self.planning_engine = AdvancedPlanningEngine()
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
        # 注册默认行动
        self._register_default_actions()
        
        # 注册启发式函数
        self._register_heuristic_functions()
        
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
        
        # 命令执行行动
        self.planning_engine.register_action(
            name="execute_command",
            preconditions={"command_parsed": True, "command_validated": True},
            effects={"command_executed": True},
            cost=2.0
        )
        
    def _register_heuristic_functions(self):
        """
        注册启发式函数到规划引擎
        """
        # 基于目标优先级的启发式函数
        self.planning_engine.register_heuristic(
            "priority_based",
            self._priority_based_heuristic
        )
        
        # 基于行动成功历史的启发式函数
        self.planning_engine.register_heuristic(
            "success_history_based",
            self._success_history_heuristic
        )
        
    def _priority_based_heuristic(self, state: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """
        基于目标优先级的启发式函数
        
        Args:
            state: 当前状态
            goal: 目标状态
            
        Returns:
            float: 估计成本
        """
        # 计算未满足目标的加权和
        total_cost = 0.0
        for key, value in goal.items():
            if key not in state or state[key] != value:
                # 获取目标优先级（默认为1.0）
                priority = self.planning_engine.goal_priorities.get(key, 1.0)
                total_cost += (1.0 / priority)  # 优先级越高，成本越低
                
        return total_cost
    
    def _success_history_heuristic(self, state: Dict[str, Any], goal: Dict[str, Any]) -> float:
        """
        基于行动成功历史的启发式函数
        
        Args:
            state: 当前状态
            goal: 目标状态
            
        Returns:
            float: 估计成本
        """
        # 基本启发式值：未满足目标数量
        base_cost = sum(1 for k, v in goal.items() if k not in state or state[k] != v)
        
        # 根据历史成功率调整
        applicable_actions = []
        for action_name in self.planning_engine.actions:
            if self.planning_engine.is_action_applicable(action_name, state):
                applicable_actions.append(action_name)
                
        if not applicable_actions:
            return float('inf')  # 无可用行动
            
        # 计算可用行动的平均成功率
        success_rates = []
        for action in applicable_actions:
            history = self.action_success_history.get(action, [])
            if history:
                success_rate = sum(1 for result in history if result) / len(history)
                success_rates.append(success_rate)
            else:
                success_rates.append(0.5)  # 默认成功率
                
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.5
        
        # 调整成本：成功率越低，成本越高
        adjusted_cost = base_cost / max(0.1, avg_success_rate)
        return adjusted_cost
    
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
            list: 行动计划
        """
        # 获取目标
        target_goal = goal or (self.core_agent.goals[0] if self.core_agent.goals else None)
        if not target_goal:
            return []
        
        # 分解目标
        sub_goals = self.decompose_goal(target_goal)
        
        # 综合计划
        complete_plan = []
        current_state = self.core_agent.get_current_state()
        
        # 为每个子目标生成计划
        for sub_goal in sub_goals:
            # 转换为状态表示
            goal_state = self._goal_to_state(sub_goal)
            
            # 设置规划引擎状态
            self.planning_engine.set_current_state(current_state)
            self.planning_engine.set_goal_state(goal_state)
            
            # 生成子计划
            sub_plan = self.planning_engine.plan(max_steps=20)
            
            if sub_plan:
                # 更新当前状态
                for action in sub_plan:
                    current_state = self.planning_engine.apply_action(action, current_state)
                
                # 添加到完整计划
                complete_plan.extend(sub_plan)
            else:
                # 如果无法规划，回退到原始方法
                fallback_plan = self.original_plan_actions(sub_goal)
                complete_plan.extend(fallback_plan)
        
        # 评估和优化计划
        optimized_plan = self._optimize_plan(complete_plan, target_goal)
        
        return optimized_plan
    
    def decompose_goal(self, goal):
        """
        将复杂目标分解为子目标
        
        Args:
            goal: 目标描述
            
        Returns:
            list: 子目标列表
        """
        # 如果是简单目标，直接返回
        if isinstance(goal, dict) and "type" in goal and goal["type"] == "simple":
            return [goal]
        
        # 使用符号推理进行目标分解
        if isinstance(goal, str):
            # 字符串目标，使用符号推理分析
            goal_facts = self.symbolic_reasoner.extract_facts(goal)
            sub_goals = []
            
            for fact in goal_facts:
                sub_goals.append({"type": "simple", "description": fact})
                
            if not sub_goals:  # 如果无法分解，作为单一目标
                sub_goals = [{"type": "simple", "description": goal}]
        
        elif isinstance(goal, dict):
            # 字典目标，根据结构分解
            if "sub_goals" in goal:
                # 已经包含子目标
                sub_goals = goal["sub_goals"]
            else:
                # 尝试根据键值对分解
                sub_goals = []
                for key, value in goal.items():
                    if key != "type":
                        sub_goals.append({"type": "simple", "attribute": key, "value": value})
        
        else:
            # 默认作为单一目标
            sub_goals = [{"type": "simple", "description": str(goal)}]
        
        # 记录分解历史
        self.goal_decomposition_history.append({
            "original_goal": goal,
            "sub_goals": sub_goals,
            "timestamp": time.time()
        })
        
        return sub_goals
    
    def evaluate_action(self, action_name, result):
        """
        评估行动执行结果
        
        Args:
            action_name: 行动名称
            result: 执行结果
            
        Returns:
            float: 评估分数 (0-1)
        """
        # 初始化行动评估记录
        if action_name not in self.action_evaluations:
            self.action_evaluations[action_name] = {
                "success_count": 0,
                "failure_count": 0,
                "execution_times": [],
                "scores": []
            }
        
        # 判断执行是否成功
        success = result.get("success", False)
        execution_time = result.get("execution_time", 0.0)
        
        # 更新评估记录
        if success:
            self.action_evaluations[action_name]["success_count"] += 1
        else:
            self.action_evaluations[action_name]["failure_count"] += 1
            
        self.action_evaluations[action_name]["execution_times"].append(execution_time)
        
        # 计算评分
        total_executions = self.action_evaluations[action_name]["success_count"] + \
                          self.action_evaluations[action_name]["failure_count"]
                          
        success_rate = self.action_evaluations[action_name]["success_count"] / total_executions \
                      if total_executions > 0 else 0.5
                      
        # 更新成功历史
        if action_name not in self.action_success_history:
            self.action_success_history[action_name] = []
            
        self.action_success_history[action_name].append(success)
        
        # 限制历史记录长度
        max_history = 100
        if len(self.action_success_history[action_name]) > max_history:
            self.action_success_history[action_name] = self.action_success_history[action_name][-max_history:]
        
        # 计算最终评分
        score = 0.7 * success_rate + 0.3 * (1.0 - min(execution_time / 10.0, 1.0))
        self.action_evaluations[action_name]["scores"].append(score)
        
        return score
    
    def predict_action_effect(self, action_name, state):
        """
        预测行动在给定状态下的效果
        
        Args:
            action_name: 行动名称
            state: 当前状态
            
        Returns:
            tuple: (预测状态, 置信度)
        """
        # 检查行动是否存在
        if action_name not in self.planning_engine.actions:
            return state.copy(), 0.0
        
        # 检查行动是否适用
        if not self.planning_engine.is_action_applicable(action_name, state):
            return state.copy(), 0.0
        
        # 获取行动定义的效果
        defined_effects = self.planning_engine.actions[action_name]["effects"]
        
        # 应用定义的效果
        predicted_state = state.copy()
        for key, value in defined_effects.items():
            predicted_state[key] = value
        
        # 基于历史数据调整预测
        if action_name in self.action_effect_predictions:
            history = self.action_effect_predictions[action_name]
            
            # 查找相似状态的历史记录
            similar_states = []
            for entry in history:
                similarity = self._calculate_state_similarity(state, entry["initial_state"])
                if similarity > 0.7:  # 相似度阈值
                    similar_states.append((entry, similarity))
            
            # 如果有相似状态，调整预测
            if similar_states:
                # 按相似度排序
                similar_states.sort(key=lambda x: x[1], reverse=True)
                
                # 取最相似的几个状态
                top_similar = similar_states[:min(3, len(similar_states))]
                
                # 根据相似状态的实际效果调整预测
                for entry, similarity in top_similar:
                    for key, value in entry["actual_effects"].items():
                        if key in predicted_state:
                            # 根据相似度加权调整
                            weight = similarity * 0.3  # 历史数据的影响权重
                            if isinstance(predicted_state[key], (int, float)) and isinstance(value, (int, float)):
                                # 数值型，加权平均
                                predicted_state[key] = (1 - weight) * predicted_state[key] + weight * value
                            elif predicted_state[key] != value and np.random.random() < weight:
                                # 非数值型，概率性替换
                                predicted_state[key] = value
        
        # 计算置信度
        if action_name in self.action_success_history and self.action_success_history[action_name]:
            # 基于历史成功率
            success_rate = sum(1 for result in self.action_success_history[action_name] if result) / \
                          len(self.action_success_history[action_name])
            confidence = 0.5 + 0.5 * success_rate  # 置信度范围：0.5-1.0
        else:
            # 无历史数据
            confidence = 0.5
        
        return predicted_state, confidence
    
    def update_action_effect_prediction(self, action_name, initial_state, predicted_state, actual_state):
        """
        更新行动效果预测模型
        
        Args:
            action_name: 行动名称
            initial_state: 初始状态
            predicted_state: 预测状态
            actual_state: 实际状态
        """
        # 初始化预测历史
        if action_name not in self.action_effect_predictions:
            self.action_effect_predictions[action_name] = []
        
        # 计算实际效果（与初始状态的差异）
        actual_effects = {}
        for key, value in actual_state.items():
            if key not in initial_state or initial_state[key] != value:
                actual_effects[key] = value
        
        # 记录预测结果
        self.action_effect_predictions[action_name].append({
            "initial_state": initial_state.copy(),
            "predicted_state": predicted_state.copy(),
            "actual_state": actual_state.copy(),
            "actual_effects": actual_effects,
            "timestamp": time.time()
        })
        
        # 限制历史记录长度
        max_history = 50
        if len(self.action_effect_predictions[action_name]) > max_history:
            self.action_effect_predictions[action_name] = self.action_effect_predictions[action_name][-max_history:]
    
    def _goal_to_state(self, goal):
        """
        将目标转换为状态表示
        
        Args:
            goal: 目标描述
            
        Returns:
            dict: 状态表示
        """
        if isinstance(goal, dict):
            if "type" in goal and goal["type"] == "simple":
                if "attribute" in goal and "value" in goal:
                    # 属性-值对形式
                    return {goal["attribute"]: goal["value"]}
                elif "description" in goal:
                    # 描述形式，使用符号推理提取状态
                    facts = self.symbolic_reasoner.extract_facts(goal["description"])
                    state = {}
                    for fact in facts:
                        # 简单解析：主语-谓语-宾语
                        parts = fact.split(" ")
                        if len(parts) >= 3:
                            state[parts[0] + "_" + parts[1]] = parts[2]
                    
                    if not state:  # 如果无法解析，使用整体描述
                        state["goal_achieved"] = goal["description"]
                    
                    return state
            else:
                # 移除类型字段，其余作为状态
                state = goal.copy()
                if "type" in state:
                    del state["type"]
                if "sub_goals" in state:
                    del state["sub_goals"]
                return state
        elif isinstance(goal, str):
            # 字符串目标
            return {"goal_achieved": goal}
        else:
            # 默认表示
            return {"goal_achieved": str(goal)}
    
    def _optimize_plan(self, plan, goal):
        """
        优化行动计划
        
        Args:
            plan: 原始行动计划
            goal: 目标
            
        Returns:
            list: 优化后的行动计划
        """
        if not plan:
            return []
        
        # 移除冗余行动
        optimized_plan = []
        current_state = self.core_agent.get_current_state().copy()
        goal_state = self._goal_to_state(goal)
        
        for action in plan:
            # 预测行动效果
            next_state, _ = self.predict_action_effect(action, current_state)
            
            # 检查行动是否必要（是否改变了状态）
            is_necessary = False
            for key, value in next_state.items():
                if key not in current_state or current_state[key] != value:
                    is_necessary = True
                    break
            
            # 如果行动必要，添加到优化计划
            if is_necessary:
                optimized_plan.append(action)
                current_state = next_state
            
            # 检查是否已达成目标
            goal_achieved = True
            for key, value in goal_state.items():
                if key not in current_state or current_state[key] != value:
                    goal_achieved = False
                    break
            
            if goal_achieved:
                break
        
        return optimized_plan
    
    def _calculate_state_similarity(self, state1, state2):
        """
        计算两个状态的相似度
        
        Args:
            state1: 状态1
            state2: 状态2
            
        Returns:
            float: 相似度 (0-1)
        """
        # 获取所有键
        all_keys = set(state1.keys()) | set(state2.keys())
        if not all_keys:
            return 1.0  # 两个空状态
        
        # 计算匹配键的数量
        matching_keys = 0
        for key in all_keys:
            if key in state1 and key in state2:
                if state1[key] == state2[key]:
                    matching_keys += 1
        
        return matching_keys / len(all_keys)