# -*- coding: utf-8 -*-
"""
目标冲突检测器 (Goal Conflict Detector)

提供智能体目标冲突检测的算法和工具
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set

class GoalConflictDetector:
    def __init__(self):
        # 目标类型映射表
        self.goal_types = {}
        # 冲突矩阵 - 记录哪些目标类型互相冲突
        self.conflict_matrix = {}
        # 资源需求映射 - 记录每种目标类型需要的资源
        self.resource_requirements = {}
        
    def register_goal_type(self, goal_type: str, resource_requirements: Dict[str, float] = None):
        """
        注册目标类型及其资源需求
        
        Args:
            goal_type: 目标类型标识
            resource_requirements: 资源需求 {resource_id: amount}
        """
        self.goal_types[goal_type] = {
            "id": goal_type,
            "registered_at": np.datetime64('now')
        }
        
        if resource_requirements:
            self.resource_requirements[goal_type] = resource_requirements
            
    def register_conflict(self, goal_type1: str, goal_type2: str, conflict_degree: float = 1.0):
        """
        注册两个目标类型之间的冲突关系
        
        Args:
            goal_type1: 目标类型1
            goal_type2: 目标类型2
            conflict_degree: 冲突程度 (0-1)
        """
        if goal_type1 not in self.conflict_matrix:
            self.conflict_matrix[goal_type1] = {}
            
        if goal_type2 not in self.conflict_matrix:
            self.conflict_matrix[goal_type2] = {}
            
        self.conflict_matrix[goal_type1][goal_type2] = conflict_degree
        self.conflict_matrix[goal_type2][goal_type1] = conflict_degree
        
    def detect_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        检测两个目标之间是否存在冲突
        
        Args:
            goal1: 目标1信息
            goal2: 目标2信息
            
        Returns:
            Tuple[bool, float, str]: (是否冲突, 冲突程度, 冲突原因)
        """
        # 提取目标类型
        goal_type1 = goal1.get("type", "unknown")
        goal_type2 = goal2.get("type", "unknown")
        
        # 检查直接冲突关系
        if goal_type1 in self.conflict_matrix and goal_type2 in self.conflict_matrix[goal_type1]:
            conflict_degree = self.conflict_matrix[goal_type1][goal_type2]
            if conflict_degree > 0:
                return True, conflict_degree, "目标类型直接冲突"
                
        # 检查资源冲突
        if goal_type1 in self.resource_requirements and goal_type2 in self.resource_requirements:
            resources1 = self.resource_requirements[goal_type1]
            resources2 = self.resource_requirements[goal_type2]
            
            # 找出共同需要的资源
            common_resources = set(resources1.keys()) & set(resources2.keys())
            
            if common_resources:
                # 计算资源需求总量是否超过阈值
                for resource in common_resources:
                    if resources1[resource] + resources2[resource] > 1.0:  # 假设1.0为资源上限
                        return True, 0.8, f"资源冲突: {resource}"
                        
        # 检查目标属性冲突
        if "attributes" in goal1 and "attributes" in goal2:
            attrs1 = goal1["attributes"]
            attrs2 = goal2["attributes"]
            
            # 检查互斥属性
            for attr, value in attrs1.items():
                if attr in attrs2 and attrs2[attr] != value and attr.startswith("exclusive_"):
                    return True, 0.9, f"互斥属性冲突: {attr}"
                    
        # 检查时间冲突
        if "timeframe" in goal1 and "timeframe" in goal2:
            time1 = goal1["timeframe"]
            time2 = goal2["timeframe"]
            
            # 简单的时间重叠检查
            if time1["start"] < time2["end"] and time2["start"] < time1["end"]:
                # 检查是否需要独占时间
                if goal1.get("exclusive_time", False) or goal2.get("exclusive_time", False):
                    return True, 0.7, "时间冲突"
                    
        return False, 0.0, "无冲突"
        
    def analyze_multiple_goals(self, goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析多个目标之间的冲突关系
        
        Args:
            goals: 目标列表
            
        Returns:
            Dict[str, Any]: 冲突分析结果
        """
        conflicts = []
        conflict_graph = {}
        
        # 初始化冲突图
        for i, goal in enumerate(goals):
            goal_id = goal.get("id", f"goal_{i}")
            conflict_graph[goal_id] = []
            
        # 检测所有目标对之间的冲突
        for i in range(len(goals)):
            for j in range(i+1, len(goals)):
                goal1 = goals[i]
                goal2 = goals[j]
                
                goal_id1 = goal1.get("id", f"goal_{i}")
                goal_id2 = goal2.get("id", f"goal_{j}")
                
                is_conflict, degree, reason = self.detect_conflict(goal1, goal2)
                
                if is_conflict:
                    conflict_info = {
                        "goal1": goal_id1,
                        "goal2": goal_id2,
                        "degree": degree,
                        "reason": reason
                    }
                    conflicts.append(conflict_info)
                    
                    # 更新冲突图
                    conflict_graph[goal_id1].append((goal_id2, degree))
                    conflict_graph[goal_id2].append((goal_id1, degree))
                    
        # 计算每个目标的冲突度
        goal_conflict_scores = {}
        for goal_id, conflicts_list in conflict_graph.items():
            if conflicts_list:
                # 计算平均冲突度
                avg_degree = sum(degree for _, degree in conflicts_list) / len(conflicts_list)
                goal_conflict_scores[goal_id] = avg_degree
            else:
                goal_conflict_scores[goal_id] = 0.0
                
        return {
            "conflicts": conflicts,
            "conflict_graph": conflict_graph,
            "goal_conflict_scores": goal_conflict_scores,
            "total_conflicts": len(conflicts)
        }
        
    def suggest_resolution(self, goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        为冲突的目标提供解决建议
        
        Args:
            goals: 目标列表
            
        Returns:
            Dict[str, Any]: 解决建议
        """
        analysis = self.analyze_multiple_goals(goals)
        
        if not analysis["conflicts"]:
            return {"status": "no_conflict", "message": "没有检测到冲突"}
            
        # 根据冲突度排序目标
        sorted_goals = sorted(
            analysis["goal_conflict_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 找出冲突最多的目标
        most_conflicting = sorted_goals[0][0]
        
        # 找出与该目标冲突的所有目标
        conflicting_goals = []
        for conflict in analysis["conflicts"]:
            if conflict["goal1"] == most_conflicting:
                conflicting_goals.append(conflict["goal2"])
            elif conflict["goal2"] == most_conflicting:
                conflicting_goals.append(conflict["goal1"])
                
        return {
            "status": "conflict_detected",
            "most_conflicting_goal": most_conflicting,
            "conflicting_goals": conflicting_goals,
            "suggestion": "考虑移除或推迟最具冲突性的目标",
            "alternative_suggestion": "尝试重新安排资源分配以减少冲突"
        }