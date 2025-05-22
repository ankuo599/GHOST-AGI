# -*- coding: utf-8 -*-
"""
目标冲突检测器测试脚本

用于测试增强版分布式协作系统中的目标冲突检测功能
"""

import unittest
from enhanced_collaboration_system import EnhancedCollaborationSystem
from goal_conflict_detector import GoalConflictDetector

class TestGoalConflictDetection(unittest.TestCase):
    def setUp(self):
        # 初始化测试环境
        self.system = EnhancedCollaborationSystem(instance_id="test_instance")
        
    def test_goal_conflict_detection(self):
        # 测试目标冲突检测功能
        # 创建两个冲突的目标
        goal1 = {
            "id": "goal1",
            "type": "resource_allocation",
            "resources": [
                {"id": "cpu", "exclusive": True},
                {"id": "memory", "exclusive": False}
            ],
            "priority": "high"
        }
        
        goal2 = {
            "id": "goal2",
            "type": "resource_conservation",
            "resources": [
                {"id": "cpu", "exclusive": True},
                {"id": "disk", "exclusive": False}
            ],
            "priority": "high"
        }
        
        # 检测冲突
        is_conflict = self.system._are_goals_conflicting(goal1, goal2)
        self.assertTrue(is_conflict)
        
        # 测试非冲突目标
        goal3 = {
            "id": "goal3",
            "type": "data_processing",
            "resources": [
                {"id": "disk", "exclusive": False},
                {"id": "network", "exclusive": False}
            ],
            "priority": "low"
        }
        
        is_conflict = self.system._are_goals_conflicting(goal1, goal3)
        self.assertFalse(is_conflict)
        
        # 测试简单字符串目标
        is_conflict = self.system._are_goals_conflicting("maximize_profit", "minimize_cost")
        self.assertTrue(is_conflict)
        
        is_conflict = self.system._are_goals_conflicting("maximize_profit", "increase_sales")
        self.assertFalse(is_conflict)
        
    def test_agent_priority(self):
        # 测试智能体优先级设置和获取
        agent_id = "agent1"
        priority = 1
        
        # 设置优先级
        success = self.system.set_agent_priority(agent_id, priority)
        self.assertTrue(success)
        
        # 获取优先级
        agent_priority = self.system.get_agent_priority(agent_id)
        self.assertEqual(agent_priority, priority)
        
        # 测试未设置优先级的情况
        agent_priority = self.system.get_agent_priority("unknown_agent")
        self.assertIsNone(agent_priority)
        
    def test_conflict_resolution_with_priority(self):
        # 测试基于优先级的冲突解决
        # 设置智能体优先级
        self.system.set_agent_priority("agent1", 1)  # 高优先级
        self.system.set_agent_priority("agent2", 2)  # 低优先级
        
        # 创建冲突
        conflict = {
            "id": "conflict1",
            "task_id": "task1",
            "type": "resource_write_conflict",
            "agents": ["agent1", "agent2"],
            "details": {"resource_id": "resource1", "access_type": "write"},
            "detected_at": 0,
            "resolved": False,
            "resolution": None
        }
        
        # 解决冲突
        resolution = self.system.resolve_conflict(conflict)
        
        # 验证结果
        self.assertEqual(resolution["status"], "success")
        self.assertEqual(resolution["strategy"], "priority_based")
        self.assertEqual(resolution["selected_agent"], "agent1")

if __name__ == "__main__":
    unittest.main()