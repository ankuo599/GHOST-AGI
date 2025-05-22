# -*- coding: utf-8 -*-
"""
分布式协作系统全面功能测试脚本

用于测试分布式协作系统的所有功能，包括：
- 基础协作系统功能
- 增强版协作系统功能
- 基于能力的任务分配
- 智能体间知识共享
- 冲突检测与解决
- 协作绩效评估
- 目标冲突检测
- 智能体优先级
"""

import unittest
import time
import logging
import sys

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 导入测试所需的模块
from collaboration_system import CollaborationSystem
from enhanced_collaboration_system import EnhancedCollaborationSystem
from goal_conflict_detector import GoalConflictDetector

class TestAllFeatures(unittest.TestCase):
    def setUp(self):
        # 初始化测试环境
        self.logger = logging.getLogger("TestAllFeatures")
        self.logger.info("初始化测试环境")
        
        # 创建基础协作系统
        self.base_system = CollaborationSystem(instance_id="base_system")
        
        # 创建增强版协作系统
        self.enhanced_system = EnhancedCollaborationSystem(instance_id="enhanced_system")
        
        # 创建目标冲突检测器
        self.conflict_detector = GoalConflictDetector()
        
    def tearDown(self):
        # 清理测试环境
        self.logger.info("清理测试环境")
        if hasattr(self, 'base_system') and self.base_system.is_running:
            self.base_system.stop()
            
        if hasattr(self, 'enhanced_system') and self.enhanced_system.is_running:
            self.enhanced_system.stop()
    
    def test_01_base_system_initialization(self):
        """测试基础协作系统初始化"""
        self.logger.info("测试基础协作系统初始化")
        
        # 验证实例ID
        self.assertIsNotNone(self.base_system.instance_id)
        
        # 验证初始状态
        self.assertFalse(self.base_system.is_running)
        self.assertFalse(self.base_system.is_coordinator)
        self.assertEqual(len(self.base_system.capabilities), 0)
        
        self.logger.info("基础协作系统初始化测试通过")
    
    def test_02_enhanced_system_initialization(self):
        """测试增强版协作系统初始化"""
        self.logger.info("测试增强版协作系统初始化")
        
        # 验证实例ID
        self.assertIsNotNone(self.enhanced_system.instance_id)
        
        # 验证初始状态
        self.assertFalse(self.enhanced_system.is_running)
        self.assertFalse(self.enhanced_system.is_coordinator)
        self.assertEqual(len(self.enhanced_system.capabilities), 0)
        
        # 验证增强功能初始化
        self.assertTrue(self.enhanced_system.task_allocation["active"])
        self.assertTrue(self.enhanced_system.knowledge_sharing["active"])
        self.assertTrue(self.enhanced_system.conflict_resolution["active"])
        self.assertTrue(self.enhanced_system.collaboration_metrics["active"])
        
        # 验证目标冲突检测器
        self.assertIsNotNone(self.enhanced_system.goal_conflict_detector)
        
        self.logger.info("增强版协作系统初始化测试通过")
    
    def test_03_capability_registration(self):
        """测试能力注册功能"""
        self.logger.info("测试能力注册功能")
        
        # 注册能力
        self.base_system.register_capability("data_processing")
        self.base_system.register_capability("image_recognition")
        
        # 验证能力注册
        self.assertEqual(len(self.base_system.capabilities), 2)
        self.assertIn("data_processing", self.base_system.capabilities)
        self.assertIn("image_recognition", self.base_system.capabilities)
        
        self.logger.info("能力注册功能测试通过")
    
    def test_04_task_allocation(self):
        """测试基于能力的任务分配"""
        self.logger.info("测试基于能力的任务分配")
        
        # 注册能力
        self.enhanced_system.register_capability("data_processing")
        self.enhanced_system.register_capability("image_recognition")
        
        # 模拟已知实例
        self.enhanced_system.known_instances = {
            "agent1": {"status": "active"},
            "agent2": {"status": "active"}
        }
        
        # 设置实例能力
        self.enhanced_system.instance_capabilities = {
            "agent1": {"data_processing"},
            "agent2": {"image_recognition", "data_processing"}
        }
        
        # 更新能力评分
        self.enhanced_system.update_capability_score("agent1", "data_processing", 0.8)
        self.enhanced_system.update_capability_score("agent2", "data_processing", 0.6)
        self.enhanced_system.update_capability_score("agent2", "image_recognition", 0.9)
        
        # 分配任务
        task_id = "task1"
        task_type = "image_analysis"
        task_data = {"image_url": "http://example.com/image.jpg"}
        required_capabilities = ["image_recognition"]
        
        agent_id = self.enhanced_system.allocate_task(task_id, task_type, task_data, required_capabilities)
        self.assertEqual(agent_id, "agent2")
        
        self.logger.info("基于能力的任务分配测试通过")
    
    def test_05_knowledge_sharing(self):
        """测试智能体间知识共享"""
        self.logger.info("测试智能体间知识共享")
        
        agent_id = "agent1"
        knowledge_id = "knowledge1"
        knowledge_data = {"key": "value"}
        
        # 共享知识
        success = self.enhanced_system.share_knowledge(agent_id, knowledge_id, knowledge_data)
        self.assertTrue(success)
        
        # 访问知识
        data = self.enhanced_system.access_shared_knowledge("agent2", knowledge_id)
        self.assertEqual(data, knowledge_data)
        
        # 测试访问控制
        knowledge_id2 = "knowledge2"
        knowledge_data2 = {"restricted": "data"}
        access_control = ["agent1"]
        
        self.enhanced_system.share_knowledge(agent_id, knowledge_id2, knowledge_data2, access_control)
        
        # agent1可以访问
        data = self.enhanced_system.access_shared_knowledge("agent1", knowledge_id2)
        self.assertEqual(data, knowledge_data2)
        
        # agent2无法访问
        data = self.enhanced_system.access_shared_knowledge("agent2", knowledge_id2)
        self.assertIsNone(data)
        
        self.logger.info("智能体间知识共享测试通过")
    
    def test_06_conflict_detection(self):
        """测试冲突检测功能"""
        self.logger.info("测试冲突检测功能")
        
        task_id = "task1"
        
        # 资源写冲突
        agent_actions = {
            "agent1": {
                "resources": [
                    {"id": "resource1", "access_type": "write"}
                ]
            },
            "agent2": {
                "resources": [
                    {"id": "resource1", "access_type": "write"}
                ]
            }
        }
        
        conflict = self.enhanced_system.detect_conflict(task_id, agent_actions)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["type"], "resource_write_conflict")
        
        # 读写冲突
        agent_actions = {
            "agent1": {
                "resources": [
                    {"id": "resource1", "access_type": "read"}
                ]
            },
            "agent2": {
                "resources": [
                    {"id": "resource1", "access_type": "write"}
                ]
            }
        }
        
        conflict = self.enhanced_system.detect_conflict(task_id, agent_actions)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["type"], "resource_read_write_conflict")
        
        self.logger.info("冲突检测功能测试通过")
    
    def test_07_conflict_resolution(self):
        """测试冲突解决功能"""
        self.logger.info("测试冲突解决功能")
        
        # 设置智能体优先级
        self.enhanced_system.set_agent_priority("agent1", 1)  # 高优先级
        self.enhanced_system.set_agent_priority("agent2", 2)  # 低优先级
        
        # 创建冲突
        conflict = {
            "id": "conflict1",
            "task_id": "task1",
            "type": "resource_write_conflict",
            "agents": ["agent1", "agent2"],
            "details": {"resource_id": "resource1", "access_type": "write"},
            "detected_at": time.time(),
            "resolved": False,
            "resolution": None
        }
        
        # 解决冲突
        resolution = self.enhanced_system.resolve_conflict(conflict)
        self.assertIsNotNone(resolution)
        self.assertTrue(resolution["resolved"])
        self.assertEqual(resolution["winner"], "agent1")
        
        self.logger.info("冲突解决功能测试通过")
    
    def test_08_goal_conflict_detection(self):
        """测试目标冲突检测功能"""
        self.logger.info("测试目标冲突检测功能")
        
        # 注册目标类型
        self.conflict_detector.register_goal_type("resource_allocation", {"cpu": 0.8, "memory": 0.5})
        self.conflict_detector.register_goal_type("resource_conservation", {"cpu": 0.3, "disk": 0.7})
        
        # 注册冲突关系
        self.conflict_detector.register_conflict("resource_allocation", "resource_conservation", 0.7)
        
        # 创建目标
        goal1 = {
            "id": "goal1",
            "type": "resource_allocation",
            "attributes": {"exclusive_mode": "high_performance"}
        }
        
        goal2 = {
            "id": "goal2",
            "type": "resource_conservation",
            "attributes": {"exclusive_mode": "energy_saving"}
        }
        
        # 检测冲突
        is_conflict, conflict_degree, reason = self.conflict_detector.detect_conflict(goal1, goal2)
        self.assertTrue(is_conflict)
        self.assertGreater(conflict_degree, 0)
        self.assertIsNotNone(reason)
        
        self.logger.info("目标冲突检测功能测试通过")
    
    def test_09_agent_priority(self):
        """测试智能体优先级功能"""
        self.logger.info("测试智能体优先级功能")
        
        agent_id = "agent1"
        priority = 1
        
        # 设置优先级
        success = self.enhanced_system.set_agent_priority(agent_id, priority)
        self.assertTrue(success)
        
        # 获取优先级
        agent_priority = self.enhanced_system.get_agent_priority(agent_id)
        self.assertEqual(agent_priority, priority)
        
        # 测试未设置优先级的情况
        agent_priority = self.enhanced_system.get_agent_priority("unknown_agent")
        self.assertIsNone(agent_priority)
        
        self.logger.info("智能体优先级功能测试通过")
    
    def test_10_collaboration_evaluation(self):
        """测试协作绩效评估功能"""
        self.logger.info("测试协作绩效评估功能")
        
        # 设置评估指标
        metrics = {
            "task_completion_rate": 0.85,
            "average_response_time": 1.2,
            "conflict_resolution_rate": 0.95
        }
        
        # 评估协作绩效
        result = self.enhanced_system.evaluate_collaboration(metrics)
        self.assertIsNotNone(result)
        self.assertIn("score", result)
        self.assertIn("details", result)
        
        self.logger.info("协作绩效评估功能测试通过")
    
    def test_11_system_integration(self):
        """测试系统集成功能"""
        self.logger.info("测试系统集成功能")
        
        # 启动系统
        self.enhanced_system.start()
        self.assertTrue(self.enhanced_system.is_running)
        
        # 注册能力
        self.enhanced_system.register_capability("data_processing")
        self.enhanced_system.register_capability("image_recognition")
        
        # 设置智能体优先级
        self.enhanced_system.set_agent_priority("agent1", 1)
        self.enhanced_system.set_agent_priority("agent2", 2)
        
        # 模拟已知实例
        self.enhanced_system.known_instances = {
            "agent1": {"status": "active"},
            "agent2": {"status": "active"}
        }
        
        # 设置实例能力
        self.enhanced_system.instance_capabilities = {
            "agent1": {"data_processing"},
            "agent2": {"image_recognition", "data_processing"}
        }
        
        # 提交任务
        task_id = self.enhanced_system.submit_task("image_analysis", {"url": "http://example.com/image.jpg"}, 
                                                required_capabilities=["image_recognition"])
        self.assertIsNotNone(task_id)
        
        # 共享知识
        success = self.enhanced_system.share_knowledge("agent1", "knowledge1", {"key": "value"})
        self.assertTrue(success)
        
        # 停止系统
        self.enhanced_system.stop()
        self.assertFalse(self.enhanced_system.is_running)
        
        self.logger.info("系统集成功能测试通过")

def run_tests():
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestAllFeatures))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n测试结果摘要:")
    print(f"运行测试用例数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    return result

if __name__ == "__main__":
    print("开始测试分布式协作系统的所有功能...\n")
    result = run_tests()
    
    # 根据测试结果设置退出码
    if len(result.failures) > 0 or len(result.errors) > 0:
        sys.exit(1)
    else:
        print("\n所有测试通过!")
        sys.exit(0)