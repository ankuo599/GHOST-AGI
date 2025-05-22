import unittest
import time
from metacognition.cognitive_monitor import CognitiveMonitor

class TestCognitiveMonitor(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.monitor = CognitiveMonitor()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertIsNotNone(self.monitor.cognitive_biases)
        self.assertIsNotNone(self.monitor.knowledge_base)
        self.assertIsNotNone(self.monitor.learning_resources)
        
    def test_reasoning_tracking(self):
        """测试推理跟踪"""
        # 创建测试推理步骤
        steps = [
            {
                "id": "step1",
                "type": "problem_definition",
                "description": "定义问题：如何提高代码质量",
                "output": "需要分析代码质量的关键指标",
                "confidence": 0.8
            },
            {
                "id": "step2",
                "type": "analysis",
                "description": "分析代码质量指标",
                "output": "包括可读性、可维护性、性能等",
                "confidence": 0.7
            },
            {
                "id": "step3",
                "type": "conclusion",
                "description": "得出结论",
                "output": "需要从多个维度提升代码质量",
                "confidence": 0.9
            }
        ]
        
        # 跟踪推理过程
        result = self.monitor.track_reasoning_process("test_reasoning", steps)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertIn("reasoning_id", result)
        self.assertIn("confidence", result)
        self.assertIn("quality_score", result)
        
    def test_cognitive_bias_detection(self):
        """测试认知偏差检测"""
        # 创建包含偏差的推理步骤
        steps = [
            {
                "id": "step1",
                "type": "assumption",
                "description": "我确定这个方案一定可行",
                "output": "方案可行性分析",
                "confidence": 0.95
            },
            {
                "id": "step2",
                "type": "analysis",
                "description": "只考虑支持这个方案的数据",
                "output": "数据支持分析",
                "confidence": 0.8
            }
        ]
        
        # 检测偏差
        biases = self.monitor.detect_cognitive_biases(steps)
        
        # 验证结果
        self.assertGreater(len(biases), 0)
        self.assertIn("bias_type", biases[0])
        self.assertIn("severity", biases[0])
        
    def test_continuous_learning(self):
        """测试持续学习"""
        # 启动持续学习
        self.monitor.start_continuous_learning()
        
        # 创建学习需求
        learning_need = {
            "type": "concept",
            "target": "机器学习",
            "priority": 0.8
        }
        
        # 识别学习需求
        needs = self.monitor.identify_learning_needs({"active_concepts": ["机器学习"]})
        
        # 验证结果
        self.assertGreater(len(needs), 0)
        self.assertEqual(needs[0]["type"], "concept")
        self.assertEqual(needs[0]["target"], "机器学习")
        
    def test_knowledge_management(self):
        """测试知识管理"""
        # 添加测试概念
        concept_content = {
            "title": "测试概念",
            "content": {
                "summary": "这是一个测试概念",
                "examples": ["示例1", "示例2"],
                "categories": ["测试", "概念"]
            }
        }
        
        # 学习概念
        self.monitor._learn_concept(concept_content)
        
        # 验证知识库
        self.assertIn("测试概念", self.monitor.knowledge_base["concepts"])
        concept_data = self.monitor.knowledge_base["concepts"]["测试概念"]
        self.assertEqual(concept_data["definition"], "这是一个测试概念")
        self.assertEqual(len(concept_data["examples"]), 2)
        
    def test_skill_learning(self):
        """测试技能学习"""
        # 添加测试技能
        skill_content = {
            "title": "测试技能",
            "content": {
                "description": "这是一个测试技能",
                "examples": ["示例1", "示例2"],
                "requirements": ["要求1", "要求2"]
            }
        }
        
        # 学习技能
        self.monitor._learn_skill(skill_content)
        
        # 验证知识库
        self.assertIn("测试技能", self.monitor.knowledge_base["skills"])
        skill_data = self.monitor.knowledge_base["skills"]["测试技能"]
        self.assertEqual(skill_data["description"], "这是一个测试技能")
        self.assertEqual(len(skill_data["examples"]), 2)
        
    def test_practice_and_evaluation(self):
        """测试练习和评估"""
        # 创建练习内容
        practice_content = {
            "type": "practice",
            "subtype": "basic",
            "target": "测试技能",
            "question": "完成基础练习",
            "answer": "练习答案"
        }
        
        # 执行练习
        result = self.monitor._execute_practice(practice_content)
        
        # 评估结果
        evaluation = self.monitor._evaluate_practice_result(result, practice_content)
        
        # 验证结果
        self.assertIn("status", evaluation)
        self.assertIn("score", evaluation)
        self.assertIn("improvement", evaluation)
        self.assertIn("feedback", evaluation)
        
    def test_metacognitive_state(self):
        """测试元认知状态"""
        # 更新元认知状态
        self.monitor.update_metacognitive_state()
        
        # 验证状态
        self.assertIn("awareness_level", self.monitor.metacognitive_state)
        self.assertIn("self_regulation", self.monitor.metacognitive_state)
        self.assertIn("learning_progress", self.monitor.metacognitive_state)
        self.assertIn("adaptation_capability", self.monitor.metacognitive_state)
        
    def test_learning_resources(self):
        """测试学习资源"""
        # 测试Wikipedia资源获取
        wiki_resources = self.monitor._fetch_wikipedia_resources({"target": "人工智能"})
        self.assertIsInstance(wiki_resources, list)
        
        # 测试GitHub资源获取
        github_resources = self.monitor._fetch_github_resources({"target": "python"})
        self.assertIsInstance(github_resources, list)
        
    def test_learning_progress(self):
        """测试学习进度"""
        # 创建学习任务
        learning_task = {
            "id": "test_task",
            "type": "concept",
            "target": "测试概念",
            "status": "started",
            "start_time": time.time(),
            "resources": [],
            "progress": 0.0
        }
        
        # 更新进度
        self.monitor._update_learning_progress(learning_task)
        
        # 验证进度
        self.assertGreaterEqual(learning_task["progress"], 0.0)
        self.assertLessEqual(learning_task["progress"], 1.0)
        
    def test_learning_effectiveness(self):
        """测试学习效果"""
        # 创建学习任务
        learning_task = {
            "id": "test_task",
            "type": "concept",
            "target": "测试概念",
            "status": "started",
            "start_time": time.time(),
            "resources": [],
            "progress": 0.8
        }
        
        # 评估学习效果
        effectiveness = self.monitor._evaluate_learning_effectiveness(learning_task)
        
        # 验证结果
        self.assertIsInstance(effectiveness, bool)
        
if __name__ == '__main__':
    unittest.main() 