"""
测试自主进化模块的功能
"""

import unittest
import os
import shutil
import time
from metacognition.self_evolution import SelfEvolutionEngine

class TestSelfEvolution(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.test_backup_dir = "test_evolution_backups"
        self.engine = SelfEvolutionEngine(backup_dir=self.test_backup_dir)
        
        # 创建测试文件
        self.test_file = "test_module.py"
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("""
# 测试模块
# TODO: 这是一个待办事项
def test_function():
    # TODO: 这是另一个待办事项
    pass
""")

    def tearDown(self):
        """测试后的清理工作"""
        if os.path.exists(self.test_backup_dir):
            shutil.rmtree(self.test_backup_dir)
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_backup_code(self):
        """测试代码备份功能"""
        result = self.engine.backup_code("test_module")
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_evolve_code(self):
        """测试代码进化功能"""
        def evolution_fn(code):
            return code.replace("TODO", "已处理")
            
        result = self.engine.evolve_code("test_module", evolution_fn)
        self.assertEqual(result["status"], "success")
        
        # 验证代码是否被修改
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertNotIn("TODO", content)
            self.assertIn("已处理", content)

    def test_restore_backup(self):
        """测试恢复备份功能"""
        # 先创建备份
        backup = self.engine.backup_code("test_module")
        self.assertIsNotNone(backup)
        
        # 修改文件
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("modified content")
            
        # 恢复备份
        result = self.engine.restore_backup("test_module")
        self.assertEqual(result["status"], "success")
        
        # 验证内容是否恢复
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("TODO", content)

    def test_auto_evolve_all(self):
        """测试自动进化所有模块功能"""
        def evolution_fn(code):
            return code.replace("TODO", "已处理")
            
        results = self.engine.auto_evolve_all(evolution_fn)
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)

    def test_evaluate_evolution(self):
        """测试进化评估功能"""
        result = self.engine.evaluate_evolution()
        self.assertIn("status", result)

if __name__ == "__main__":
    unittest.main() 