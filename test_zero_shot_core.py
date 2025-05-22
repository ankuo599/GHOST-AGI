#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GHOST AGI 零样本学习核心功能测试脚本

测试改进的零样本学习模块、知识图谱功能和类比推理能力
"""

import os
import sys
import time
import unittest
from typing import Dict, Any, List

# 添加当前目录到路径，使能够导入主项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主要模块
from memory.vector_store import VectorStore
from learning.zero_shot_learning import ZeroShotLearningModule
from learning.integrator import LearningIntegrator
from utils.event_system import EventSystem

class TestZeroShotLearning(unittest.TestCase):
    """测试零样本学习功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 初始化事件系统
        self.event_system = EventSystem()
        
        # 初始化向量存储
        self.vector_store = VectorStore(dimension=768)
        
        # 初始化零样本学习模块
        self.zero_shot = ZeroShotLearningModule(
            vector_store=self.vector_store,
            event_system=self.event_system
        )
        
        # 初始化学习集成器
        self.integrator = LearningIntegrator(
            vector_store=self.vector_store,
            event_system=self.event_system
        )
        
        # 手动添加零样本模块到集成器
        self.integrator.learning_modules["zero_shot"] = self.zero_shot
        self.integrator.strategy_to_module["zero_shot_classification"] = "zero_shot"
        self.integrator.strategy_to_module["zero_shot_generation"] = "zero_shot"
        self.integrator.strategy_to_module["zero_shot_relation"] = "zero_shot"
        self.integrator.strategy_to_module["zero_shot_analogy"] = "zero_shot"
        
        # 初始化测试用知识图谱
        self._initialize_test_knowledge_graph()
        
    def _initialize_test_knowledge_graph(self):
        """初始化测试用知识图谱"""
        # 添加基础概念
        concepts = [
            # 技术领域概念
            {"name": "技术", "properties": {"domain": "general", "abstract": True}},
            {"name": "编程语言", "properties": {"domain": "technology"}},
            {"name": "Python", "properties": {"domain": "technology", "year": 1991}},
            {"name": "JavaScript", "properties": {"domain": "technology", "year": 1995}},
            {"name": "人工智能", "properties": {"domain": "technology"}},
            {"name": "机器学习", "properties": {"domain": "technology"}},
            {"name": "深度学习", "properties": {"domain": "technology"}},
            
            # 动物领域概念
            {"name": "动物", "properties": {"domain": "biology", "abstract": True}},
            {"name": "哺乳动物", "properties": {"domain": "biology"}},
            {"name": "猫", "properties": {"domain": "biology", "lifespan": 15}},
            {"name": "狗", "properties": {"domain": "biology", "lifespan": 12}},
            {"name": "鸟类", "properties": {"domain": "biology"}},
            {"name": "鹦鹉", "properties": {"domain": "biology", "can_talk": True}},
            {"name": "鹰", "properties": {"domain": "biology", "can_fly": True}},
            
            # 地理领域概念
            {"name": "地点", "properties": {"domain": "geography", "abstract": True}},
            {"name": "国家", "properties": {"domain": "geography"}},
            {"name": "中国", "properties": {"domain": "geography", "continent": "亚洲"}},
            {"name": "美国", "properties": {"domain": "geography", "continent": "北美洲"}},
            {"name": "城市", "properties": {"domain": "geography"}},
            {"name": "北京", "properties": {"domain": "geography", "capital": True}},
            {"name": "上海", "properties": {"domain": "geography", "capital": False}},
            {"name": "纽约", "properties": {"domain": "geography", "capital": False}}
        ]
        
        # 添加概念
        concept_ids = {}
        for concept in concepts:
            concept_id = self.vector_store.add_concept(
                concept_name=concept["name"],
                properties=concept["properties"]
            )
            concept_ids[concept["name"]] = concept_id
            
        # 添加关系
        relations = [
            # 技术领域关系
            {"source": "编程语言", "target": "技术", "type": "is_a"},
            {"source": "Python", "target": "编程语言", "type": "is_a"},
            {"source": "JavaScript", "target": "编程语言", "type": "is_a"},
            {"source": "人工智能", "target": "技术", "type": "is_a"},
            {"source": "机器学习", "target": "人工智能", "type": "is_a"},
            {"source": "深度学习", "target": "机器学习", "type": "is_a"},
            
            # 动物领域关系
            {"source": "哺乳动物", "target": "动物", "type": "is_a"},
            {"source": "鸟类", "target": "动物", "type": "is_a"},
            {"source": "猫", "target": "哺乳动物", "type": "is_a"},
            {"source": "狗", "target": "哺乳动物", "type": "is_a"},
            {"source": "鹦鹉", "target": "鸟类", "type": "is_a"},
            {"source": "鹰", "target": "鸟类", "type": "is_a"},
            
            # 地理领域关系
            {"source": "国家", "target": "地点", "type": "is_a"},
            {"source": "城市", "target": "地点", "type": "is_a"},
            {"source": "中国", "target": "国家", "type": "is_a"},
            {"source": "美国", "target": "国家", "type": "is_a"},
            {"source": "北京", "target": "城市", "type": "is_a"},
            {"source": "上海", "target": "城市", "type": "is_a"},
            {"source": "纽约", "target": "城市", "type": "is_a"},
            {"source": "北京", "target": "中国", "type": "part_of"},
            {"source": "上海", "target": "中国", "type": "part_of"},
            {"source": "纽约", "target": "美国", "type": "part_of"}
        ]
        
        for relation in relations:
            source_id = concept_ids.get(relation["source"])
            target_id = concept_ids.get(relation["target"])
            
            if source_id and target_id:
                self.vector_store.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation["type"]
                )
                
    def test_zero_shot_classification(self):
        """测试零样本分类功能"""
        # 准备测试数据
        test_cases = [
            {
                "input": "我正在学习Python编程",
                "target_classes": ["技术", "音乐", "体育", "医学"],
                "expected": "技术"
            },
            {
                "input": "我家的猫咪今天吃了很多东西",
                "target_classes": ["动物", "植物", "技术", "地点"],
                "expected": "动物"
            },
            {
                "input": "我去年去北京旅游了两周",
                "target_classes": ["地点", "时间", "事件", "技术"],
                "expected": "地点"
            }
        ]
        
        for case in test_cases:
            # 构建请求
            query = {
                "task_type": "classification",
                "data": {"input": case["input"]},
                "context": {"target_classes": case["target_classes"]}
            }
            
            # 调用零样本分类
            result = self.zero_shot.zero_shot_inference(query)
            
            # 验证结果
            self.assertEqual(result["status"], "success", f"分类失败: {case['input']}")
            self.assertEqual(result["predicted_class"], case["expected"], 
                            f"分类错误: 预期 {case['expected']}, 实际 {result['predicted_class']}")
    
    def test_zero_shot_relation_prediction(self):
        """测试零样本关系预测功能"""
        # 准备测试数据
        test_cases = [
            {
                "entity1": "Python",
                "entity2": "编程语言",
                "expected_relation": "is_a"
            },
            {
                "entity1": "北京",
                "entity2": "中国",
                "expected_relation": "part_of"
            },
            {
                "entity1": "猫",
                "entity2": "动物",
                "expected_relation": "is_a"  # 通过哺乳动物间接关系
            },
            {
                "entity1": "JavaScript", 
                "entity2": "Python",
                "expected_relationship_type": "相似概念"  # 都是编程语言
            }
        ]
        
        for case in test_cases:
            # 构建请求
            query = {
                "task_type": "relation_prediction",
                "data": {
                    "entity1": case["entity1"],
                    "entity2": case["entity2"]
                },
                "context": {}
            }
            
            # 调用零样本关系预测
            result = self.zero_shot.zero_shot_inference(query)
            
            # 验证结果
            self.assertEqual(result["status"], "success", f"关系预测失败: {case['entity1']} -> {case['entity2']}")
            
            # 检查是否发现了预期关系
            if "expected_relation" in case:
                found_relation = False
                for relation in result.get("relations", []):
                    if relation.get("relation") == case["expected_relation"]:
                        found_relation = True
                        break
                
                self.assertTrue(found_relation, 
                               f"未找到预期关系 {case['expected_relation']}: {case['entity1']} -> {case['entity2']}")
    
    def test_zero_shot_analogy(self):
        """测试零样本类比推理功能"""
        # 准备测试数据
        test_cases = [
            {
                "a": "北京",
                "b": "中国",
                "c": "纽约",
                "expected": "美国"  # 类比: 北京:中国::纽约:?
            },
            {
                "a": "Python",
                "b": "编程语言",
                "c": "猫",
                "expected": "哺乳动物"  # 类比: Python:编程语言::猫:?
            },
            {
                "a": "深度学习",
                "b": "机器学习",
                "c": "机器学习",
                "expected": "人工智能"  # 类比: 深度学习:机器学习::机器学习:?
            }
        ]
        
        for case in test_cases:
            # 构建请求
            query = {
                "task_type": "analogical_reasoning",
                "data": {
                    "term_a": case["a"],
                    "term_b": case["b"],
                    "term_c": case["c"]
                },
                "context": {}
            }
            
            # 调用零样本类比推理
            result = self.zero_shot.zero_shot_inference(query)
            
            # 验证结果
            self.assertEqual(result["status"], "success", 
                            f"类比推理失败: {case['a']}:{case['b']}::{case['c']}:?")
            
            # 检查结果是否符合预期
            self.assertEqual(result["best_match"], case["expected"], 
                            f"类比错误: 预期 {case['expected']}, 实际 {result['best_match']}")
    
    def test_pattern_extraction(self):
        """测试模式提取功能"""
        # 提取模式
        patterns = self.vector_store.extract_patterns()
        
        # 验证是否成功提取了模式
        self.assertTrue(len(patterns) > 0, "未能提取到任何模式")
        
        # 检查是否提取了层次结构模式
        hierarchy_patterns = [p for p in patterns if p.get("pattern") == "层次结构"]
        self.assertTrue(len(hierarchy_patterns) > 0, "未能提取到层次结构模式")
        
        # 检查是否提取了树形结构模式
        tree_patterns = [p for p in patterns if p.get("pattern") == "树形结构"]
        self.assertTrue(len(tree_patterns) > 0, "未能提取到树形结构模式")
        
        # 打印发现的模式
        print(f"提取到 {len(patterns)} 个模式:")
        for i, pattern in enumerate(patterns[:5]):
            print(f"{i+1}. {pattern.get('pattern')}: {pattern.get('description')}")
    
    def test_meta_learning(self):
        """测试元学习功能"""
        # 为零样本模块添加一些测试用任务性能记录
        self.zero_shot.task_performances = {
            "classification": [
                {"success": True, "query": {"data": {"input": "Python是一种编程语言"}}, 
                 "result": {"predicted_class": "技术", "confidence": 0.9}},
                {"success": True, "query": {"data": {"input": "猫喜欢吃鱼"}}, 
                 "result": {"predicted_class": "动物", "confidence": 0.85}},
                {"success": False, "query": {"data": {"input": "复杂的量子力学理论"}}, 
                 "result": {"message": "无法确定类别"}}
            ],
            "analogical_reasoning": [
                {"success": True, "task_subtype": "analogy", "query": {"data": {"term_a": "北京", "term_b": "中国", "term_c": "纽约"}}, 
                 "result": {"best_match": "美国", "method": "relation_inference"}},
                {"success": True, "task_subtype": "analogy", "query": {"data": {"term_a": "猫", "term_b": "哺乳动物", "term_c": "鹰"}}, 
                 "result": {"best_match": "鸟类", "method": "relation_inference"}}
            ]
        }
        
        # 执行元学习
        result = self.zero_shot.meta_learn()
        
        # 验证元学习结果
        self.assertEqual(result["status"], "success", "元学习失败")
        
        # 执行学习集成器的元优化
        integrator_result = self.integrator.meta_optimize()
        
        # 验证集成器元优化结果
        self.assertEqual(integrator_result["status"], "success", "学习集成器元优化失败")
        
        # 打印元学习改进
        print("\n元学习改进结果:")
        for i, improvement in enumerate(result.get("improvements", [])):
            print(f"{i+1}. {improvement.get('improvement')}: {improvement.get('reason')}")
            
        # 打印集成器元优化改进
        print("\n集成器元优化改进结果:")
        for i, improvement in enumerate(integrator_result.get("improvements", [])):
            print(f"{i+1}. 类型: {improvement.get('type')}, "
                  f"原因: {improvement.get('reason', '未知')}")

def main():
    """运行测试"""
    print("开始测试零样本学习核心功能...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestZeroShotLearning)
    unittest.TextTestRunner(verbosity=2).run(test_suite)

if __name__ == "__main__":
    main() 