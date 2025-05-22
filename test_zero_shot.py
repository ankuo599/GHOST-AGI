# -*- coding: utf-8 -*-
"""
GHOST AGI 零样本学习与知识图谱测试脚本
测试系统的零样本学习和知识表示能力
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("zero_shot_test.log")
    ]
)
logger = logging.getLogger("ZeroShotTest")

# 导入系统
from main import GhostAGI

def test_zero_shot_learning():
    """测试零样本学习能力"""
    logger.info("开始测试零样本学习能力...")
    
    # 创建AGI系统实例（禁用安全限制）
    agi = GhostAGI(config={
        "sandbox_enabled": False,
        "safety_checks": False
    })
    
    # 启动系统
    agi.start()
    logger.info("系统已启动，安全限制已禁用")
    
    # 测试零样本分类任务
    test_zero_shot_classification(agi)
    
    # 测试零样本生成任务
    test_zero_shot_generation(agi)
    
    # 测试类比推理能力
    test_zero_shot_analogy(agi)
    
    # 测试零样本关系预测
    test_zero_shot_relation_prediction(agi)
    
    # 测试知识图谱功能
    test_knowledge_graph(agi)
    
    # 测试元优化功能
    test_meta_optimization(agi)
    
    # 停止系统
    agi.stop()
    logger.info("测试完成，系统已停止")
    
def test_zero_shot_classification(agi):
    """测试零样本分类能力"""
    logger.info("测试零样本分类...")
    
    # 准备测试数据
    classification_tests = [
        {
            "input": "今天天气真好，阳光明媚",
            "target_classes": ["天气", "情绪", "新闻", "技术"]
        },
        {
            "input": "Python是一种解释型高级编程语言",
            "target_classes": ["编程语言", "操作系统", "网络协议", "数据库"]
        },
        {
            "input": "深度学习是机器学习的一个分支，它使用多层神经网络进行学习",
            "target_classes": ["人工智能", "生物学", "化学", "物理学"]
        }
    ]
    
    # 执行测试
    for i, test_data in enumerate(classification_tests):
        logger.info(f"分类测试 {i+1}: {test_data['input']}")
        
        # 构建查询
        query = {
            "task_type": "classification",
            "data": {"input": test_data["input"]},
            "context": {"target_classes": test_data["target_classes"]}
        }
        
        # 执行零样本学习
        result = execute_zero_shot_query(agi, query)
        
        if result and result.get("status") == "success":
            predicted = result.get("predicted_class", "未知")
            confidence = result.get("confidence", 0)
            logger.info(f"分类成功: {predicted} (置信度: {confidence:.2f})")
        else:
            logger.warning(f"分类失败: {result.get('message', '未知错误')}")
            
        # 暂停以免过载系统
        time.sleep(1)
    
def test_zero_shot_generation(agi):
    """测试零样本生成能力"""
    logger.info("测试零样本生成...")
    
    # 准备测试数据
    generation_tests = [
        {
            "prompt": "请解释什么是人工智能",
            "constraints": {"max_length": 100}
        },
        {
            "prompt": "写一个关于春天的短诗",
            "constraints": {"style": "classical"}
        },
        {
            "prompt": "提供三个提高工作效率的建议",
            "constraints": {"format": "list"}
        }
    ]
    
    # 执行测试
    for i, test_data in enumerate(generation_tests):
        logger.info(f"生成测试 {i+1}: {test_data['prompt']}")
        
        # 构建查询
        query = {
            "task_type": "generation",
            "data": {"prompt": test_data["prompt"]},
            "context": {"constraints": test_data["constraints"]}
        }
        
        # 执行零样本学习
        result = execute_zero_shot_query(agi, query)
        
        if result and result.get("status") in ["success", "limited"]:
            generated_text = result.get("generated_text", "无生成内容")
            logger.info(f"生成结果: {generated_text}")
        else:
            logger.warning(f"生成失败: {result.get('message', '未知错误')}")
            
        # 暂停以免过载系统
        time.sleep(1)
        
def test_zero_shot_analogy(agi):
    """测试类比推理能力"""
    logger.info("测试类比推理...")
    
    # 确保知识图谱中有一些概念
    ensure_concepts_exist(agi)
    
    # 准备测试数据
    analogy_tests = [
        {
            "a": "国王",
            "b": "王后",
            "c": "男人"
        },
        {
            "a": "书籍",
            "b": "阅读",
            "c": "电影"
        },
        {
            "a": "水",
            "b": "船",
            "c": "陆地"
        }
    ]
    
    # 执行测试
    for i, test_data in enumerate(analogy_tests):
        logger.info(f"类比测试 {i+1}: {test_data['a']}:{test_data['b']}::{test_data['c']}:?")
        
        # 构建查询
        query = {
            "task_type": "analogical_reasoning",
            "data": {
                "term_a": test_data["a"],
                "term_b": test_data["b"],
                "term_c": test_data["c"]
            },
            "context": {}
        }
        
        # 执行零样本学习
        result = execute_zero_shot_query(agi, query)
        
        if result and result.get("status") in ["success", "limited"]:
            if "results" in result and result["results"]:
                top_result = result["results"][0]
                logger.info(f"类比结果: {test_data['c']}:{top_result.get('name', '?')} (相似度: {top_result.get('similarity', 0):.2f})")
                
                # 显示其他可能的结果
                other_results = result["results"][1:3] if len(result["results"]) > 1 else []
                if other_results:
                    other_names = [r["name"] for r in other_results]
                    logger.info(f"其他可能结果: {', '.join(other_names)}")
            else:
                logger.info("未找到类比结果")
        else:
            logger.warning(f"类比推理失败: {result.get('message', '未知错误')}")
            
        # 暂停以免过载系统
        time.sleep(1)

def test_zero_shot_relation_prediction(agi):
    """测试零样本关系预测"""
    logger.info("测试关系预测...")
    
    # 确保知识图谱中有一些概念
    ensure_concepts_exist(agi)
    
    # 准备测试数据
    relation_tests = [
        {
            "entity1": "狗",
            "entity2": "动物"
        },
        {
            "entity1": "苹果",
            "entity2": "水果"
        },
        {
            "entity1": "太阳",
            "entity2": "地球"
        }
    ]
    
    # 执行测试
    for i, test_data in enumerate(relation_tests):
        logger.info(f"关系测试 {i+1}: {test_data['entity1']} - {test_data['entity2']}")
        
        # 构建查询
        query = {
            "task_type": "relation_prediction",
            "data": {
                "entity1": test_data["entity1"],
                "entity2": test_data["entity2"]
            },
            "context": {}
        }
        
        # 执行零样本学习
        result = execute_zero_shot_query(agi, query)
        
        if result and result.get("status") in ["success", "limited"]:
            if "predicted_relations" in result and result["predicted_relations"]:
                relations = result["predicted_relations"]
                for relation in relations:
                    rel_type = relation.get("relation", "")
                    confidence = relation.get("confidence", 0)
                    inferred = "（推断）" if relation.get("inferred", False) else ""
                    logger.info(f"预测关系: {rel_type} (置信度: {confidence:.2f}) {inferred}")
            else:
                logger.info("未找到关系")
        else:
            logger.warning(f"关系预测失败: {result.get('message', '未知错误')}")
            
        # 暂停以免过载系统
        time.sleep(1)
        
def test_knowledge_graph(agi):
    """测试知识图谱功能"""
    logger.info("测试知识图谱...")
    
    # 确保有向量存储
    if not hasattr(agi, "vector_store") or not agi.vector_store:
        logger.error("向量存储不可用")
        return
        
    # 检查是否支持知识图谱功能
    if not hasattr(agi.vector_store, "add_concept"):
        logger.error("向量存储不支持知识图谱功能")
        return
        
    # 添加测试概念
    concepts = [
        {"name": "人类", "properties": {"description": "智能生物"}},
        {"name": "计算机", "properties": {"description": "电子设备"}},
        {"name": "软件", "properties": {"description": "计算机程序"}},
        {"name": "硬件", "properties": {"description": "计算机物理组件"}},
        {"name": "操作系统", "properties": {"description": "基础系统软件"}}
    ]
    
    concept_ids = {}
    for concept in concepts:
        try:
            concept_id = agi.vector_store.add_concept(
                concept_name=concept["name"],
                properties=concept["properties"]
            )
            concept_ids[concept["name"]] = concept_id
            logger.info(f"添加概念: {concept['name']} -> {concept_id}")
        except Exception as e:
            logger.error(f"添加概念失败: {str(e)}")
            
    # 添加关系
    relations = [
        {"source": "软件", "target": "计算机", "type": "runs_on"},
        {"source": "硬件", "target": "计算机", "type": "part_of"},
        {"source": "操作系统", "target": "软件", "type": "is_a"},
        {"source": "人类", "target": "计算机", "type": "uses"}
    ]
    
    for relation in relations:
        try:
            source_id = concept_ids.get(relation["source"])
            target_id = concept_ids.get(relation["target"])
            
            if source_id and target_id:
                agi.vector_store.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation["type"]
                )
                logger.info(f"添加关系: {relation['source']} {relation['type']} {relation['target']}")
                
        except Exception as e:
            logger.error(f"添加关系失败: {str(e)}")
            
    # 查找相关概念
    try:
        human_id = concept_ids.get("人类")
        if human_id:
            related = agi.vector_store.find_related_concepts(human_id)
            logger.info(f"与'人类'相关的概念: {[r['name'] for r in related]}")
    except Exception as e:
        logger.error(f"查找相关概念失败: {str(e)}")
        
    # 提取模式
    try:
        patterns = agi.vector_store.extract_patterns()
        logger.info(f"提取到 {len(patterns)} 个模式")
        for pattern in patterns[:3]:  # 只显示前3个
            logger.info(f"模式: {pattern}")
    except Exception as e:
        logger.error(f"提取模式失败: {str(e)}")
        
    # 保存知识图谱
    try:
        knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        os.makedirs(knowledge_dir, exist_ok=True)
        
        knowledge_graph_path = os.path.join(knowledge_dir, "test_knowledge_graph.json")
        if agi.vector_store.save_knowledge_graph(knowledge_graph_path):
            logger.info(f"知识图谱保存成功: {knowledge_graph_path}")
        else:
            logger.warning("知识图谱保存失败")
    except Exception as e:
        logger.error(f"知识图谱保存失败: {str(e)}")
        
def test_meta_optimization(agi):
    """测试元优化功能"""
    logger.info("测试元优化...")
    
    try:
        # 调用元优化
        result = agi.meta_optimize()
        
        if result and result.get("status") == "completed":
            logger.info("元优化完成")
            
            # 打印各模块的优化结果
            for module, module_result in result.get("results", {}).items():
                status = module_result.get("status", "未知")
                logger.info(f"{module} 优化结果: {status}")
                
                # 如果有知识模式，打印发现的模式数量
                if module == "knowledge_patterns" and status == "success":
                    patterns_found = module_result.get("patterns_found", 0)
                    logger.info(f"发现了 {patterns_found} 个知识模式")
        else:
            logger.warning(f"元优化失败: {result}")
    except Exception as e:
        logger.error(f"元优化失败: {str(e)}")
        
def execute_zero_shot_query(agi, query):
    """
    执行零样本学习查询
    
    Args:
        agi: AGI实例
        query: 查询数据
        
    Returns:
        Dict: 查询结果
    """
    try:
        # 对不同集成方式做不同处理
        if hasattr(agi, "learning_integrator") and agi.learning_integrator:
            # 使用学习集成器
            query_data = {
                "task_type": "zero_shot",
                "query": query
            }
            return agi.learning_integrator.learn(query_data)
            
        elif hasattr(agi, "zero_shot_learning") and agi.zero_shot_learning:
            # 直接使用零样本学习模块
            return agi.zero_shot_learning.zero_shot_inference(query)
            
        else:
            # 尝试使用自动切换机制
            if "data" in query and "prompt" in query["data"]:
                return agi._try_zero_shot_learning(query["data"]["prompt"])
            else:
                logger.warning("系统没有可用的零样本学习模块")
                return {"status": "error", "message": "零样本学习不可用"}
    except Exception as e:
        logger.error(f"执行零样本查询失败: {str(e)}")
        return {"status": "error", "message": str(e)}
        
def ensure_concepts_exist(agi):
    """确保知识图谱中存在一些基本概念用于测试"""
    # 检查是否支持知识图谱功能
    if not hasattr(agi, "vector_store") or not hasattr(agi.vector_store, "add_concept"):
        logger.warning("向量存储不支持知识图谱功能，跳过概念创建")
        return
        
    # 基本概念
    basic_concepts = [
        "国王", "王后", "男人", "女人", 
        "书籍", "阅读", "电影", "观看", 
        "水", "船", "陆地", "汽车",
        "狗", "动物", "苹果", "水果", "太阳", "地球"
    ]
    
    for concept in basic_concepts:
        try:
            # 检查概念是否已存在
            concept_id = f"concept:{concept.lower().replace(' ', '_')}"
            
            # 如果没有concept_vectors属性或该概念不在其中，添加概念
            if not hasattr(agi.vector_store, "concept_vectors") or concept_id not in agi.vector_store.concept_vectors:
                agi.vector_store.add_concept(concept_name=concept)
                logger.debug(f"添加概念: {concept}")
                
        except Exception as e:
            logger.warning(f"添加概念 {concept} 失败: {str(e)}")
            
# 主程序入口
if __name__ == "__main__":
    test_zero_shot_learning() 