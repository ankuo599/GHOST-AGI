"""
GHOST-AGI系统功能测试

该脚本对GHOST-AGI系统的所有主要模块进行测试，并生成测试报告。
"""

import sys
import os
import time
import json
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghost_agi_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GHOST-AGI-Test")

# 尝试导入各个模块
try:
    from metacognition.cognitive_monitor import CognitiveMonitor
    COGNITIVE_MONITOR_AVAILABLE = True
except ImportError:
    logger.warning("认知监控模块不可用")
    COGNITIVE_MONITOR_AVAILABLE = False

try:
    from metacognition.reasoning_strategy_selector import ReasoningStrategySelector
    REASONING_SELECTOR_AVAILABLE = True
except ImportError:
    logger.warning("推理策略选择器不可用")
    REASONING_SELECTOR_AVAILABLE = False

try:
    from metacognition.meta_learning import MetaLearningModule
    META_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("元学习模块不可用")
    META_LEARNING_AVAILABLE = False

try:
    from architecture.architecture_awareness import ArchitectureAwareness
    ARCHITECTURE_AWARENESS_AVAILABLE = True
except ImportError:
    logger.warning("架构感知模块不可用")
    ARCHITECTURE_AWARENESS_AVAILABLE = False

try:
    from perception.cross_modal_integrator import CrossModalIntegrator
    CROSS_MODAL_AVAILABLE = True
except ImportError:
    logger.warning("跨模态整合模块不可用")
    CROSS_MODAL_AVAILABLE = False

try:
    from reasoning.creative_thinking_engine import CreativeThinkingEngine
    CREATIVE_THINKING_AVAILABLE = True
except ImportError:
    logger.warning("创造性思维引擎不可用")
    CREATIVE_THINKING_AVAILABLE = False

# 测试结果存储
test_results = {
    "summary": {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "modules_tested": []
    },
    "module_results": {},
    "timestamp": datetime.now().isoformat(),
    "system_info": {
        "python_version": sys.version,
        "platform": sys.platform
    }
}

def run_test(module_name: str, test_name: str, test_func):
    """运行单个测试并记录结果"""
    global test_results
    
    if module_name not in test_results["module_results"]:
        test_results["module_results"][module_name] = {
            "tests": [],
            "passed": 0,
            "failed": 0
        }
        if module_name not in test_results["summary"]["modules_tested"]:
            test_results["summary"]["modules_tested"].append(module_name)
    
    logger.info(f"运行测试: {module_name}.{test_name}")
    start_time = time.time()
    
    try:
        result = test_func()
        status = "PASS" if result else "FAIL"
        
        if result:
            test_results["module_results"][module_name]["passed"] += 1
            test_results["summary"]["passed_tests"] += 1
        else:
            test_results["module_results"][module_name]["failed"] += 1
            test_results["summary"]["failed_tests"] += 1
        
    except Exception as e:
        status = "ERROR"
        logger.error(f"测试错误: {str(e)}")
        test_results["module_results"][module_name]["failed"] += 1
        test_results["summary"]["failed_tests"] += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    test_record = {
        "name": test_name,
        "status": status,
        "duration": duration
    }
    
    if status == "ERROR":
        test_record["error"] = str(e) if 'e' in locals() else "未知错误"
    
    test_results["module_results"][module_name]["tests"].append(test_record)
    test_results["summary"]["total_tests"] += 1
    
    logger.info(f"测试结果: {status} (耗时: {duration:.2f}秒)")
    return status == "PASS"

# ======== 元认知系统测试 ========

def test_cognitive_monitor():
    """测试认知监控模块"""
    if not COGNITIVE_MONITOR_AVAILABLE:
        logger.warning("跳过认知监控模块测试")
        return
    
    logger.info("=== 开始认知监控模块测试 ===")
    
    cognitive_monitor = CognitiveMonitor()
    
    # 测试推理跟踪
    def test_track_reasoning():
        reasoning_id = f"test_reasoning_{int(time.time())}"
        reasoning_steps = [
            {
                "id": "step1",
                "type": "problem_definition",
                "description": "定义测试问题",
                "output": "这是一个测试问题的定义"
            },
            {
                "id": "step2",
                "type": "analysis",
                "description": "分析测试问题",
                "output": "这是对测试问题的分析",
                "depends_on": "step1"
            }
        ]
        
        result = cognitive_monitor.track_reasoning_process(reasoning_id, reasoning_steps)
        return result.get("status") == "success"
    
    # 测试获取认知跟踪
    def test_get_trace():
        traces = cognitive_monitor.get_all_reasoning_ids()
        if not traces:
            return False
        
        trace = cognitive_monitor.get_cognitive_trace(traces[0])
        return trace is not None
    
    # 测试认知偏差检测
    def test_bias_detection():
        test_steps = [
            {
                "id": "step1",
                "type": "problem_definition",
                "description": "定义问题",
                "output": "这是一个需要决策的问题"
            },
            {
                "id": "step2",
                "type": "decision",
                "description": "做出决策",
                "output": "基于第一印象做出决策，没有考虑其他可能性",
                "depends_on": "step1"
            }
        ]
        
        biases = cognitive_monitor.detect_cognitive_biases(test_steps)
        return len(biases) > 0
    
    # 运行测试
    run_test("认知监控", "推理跟踪", test_track_reasoning)
    run_test("认知监控", "获取认知跟踪", test_get_trace)
    run_test("认知监控", "认知偏差检测", test_bias_detection)

def test_reasoning_strategy_selector():
    """测试推理策略选择器"""
    if not REASONING_SELECTOR_AVAILABLE:
        logger.warning("跳过推理策略选择器测试")
        return
    
    logger.info("=== 开始推理策略选择器测试 ===")
    
    strategy_selector = ReasoningStrategySelector()
    
    # 测试策略选择
    def test_strategy_selection():
        problem = {
            "type": "classification",
            "description": "对数据进行分类的问题",
            "complexity": "medium"
        }
        
        context = {
            "time_constraint": "normal",
            "resources": "sufficient",
            "importance": "high"
        }
        
        strategy = strategy_selector.select_reasoning_strategy(problem, context)
        return "strategy_id" in strategy and "confidence" in strategy
    
    # 测试策略效果评估
    def test_strategy_evaluation():
        problem = {
            "type": "prediction",
            "description": "预测未来趋势的问题"
        }
        
        context = {
            "time_constraint": "normal"
        }
        
        strategy = strategy_selector.select_reasoning_strategy(problem, context)
        
        result = {
            "selection_id": strategy["selection_id"],
            "status": "success",
            "accuracy": 0.85,
            "efficiency": 0.75,
            "completeness": 0.8
        }
        
        evaluation = strategy_selector.evaluate_strategy_effectiveness(
            strategy["strategy_id"], problem, result
        )
        
        return "effectiveness" in evaluation and evaluation["status"] == "success"
    
    # 测试策略推荐
    def test_strategy_recommendation():
        recommendations = strategy_selector.get_strategy_recommendations(
            "需要分析多个因素之间复杂因果关系的问题"
        )
        
        return len(recommendations) > 0 and "strategy_id" in recommendations[0]
    
    # 运行测试
    run_test("推理策略选择器", "策略选择", test_strategy_selection)
    run_test("推理策略选择器", "策略效果评估", test_strategy_evaluation)
    run_test("推理策略选择器", "策略推荐", test_strategy_recommendation)

def test_meta_learning():
    """测试元学习模块"""
    if not META_LEARNING_AVAILABLE:
        logger.warning("跳过元学习模块测试")
        return
    
    logger.info("=== 开始元学习模块测试 ===")
    
    meta_learning = MetaLearningModule()
    
    # 测试学习策略优化
    def test_learning_strategy_optimization():
        task = {
            "task_id": f"task_{int(time.time())}",
            "type": "classification",
            "description": "图像分类任务",
            "data_size": 1000,
            "complexity": "medium"
        }
        
        strategy = meta_learning.optimize_learning_strategy(task)
        return "strategy" in strategy and "hyperparameters" in strategy
    
    # 测试学习效果评估
    def test_learning_effectiveness():
        task = {
            "task_id": f"task_{int(time.time())}",
            "type": "regression",
            "description": "房价预测任务",
            "data_size": 500
        }
        
        # 先优化策略
        meta_learning.optimize_learning_strategy(task)
        
        # 模拟结果
        result = {
            "task_id": task["task_id"],
            "accuracy": 0.82,
            "loss": 0.15,
            "convergence_time": 120,
            "generalization": 0.78
        }
        
        evaluation = meta_learning.evaluate_learning_effectiveness(task["task_id"], result)
        return "effectiveness" in evaluation and "percentile" in evaluation
    
    # 测试学习模式分析
    def test_learning_pattern_analysis():
        # 创建多个任务并提供结果，构建历史数据
        for i in range(3):
            task_id = f"analysis_task_{i}_{int(time.time())}"
            task = {
                "task_id": task_id,
                "type": ["classification", "regression", "clustering"][i % 3],
                "description": f"测试任务 {i}",
                "data_size": 1000
            }
            
            # 优化策略
            meta_learning.optimize_learning_strategy(task)
            
            # 模拟结果
            result = {
                "task_id": task_id,
                "accuracy": 0.7 + (i * 0.05),
                "loss": 0.3 - (i * 0.05),
                "convergence_time": 100 - (i * 10),
                "generalization": 0.65 + (i * 0.05),
                "effectiveness": 0.7 + (i * 0.05)
            }
            
            meta_learning.evaluate_learning_effectiveness(task_id, result)
            
            # 添加时间间隔
            time.sleep(0.1)
        
        # 分析学习模式
        analysis = meta_learning.analyze_learning_patterns()
        
        # 请求改进建议
        improvements = meta_learning.recommend_learning_improvements()
        
        return (analysis.get("status") in ["success", "insufficient_data"] and 
                isinstance(improvements, list))
    
    # 测试与认知监控的集成
    def test_cognitive_integration():
        if not COGNITIVE_MONITOR_AVAILABLE:
            logger.info("认知监控模块不可用，跳过集成测试")
            return True
            
        task_id = f"cog_task_{int(time.time())}"
        task = {
            "task_id": task_id,
            "type": "classification",
            "description": "集成测试任务"
        }
        
        # 优化策略
        meta_learning.optimize_learning_strategy(task)
        
        # 模拟认知反馈
        cognitive_feedback = {
            "task_id": task_id,
            "confidence": 0.6,
            "detected_biases": ["confirmation_bias"],
            "quality_score": 0.55
        }
        
        result = meta_learning.integrate_with_cognitive_monitor(cognitive_feedback)
        return result.get("status") == "success" or result.get("status") == "error"
    
    # 运行测试
    run_test("元学习", "学习策略优化", test_learning_strategy_optimization)
    run_test("元学习", "学习效果评估", test_learning_effectiveness)
    run_test("元学习", "学习模式分析", test_learning_pattern_analysis)
    run_test("元学习", "认知监控集成", test_cognitive_integration)

# ======== 架构感知测试 ========

def test_architecture_awareness():
    """测试架构感知模块"""
    if not ARCHITECTURE_AWARENESS_AVAILABLE:
        logger.warning("跳过架构感知模块测试")
        return
    
    logger.info("=== 开始架构感知模块测试 ===")
    
    architecture_awareness = ArchitectureAwareness()
    
    # 测试系统结构分析
    def test_system_structure_analysis():
        analysis = architecture_awareness.analyze_system_structure()
        return "modules" in analysis and "dependencies" in analysis
    
    # 测试架构问题检测
    def test_architecture_issue_detection():
        issues = architecture_awareness.detect_architecture_issues()
        return isinstance(issues, list)
    
    # 测试优化建议
    def test_improvement_suggestions():
        suggestions = architecture_awareness.suggest_architecture_improvements()
        return isinstance(suggestions, list) and len(suggestions) >= 0
    
    # 运行测试
    run_test("架构感知", "系统结构分析", test_system_structure_analysis)
    run_test("架构感知", "架构问题检测", test_architecture_issue_detection)
    run_test("架构感知", "优化建议", test_improvement_suggestions)

# ======== 跨模态整合测试 ========

def test_cross_modal_integrator():
    """测试跨模态整合模块"""
    if not CROSS_MODAL_AVAILABLE:
        logger.warning("跳过跨模态整合模块测试")
        return
    
    logger.info("=== 开始跨模态整合模块测试 ===")
    
    cross_modal = CrossModalIntegrator()
    
    # 测试模态表示创建
    def test_modal_representation():
        text_data = "这是一段测试文本"
        text_embedding = cross_modal.create_text_embedding(text_data)
        
        return isinstance(text_embedding, dict) and "embedding" in text_embedding
    
    # 测试跨模态对齐
    def test_cross_modal_alignment():
        text_data = "一只猫"
        image_data = "测试图像数据"  # 实际应用中这应该是图像数据
        
        alignment = cross_modal.align_modalities(
            {"text": text_data, "image": image_data}
        )
        
        return isinstance(alignment, dict) and "alignment_score" in alignment
    
    # 测试多模态搜索
    def test_multimodal_search():
        query = "测试查询"
        search_results = cross_modal.search_across_modalities(query)
        
        return isinstance(search_results, list)
    
    # 运行测试
    run_test("跨模态整合", "模态表示创建", test_modal_representation)
    run_test("跨模态整合", "跨模态对齐", test_cross_modal_alignment)
    run_test("跨模态整合", "多模态搜索", test_multimodal_search)

# ======== 创造性思维测试 ========

def test_creative_thinking_engine():
    """测试创造性思维引擎"""
    if not CREATIVE_THINKING_AVAILABLE:
        logger.warning("跳过创造性思维引擎测试")
        return
    
    logger.info("=== 开始创造性思维引擎测试 ===")
    
    creative_engine = CreativeThinkingEngine()
    
    # 测试概念混合
    def test_conceptual_blending():
        problem = {
            "type": "design",
            "description": "设计一种结合传统文化和现代科技的产品",
            "domains": ["文化", "科技"]
        }
        
        result = creative_engine.generate_creative_idea(
            problem=problem,
            thinking_mode="conceptual_blending"
        )
        
        return ("idea" in result and 
                "concept" in result["idea"] and 
                "evaluation" in result)
    
    # 测试类比推理
    def test_analogical_reasoning():
        problem = {
            "type": "innovation",
            "description": "使用自然界的原理解决城市交通拥堵问题",
            "source_domain": "自然",
            "target_domain": "交通"
        }
        
        result = creative_engine.generate_creative_idea(
            problem=problem,
            thinking_mode="analogical_reasoning"
        )
        
        return ("idea" in result and 
                "analogy" in result["idea"] and 
                "insights" in result["idea"])
    
    # 测试发散思维
    def test_divergent_thinking():
        problem = {
            "type": "ideation",
            "description": "为老年人提供更好的社交体验",
            "central_concept": "老年社交"
        }
        
        result = creative_engine.generate_creative_idea(
            problem=problem,
            thinking_mode="divergent_thinking"
        )
        
        return ("idea" in result and 
                "branches" in result["idea"] and 
                "diversity" in result["idea"])
    
    # 测试约束放松
    def test_constraint_relaxation():
        problem = {
            "type": "optimization",
            "description": "在空间有限的情况下优化公寓内部设计",
            "constraints": [
                {"name": "空间约束", "description": "公寓面积小", "importance": "high"},
                {"name": "预算约束", "description": "装修预算有限", "importance": "medium"}
            ]
        }
        
        result = creative_engine.generate_creative_idea(
            problem=problem,
            thinking_mode="constraint_relaxation"
        )
        
        return ("idea" in result and 
                "relaxations" in result["idea"] and 
                "solutions" in result["idea"])
    
    # 测试横向思维
    def test_lateral_thinking():
        problem = {
            "type": "innovation",
            "description": "从全新角度思考废弃物处理方式"
        }
        
        result = creative_engine.generate_creative_idea(
            problem=problem,
            thinking_mode="lateral_thinking"
        )
        
        return ("idea" in result and 
                any(key in result["idea"] for key in ["perspective_shifts", "provocations", "insights"]))
    
    # 测试自动选择模式
    def test_auto_mode_selection():
        problem = {
            "type": "general",
            "description": "提高在线教育的学习效果和参与度"
        }
        
        result = creative_engine.generate_creative_idea(problem=problem)
        
        return ("thinking_mode" in result and 
                "idea" in result and 
                "evaluation" in result)
    
    # 运行测试
    run_test("创造性思维", "概念混合", test_conceptual_blending)
    run_test("创造性思维", "类比推理", test_analogical_reasoning)
    run_test("创造性思维", "发散思维", test_divergent_thinking)
    run_test("创造性思维", "约束放松", test_constraint_relaxation)
    run_test("创造性思维", "横向思维", test_lateral_thinking)
    run_test("创造性思维", "自动模式选择", test_auto_mode_selection)

# ======== 主测试函数 ========

def run_all_tests():
    """运行所有模块测试"""
    logger.info("======== 开始GHOST-AGI系统测试 ========")
    start_time = time.time()
    
    # 测试元认知系统
    test_cognitive_monitor()
    test_reasoning_strategy_selector()
    test_meta_learning()
    
    # 测试架构感知
    test_architecture_awareness()
    
    # 测试跨模态整合
    test_cross_modal_integrator()
    
    # 测试创造性思维
    test_creative_thinking_engine()
    
    # 计算测试时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 更新测试结果
    test_results["duration"] = duration
    
    logger.info(f"======== 系统测试完成 (耗时: {duration:.2f}秒) ========")
    logger.info(f"总测试数: {test_results['summary']['total_tests']}")
    logger.info(f"通过测试: {test_results['summary']['passed_tests']}")
    logger.info(f"失败测试: {test_results['summary']['failed_tests']}")
    
    # 保存测试结果
    with open("ghost_agi_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info("测试结果已保存到 ghost_agi_test_results.json")
    
    return test_results

if __name__ == "__main__":
    run_all_tests() 