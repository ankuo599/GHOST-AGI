"""
GHOST-AGI 模块集成测试

该脚本测试不同模块之间的协作和整合
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghost_agi_integration_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GHOST-AGI-Integration-Test")

# 尝试导入模块
try:
    from metacognition.cognitive_monitor import CognitiveMonitor
    from metacognition.reasoning_strategy_selector import ReasoningStrategySelector
    from metacognition.meta_learning import MetaLearningModule
    
    # 导入测试配置
    from tests.config import TEST_DATA
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    MODULES_AVAILABLE = False

def test_metacognition_integration():
    """测试元认知模块的集成"""
    if not MODULES_AVAILABLE:
        logger.error("必要模块不可用，无法运行集成测试")
        return False
    
    logger.info("=== 开始元认知模块集成测试 ===")
    
    # 初始化模块
    cognitive_monitor = CognitiveMonitor()
    strategy_selector = ReasoningStrategySelector(cognitive_monitor=cognitive_monitor)
    meta_learning = MetaLearningModule(cognitive_monitor=cognitive_monitor)
    
    # 测试识别问题 -> 选择策略 -> 执行 -> 学习改进 的流程
    
    # 1. 设置测试问题
    problem = TEST_DATA["problems"][0]
    task_id = f"integration_task_{int(time.time())}"
    
    # 2. 开始认知过程监控
    reasoning_steps = TEST_DATA["reasoning_processes"][0]["steps"]
    reasoning_id = f"reasoning_{int(time.time())}"
    
    monitoring_result = cognitive_monitor.track_reasoning_process(reasoning_id, reasoning_steps)
    
    logger.info(f"推理跟踪结果: {json.dumps(monitoring_result, ensure_ascii=False)}")
    
    # 3. 选择推理策略
    strategy = strategy_selector.select_reasoning_strategy(
        problem, problem.get("context", {})
    )
    
    logger.info(f"选择的策略: {strategy['strategy_id']}")
    
    # 4. 基于策略创建学习任务
    task = {
        "task_id": task_id,
        "type": problem["type"],
        "description": problem["description"],
        "data_size": 5000,
        "complexity": problem.get("complexity", "medium"),
        "reasoning_id": reasoning_id
    }
    
    learning_strategy = meta_learning.optimize_learning_strategy(task)
    
    logger.info(f"优化的学习策略: {learning_strategy['strategy']}")
    
    # 5. 模拟执行结果
    execution_result = {
        "selection_id": strategy["selection_id"],
        "status": "success",
        "accuracy": 0.88,
        "efficiency": 0.82,
        "completeness": 0.85
    }
    
    # 6. 评估策略效果
    evaluation = strategy_selector.evaluate_strategy_effectiveness(
        strategy["strategy_id"], problem, execution_result
    )
    
    logger.info(f"策略效果评估: {evaluation['effectiveness']:.2f}")
    
    # 7. 提供认知反馈
    cognitive_feedback = {
        "task_id": task_id,
        "confidence": monitoring_result["confidence"],
        "detected_biases": monitoring_result.get("detected_biases", []),
        "quality_score": monitoring_result.get("quality_score", 0.7)
    }
    
    feedback_result = meta_learning.integrate_with_cognitive_monitor(cognitive_feedback)
    
    logger.info(f"认知反馈集成结果: {json.dumps(feedback_result, ensure_ascii=False)}")
    
    # 8. 模拟学习结果
    learning_result = {
        "task_id": task_id,
        "accuracy": 0.86,
        "loss": 0.12,
        "convergence_time": 95,
        "generalization": 0.83,
        "effectiveness": 0.85
    }
    
    learning_evaluation = meta_learning.evaluate_learning_effectiveness(task_id, learning_result)
    
    logger.info(f"学习效果评估: {learning_evaluation['effectiveness']:.2f}")
    
    # 9. 分析学习模式并提供改进建议
    if len(meta_learning.learning_history) >= 3:
        analysis = meta_learning.analyze_learning_patterns()
        logger.info(f"学习模式分析状态: {analysis.get('status')}")
        
        improvements = meta_learning.recommend_learning_improvements()
        logger.info(f"提供了{len(improvements)}条改进建议")
    
    # 测试通过的条件
    success = (
        monitoring_result.get("status") == "success" and
        "strategy_id" in strategy and
        "strategy" in learning_strategy and
        evaluation.get("status") == "success" and
        learning_evaluation.get("status") == "success" 
    )
    
    logger.info(f"集成测试结果: {'通过' if success else '失败'}")
    return success

if __name__ == "__main__":
    if test_metacognition_integration():
        print("元认知模块集成测试通过")
    else:
        print("元认知模块集成测试失败") 