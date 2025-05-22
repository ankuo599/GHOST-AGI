"""
GHOST-AGI 元认知系统使用示例

该示例展示了如何使用GHOST-AGI的元认知模块进行认知监控、推理策略选择和元学习。
"""

import sys
import os
import time
import json
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metacognition.cognitive_monitor import CognitiveMonitor
from metacognition.reasoning_strategy_selector import ReasoningStrategySelector
from metacognition.meta_learning import MetaLearningModule

def simulate_reasoning_process():
    """模拟一个推理过程，用于演示认知监控功能"""
    return [
        {
            "id": "step1",
            "type": "problem_definition",
            "description": "定义问题：识别图像中的物体",
            "output": "需要对输入图像进行物体识别，确定图像中包含的主要对象"
        },
        {
            "id": "step2",
            "type": "analysis",
            "description": "分析问题特征",
            "output": "这是一个计算机视觉分类问题，需要使用图像特征提取和分类技术",
            "depends_on": "step1"
        },
        {
            "id": "step3",
            "type": "decision",
            "description": "选择解决方案",
            "options": ["传统计算机视觉方法", "深度学习方法"],
            "selected_option": "深度学习方法",
            "output": "考虑到问题的复杂性，选择使用卷积神经网络进行处理"
        },
        {
            "id": "step4",
            "type": "implementation",
            "description": "实现解决方案",
            "output": "使用预训练的ResNet50模型处理图像"
        },
        {
            "id": "step5",
            "type": "evaluation",
            "description": "评估结果",
            "output": "模型识别准确率为92%，满足需求",
            "evidence": "在测试集上的准确率指标"
        },
        {
            "id": "step6",
            "type": "conclusion",
            "description": "总结结论",
            "output": "成功使用深度学习方法完成图像识别任务，识别准确率满足要求"
        }
    ]

def simulate_learning_task(task_type: str) -> Dict[str, Any]:
    """模拟一个学习任务，用于演示元学习功能"""
    task = {
        "task_id": f"task_{int(time.time())}",
        "type": task_type,
        "description": f"学习{task_type}任务",
        "data_size": 1000,
        "complexity": "medium",
        "time_constraint": "normal"
    }
    
    # 根据任务类型添加特定参数
    if task_type == "classification":
        task["classes"] = 10
        task["balance"] = "balanced"
    elif task_type == "regression":
        task["output_dim"] = 1
        task["distribution"] = "normal"
    elif task_type == "clustering":
        task["expected_clusters"] = 5
        task["dimensionality"] = 50
    
    return task

def simulate_learning_result(task_id: str, effectiveness: float) -> Dict[str, Any]:
    """模拟学习结果，用于演示元学习评估功能"""
    return {
        "task_id": task_id,
        "accuracy": min(0.98, effectiveness + 0.2),
        "loss": max(0.02, 1.0 - effectiveness),
        "convergence_time": 100 / (effectiveness + 0.5),
        "generalization": min(0.95, effectiveness - 0.1),
        "effectiveness": effectiveness
    }

def main():
    print("=== GHOST-AGI 元认知系统示例 ===")
    
    # 初始化元认知模块
    cognitive_monitor = CognitiveMonitor()
    strategy_selector = ReasoningStrategySelector(cognitive_monitor=cognitive_monitor)
    meta_learning = MetaLearningModule(cognitive_monitor=cognitive_monitor)
    
    print("\n1. 认知监控示例")
    print("-" * 40)
    
    # 模拟推理过程并进行认知监控
    reasoning_steps = simulate_reasoning_process()
    reasoning_id = "reasoning_" + str(int(time.time()))
    
    # 监控推理过程
    monitoring_result = cognitive_monitor.track_reasoning_process(reasoning_id, reasoning_steps)
    
    print(f"推理跟踪结果: {json.dumps(monitoring_result, indent=2, ensure_ascii=False)}")
    
    # 提取推理过程跟踪
    trace = cognitive_monitor.get_cognitive_trace(reasoning_id)
    
    # 检测认知偏差
    biases = cognitive_monitor.detect_cognitive_biases(reasoning_steps)
    print(f"检测到的认知偏差: {len(biases)}")
    for bias in biases:
        print(f" - {bias['bias_type']}: {bias['description']}")
    
    print("\n2. 推理策略选择示例")
    print("-" * 40)
    
    # 定义问题和上下文
    problem = {
        "type": "classification",
        "description": "对图像进行分类，识别其中的物体",
        "complexity": "medium"
    }
    
    context = {
        "time_constraint": "normal",
        "resources": "sufficient",
        "importance": "high"
    }
    
    # 选择推理策略
    strategy = strategy_selector.select_reasoning_strategy(problem, context)
    
    print(f"为问题选择的推理策略: {strategy['strategy_id']}")
    print(f"策略名称: {strategy['name']}")
    print(f"策略描述: {strategy['description']}")
    print(f"选择置信度: {strategy['confidence']:.2f}")
    
    # 评估策略效果
    result = {
        "selection_id": strategy["selection_id"],
        "status": "success",
        "accuracy": 0.92,
        "efficiency": 0.85,
        "completeness": 0.9
    }
    
    evaluation = strategy_selector.evaluate_strategy_effectiveness(
        strategy["strategy_id"], problem, result
    )
    
    print(f"策略效果评估: {evaluation['effectiveness']:.2f}")
    
    print("\n3. 元学习示例")
    print("-" * 40)
    
    # 创建不同类型的学习任务
    tasks = [
        simulate_learning_task("classification"),
        simulate_learning_task("regression"),
        simulate_learning_task("clustering")
    ]
    
    for i, task in enumerate(tasks):
        # 为任务优化学习策略
        print(f"\n任务 {i+1}: {task['type']}")
        strategy = meta_learning.optimize_learning_strategy(task)
        
        print(f"学习策略: {strategy['strategy']}")
        print(f"超参数: {json.dumps(strategy['hyperparameters'], indent=2)}")
        
        # 模拟学习结果
        effectiveness = 0.7 + i * 0.05  # 随着索引增加效果略有提高
        result = simulate_learning_result(task["task_id"], effectiveness)
        
        # 评估学习效果
        evaluation = meta_learning.evaluate_learning_effectiveness(task["task_id"], result)
        
        print(f"学习效果评估: {evaluation['effectiveness']:.2f}")
        print(f"百分位排名: {evaluation['percentile']:.1f}%")
        
        if "improvement_suggestions" in evaluation:
            print("改进建议:")
            for suggestion in evaluation["improvement_suggestions"]:
                print(f" - {suggestion}")
    
    # 分析学习模式
    print("\n学习模式分析:")
    analysis = meta_learning.analyze_learning_patterns()
    
    if analysis.get("status") == "success":
        print(f"总任务数: {analysis['total_tasks_analyzed']}")
        
        trends = analysis.get("effectiveness_trend", {})
        print(f"学习趋势: {trends.get('trend', 'unknown')}")
        
        bottlenecks = analysis.get("learning_bottlenecks", [])
        if bottlenecks:
            print("学习瓶颈:")
            for bottleneck in bottlenecks:
                print(f" - {bottleneck['description']}")
    
    # 推荐学习改进措施
    improvements = meta_learning.recommend_learning_improvements()
    
    print("\n学习改进建议:")
    for improvement in improvements:
        print(f" - {improvement['description']}: {improvement['implementation']}")

if __name__ == "__main__":
    main() 