from metacognition.cognitive_monitor import CognitiveMonitor
import time

def main():
    # 初始化认知监控器
    monitor = CognitiveMonitor()
    print("认知监控器初始化完成")

    # 启动持续学习
    monitor.start_continuous_learning()
    print("持续学习已启动")

    # 示例：追踪推理过程
    reasoning_steps = [
        "分析问题：计算两个数的最大公约数",
        "应用欧几里得算法",
        "验证结果"
    ]
    
    context = {
        "task": "计算最大公约数",
        "input": [48, 36],
        "expected_output": 12
    }

    result = monitor.track_reasoning_process(
        reasoning_steps=reasoning_steps,
        context=context
    )
    print("\n推理追踪结果:")
    print(f"推理质量: {result['reasoning_quality']}")
    print(f"置信度: {result['confidence']}")
    print(f"效率得分: {result['efficiency_score']}")

    # 示例：检测认知偏差
    reasoning_traces = [
        "这个算法一定是最优的，因为它在所有测试用例中都表现良好",
        "我倾向于使用熟悉的解决方案，而不是探索新的方法"
    ]

    biases = monitor.detect_cognitive_biases(reasoning_traces)
    print("\n认知偏差检测结果:")
    for bias in biases:
        print(f"- {bias['bias_type']}: {bias['description']}")
        print(f"  严重程度: {bias['severity']}")
        print(f"  缓解建议: {bias['mitigation']}")

    # 示例：学习新概念
    concept = {
        "name": "动态规划",
        "description": "一种通过将复杂问题分解为子问题来解决问题的方法",
        "category": "算法",
        "difficulty": "中等"
    }

    monitor.knowledge_base["concepts"][concept["name"]] = concept
    print("\n已添加新概念到知识库")

    # 示例：生成练习
    practice = monitor._generate_practice_questions(concept)
    print("\n生成的练习:")
    for i, question in enumerate(practice, 1):
        print(f"{i}. {question['question']}")
        print(f"   类型: {question['type']}")
        print(f"   难度: {question['difficulty']}")

    # 等待一段时间以观察持续学习的效果
    print("\n等待10秒观察持续学习效果...")
    time.sleep(10)

    # 检查学习进度
    print("\n当前学习状态:")
    print(f"活跃学习任务数: {len(monitor.active_learning_tasks)}")
    print(f"知识库概念数: {len(monitor.knowledge_base['concepts'])}")
    print(f"知识库技能数: {len(monitor.knowledge_base['skills'])}")

if __name__ == "__main__":
    main() 