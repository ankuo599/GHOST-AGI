"""
GHOST-AGI 测试配置
"""

# 测试数据配置
TEST_DATA = {
    # 认知监控测试数据
    "reasoning_processes": [
        {
            "steps": [
                {
                    "id": "step1",
                    "type": "problem_definition",
                    "description": "定义问题：优化排序算法",
                    "output": "需要在大数据集上提高排序算法的效率"
                },
                {
                    "id": "step2",
                    "type": "analysis",
                    "description": "分析问题特征",
                    "output": "该问题涉及时间复杂度优化，需要考虑空间复杂度平衡",
                    "depends_on": "step1"
                },
                {
                    "id": "step3",
                    "type": "decision",
                    "description": "选择解决方案",
                    "options": ["归并排序", "快速排序", "堆排序", "基数排序"],
                    "selected_option": "快速排序",
                    "output": "选择快速排序并进行优化",
                    "depends_on": "step2"
                }
            ]
        }
    ],
    
    # 推理策略测试数据
    "problems": [
        {
            "type": "classification",
            "description": "对电子邮件进行垃圾邮件分类",
            "complexity": "medium",
            "context": {
                "time_constraint": "normal",
                "resources": "sufficient",
                "importance": "high"
            }
        },
        {
            "type": "diagnosis",
            "description": "诊断网络连接问题",
            "complexity": "high",
            "context": {
                "time_constraint": "urgent",
                "resources": "limited",
                "importance": "critical"
            }
        }
    ],
    
    # 元学习测试数据
    "learning_tasks": [
        {
            "type": "classification",
            "description": "图像分类任务",
            "data_size": 10000,
            "complexity": "medium",
            "time_constraint": "normal"
        },
        {
            "type": "regression",
            "description": "房价预测任务",
            "data_size": 5000,
            "complexity": "medium",
            "time_constraint": "relaxed"
        },
        {
            "type": "clustering",
            "description": "客户分群任务",
            "data_size": 20000,
            "complexity": "high",
            "time_constraint": "normal"
        }
    ]
} 