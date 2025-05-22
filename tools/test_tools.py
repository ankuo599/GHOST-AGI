# -*- coding: utf-8 -*-
"""
测试工具模块 (Test Tools)

提供用于测试系统各种功能的工具
"""

import time
import random
import json
import os
import platform
from typing import Dict, List, Any, Optional
from .tool_executor import tool

@tool(
    name="test_memory",
    description="测试记忆系统存取功能",
    required_params=["content"],
    optional_params={"memory_type": "short_term"}
)
def test_memory(content: str, memory_type: str = "short_term") -> Dict[str, Any]:
    """
    测试记忆系统存取功能
    
    Args:
        content: 要存储的内容
        memory_type: 记忆类型 (short_term/long_term)
        
    Returns:
        Dict: 测试结果
    """
    memory_id = f"test_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return {
        "status": "success",
        "message": f"已将内容存储到{memory_type}记忆",
        "memory_id": memory_id,
        "content": content,
        "timestamp": time.time()
    }

@tool(
    name="test_reasoning",
    description="测试推理引擎功能",
    required_params=["query"],
    optional_params={"context": {}}
)
def test_reasoning(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    测试推理引擎功能
    
    Args:
        query: 推理查询
        context: 推理上下文
        
    Returns:
        Dict: 推理结果
    """
    # 简单演示推理
    context = context or {}
    
    # 生成随机推理结果
    result_options = [
        "这是基于已有知识的推理结果。",
        "根据符号推理引擎的分析，得出结论。",
        "通过逻辑推导，可以判断这个结论是合理的。",
        "综合考虑多方面因素，这个结论是最优的。"
    ]
    
    return {
        "status": "success",
        "query": query,
        "conclusion": random.choice(result_options),
        "confidence": round(random.uniform(0.7, 0.98), 2),
        "reasoning_steps": random.randint(3, 8),
        "execution_time": round(random.uniform(0.1, 1.2), 3),
        "timestamp": time.time()
    }

@tool(
    name="test_planning",
    description="测试规划引擎功能",
    required_params=["goal"],
    optional_params={"constraints": []}
)
def test_planning(goal: str, constraints: List[str] = None) -> Dict[str, Any]:
    """
    测试规划引擎功能
    
    Args:
        goal: 规划目标
        constraints: 规划约束条件
        
    Returns:
        Dict: 规划结果
    """
    constraints = constraints or []
    
    # 生成一个简单的计划
    steps = []
    step_count = random.randint(3, 6)
    
    for i in range(1, step_count + 1):
        steps.append({
            "id": i,
            "description": f"计划步骤 {i} - {random.choice(['分析', '执行', '评估', '总结'])}任务",
            "estimated_time": random.randint(1, 10)
        })
    
    return {
        "status": "success",
        "plan_id": f"plan_{int(time.time())}",
        "goal": goal,
        "constraints": constraints,
        "steps": steps,
        "estimated_completion_time": sum(step["estimated_time"] for step in steps),
        "timestamp": time.time()
    }

@tool(
    name="test_event_system",
    description="测试事件系统功能",
    required_params=["event_type"],
    optional_params={"event_data": {}}
)
def test_event_system(event_type: str, event_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    测试事件系统功能
    
    Args:
        event_type: 事件类型
        event_data: 事件数据
        
    Returns:
        Dict: 测试结果
    """
    event_data = event_data or {}
    
    # 补充事件数据
    event_data.update({
        "test_id": f"test_{int(time.time())}",
        "timestamp": time.time()
    })
    
    return {
        "status": "success",
        "message": f"已发布事件 {event_type}",
        "event_type": event_type,
        "event_data": event_data,
        "timestamp": time.time()
    }

@tool(
    name="test_performance",
    description="测试系统性能",
    required_params=["test_duration"],
    optional_params={"complexity": "medium"}
)
def test_performance(test_duration: int, complexity: str = "medium") -> Dict[str, Any]:
    """
    测试系统性能
    
    Args:
        test_duration: 测试持续时间(秒)
        complexity: 测试复杂度(low/medium/high)
        
    Returns:
        Dict: 性能测试结果
    """
    # 限制测试时间范围
    test_duration = min(max(test_duration, 1), 30)
    
    # 复杂度映射
    complexity_map = {
        "low": 1000,
        "medium": 10000,
        "high": 100000
    }
    
    iteration_count = complexity_map.get(complexity.lower(), 10000)
    
    # 模拟测试过程
    start_time = time.time()
    
    # 简单计算
    result = 0
    for i in range(iteration_count):
        result += i
        
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # 获取系统信息
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }
    
    return {
        "status": "success",
        "test_type": "performance",
        "requested_duration": test_duration,
        "actual_duration": round(actual_duration, 3),
        "complexity": complexity,
        "iterations": iteration_count,
        "system_info": system_info,
        "timestamp": time.time()
    }

@tool(
    name="run_demo",
    description="运行系统演示",
    required_params=["demo_type"],
    optional_params={"duration": 10}
)
def run_demo(demo_type: str, duration: int = 10) -> Dict[str, Any]:
    """
    运行系统演示
    
    Args:
        demo_type: 演示类型(conversation/planning/memory/reasoning)
        duration: 演示持续时间(秒)
        
    Returns:
        Dict: 演示结果
    """
    # 限制演示时间
    duration = min(max(duration, 1), 60)
    
    # 模拟演示
    demo_types = ["conversation", "planning", "memory", "reasoning"]
    
    if demo_type.lower() not in demo_types:
        demo_type = random.choice(demo_types)
    
    demo_results = {
        "conversation": {
            "message": "演示了基本对话功能",
            "interactions": random.randint(3, 8),
            "topics": ["天气", "新闻", "技术", "健康", "娱乐"][:random.randint(1, 5)]
        },
        "planning": {
            "message": "演示了规划引擎功能",
            "plans_created": random.randint(1, 3),
            "goals": ["信息获取", "任务管理", "知识学习"][:random.randint(1, 3)]
        },
        "memory": {
            "message": "演示了记忆系统功能",
            "memories_created": random.randint(5, 15),
            "memories_retrieved": random.randint(3, 8)
        },
        "reasoning": {
            "message": "演示了推理引擎功能",
            "reasoning_tasks": random.randint(2, 5),
            "accuracy": round(random.uniform(0.85, 0.98), 2)
        }
    }
    
    # 休眠一段时间模拟演示过程
    time.sleep(min(duration, 2))
    
    return {
        "status": "success",
        "demo_type": demo_type,
        "duration": duration,
        "results": demo_results.get(demo_type.lower(), demo_results["conversation"]),
        "timestamp": time.time()
    } 