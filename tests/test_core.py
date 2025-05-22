"""
核心功能测试
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
import json
import tempfile

from core.event_bus import EventBus, Event
from core.task_scheduler import TaskScheduler, Task, TaskStatus
from core.health_check import HealthCheck
from knowledge.knowledge_base import KnowledgeBase, KnowledgeNode
from reasoning.reasoning_engine import ReasoningEngine

@pytest.fixture
async def event_bus():
    """事件总线测试夹具"""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()
    
@pytest.fixture
async def task_scheduler():
    """任务调度器测试夹具"""
    scheduler = TaskScheduler(max_workers=2)
    await scheduler.start()
    yield scheduler
    await scheduler.stop()
    
@pytest.fixture
async def health_check():
    """健康检查测试夹具"""
    checker = HealthCheck(check_interval=1)
    await checker.start()
    yield checker
    await checker.stop()
    
@pytest.fixture
def knowledge_base():
    """知识库测试夹具"""
    kb = KnowledgeBase()
    # 添加测试数据
    node1 = KnowledgeNode("测试知识1", "fact", 0.9)
    node2 = KnowledgeNode("测试知识2", "rule", 0.8)
    kb.add_node(node1)
    kb.add_node(node2)
    return kb
    
@pytest.fixture
def reasoning_engine(knowledge_base):
    """推理引擎测试夹具"""
    engine = ReasoningEngine(knowledge_base)
    # 添加测试规则
    engine.add_inference_rule(
        "test_rule",
        ["测试知识1", "测试知识2"],
        "测试结论",
        0.9
    )
    return engine

@pytest.mark.asyncio
async def test_event_bus(event_bus):
    """测试事件总线"""
    events = []
    
    async def handler(event):
        events.append(event)
        
    # 订阅事件
    event_bus.subscribe("test_event", handler)
    
    # 发布事件
    event = Event("test_event", {"data": "test"})
    await event_bus.publish(event)
    
    # 等待事件处理
    await asyncio.sleep(0.1)
    
    assert len(events) == 1
    assert events[0].type == "test_event"
    assert events[0].data["data"] == "test"
    
@pytest.mark.asyncio
async def test_task_scheduler(task_scheduler):
    """测试任务调度器"""
    results = []
    
    async def test_handler(data):
        results.append(data)
        return {"status": "success"}
        
    # 注册处理器
    task_scheduler.register_handler("test_task", test_handler)
    
    # 提交任务
    task = Task("test_task", {"data": "test"})
    task_id = await task_scheduler.submit_task(task)
    
    # 等待任务完成
    await asyncio.sleep(0.1)
    
    # 检查任务状态
    task = await task_scheduler.get_task_status(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.result["status"] == "success"
    assert len(results) == 1
    assert results[0]["data"] == "test"
    
@pytest.mark.asyncio
async def test_health_check(health_check):
    """测试健康检查"""
    # 注册自定义检查器
    async def test_checker():
        return {"healthy": True, "message": "测试检查通过"}
        
    health_check.register_checker("test", test_checker)
    
    # 等待检查执行
    await asyncio.sleep(2)
    
    # 检查状态
    status = health_check.get_health_status()
    assert status["status"] == "healthy"
    assert "test" in status["checks"]
    assert status["checks"]["test"]["healthy"] is True
    
@pytest.mark.asyncio
async def test_knowledge_base(knowledge_base):
    """测试知识库"""
    # 搜索知识
    results = knowledge_base.search("测试知识")
    assert len(results) == 2
    
    # 添加关系
    node1 = results[0]
    node2 = results[1]
    knowledge_base.add_relation(node1.id, node2.id, "related")
    
    # 获取相关节点
    related = knowledge_base.get_related_nodes(node1.id)
    assert len(related) == 1
    assert related[0].id == node2.id
    
@pytest.mark.asyncio
async def test_reasoning_engine(reasoning_engine):
    """测试推理引擎"""
    # 执行推理
    result = await reasoning_engine.reason("测试知识")
    
    assert "conclusions" in result
    assert len(result["conclusions"]) > 0
    assert result["conclusions"][0]["conclusion"] == "测试结论"
    
def test_state_persistence():
    """测试状态持久化"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试数据
        test_data = {
            "test_key": "test_value",
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存状态
        state_file = Path(temp_dir) / "test_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            
        # 加载状态
        with open(state_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            
        assert loaded_data["test_key"] == test_data["test_key"] 