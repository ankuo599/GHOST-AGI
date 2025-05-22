"""
事件总线系统
实现模块间的事件通信
"""

from typing import Dict, List, Any, Optional, Callable, Set
import logging
import asyncio
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import uuid
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    """事件"""
    id: str
    type: str
    data: Any
    source: str
    timestamp: str
    metadata: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    validation_status: bool = True

class EventFilter:
    """事件过滤器"""
    def __init__(self, event_types: Optional[Set[str]] = None,
                 sources: Optional[Set[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.event_types = event_types
        self.sources = sources
        self.metadata = metadata
        
    def matches(self, event: Event) -> bool:
        """检查事件是否匹配过滤器"""
        # 检查事件类型
        if self.event_types and event.type not in self.event_types:
            return False
            
        # 检查事件源
        if self.sources and event.source not in self.sources:
            return False
            
        # 检查元数据
        if self.metadata:
            for key, value in self.metadata.items():
                if key not in event.metadata or event.metadata[key] != value:
                    return False
                    
        return True

class EventHandler:
    """事件处理器"""
    def __init__(self, handler: Callable, filter: Optional[EventFilter] = None):
        self.handler = handler
        self.filter = filter
        self.is_async = asyncio.iscoroutinefunction(handler)
        
    async def handle(self, event: Event):
        """处理事件"""
        if self.filter and not self.filter.matches(event):
            return
            
        try:
            if self.is_async:
                await self.handler(event)
            else:
                self.handler(event)
        except Exception as e:
            logging.error(f"事件处理失败: {str(e)}")

class EventValidator:
    """事件验证器"""
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        
    def register_validator(self, event_type: str, validator: Callable):
        """注册事件验证器"""
        if event_type not in self.validators:
            self.validators[event_type] = []
        self.validators[event_type].append(validator)
        
    def validate(self, event: Event) -> bool:
        """验证事件"""
        if event.type not in self.validators:
            return True
            
        for validator in self.validators[event.type]:
            try:
                if not validator(event):
                    return False
            except Exception as e:
                logging.error(f"事件验证失败: {str(e)}")
                return False
        return True

class EventMonitor:
    """事件监控器"""
    def __init__(self):
        self.stats = {
            "total_events": 0,
            "failed_events": 0,
            "retried_events": 0,
            "event_types": {},
            "processing_times": []
        }
        
    def record_event(self, event: Event, processing_time: float):
        """记录事件统计"""
        self.stats["total_events"] += 1
        self.stats["processing_times"].append(processing_time)
        
        if event.type not in self.stats["event_types"]:
            self.stats["event_types"][event.type] = 0
        self.stats["event_types"][event.type] += 1
        
    def record_failure(self):
        """记录失败事件"""
        self.stats["failed_events"] += 1
        
    def record_retry(self):
        """记录重试事件"""
        self.stats["retried_events"] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
        else:
            stats["avg_processing_time"] = 0
        return stats

class EventBus:
    """事件总线"""
    def __init__(self):
        self.logger = logging.getLogger("EventBus")
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.event_history: List[Event] = []
        self.max_history_size = 1000
        self.validator = EventValidator()
        self.monitor = EventMonitor()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in EventPriority
        }
        
    def subscribe(self, event_type: str, handler: Callable,
                 filter: Optional[EventFilter] = None):
        """订阅事件"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
            
        self.handlers[event_type].append(EventHandler(handler, filter))
        
    def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
        if event_type in self.handlers:
            self.handlers[event_type] = [
                h for h in self.handlers[event_type]
                if h.handler != handler
            ]
            
    async def process_priority_queue(self, priority: EventPriority):
        """处理优先级队列"""
        while True:
            event = await self.priority_queues[priority].get()
            try:
                start_time = time.time()
                
                # 验证事件
                if not self.validator.validate(event):
                    event.validation_status = False
                    self.monitor.record_failure()
                    continue
                    
                # 处理事件
                if event.type in self.handlers:
                    for handler in self.handlers[event.type]:
                        try:
                            await handler.handle(event)
                        except Exception as e:
                            self.logger.error(f"事件处理失败: {str(e)}")
                            if event.retry_count < event.max_retries:
                                event.retry_count += 1
                                self.monitor.record_retry()
                                await self.priority_queues[priority].put(event)
                            else:
                                self.monitor.record_failure()
                                
                processing_time = time.time() - start_time
                self.monitor.record_event(event, processing_time)
                
            except Exception as e:
                self.logger.error(f"优先级队列处理失败: {str(e)}")
                self.monitor.record_failure()
            finally:
                self.priority_queues[priority].task_done()
                
    async def start(self):
        """启动事件总线"""
        for priority in EventPriority:
            asyncio.create_task(self.process_priority_queue(priority))
            
    async def stop(self):
        """停止事件总线"""
        for queue in self.priority_queues.values():
            await queue.join()
        self.executor.shutdown()
        
    async def publish(self, event_type: str, data: Any,
                     source: str, metadata: Optional[Dict[str, Any]] = None,
                     priority: EventPriority = EventPriority.NORMAL):
        """发布事件"""
        try:
            event = Event(
                id=str(uuid.uuid4()),
                type=event_type,
                data=data,
                source=source,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
                priority=priority
            )
            
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
                
            await self.priority_queues[priority].put(event)
            return event
            
        except Exception as e:
            self.logger.error(f"事件发布失败: {str(e)}")
            raise
            
    def get_event_history(self, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        return self.event_history[-limit:]
        
    def clear_history(self):
        """清除事件历史"""
        self.event_history.clear()
        
    def save_state(self, path: str):
        """保存状态"""
        data = {
            "event_history": [asdict(e) for e in self.event_history]
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_state(self, path: str):
        """加载状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.event_history = [Event(**e) for e in data["event_history"]]
            
        except FileNotFoundError:
            self.logger.warning("状态文件不存在，使用默认状态")

    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return self.monitor.get_stats()

class EventManager:
    """事件管理器"""
    def __init__(self):
        self.logger = logging.getLogger("EventManager")
        self.event_bus = EventBus()
        self.event_types: Set[str] = set()
        self.event_sources: Set[str] = set()
        
    def register_event_type(self, event_type: str):
        """注册事件类型"""
        self.event_types.add(event_type)
        
    def register_event_source(self, source: str):
        """注册事件源"""
        self.event_sources.add(source)
        
    def subscribe(self, event_type: str, handler: Callable,
                 filter: Optional[EventFilter] = None):
        """订阅事件"""
        if event_type not in self.event_types:
            self.logger.warning(f"未注册的事件类型: {event_type}")
            
        self.event_bus.subscribe(event_type, handler, filter)
        
    def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
        self.event_bus.unsubscribe(event_type, handler)
        
    async def publish(self, event_type: str, data: Any,
                     source: str, metadata: Optional[Dict[str, Any]] = None):
        """发布事件"""
        if event_type not in self.event_types:
            self.logger.warning(f"未注册的事件类型: {event_type}")
            
        if source not in self.event_sources:
            self.logger.warning(f"未注册的事件源: {source}")
            
        return await self.event_bus.publish(event_type, data, source, metadata)
        
    def get_event_history(self, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        return self.event_bus.get_event_history(limit)
        
    def clear_history(self):
        """清除事件历史"""
        self.event_bus.clear_history()
        
    def save_state(self, path: str):
        """保存状态"""
        data = {
            "event_types": list(self.event_types),
            "event_sources": list(self.event_sources),
            "event_bus": self.event_bus
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_state(self, path: str):
        """加载状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.event_types = set(data["event_types"])
            self.event_sources = set(data["event_sources"])
            self.event_bus = data["event_bus"]
            
        except FileNotFoundError:
            self.logger.warning("状态文件不存在，使用默认状态") 