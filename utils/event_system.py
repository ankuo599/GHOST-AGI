# -*- coding: utf-8 -*-
"""
事件系统 (Event System)

提供基于发布-订阅模式的事件通信机制，使系统各组件能有效协作
支持事件过滤、优先级和异步处理
"""

import time
import uuid
import threading
import queue
from typing import Dict, List, Any, Callable, Optional, Set, Union

class EventSystem:
    def __init__(self):
        """
        初始化事件系统
        """
        self.handlers = {}  # 事件处理器映射
        self.event_history = []  # 事件历史记录
        self.max_history = 1000  # 最大历史记录数
        self.async_mode = False  # 异步模式
        self.event_queue = queue.PriorityQueue()  # 事件队列（用于异步处理）
        self.running = False  # 标记事件循环是否运行
        self.event_thread = None  # 事件处理线程
        self.lock = threading.RLock()  # 线程锁
        
    def subscribe(self, event_type: str, handler: Callable, 
                  filter_condition: Optional[Callable] = None, 
                  priority: int = 0) -> str:
        """
        订阅特定类型的事件
        
        Args:
            event_type (str): 事件类型
            handler (Callable): 事件处理函数
            filter_condition (Callable, optional): 事件过滤条件
            priority (int): 处理优先级（较低的值表示较高的优先级）
            
        Returns:
            str: 订阅ID，用于取消订阅
        """
        subscription_id = str(uuid.uuid4())
        
        with self.lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
                
            self.handlers[event_type].append({
                "id": subscription_id,
                "handler": handler,
                "filter": filter_condition,
                "priority": priority
            })
            
            # 按优先级排序
            self.handlers[event_type].sort(key=lambda x: x["priority"])
            
        return subscription_id
        
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_id (str): 订阅ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        with self.lock:
            for event_type, handlers in self.handlers.items():
                for i, handler_info in enumerate(handlers):
                    if handler_info["id"] == subscription_id:
                        self.handlers[event_type].pop(i)
                        return True
                        
        return False
        
    def publish(self, event_type: str, data: Any, 
                metadata: Optional[Dict[str, Any]] = None, 
                priority: int = 0) -> str:
        """
        发布事件并通知订阅者
        
        Args:
            event_type (str): 事件类型
            data (Any): 事件数据
            metadata (Dict[str, Any], optional): 事件元数据
            priority (int): 事件优先级（较低的值表示较高的优先级）
            
        Returns:
            str: 事件ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "type": event_type,
            "data": data,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "priority": priority
        }
        
        # 记录事件
        with self.lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # 异步模式：将事件加入队列
        if self.async_mode and self.running:
            self.event_queue.put((priority, event))
            return event_id
        
        # 同步模式：直接通知订阅者
        self._notify_subscribers(event)
        return event_id
        
    def _notify_subscribers(self, event: Dict[str, Any]) -> None:
        """
        通知所有订阅者
        
        Args:
            event (Dict[str, Any]): 事件对象
        """
        event_type = event["type"]
        
        if event_type not in self.handlers:
            return
            
        for handler_info in self.handlers[event_type]:
            filter_fn = handler_info["filter"]
            
            # 应用过滤条件
            if filter_fn is not None and not filter_fn(event["data"]):
                continue
                
            try:
                # 调用处理器
                handler_info["handler"](event)
            except Exception as e:
                import traceback
                error_info = {
                    "error": str(e),
                    "event_id": event["id"],
                    "event_type": event_type,
                    "traceback": traceback.format_exc()
                }
                print(f"事件处理错误: {error_info}")
                
    def start_async(self) -> bool:
        """
        启动异步事件处理循环
        
        Returns:
            bool: 是否成功启动
        """
        if self.running:
            return False
            
        self.async_mode = True
        self.running = True
        self.event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self.event_thread.start()
        
        return True
        
    def stop_async(self) -> bool:
        """
        停止异步事件处理循环
        
        Returns:
            bool: 是否成功停止
        """
        if not self.running:
            return False
            
        self.running = False
        if self.event_thread:
            self.event_thread.join(timeout=1.0)
            
        return True
        
    def _event_loop(self) -> None:
        """
        事件处理循环
        """
        while self.running:
            try:
                # 尝试获取事件，设置超时以便能响应停止请求
                try:
                    priority, event = self.event_queue.get(timeout=0.1)
                    self._notify_subscribers(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"事件循环错误: {str(e)}")
                
    def get_history(self, event_type: Optional[str] = None, 
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取事件历史
        
        Args:
            event_type (str, optional): 事件类型过滤
            limit (int, optional): 返回的事件数量限制
            
        Returns:
            List[Dict[str, Any]]: 事件历史列表
        """
        with self.lock:
            if event_type:
                filtered = [e for e in self.event_history if e["type"] == event_type]
            else:
                filtered = self.event_history[:]
                
            if limit and limit > 0:
                return filtered[-limit:]
            return filtered
            
    def clear_history(self) -> None:
        """
        清除事件历史
        """
        with self.lock:
            self.event_history = []
            
    def get_subscription_count(self, event_type: Optional[str] = None) -> int:
        """
        获取订阅数量
        
        Args:
            event_type (str, optional): 事件类型
            
        Returns:
            int: 订阅数量
        """
        with self.lock:
            if event_type:
                return len(self.handlers.get(event_type, []))
            return sum(len(handlers) for handlers in self.handlers.values())
            
    def get_subscribers(self, event_type: str) -> List[Dict[str, Any]]:
        """
        获取特定事件类型的所有订阅者信息
        
        Args:
            event_type (str): 事件类型
            
        Returns:
            List[Dict[str, Any]]: 订阅者信息列表
        """
        with self.lock:
            return [
                {
                    "id": h["id"],
                    "priority": h["priority"],
                    "has_filter": h["filter"] is not None
                }
                for h in self.handlers.get(event_type, [])
            ]