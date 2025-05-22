# -*- coding: utf-8 -*-
"""
事件循环系统 (Event Loop)

负责系统的持续运行、事件处理和智能体调度
实现系统的自我监控和自我重构能力
"""

import time
import threading
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from queue import PriorityQueue

from utils.event_system import EventSystem

class EventLoop:
    def __init__(self, event_system=None):
        """
        初始化事件循环系统
        
        Args:
            event_system: 事件系统实例，如果为None则创建新实例
        """
        self.event_system = event_system or EventSystem()
        self.running = False
        self.tasks = PriorityQueue()  # 优先级任务队列
        self.scheduled_tasks = []  # 定时任务列表
        self.loop_interval = 0.01  # 主循环间隔（秒）
        self.main_thread = None
        self.last_health_check = time.time()
        self.health_check_interval = 60  # 健康检查间隔（秒）
        self.system_metrics = {}
        self.logger = logging.getLogger("EventLoop")
        
    def start(self):
        """
        启动事件循环
        """
        if self.running:
            self.logger.warning("事件循环已经在运行中")
            return False
            
        self.running = True
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        self.logger.info("事件循环已启动")
        return True
        
    def stop(self):
        """
        停止事件循环
        """
        self.running = False
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
            
        self.logger.info("事件循环已停止")
        return True
        
    def _main_loop(self):
        """
        主循环逻辑
        """
        while self.running:
            try:
                # 处理优先级任务
                self._process_tasks()
                
                # 处理定时任务
                self._process_scheduled_tasks()
                
                # 系统健康检查
                self._health_check()
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(self.loop_interval)
                
            except Exception as e:
                self.logger.error(f"事件循环发生错误: {str(e)}")
                # 记录错误但不中断循环
                time.sleep(1.0)  # 错误后稍长暂停
    
    def _process_tasks(self):
        """
        处理优先级任务队列中的任务
        """
        # 处理当前队列中的所有任务，但限制每次循环处理的任务数量
        max_tasks_per_cycle = 10
        tasks_processed = 0
        
        while not self.tasks.empty() and tasks_processed < max_tasks_per_cycle:
            try:
                # 获取最高优先级的任务
                priority, task_id, task_func, args, kwargs = self.tasks.get_nowait()
                
                # 执行任务
                task_func(*args, **kwargs)
                
                # 标记任务完成
                self.tasks.task_done()
                tasks_processed += 1
                
            except Exception as e:
                self.logger.error(f"任务执行错误: {str(e)}")
    
    def _process_scheduled_tasks(self):
        """
        处理定时任务
        """
        current_time = time.time()
        tasks_to_run = []
        
        # 找出需要执行的定时任务
        for task in self.scheduled_tasks:
            if current_time >= task["next_run"]:
                tasks_to_run.append(task)
                
        # 执行需要运行的任务
        for task in tasks_to_run:
            try:
                # 执行任务
                task["func"](*task["args"], **task["kwargs"])
                
                # 更新下次执行时间
                if task["interval"] > 0:  # 重复任务
                    task["next_run"] = current_time + task["interval"]
                else:  # 一次性任务
                    self.scheduled_tasks.remove(task)
                    
            except Exception as e:
                self.logger.error(f"定时任务执行错误: {str(e)}")
    
    def _health_check(self):
        """
        系统健康检查
        """
        current_time = time.time()
        
        # 定期执行健康检查
        if current_time - self.last_health_check >= self.health_check_interval:
            self.last_health_check = current_time
            
            # 收集系统指标
            self.system_metrics = {
                "timestamp": current_time,
                "tasks_in_queue": self.tasks.qsize(),
                "scheduled_tasks": len(self.scheduled_tasks),
                "memory_usage": self._get_memory_usage(),
                "uptime": self._get_uptime()
            }
            
            # 发布健康状态事件
            self.event_system.publish("system.health_check", self.system_metrics)
            
    def _get_memory_usage(self):
        """
        获取当前内存使用情况
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # 转换为MB
        except ImportError:
            return 0
    
    def _get_uptime(self):
        """
        获取系统运行时间
        """
        # 这里应该返回系统启动至今的时间
        # 简化实现，返回0
        return 0
    
    def add_task(self, task_func, *args, priority=0, **kwargs):
        """
        添加任务到优先级队列
        
        Args:
            task_func: 任务函数
            *args: 位置参数
            priority: 优先级，数字越小优先级越高
            **kwargs: 关键字参数
            
        Returns:
            str: 任务ID
        """
        task_id = str(time.time())
        self.tasks.put((priority, task_id, task_func, args, kwargs))
        return task_id
    
    def schedule_task(self, task_func, interval, *args, delay=0, **kwargs):
        """
        调度定时任务
        
        Args:
            task_func: 任务函数
            interval: 执行间隔（秒），0表示一次性任务
            *args: 位置参数
            delay: 首次执行延迟（秒）
            **kwargs: 关键字参数
            
        Returns:
            str: 任务ID
        """
        task_id = str(time.time())
        task = {
            "id": task_id,
            "func": task_func,
            "interval": interval,
            "next_run": time.time() + delay,
            "args": args,
            "kwargs": kwargs
        }
        
        self.scheduled_tasks.append(task)
        return task_id
    
    def cancel_scheduled_task(self, task_id):
        """
        取消定时任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        for i, task in enumerate(self.scheduled_tasks):
            if task["id"] == task_id:
                del self.scheduled_tasks[i]
                return True
                
        return False