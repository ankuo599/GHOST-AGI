"""
动态内存管理器 (Dynamic Memory Manager)

提供智能的内存分配、优化和回收机制，实现系统资源的高效利用。
支持按优先级、使用频率和重要性智能管理内存资源。
"""

import time
import logging
import uuid
import os
import gc
import psutil
import threading
from typing import Dict, List, Any, Optional, Union, Set, Callable
from collections import defaultdict, deque, OrderedDict
import numpy as np

class MemoryItem:
    """内存项，表示存储在内存中的数据项"""
    
    def __init__(self, key: str, value: Any, size_bytes: int = 0):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes or self._estimate_size(value)
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.priority = 0.5  # 默认中等优先级
        self.locked = False  # 锁定状态，防止被回收
    
    def _estimate_size(self, obj: Any) -> int:
        """估计对象大小（字节）"""
        if hasattr(obj, "__sizeof__"):
            return obj.__sizeof__()
        return 100  # 默认估计大小

class LRUCache:
    """最近最少使用缓存实现"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()  # 有序字典，按访问顺序排列
        self.size_bytes = 0
    
    def get(self, key: str) -> Optional[MemoryItem]:
        """获取缓存项"""
        if key not in self.cache:
            return None
        
        # 移动到末尾（最近访问）
        item = self.cache.pop(key)
        self.cache[key] = item
        
        # 更新访问信息
        item.last_accessed = time.time()
        item.access_count += 1
        
        return item
    
    def put(self, item: MemoryItem) -> List[MemoryItem]:
        """
        添加缓存项
        
        Returns:
            List[MemoryItem]: 被驱逐的项
        """
        evicted = []
        
        # 如果已存在，先移除
        if item.key in self.cache:
            old_item = self.cache.pop(item.key)
            self.size_bytes -= old_item.size_bytes
        
        # 添加新项
        self.cache[item.key] = item
        self.size_bytes += item.size_bytes
        
        # 如果超过容量，驱逐最早的项
        while len(self.cache) > self.capacity:
            key, evicted_item = self.cache.popitem(last=False)  # 移除最早的项
            self.size_bytes -= evicted_item.size_bytes
            evicted.append(evicted_item)
        
        return evicted
    
    def remove(self, key: str) -> Optional[MemoryItem]:
        """移除缓存项"""
        if key not in self.cache:
            return None
        
        item = self.cache.pop(key)
        self.size_bytes -= item.size_bytes
        return item
    
    def clear(self) -> List[MemoryItem]:
        """清空缓存"""
        items = list(self.cache.values())
        self.cache.clear()
        self.size_bytes = 0
        return items

class PriorityMemoryStore:
    """基于优先级的内存存储"""
    
    def __init__(self, max_size_bytes: int = 1024 * 1024 * 100):  # 默认100MB
        self.max_size_bytes = max_size_bytes
        self.store = {}  # {key: MemoryItem}
        self.size_bytes = 0
    
    def get(self, key: str) -> Optional[MemoryItem]:
        """获取存储项"""
        if key not in self.store:
            return None
        
        item = self.store[key]
        
        # 更新访问信息
        item.last_accessed = time.time()
        item.access_count += 1
        
        return item
    
    def put(self, item: MemoryItem) -> List[MemoryItem]:
        """
        添加存储项
        
        Returns:
            List[MemoryItem]: 被驱逐的项
        """
        evicted = []
        
        # 如果已存在，先移除
        if item.key in self.store:
            old_item = self.store.pop(item.key)
            self.size_bytes -= old_item.size_bytes
        
        # 添加新项
        self.store[item.key] = item
        self.size_bytes += item.size_bytes
        
        # 如果超过容量，驱逐优先级最低的项
        if self.size_bytes > self.max_size_bytes:
            evicted = self._evict_items()
        
        return evicted
    
    def remove(self, key: str) -> Optional[MemoryItem]:
        """移除存储项"""
        if key not in self.store:
            return None
        
        item = self.store.pop(key)
        self.size_bytes -= item.size_bytes
        return item
    
    def _evict_items(self) -> List[MemoryItem]:
        """驱逐项目以释放空间"""
        # 计算需要释放的空间
        target_size = int(self.max_size_bytes * 0.8)  # 释放到80%容量
        to_free = self.size_bytes - target_size
        
        if to_free <= 0:
            return []
        
        # 按优先级和访问时间排序
        items = [(k, v) for k, v in self.store.items() if not v.locked]
        items.sort(key=lambda x: (x[1].priority, -time.time() + x[1].last_accessed))
        
        evicted = []
        freed = 0
        
        for key, item in items:
            if freed >= to_free:
                break
                
            self.store.pop(key)
            self.size_bytes -= item.size_bytes
            freed += item.size_bytes
            evicted.append(item)
        
        return evicted

class DynamicMemoryManager:
    """动态内存管理器，负责系统内存的智能管理"""
    
    def __init__(self, logger=None):
        """
        初始化动态内存管理器
        
        Args:
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 内存存储
        self.short_term_memory = LRUCache(capacity=10000)  # 短期内存（缓存）
        self.working_memory = PriorityMemoryStore(max_size_bytes=1024 * 1024 * 200)  # 工作内存
        self.long_term_memory = PriorityMemoryStore(max_size_bytes=1024 * 1024 * 500)  # 长期内存
        
        # 内存监控
        self.memory_usage_history = deque(maxlen=100)
        self.eviction_history = []
        self.collection_history = []
        
        # 内存统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
            "collections": 0,
            "total_stored_bytes": 0
        }
        
        # 回调函数
        self.on_evict_callbacks = []
        
        # 内存管理参数
        self.config = {
            "promotion_threshold": 10,  # 访问次数超过此值，从短期存储升级到工作存储
            "working_to_long_threshold": 30,  # 从工作存储升级到长期存储的阈值
            "auto_collect_interval": 300,  # 自动内存回收的间隔（秒）
            "max_memory_percent": 80,  # 系统内存使用的最大百分比，超过此值触发回收
            "priority_boost_rate": 0.05  # 每次访问提升的优先级
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 垃圾回收线程
        self.gc_thread = None
        self.is_running = False
        
        self.logger.info("动态内存管理器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("DynamicMemoryManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("memory_manager.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def start(self):
        """启动内存管理器和垃圾回收线程"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # 启动垃圾回收线程
        self.gc_thread = threading.Thread(target=self._gc_loop)
        self.gc_thread.daemon = True
        self.gc_thread.start()
        
        self.logger.info("动态内存管理器已启动")
    
    def stop(self):
        """停止内存管理器"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.gc_thread:
            self.gc_thread.join(timeout=2.0)
            
        self.logger.info("动态内存管理器已停止")
    
    def store(self, key: str, value: Any, 
             memory_type: str = "short_term", 
             priority: float = 0.5,
             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        存储数据
        
        Args:
            key: 数据键
            value: 数据值
            memory_type: 内存类型（short_term, working, long_term）
            priority: 优先级（0-1，越高越不易被驱逐）
            metadata: 元数据
            
        Returns:
            Dict: 存储结果
        """
        with self.lock:
            # 创建内存项
            item = MemoryItem(key, value)
            item.priority = priority
            
            # 添加元数据
            if metadata:
                item.metadata = metadata
            
            # 根据内存类型选择存储
            evicted = []
            if memory_type == "short_term":
                evicted = self.short_term_memory.put(item)
            elif memory_type == "working":
                evicted = self.working_memory.put(item)
            elif memory_type == "long_term":
                evicted = self.long_term_memory.put(item)
            else:
                return {
                    "status": "error",
                    "message": f"未知的内存类型: {memory_type}"
                }
            
            # 处理被驱逐的项
            for evicted_item in evicted:
                self._handle_eviction(evicted_item)
            
            # 更新统计
            self.stats["writes"] += 1
            self.stats["evictions"] += len(evicted)
            self.stats["total_stored_bytes"] = (
                self.short_term_memory.size_bytes + 
                self.working_memory.size_bytes + 
                self.long_term_memory.size_bytes
            )
            
            # 记录内存使用情况
            self._record_memory_usage()
            
            self.logger.info(f"已存储数据: {key} (大小: {item.size_bytes} 字节, 类型: {memory_type})")
            
            return {
                "status": "success",
                "key": key,
                "memory_type": memory_type,
                "size_bytes": item.size_bytes,
                "evicted_count": len(evicted)
            }
    
    def retrieve(self, key: str, promote: bool = True) -> Dict[str, Any]:
        """
        检索数据
        
        Args:
            key: 数据键
            promote: 是否提升优先级
            
        Returns:
            Dict: 检索结果
        """
        with self.lock:
            # 在各个内存区域查找
            memory_type = None
            item = None
            
            # 短期内存
            short_term_item = self.short_term_memory.get(key)
            if short_term_item:
                item = short_term_item
                memory_type = "short_term"
                
                # 如果访问次数超过阈值，考虑升级到工作内存
                if promote and item.access_count >= self.config["promotion_threshold"]:
                    self.short_term_memory.remove(key)
                    self.working_memory.put(item)
                    memory_type = "working"
                    self.logger.info(f"数据已从短期内存升级到工作内存: {key}")
            
            # 工作内存
            if not item:
                working_item = self.working_memory.get(key)
                if working_item:
                    item = working_item
                    memory_type = "working"
                    
                    # 如果访问次数超过阈值，考虑升级到长期内存
                    if promote and item.access_count >= self.config["working_to_long_threshold"]:
                        self.working_memory.remove(key)
                        self.long_term_memory.put(item)
                        memory_type = "long_term"
                        self.logger.info(f"数据已从工作内存升级到长期内存: {key}")
            
            # 长期内存
            if not item:
                long_term_item = self.long_term_memory.get(key)
                if long_term_item:
                    item = long_term_item
                    memory_type = "long_term"
            
            # 更新统计
            if item:
                self.stats["hits"] += 1
                
                # 提升优先级
                if promote:
                    item.priority = min(1.0, item.priority + self.config["priority_boost_rate"])
                
                return {
                    "status": "success",
                    "key": key,
                    "value": item.value,
                    "memory_type": memory_type,
                    "access_count": item.access_count,
                    "last_accessed": item.last_accessed,
                    "size_bytes": item.size_bytes
                }
            else:
                self.stats["misses"] += 1
                
                return {
                    "status": "not_found",
                    "key": key
                }
    
    def remove(self, key: str) -> Dict[str, Any]:
        """
        移除数据
        
        Args:
            key: 数据键
            
        Returns:
            Dict: 移除结果
        """
        with self.lock:
            # 在各个内存区域查找并移除
            item = None
            memory_type = None
            
            # 短期内存
            short_term_item = self.short_term_memory.remove(key)
            if short_term_item:
                item = short_term_item
                memory_type = "short_term"
            
            # 工作内存
            if not item:
                working_item = self.working_memory.remove(key)
                if working_item:
                    item = working_item
                    memory_type = "working"
            
            # 长期内存
            if not item:
                long_term_item = self.long_term_memory.remove(key)
                if long_term_item:
                    item = long_term_item
                    memory_type = "long_term"
            
            if item:
                # 更新统计
                self.stats["total_stored_bytes"] = (
                    self.short_term_memory.size_bytes + 
                    self.working_memory.size_bytes + 
                    self.long_term_memory.size_bytes
                )
                
                self.logger.info(f"已移除数据: {key} (来自: {memory_type})")
                
                return {
                    "status": "success",
                    "key": key,
                    "memory_type": memory_type,
                    "size_bytes": item.size_bytes
                }
            else:
                return {
                    "status": "not_found",
                    "key": key
                }
    
    def lock_item(self, key: str, lock_state: bool = True) -> Dict[str, Any]:
        """
        锁定或解锁内存项（防止被驱逐）
        
        Args:
            key: 数据键
            lock_state: 锁定状态
            
        Returns:
            Dict: 锁定结果
        """
        with self.lock:
            # 在各个内存区域查找
            item = None
            memory_type = None
            
            # 短期内存
            short_term_item = self.short_term_memory.get(key)
            if short_term_item:
                item = short_term_item
                memory_type = "short_term"
            
            # 工作内存
            if not item:
                working_item = self.working_memory.get(key)
                if working_item:
                    item = working_item
                    memory_type = "working"
            
            # 长期内存
            if not item:
                long_term_item = self.long_term_memory.get(key)
                if long_term_item:
                    item = long_term_item
                    memory_type = "long_term"
            
            if item:
                item.locked = lock_state
                
                self.logger.info(f"数据 {key} {'锁定' if lock_state else '解锁'} (来自: {memory_type})")
                
                return {
                    "status": "success",
                    "key": key,
                    "locked": lock_state,
                    "memory_type": memory_type
                }
            else:
                return {
                    "status": "not_found",
                    "key": key
                }
    
    def collect_garbage(self, force: bool = False) -> Dict[str, Any]:
        """
        执行垃圾回收
        
        Args:
            force: 强制回收
            
        Returns:
            Dict: 回收结果
        """
        with self.lock:
            start_time = time.time()
            system_memory_used = self._get_system_memory_percent()
            
            # 只有在内存使用超过阈值或强制回收时才执行
            if not force and system_memory_used < self.config["max_memory_percent"]:
                return {
                    "status": "skipped",
                    "reason": "内存使用率低于阈值",
                    "memory_used_percent": system_memory_used,
                    "threshold_percent": self.config["max_memory_percent"]
                }
            
            # 回收前的内存使用
            before_size = (
                self.short_term_memory.size_bytes + 
                self.working_memory.size_bytes + 
                self.long_term_memory.size_bytes
            )
            
            # 执行回收
            evicted = []
            
            # 回收短期内存中的旧项
            if force or system_memory_used > self.config["max_memory_percent"] * 0.8:
                old_threshold = time.time() - 3600  # 1小时前
                keys_to_remove = []
                
                for key, item in list(self.short_term_memory.cache.items()):
                    if not item.locked and item.last_accessed < old_threshold:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    item = self.short_term_memory.remove(key)
                    if item:
                        evicted.append(item)
            
            # 回收长期内存中的低优先级项
            if force or system_memory_used > self.config["max_memory_percent"] * 0.9:
                target_size = int(self.long_term_memory.max_size_bytes * 0.8)  # 减少到80%
                
                if self.long_term_memory.size_bytes > target_size:
                    items = [(k, v) for k, v in self.long_term_memory.store.items() if not v.locked]
                    items.sort(key=lambda x: x[1].priority)
                    
                    to_free = self.long_term_memory.size_bytes - target_size
                    freed = 0
                    
                    for key, item in items:
                        if freed >= to_free:
                            break
                            
                        self.long_term_memory.remove(key)
                        freed += item.size_bytes
                        evicted.append(item)
            
            # 调用Python垃圾回收
            gc.collect()
            
            # 处理被驱逐的项
            for item in evicted:
                self._handle_eviction(item)
            
            # 更新统计
            self.stats["collections"] += 1
            self.stats["evictions"] += len(evicted)
            self.stats["total_stored_bytes"] = (
                self.short_term_memory.size_bytes + 
                self.working_memory.size_bytes + 
                self.long_term_memory.size_bytes
            )
            
            # 计算释放的空间
            after_size = self.stats["total_stored_bytes"]
            freed_bytes = before_size - after_size
            
            # 记录回收历史
            collection_record = {
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "evicted_count": len(evicted),
                "freed_bytes": freed_bytes,
                "before_size": before_size,
                "after_size": after_size,
                "system_memory_before": system_memory_used
            }
            self.collection_history.append(collection_record)
            
            self.logger.info(f"垃圾回收完成，释放了 {freed_bytes} 字节，驱逐了 {len(evicted)} 个项")
            
            return {
                "status": "success",
                "evicted_count": len(evicted),
                "freed_bytes": freed_bytes,
                "duration": time.time() - start_time,
                "memory_used_percent": self._get_system_memory_percent()
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self.lock:
            total_items = (
                len(self.short_term_memory.cache) + 
                len(self.working_memory.store) + 
                len(self.long_term_memory.store)
            )
            
            # 计算命中率
            total_retrievals = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(1, total_retrievals)
            
            # 获取系统内存使用情况
            system_memory = psutil.virtual_memory()
            
            return {
                "status": "success",
                "timestamp": time.time(),
                "total_items": total_items,
                "short_term_items": len(self.short_term_memory.cache),
                "working_items": len(self.working_memory.store),
                "long_term_items": len(self.long_term_memory.store),
                "total_size_bytes": self.stats["total_stored_bytes"],
                "short_term_size_bytes": self.short_term_memory.size_bytes,
                "working_size_bytes": self.working_memory.size_bytes,
                "long_term_size_bytes": self.long_term_memory.size_bytes,
                "hit_rate": hit_rate,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "writes": self.stats["writes"],
                "evictions": self.stats["evictions"],
                "collections": self.stats["collections"],
                "system_memory_used_percent": system_memory.percent,
                "system_memory_available_bytes": system_memory.available
            }
    
    def register_eviction_callback(self, callback: Callable) -> str:
        """
        注册驱逐回调函数
        
        Args:
            callback: 回调函数，接收被驱逐的内存项
            
        Returns:
            str: 回调ID
        """
        callback_id = str(uuid.uuid4())
        
        self.on_evict_callbacks.append({
            "id": callback_id,
            "callback": callback
        })
        
        return callback_id
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新配置
        
        Args:
            config_updates: 配置更新
            
        Returns:
            Dict: 更新后的配置
        """
        with self.lock:
            for key, value in config_updates.items():
                if key in self.config:
                    self.config[key] = value
                    
            return dict(self.config)
    
    def clear_memory(self, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        清空内存
        
        Args:
            memory_type: 内存类型，如果为None则清空所有
            
        Returns:
            Dict: 清空结果
        """
        with self.lock:
            cleared = {
                "short_term": 0,
                "working": 0,
                "long_term": 0
            }
            
            # 清空指定内存或所有内存
            if memory_type in [None, "short_term"]:
                items = self.short_term_memory.clear()
                cleared["short_term"] = len(items)
                
                # 处理被驱逐的项
                for item in items:
                    self._handle_eviction(item)
            
            if memory_type in [None, "working"]:
                items = list(self.working_memory.store.values())
                self.working_memory.store.clear()
                self.working_memory.size_bytes = 0
                cleared["working"] = len(items)
                
                # 处理被驱逐的项
                for item in items:
                    self._handle_eviction(item)
            
            if memory_type in [None, "long_term"]:
                items = list(self.long_term_memory.store.values())
                self.long_term_memory.store.clear()
                self.long_term_memory.size_bytes = 0
                cleared["long_term"] = len(items)
                
                # 处理被驱逐的项
                for item in items:
                    self._handle_eviction(item)
            
            # 更新统计
            self.stats["total_stored_bytes"] = (
                self.short_term_memory.size_bytes + 
                self.working_memory.size_bytes + 
                self.long_term_memory.size_bytes
            )
            
            total_cleared = sum(cleared.values())
            self.logger.info(f"内存已清空: {memory_type or '所有'}，清除了 {total_cleared} 个项")
            
            return {
                "status": "success",
                "cleared": cleared,
                "total_cleared": total_cleared
            }
    
    def _gc_loop(self):
        """垃圾回收循环"""
        last_collection_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 检查是否需要执行垃圾回收
                if current_time - last_collection_time > self.config["auto_collect_interval"]:
                    result = self.collect_garbage()
                    
                    if result["status"] == "success":
                        last_collection_time = current_time
                
                # 检查系统内存使用情况
                system_memory_used = self._get_system_memory_percent()
                
                # 如果内存使用率过高，立即执行回收
                if system_memory_used > self.config["max_memory_percent"]:
                    self.collect_garbage(force=True)
                    last_collection_time = time.time()
                
                # 记录内存使用情况
                self._record_memory_usage()
                
                # 避免过度占用CPU
                time.sleep(10)
            except Exception as e:
                self.logger.error(f"垃圾回收循环出错: {str(e)}")
                time.sleep(30)  # 出错后延长休眠时间
    
    def _get_system_memory_percent(self) -> float:
        """获取系统内存使用百分比"""
        return psutil.virtual_memory().percent
    
    def _record_memory_usage(self):
        """记录内存使用情况"""
        system_memory = psutil.virtual_memory()
        
        usage_record = {
            "timestamp": time.time(),
            "managed_bytes": self.stats["total_stored_bytes"],
            "system_percent": system_memory.percent,
            "system_available": system_memory.available,
            "short_term_bytes": self.short_term_memory.size_bytes,
            "working_bytes": self.working_memory.size_bytes,
            "long_term_bytes": self.long_term_memory.size_bytes
        }
        
        self.memory_usage_history.append(usage_record)
    
    def _handle_eviction(self, item: MemoryItem):
        """处理被驱逐的内存项"""
        # 记录驱逐历史
        eviction_record = {
            "key": item.key,
            "size_bytes": item.size_bytes,
            "created_at": item.created_at,
            "last_accessed": item.last_accessed,
            "access_count": item.access_count,
            "priority": item.priority,
            "timestamp": time.time()
        }
        self.eviction_history.append(eviction_record)
        
        # 调用回调函数
        for callback_info in self.on_evict_callbacks:
            try:
                callback_info["callback"](item)
            except Exception as e:
                self.logger.error(f"驱逐回调函数出错: {str(e)}") 