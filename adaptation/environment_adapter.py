# -*- coding: utf-8 -*-
"""
环境适应模块 (Environment Adapter)

负责感知和适应系统运行环境，动态调整系统行为和资源使用
实现系统在不同环境下的最优性能
"""

import os
import sys
import time
import logging
import platform
import psutil
from typing import Dict, List, Any, Optional, Tuple

class EnvironmentAdapter:
    def __init__(self):
        """
        初始化环境适应模块
        """
        self.logger = logging.getLogger("EnvironmentAdapter")
        
        # 环境信息
        self.environment_info = self._collect_environment_info()
        
        # 资源使用阈值
        self.resource_thresholds = {
            "cpu_percent": 80.0,  # CPU使用率阈值
            "memory_percent": 75.0,  # 内存使用率阈值
            "disk_percent": 90.0,  # 磁盘使用率阈值
            "network_usage": 80.0  # 网络使用率阈值
        }
        
        # 适应策略
        self.adaptation_strategies = {
            "high_cpu": self._adapt_high_cpu,
            "high_memory": self._adapt_high_memory,
            "high_disk": self._adapt_high_disk,
            "high_network": self._adapt_high_network,
            "low_resources": self._adapt_low_resources,
            "high_resources": self._adapt_high_resources
        }
        
        # 当前适应状态
        self.current_adaptations = set()
        
        # 资源监控间隔（秒）
        self.monitoring_interval = 30
        self.last_monitoring_time = 0
        
        # 环境变化历史
        self.environment_changes = []
        
        self.logger.info("环境适应模块初始化完成")
        self.logger.info(f"当前环境: {self.environment_info['os_name']} {self.environment_info['os_version']}")
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """
        收集环境信息
        
        Returns:
            Dict[str, Any]: 环境信息
        """
        info = {
            "os_name": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory": psutil.virtual_memory().total,
            "total_disk": {}
        }
        
        # 收集磁盘信息
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                info["total_disk"][partition.mountpoint] = partition_usage.total
            except Exception:
                pass
        
        return info
    
    def monitor_resources(self) -> Dict[str, Any]:
        """
        监控系统资源使用情况
        
        Returns:
            Dict[str, Any]: 资源使用情况
        """
        current_time = time.time()
        
        # 限制监控频率
        if current_time - self.last_monitoring_time < self.monitoring_interval:
            return {}
            
        self.last_monitoring_time = current_time
        
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory(),
            "disk": {},
            "network": psutil.net_io_counters(),
            "timestamp": current_time
        }
        
        # 收集磁盘使用情况
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                resources["disk"][partition.mountpoint] = {
                    "total": partition_usage.total,
                    "used": partition_usage.used,
                    "free": partition_usage.free,
                    "percent": partition_usage.percent
                }
            except Exception:
                pass
        
        # 检查是否需要适应
        self._check_adaptation_needed(resources)
        
        return resources
    
    def _check_adaptation_needed(self, resources: Dict[str, Any]):
        """
        检查是否需要进行环境适应
        
        Args:
            resources: 资源使用情况
        """
        adaptations_needed = set()
        
        # 检查CPU使用率
        if resources["cpu_percent"] > self.resource_thresholds["cpu_percent"]:
            adaptations_needed.add("high_cpu")
            
        # 检查内存使用率
        if resources["memory"].percent > self.resource_thresholds["memory_percent"]:
            adaptations_needed.add("high_memory")
            
        # 检查磁盘使用率
        for mount, usage in resources["disk"].items():
            if usage["percent"] > self.resource_thresholds["disk_percent"]:
                adaptations_needed.add("high_disk")
                break
        
        # 检查整体资源状态
        if len(adaptations_needed) >= 2:
            adaptations_needed.add("low_resources")
        elif len(adaptations_needed) == 0 and resources["cpu_percent"] < 30 and resources["memory"].percent < 30:
            adaptations_needed.add("high_resources")
        
        # 应用需要的适应策略
        for adaptation in adaptations_needed:
            if adaptation not in self.current_adaptations:
                self._apply_adaptation(adaptation)
                
        # 移除不再需要的适应
        for adaptation in list(self.current_adaptations):
            if adaptation not in adaptations_needed and adaptation not in ["low_resources", "high_resources"]:
                self._remove_adaptation(adaptation)
    
    def _apply_adaptation(self, adaptation_type: str):
        """
        应用适应策略
        
        Args:
            adaptation_type: 适应类型
        """
        if adaptation_type in self.adaptation_strategies:
            strategy = self.adaptation_strategies[adaptation_type]
            result = strategy(True)
            
            if result:
                self.current_adaptations.add(adaptation_type)
                self.logger.info(f"已应用环境适应策略: {adaptation_type}")
                
                # 记录环境变化
                self.environment_changes.append({
                    "type": "adaptation_applied",
                    "adaptation": adaptation_type,
                    "timestamp": time.time()
                })
    
    def _remove_adaptation(self, adaptation_type: str):
        """
        移除适应策略
        
        Args:
            adaptation_type: 适应类型
        """
        if adaptation_type in self.adaptation_strategies:
            strategy = self.adaptation_strategies[adaptation_type]
            result = strategy(False)
            
            if result and adaptation_type in self.current_adaptations:
                self.current_adaptations.remove(adaptation_type)
                self.logger.info(f"已移除环境适应策略: {adaptation_type}")
                
                # 记录环境变化
                self.environment_changes.append({
                    "type": "adaptation_removed",
                    "adaptation": adaptation_type,
                    "timestamp": time.time()
                })
    
    # 适应策略实现
    def _adapt_high_cpu(self, apply: bool) -> bool:
        """
        适应高CPU使用率
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现高CPU使用率适应策略
            # 例如：降低任务并行度、延迟非关键任务等
            return True
        else:
            # 恢复正常CPU使用策略
            return True
    
    def _adapt_high_memory(self, apply: bool) -> bool:
        """
        适应高内存使用率
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现高内存使用率适应策略
            # 例如：清理缓存、限制内存密集型操作等
            return True
        else:
            # 恢复正常内存使用策略
            return True
    
    def _adapt_high_disk(self, apply: bool) -> bool:
        """
        适应高磁盘使用率
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现高磁盘使用率适应策略
            # 例如：清理临时文件、限制日志大小等
            return True
        else:
            # 恢复正常磁盘使用策略
            return True
    
    def _adapt_high_network(self, apply: bool) -> bool:
        """
        适应高网络使用率
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现高网络使用率适应策略
            # 例如：限制非关键网络请求、降低数据传输频率等
            return True
        else:
            # 恢复正常网络使用策略
            return True
    
    def _adapt_low_resources(self, apply: bool) -> bool:
        """
        适应低资源环境
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现低资源环境适应策略
            # 例如：禁用非核心功能、降低处理精度等
            return True
        else:
            # 恢复正常资源使用策略
            return True
    
    def _adapt_high_resources(self, apply: bool) -> bool:
        """
        适应高资源环境
        
        Args:
            apply: 是否应用策略
            
        Returns:
            bool: 是否成功
        """
        if apply:
            # 实现高资源环境适应策略
            # 例如：启用高级功能、提高处理精度等
            return True
        else:
            # 恢复正常资源使用策略
            return True
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """
        获取当前适应状态
        
        Returns:
            Dict[str, Any]: 适应状态
        """
        return {
            "environment": self.environment_info,
            "current_adaptations": list(self.current_adaptations),
            "resource_thresholds": self.resource_thresholds,
            "last_monitoring_time": self.last_monitoring_time,
            "monitoring_interval": self.monitoring_interval,
            "recent_changes": self.environment_changes[-10:] if self.environment_changes else []
        }
    
    def set_resource_thresholds(self, thresholds: Dict[str, float]):
        """
        设置资源使用阈值
        
        Args:
            thresholds: 资源阈值配置
        """
        for key, value in thresholds.items():
            if key in self.resource_thresholds and isinstance(value, (int, float)):
                self.resource_thresholds[key] = value
                
        self.logger.info(f"已更新资源阈值: {self.resource_thresholds}")
    
    def set_monitoring_interval(self, interval: int):
        """
        设置监控间隔
        
        Args:
            interval: 监控间隔（秒）
        """
        if interval > 0:
            self.monitoring_interval = interval
            self.logger.info(f"已更新监控间隔: {interval}秒")