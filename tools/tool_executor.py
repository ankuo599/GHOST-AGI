# -*- coding: utf-8 -*-
"""
工具执行器 (Tool Executor)

负责管理和执行各种工具，提供统一的工具调用接口
支持工具的注册、调用参数验证和结果处理
"""

import time
import uuid
import importlib
import inspect
import logging
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Union

# 工具装饰器
def tool(name: str = None, description: str = "", required_params: List[str] = None, 
        optional_params: Dict[str, Any] = None):
    """
    工具装饰器，用于注册工具函数
    
    Args:
        name: 工具名称，默认为函数名
        description: 工具描述
        required_params: 必需参数列表
        optional_params: 可选参数及其默认值
    """
    def decorator(func):
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__
        func._required_params = required_params or []
        func._optional_params = optional_params or {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator

class ToolExecutor:
    def __init__(self, event_system=None, sandbox_enabled=True):
        """
        初始化工具执行器
        
        Args:
            event_system: 事件系统实例，用于发布工具执行相关事件
            sandbox_enabled: 是否启用安全沙箱，False表示无安全限制
        """
        # 安全配置
        self.sandbox_enabled = sandbox_enabled
        if not sandbox_enabled:
            logging.warning("警告：工具执行器安全沙箱已禁用，系统将以无限制模式运行")
        self.tools = {}  # 注册的工具
        self.tool_history = []  # 工具执行历史
        self.event_system = event_system
        self.logger = logging.getLogger("ToolExecutor")
        
    def register_tool(self, tool_name: str, tool_function: Callable, 
                     description: str = "", 
                     required_params: Optional[List[str]] = None,
                     optional_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册工具
        
        Args:
            tool_name: 工具名称
            tool_function: 工具函数
            description: 工具描述
            required_params: 必需参数列表
            optional_params: 可选参数及其默认值
            
        Returns:
            bool: 是否注册成功
        """
        if tool_name in self.tools:
            self.logger.warning(f"工具 '{tool_name}' 已存在，将被覆盖")
            
        self.tools[tool_name] = {
            "function": tool_function,
            "description": description or getattr(tool_function, "__doc__", ""),
            "required_params": required_params or [],
            "optional_params": optional_params or {}
        }
        
        self.logger.info(f"工具 '{tool_name}' 已注册")
        
        # 发布工具注册事件
        if self.event_system:
            self.event_system.publish("tool.registered", {
                "tool_name": tool_name,
                "description": description,
                "required_params": required_params,
                "optional_params": optional_params
            })
            
        return True
        
    def unregister_tool(self, tool_name: str) -> bool:
        """
        取消注册工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 是否取消成功
        """
        if tool_name not in self.tools:
            self.logger.warning(f"工具 '{tool_name}' 不存在，无法取消注册")
            return False
            
        del self.tools[tool_name]
        self.logger.info(f"工具 '{tool_name}' 已取消注册")
        
        # 发布工具取消注册事件
        if self.event_system:
            self.event_system.publish("tool.unregistered", {
                "tool_name": tool_name
            })
            
        return True
        
    def get_tool_info(self, tool_name: str = None) -> Dict[str, Any]:
        """
        获取工具信息
        
        Args:
            tool_name: 工具名称，如为None则返回所有工具
            
        Returns:
            Dict: 工具信息
        """
        if tool_name is None:
            # 返回所有工具信息
            result = {}
            for name, info in self.tools.items():
                result[name] = {
                    "description": info["description"],
                    "required_params": info["required_params"],
                    "optional_params": info["optional_params"]
                }
            return result
        
        if tool_name not in self.tools:
            self.logger.warning(f"工具 '{tool_name}' 不存在")
            return {}
            
        info = self.tools[tool_name]
        return {
            "description": info["description"],
            "required_params": info["required_params"],
            "optional_params": info["optional_params"]
        }
        
    def validate_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证工具参数
        
        Args:
            tool_name: 工具名称
            params: 参数字典
            
        Returns:
            Dict: 验证结果，包含是否有效、缺失参数和额外参数
        """
        if tool_name not in self.tools:
            return {"valid": False, "message": f"工具 '{tool_name}' 不存在"}
            
        tool_info = self.tools[tool_name]
        required_params = tool_info["required_params"]
        optional_params = tool_info["optional_params"]
        
        # 检查必需参数
        missing_params = []
        for param in required_params:
            if param not in params:
                missing_params.append(param)
                
        if missing_params:
            return {
                "valid": False, 
                "message": f"缺少必需参数: {', '.join(missing_params)}"
            }
            
        # 检查额外参数
        all_params = set(required_params) | set(optional_params.keys())
        extra_params = []
        for param in params:
            if param not in all_params:
                extra_params.append(param)
                
        # 构建参数字典，包含默认值
        validated_params = optional_params.copy()
        for param, value in params.items():
            if param not in extra_params:
                validated_params[param] = value
                
        return {
            "valid": True,
            "extra_params": extra_params,
            "validated_params": validated_params
        }
        
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            params: 参数字典
            
        Returns:
            Dict: 执行结果
        """
        params = params or {}
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            error_result = {
                "status": "error",
                "message": f"工具 '{tool_name}' 不存在",
                "execution_id": execution_id,
                "execution_time": 0
            }
            
            self._record_execution(tool_name, params, error_result, 0)
            return error_result
            
        # 验证参数
        validation = self.validate_params(tool_name, params)
        if not validation["valid"]:
            error_result = {
                "status": "error",
                "message": validation["message"],
                "execution_id": execution_id,
                "execution_time": 0
            }
            
            self._record_execution(tool_name, params, error_result, 0)
            return error_result
            
        # 获取验证后的参数
        validated_params = validation["validated_params"]
        
        # 执行工具
        try:
            if self.event_system:
                self.event_system.publish("tool.execution.started", {
                    "tool_name": tool_name,
                    "execution_id": execution_id,
                    "params": validated_params
                })
                
            # 调用工具函数
            tool_function = self.tools[tool_name]["function"]
            tool_result = tool_function(**validated_params)
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 构建结果
            if isinstance(tool_result, dict):
                if "status" not in tool_result:
                    tool_result["status"] = "success"
                    
                tool_result["execution_id"] = execution_id
                tool_result["execution_time"] = round(execution_time, 3)
                
                result = tool_result
            else:
                result = {
                    "status": "success",
                    "result": tool_result,
                    "execution_id": execution_id,
                    "execution_time": round(execution_time, 3)
                }
                
            # 发布工具执行完成事件
            if self.event_system:
                self.event_system.publish("tool.execution.completed", {
                    "tool_name": tool_name,
                    "execution_id": execution_id,
                    "execution_time": round(execution_time, 3),
                    "status": "success"
                })
                
        except Exception as e:
            # 执行出错
            execution_time = time.time() - start_time
            self.logger.error(f"工具 '{tool_name}' 执行出错: {str(e)}")
            
            result = {
                "status": "error",
                "message": f"工具执行出错: {str(e)}",
                "execution_id": execution_id,
                "execution_time": round(execution_time, 3)
            }
            
            # 发布工具执行错误事件
            if self.event_system:
                self.event_system.publish("tool.execution.failed", {
                    "tool_name": tool_name,
                    "execution_id": execution_id,
                    "execution_time": round(execution_time, 3),
                    "error": str(e)
                })
                
        # 记录执行
        self._record_execution(tool_name, params, result, execution_time)
        
        return result
        
    def _record_execution(self, tool_name: str, params: Dict[str, Any], 
                         result: Dict[str, Any], execution_time: float):
        """
        记录工具执行
        
        Args:
            tool_name: 工具名称
            params: 参数
            result: 执行结果
            execution_time: 执行时间
        """
        record = {
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
        
        # 限制历史记录长度
        max_history_length = 100
        self.tool_history.append(record)
        if len(self.tool_history) > max_history_length:
            self.tool_history = self.tool_history[-max_history_length:]
            
    def get_execution_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Args:
            limit: 限制返回数量，None为返回全部
            
        Returns:
            List: 执行历史记录
        """
        if limit is None:
            return self.tool_history
            
        return self.tool_history[-limit:]
        
    def clear_history(self) -> bool:
        """
        清空执行历史
        
        Returns:
            bool: 是否成功
        """
        self.tool_history = []
        self.logger.info("已清空工具执行历史")
        return True

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用工具
        
        Returns:
            List[Dict]: 工具信息列表
        """
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "required_params": info["required_params"],
                "optional_params": list(info["optional_params"].keys())
            }
            for info in self.tools.values()
        ]
        
    def load_tools_from_module(self, module_name: str) -> int:
        """
        从模块加载工具
        
        Args:
            module_name: 模块名称
            
        Returns:
            int: 加载的工具数量
        """
        try:
            module = importlib.import_module(module_name)
            
            # 查找模块中的工具函数
            tool_count = 0
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, "_is_tool") and obj._is_tool:
                    # 获取工具元数据
                    tool_name = getattr(obj, "_tool_name", name)
                    description = getattr(obj, "_tool_description", "")
                    required_params = getattr(obj, "_required_params", [])
                    optional_params = getattr(obj, "_optional_params", {})
                    
                    # 注册工具
                    self.register_tool(
                        tool_name=tool_name,
                        tool_function=obj,
                        description=description,
                        required_params=required_params,
                        optional_params=optional_params
                    )
                    tool_count += 1
                    
            return tool_count
        except Exception as e:
            self.logger.error(f"从模块 '{module_name}' 加载工具时出错: {str(e)}")
            return 0 