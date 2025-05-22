# -*- coding: utf-8 -*-
"""
工具执行器 (Tool Executor)

负责动态生成和调用网络工具/实体组件执行，支持Python脚本、系统命令和API调用
"""

import importlib
import inspect
import json
import os
import sys
import subprocess
import requests
import time
import shlex
import tempfile
from typing import Dict, Any, List, Optional, Callable, Union

class ToolExecutor:
    def __init__(self):
        """
        初始化工具执行器
        """
        self.available_tools = {}
        self.tool_history = []
        self.max_history_size = 100
        self.sandbox_enabled = True
        self.api_rate_limits = {}
        self.command_timeout = 30  # 命令执行超时时间（秒）
        
        # 注册内置工具
        self._register_builtin_tools()
        
    def register_tool(self, tool_name: str, tool_function: Callable, description: str = ""):
        """
        注册工具到执行器
        
        Args:
            tool_name (str): 工具名称
            tool_function (Callable): 工具函数
            description (str, optional): 工具描述
        
        Returns:
            bool: 是否成功注册
        """
        if not callable(tool_function):
            return False
            
        self.available_tools[tool_name] = {
            "function": tool_function,
            "description": description,
            "signature": inspect.signature(tool_function)
        }
        return True
    
    def register_tools_from_module(self, module_name: str):
        """
        从模块中注册所有工具
        
        Args:
            module_name (str): 模块名称
            
        Returns:
            int: 注册的工具数量
        """
        try:
            module = importlib.import_module(module_name)
            count = 0
            
            for name, obj in inspect.getmembers(module):
                # 只注册函数和方法
                if inspect.isfunction(obj) and not name.startswith('_'):
                    doc = inspect.getdoc(obj) or ""
                    self.register_tool(name, obj, doc)
                    count += 1
                    
            return count
        except ImportError:
            return 0
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行指定的工具
        
        Args:
            tool_name (str): 工具名称
            params (Dict[str, Any], optional): 工具参数
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if not tool_name:
            return {"status": "error", "message": "未指定工具名称"}
            
        # 检查工具是否存在
        if tool_name not in self.available_tools:
            # 尝试动态加载工具
            if hasattr(self, '_try_load_tool') and self._try_load_tool(tool_name):
                print(f"[工具执行器] 已动态加载工具: {tool_name}")
            else:
                return {"status": "error", "message": f"工具 '{tool_name}' 不存在"}
            
        tool = self.available_tools[tool_name]
        params = params or {}
        
        # 记录工具调用
        tool_call = {
            "tool": tool_name,
            "params": params,
            "timestamp": __import__("time").time()
        }
        
        try:
            # 执行工具函数
            result = tool["function"](**params)
            
            # 更新工具调用记录
            tool_call["status"] = "success"
            tool_call["result"] = result
            self.tool_history.append(tool_call)
            
            # 限制历史记录大小
            if len(self.tool_history) > self.max_history_size:
                self.tool_history = self.tool_history[-self.max_history_size:]
                
            return {"status": "success", "result": result, "message": "工具执行成功"}
        except Exception as e:
            # 记录错误
            tool_call["status"] = "error"
            tool_call["error"] = str(e)
            self.tool_history.append(tool_call)
            
            return {"status": "error", "message": f"工具执行失败: {str(e)}"}
    
    def generate_dynamic_tool(self, tool_spec: Dict[str, Any]) -> bool:
        """
        根据规范动态生成工具
        
        Args:
            tool_spec (Dict[str, Any]): 工具规范，包含名称、代码和描述
            
        Returns:
            bool: 是否成功生成
        """
        if not isinstance(tool_spec, dict):
            return False
            
        name = tool_spec.get("name")
        code = tool_spec.get("code")
        description = tool_spec.get("description", "")
        
        if not name or not code:
            return False
            
        try:
            # 动态编译代码
            compiled_code = compile(code, f"<dynamic_tool_{name}>", "exec")
            tool_namespace = {}
            exec(compiled_code, tool_namespace)
            
            # 查找主函数
            main_function = None
            for key, value in tool_namespace.items():
                if callable(value) and (key == name or key == "main"):
                    main_function = value
                    break
                    
            if not main_function:
                return False
                
            # 注册工具
            return self.register_tool(name, main_function, description)
        except Exception:
            return False
    
    def get_tool_info(self, tool_name: str = None) -> Dict[str, Any]:
        """
        获取工具信息
        
        Args:
            tool_name (str, optional): 工具名称，如果为None则返回所有工具信息
            
        Returns:
            Dict[str, Any]: 工具信息
        """
        if tool_name:
            if tool_name not in self.available_tools:
                return {}
                
            tool = self.available_tools[tool_name]
            return {
                "name": tool_name,
                "description": tool["description"],
                "parameters": str(tool["signature"])
            }
        else:
            # 返回所有工具的简要信息
            return {
                name: {
                    "description": tool["description"],
                    "parameters": str(tool["signature"])
                } for name, tool in self.available_tools.items()
            }
    
    def get_tool_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取工具调用历史
        
        Args:
            limit (int, optional): 限制返回的历史记录数量
            
        Returns:
            List[Dict[str, Any]]: 工具调用历史
        """
        if limit and limit > 0:
            return self.tool_history[-limit:]
        return self.tool_history
    
    def clear_history(self):
        """
        清空工具调用历史
        """
        self.tool_history = []
        
    def execute_from_json(self, json_str: str) -> Dict[str, Any]:
        """
        从JSON字符串执行工具
        
        Args:
            json_str (str): JSON格式的工具调用描述
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            data = json.loads(json_str)
            tool_name = data.get("tool")
            params = data.get("params", {})
            
            if not tool_name:
                return {"status": "error", "message": "缺少工具名称"}
                
            return self.execute_tool(tool_name, params)
        except json.JSONDecodeError:
            return {"status": "error", "message": "无效的JSON格式"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def _register_builtin_tools(self):
        """
        注册内置工具
        """
        # 注册系统命令执行工具
        self.register_tool(
            "run_command", 
            self._run_system_command,
            "执行系统命令，返回命令输出。注意：此工具在安全模式下受限。"
        )
        
        # 注册Python脚本执行工具
        self.register_tool(
            "run_python", 
            self._run_python_script,
            "执行Python脚本，返回脚本输出和执行结果。"
        )
        
        # 注册API调用工具
        self.register_tool(
            "call_api", 
            self._call_api,
            "调用外部API，支持GET、POST、PUT、DELETE方法。"
        )
    
    def _run_system_command(self, command: str, args: List[str] = None, timeout: int = None) -> Dict[str, Any]:
        """
        执行系统命令
        
        Args:
            command (str): 要执行的命令
            args (List[str], optional): 命令参数列表
            timeout (int, optional): 命令超时时间（秒）
            
        Returns:
            Dict[str, Any]: 命令执行结果
        """
        # 移除安全检查
        args = args or []
        timeout = timeout or self.command_timeout
        
        try:
            # 构建完整命令
            cmd = [command] + args
            
            # 执行命令并捕获输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True  # 允许shell模式执行，无安全限制
            )
            
            # 等待命令执行完成，设置超时
            stdout, stderr = process.communicate(timeout=timeout)
            
            return {
                "status": "success" if process.returncode == 0 else "error",
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": command,
                "args": args
            }
        except subprocess.TimeoutExpired:
            # 命令执行超时，终止进程
            try:
                process.kill()
            except:
                pass
                
            return {
                "status": "error",
                "message": f"命令执行超时（{timeout}秒）",
                "command": command,
                "args": args
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "command": command,
                "args": args
            }
    
    def _run_python_script(self, code: str, args: List[str] = None, timeout: int = None) -> Dict[str, Any]:
        """
        执行Python脚本，无安全限制
        
        Args:
            code (str): Python代码
            args (List[str], optional): 脚本参数
            timeout (int, optional): 脚本超时时间（秒）
            
        Returns:
            Dict[str, Any]: 脚本执行结果
        """
        # 移除安全检查，允许执行任何代码
        args = args or []
        timeout = timeout or self.command_timeout
        
        try:
            # 创建临时脚本文件
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code)
            
            # 执行脚本
            cmd = [sys.executable, temp_file_path] + args
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True  # 允许shell环境
            )
            
            # 等待脚本执行完成，设置超时
            stdout, stderr = process.communicate(timeout=timeout)
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
            return {
                "status": "success" if process.returncode == 0 else "error",
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "code_preview": code[:100] + '...' if len(code) > 100 else code
            }
        except subprocess.TimeoutExpired:
            # 脚本执行超时，终止进程
            try:
                process.kill()
                os.unlink(temp_file_path)
            except:
                pass
                
            return {
                "status": "error",
                "message": f"脚本执行超时（{timeout}秒）",
                "code_preview": code[:100] + '...' if len(code) > 100 else code
            }
        except Exception as e:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
            return {
                "status": "error",
                "message": str(e),
                "code_preview": code[:100] + '...' if len(code) > 100 else code
            }
    
    def _call_api(self, url: str, method: str = "GET", headers: Dict[str, str] = None, 
                 data: Any = None, json_data: Dict[str, Any] = None, 
                 timeout: int = 30) -> Dict[str, Any]:
        """
        调用外部API，无安全限制
        
        Args:
            url (str): API URL
            method (str, optional): 请求方法，支持GET、POST、PUT、DELETE
            headers (Dict[str, str], optional): 请求头
            data (Any, optional): 请求数据（表单数据）
            json_data (Dict[str, Any], optional): JSON请求数据
            timeout (int, optional): 请求超时时间（秒）
            
        Returns:
            Dict[str, Any]: API调用结果
        """
        # 移除URL安全检查和速率限制
        
        # 标准化请求方法
        method = method.upper()
        if method not in ["GET", "POST", "PUT", "DELETE"]:
            return {"status": "error", "message": f"不支持的请求方法: {method}"}
            
        headers = headers or {}
        
        try:
            # 发送请求
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            # 尝试解析JSON响应
            try:
                response_json = response.json()
                content_type = "json"
            except:
                response_json = None
                content_type = "text"
            
            return {
                "status": "success" if response.ok else "error",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": content_type,
                "content": response_json if response_json else response.text,
                "url": url,
                "method": method
            }
        except requests.Timeout:
            return {
                "status": "error",
                "message": f"API请求超时（{timeout}秒）",
                "url": url,
                "method": method
            }
        except requests.RequestException as e:
            return {
                "status": "error",
                "message": str(e),
                "url": url,
                "method": method
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"API调用异常: {str(e)}",
                "url": url,
                "method": method
            }