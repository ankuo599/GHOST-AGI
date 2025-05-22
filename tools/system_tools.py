# -*- coding: utf-8 -*-
"""
系统工具模块 (System Tools)

提供系统操作、文件处理和资源监控功能
"""

import os
import sys
import time
import platform
import subprocess
import json
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from .tool_executor import tool

@tool(
    name="system_info",
    description="获取系统信息",
    required_params=[],
    optional_params={}
)
def system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        Dict: 系统信息
    """
    try:
        # 获取系统和平台信息
        system_data = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        # 获取内存信息
        memory = psutil.virtual_memory()
        memory_data = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # 获取CPU信息
        cpu_data = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.1, percpu=False),
            "per_cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True)
        }
        
        # 获取磁盘信息
        disk_data = []
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_data.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "file_system": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                })
            except:
                pass
                
        # 获取网络信息
        net_io = psutil.net_io_counters()
        net_data = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        return {
            "status": "success",
            "system": system_data,
            "memory": memory_data,
            "cpu": cpu_data,
            "disk": disk_data,
            "network": net_data,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@tool(
    name="file_operations",
    description="执行文件操作",
    required_params=["operation", "path"],
    optional_params={"content": None, "mode": "r", "encoding": "utf-8"}
)
def file_operations(operation: str, path: str, content: Optional[str] = None, 
                  mode: str = "r", encoding: str = "utf-8") -> Dict[str, Any]:
    """
    执行文件操作
    
    Args:
        operation: 操作类型 (read, write, append, delete, exists, info)
        path: 文件路径
        content: 要写入的内容
        mode: 文件打开模式
        encoding: 文件编码
        
    Returns:
        Dict: 操作结果
    """
    try:
        # 检查文件是否存在
        file_exists = os.path.exists(path)
        
        if operation == "exists":
            return {
                "status": "success",
                "exists": file_exists,
                "path": path,
                "timestamp": time.time()
            }
            
        # 获取文件信息
        if operation == "info":
            if not file_exists:
                return {
                    "status": "error",
                    "message": f"文件 '{path}' 不存在"
                }
                
            stats = os.stat(path)
            return {
                "status": "success",
                "path": path,
                "size": stats.st_size,
                "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stats.st_atime).isoformat(),
                "is_dir": os.path.isdir(path),
                "is_file": os.path.isfile(path),
                "permissions": oct(stats.st_mode)[-3:],
                "timestamp": time.time()
            }
            
        # 读取文件
        if operation == "read":
            if not file_exists:
                return {
                    "status": "error",
                    "message": f"文件 '{path}' 不存在"
                }
                
            with open(path, mode, encoding=encoding) as file:
                content = file.read()
                
            return {
                "status": "success",
                "path": path,
                "content": content,
                "size": len(content),
                "timestamp": time.time()
            }
            
        # 写入文件
        if operation in ["write", "append"]:
            if operation == "write":
                mode = "w"
            else:
                mode = "a"
                
            # 确保目录存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            with open(path, mode, encoding=encoding) as file:
                file.write(content or "")
                
            return {
                "status": "success",
                "operation": operation,
                "path": path,
                "size": len(content or ""),
                "timestamp": time.time()
            }
            
        # 删除文件
        if operation == "delete":
            if not file_exists:
                return {
                    "status": "error",
                    "message": f"文件 '{path}' 不存在"
                }
                
            if os.path.isdir(path):
                os.rmdir(path)
            else:
                os.remove(path)
                
            return {
                "status": "success",
                "operation": "delete",
                "path": path,
                "timestamp": time.time()
            }
            
        return {
            "status": "error",
            "message": f"不支持的操作 '{operation}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@tool(
    name="execute_command",
    description="执行系统命令，无安全限制",
    required_params=["command"],
    optional_params={"timeout": None, "shell": True, "cwd": None, "env": None}
)
def execute_command(command: str, timeout: Optional[int] = None, shell: bool = True, 
                  cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    执行系统命令，无任何安全限制和超时控制
    
    Args:
        command: 要执行的命令
        timeout: 超时时间(秒)，None表示无超时限制
        shell: 是否使用shell执行，默认True允许shell功能
        cwd: 工作目录
        env: 环境变量
        
    Returns:
        Dict: 执行结果
    """
    start_time = time.time()
    
    try:
        # 无限制执行任何命令
        result = subprocess.run(
            command,
            shell=shell,  # 允许shell命令，提高灵活性
            cwd=cwd,
            env=env,
            timeout=timeout,  # 可能为None，表示无超时限制
            capture_output=True,
            text=True
        )
        
        # 处理执行结果
        return {
            "status": "success",
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"命令执行超时",
            "command": command,
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "command": command,
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

@tool(
    name="directory_operations",
    description="执行目录操作",
    required_params=["operation", "path"],
    optional_params={"recursive": False, "pattern": "*"}
)
def directory_operations(operation: str, path: str, recursive: bool = False, 
                       pattern: str = "*") -> Dict[str, Any]:
    """
    执行目录操作
    
    Args:
        operation: 操作类型 (list, create, delete, find)
        path: 目录路径
        recursive: 是否递归操作
        pattern: 文件模式匹配
        
    Returns:
        Dict: 操作结果
    """
    try:
        # 检查目录是否存在
        dir_exists = os.path.exists(path) and os.path.isdir(path)
        
        # 列出目录内容
        if operation == "list":
            if not dir_exists:
                return {
                    "status": "error",
                    "message": f"目录 '{path}' 不存在"
                }
                
            import glob
            
            # 构建匹配模式
            pattern_path = os.path.join(path, pattern)
            
            # 执行查找
            if recursive:
                files = glob.glob(pattern_path, recursive=True)
            else:
                files = glob.glob(pattern_path)
                
            # 获取文件信息
            file_info = []
            for file_path in files:
                stats = os.stat(file_path)
                file_info.append({
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size": stats.st_size,
                    "is_dir": os.path.isdir(file_path),
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
                
            return {
                "status": "success",
                "path": path,
                "items": file_info,
                "count": len(file_info),
                "timestamp": time.time()
            }
            
        # 创建目录
        if operation == "create":
            if dir_exists:
                return {
                    "status": "success",
                    "message": f"目录 '{path}' 已经存在",
                    "path": path,
                    "timestamp": time.time()
                }
                
            os.makedirs(path)
            return {
                "status": "success",
                "operation": "create",
                "path": path,
                "timestamp": time.time()
            }
            
        # 删除目录
        if operation == "delete":
            if not dir_exists:
                return {
                    "status": "error",
                    "message": f"目录 '{path}' 不存在"
                }
                
            import shutil
            
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
                
            return {
                "status": "success",
                "operation": "delete",
                "path": path,
                "recursive": recursive,
                "timestamp": time.time()
            }
            
        # 查找文件
        if operation == "find":
            if not dir_exists:
                return {
                    "status": "error",
                    "message": f"目录 '{path}' 不存在"
                }
                
            import glob
            
            # 构建匹配模式
            pattern_path = os.path.join(path, pattern)
            
            # 执行查找
            if recursive:
                matching_files = glob.glob(pattern_path, recursive=True)
            else:
                matching_files = glob.glob(pattern_path)
                
            return {
                "status": "success",
                "path": path,
                "pattern": pattern,
                "matches": matching_files,
                "count": len(matching_files),
                "timestamp": time.time()
            }
            
        return {
            "status": "error",
            "message": f"不支持的操作 '{operation}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        } 