# -*- coding: utf-8 -*-
"""
工具包 (Tools Package)

提供各类工具和工具执行器
"""

from .tool_executor import ToolExecutor, tool
from .network_tools import web_request, api_call, fetch_weather, search_wikipedia
from .system_tools import system_info, file_operations, execute_command, directory_operations
from .test_tools import test_memory, test_reasoning, test_planning, test_event_system, test_performance, run_demo

# 导出可用工具列表
available_tools = [
    web_request,
    api_call,
    fetch_weather,
    search_wikipedia,
    system_info,
    file_operations,
    execute_command,
    directory_operations,
    test_memory,
    test_reasoning,
    test_planning,
    test_event_system,
    test_performance,
    run_demo
]

__all__ = [
    'ToolExecutor',
    'tool',
    'web_request',
    'api_call',
    'fetch_weather',
    'search_wikipedia',
    'system_info',
    'file_operations',
    'execute_command',
    'directory_operations',
    'test_memory',
    'test_reasoning',
    'test_planning',
    'test_event_system',
    'test_performance',
    'run_demo',
    'available_tools'
] 