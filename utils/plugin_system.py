# -*- coding: utf-8 -*-
"""
插件系统 (Plugin System)

负责模块解耦和动态加载，使各个智能体可独立扩展
支持插件注册、加载、卸载和管理
"""

import os
import sys
import time
import json
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Type

class PluginSystem:
    def __init__(self, plugin_dir="plugins"):
        """
        初始化插件系统
        
        Args:
            plugin_dir: 插件目录
        """
        self.plugin_dir = plugin_dir
        self.plugins = {}  # 已加载的插件 {plugin_id: plugin_instance}
        self.plugin_info = {}  # 插件信息 {plugin_id: plugin_metadata}
        self.hooks = {}  # 钩子点 {hook_name: [callbacks]}
        self.dependencies = {}  # 插件依赖关系 {plugin_id: [dependency_ids]}
        self.load_order = []  # 插件加载顺序
        
        # 确保插件目录存在
        os.makedirs(plugin_dir, exist_ok=True)
        
        # 添加插件目录到Python路径
        if plugin_dir not in sys.path:
            sys.path.append(os.path.abspath(plugin_dir))
            
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """
        发现可用插件
        
        Returns:
            List[Dict[str, Any]]: 可用插件列表
        """
        discovered_plugins = []
        
        # 遍历插件目录
        for item in os.listdir(self.plugin_dir):
            if os.path.isdir(os.path.join(self.plugin_dir, item)) and not item.startswith('_'):
                # 检查是否有manifest.json文件
                manifest_path = os.path.join(self.plugin_dir, item, 'manifest.json')
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                            
                        # 添加插件路径信息
                        manifest['path'] = os.path.join(self.plugin_dir, item)
                        manifest['id'] = item
                        
                        discovered_plugins.append(manifest)
                    except Exception as e:
                        print(f"读取插件清单文件失败: {item}, 错误: {e}")
                        
        return discovered_plugins
        
    def load_plugin(self, plugin_id: str) -> bool:
        """
        加载插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 是否成功加载
        """
        # 如果插件已加载，直接返回成功
        if plugin_id in self.plugins:
            return True
            
        # 获取插件信息
        plugin_path = os.path.join(self.plugin_dir, plugin_id)
        manifest_path = os.path.join(plugin_path, 'manifest.json')
        
        if not os.path.exists(manifest_path):
            print(f"插件清单文件不存在: {plugin_id}")
            return False
            
        try:
            # 读取插件清单
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            # 检查依赖
            if 'dependencies' in manifest:
                for dep_id in manifest['dependencies']:
                    if dep_id not in self.plugins:
                        # 尝试加载依赖
                        if not self.load_plugin(dep_id):
                            print(f"无法加载插件 {plugin_id} 的依赖: {dep_id}")
                            return False
                            
            # 导入主模块
            main_module = manifest.get('main_module', 'main')
            module_path = f"{plugin_id}.{main_module}"
            
            try:
                module = importlib.import_module(module_path)
                
                # 查找插件类
                plugin_class = None
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, 'initialize') and hasattr(obj, 'shutdown'):
                        plugin_class = obj
                        break
                        
                if not plugin_class:
                    print(f"插件 {plugin_id} 中未找到有效的插件类")
                    return False
                    
                # 实例化插件
                plugin_instance = plugin_class()
                
                # 初始化插件
                plugin_instance.initialize(self)
                
                # 注册插件
                self.plugins[plugin_id] = plugin_instance
                self.plugin_info[plugin_id] = manifest
                self.load_order.append(plugin_id)
                
                # 记录依赖关系
                if 'dependencies' in manifest:
                    self.dependencies[plugin_id] = manifest['dependencies']
                    
                print(f"成功加载插件: {plugin_id} ({manifest.get('name', 'Unknown')})")
                return True
                
            except Exception as e:
                print(f"加载插件模块失败: {plugin_id}, 错误: {e}")
                return False
                
        except Exception as e:
            print(f"加载插件失败: {plugin_id}, 错误: {e}")
            return False
            
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 是否成功卸载
        """
        if plugin_id not in self.plugins:
            return False
            
        # 检查是否有其他插件依赖于此插件
        for dep_id, deps in self.dependencies.items():
            if plugin_id in deps and dep_id in self.plugins:
                print(f"无法卸载插件 {plugin_id}，因为插件 {dep_id} 依赖于它")
                return False
                
        try:
            # 调用插件的关闭方法
            self.plugins[plugin_id].shutdown()
            
            # 移除插件
            del self.plugins[plugin_id]
            del self.plugin_info[plugin_id]
            if plugin_id in self.dependencies:
                del self.dependencies[plugin_id]
                
            # 从加载顺序中移除
            if plugin_id in self.load_order:
                self.load_order.remove(plugin_id)
                
            # 移除相关钩子
            for hook_name in list(self.hooks.keys()):
                self.hooks[hook_name] = [cb for cb in self.hooks[hook_name] 
                                        if not (hasattr(cb, '__self__') and 
                                               cb.__self__ == self.plugins[plugin_id])]
                                               
            print(f"成功卸载插件: {plugin_id}")
            return True
            
        except Exception as e:
            print(f"卸载插件失败: {plugin_id}, 错误: {e}")
            return False
            
    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """
        注册钩子
        
        Args:
            hook_name: 钩子名称
            callback: 回调函数
            
        Returns:
            bool: 是否成功注册
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
            
        self.hooks[hook_name].append(callback)
        return True
        
    def unregister_hook(self, hook_name: str, callback: Callable) -> bool:
        """
        取消注册钩子
        
        Args:
            hook_name: 钩子名称
            callback: 回调函数
            
        Returns:
            bool: 是否成功取消注册
        """
        if hook_name not in self.hooks:
            return False
            
        try:
            self.hooks[hook_name].remove(callback)
            return True
        except ValueError:
            return False
            
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        触发钩子
        
        Args:
            hook_name: 钩子名称
            *args, **kwargs: 传递给钩子函数的参数
            
        Returns:
            List[Any]: 钩子函数返回值列表
        """
        if hook_name not in self.hooks:
            return []
            
        results = []
        for callback in self.hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"执行钩子 {hook_name} 时出错: {e}")
                results.append(None)
                
        return results
        
    def get_plugin_info(self, plugin_id: str = None) -> Dict[str, Any]:
        """
        获取插件信息
        
        Args:
            plugin_id: 插件ID，如果不提供则返回所有插件信息
            
        Returns:
            Dict[str, Any]: 插件信息
        """
        if plugin_id:
            return self.plugin_info.get(plugin_id, {})
        else:
            return self.plugin_info
            
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        加载所有可用插件
        
        Returns:
            Dict[str, bool]: 插件加载结果 {plugin_id: success}
        """
        discovered = self.discover_plugins()
        results = {}
        
        # 首先构建依赖图
        dependency_graph = {}
        for plugin in discovered:
            plugin_id = plugin['id']
            dependency_graph[plugin_id] = plugin.get('dependencies', [])
            
        # 拓扑排序，确保依赖先加载
        load_order = self._topological_sort(dependency_graph)
        
        # 按顺序加载插件
        for plugin_id in load_order:
            results[plugin_id] = self.load_plugin(plugin_id)
            
        return results
        
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        对依赖图进行拓扑排序
        
        Args:
            graph: 依赖图 {node: [dependencies]}
            
        Returns:
            List[str]: 排序后的节点列表
        """
        # 计算入度
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[dep] += 1
                    
        # 找出入度为0的节点
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        # 拓扑排序
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # 减少相邻节点的入度
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
                        
        # 检查是否有环
        if len(result) != len(graph):
            print("警告: 插件依赖中存在循环依赖")
            
        return result
        
    def create_plugin_template(self, plugin_id: str, name: str, description: str, version: str = "0.1.0") -> bool:
        """
        创建插件模板
        
        Args:
            plugin_id: 插件ID
            name: 插件名称
            description: 插件描述
            version: 插件版本
            
        Returns:
            bool: 是否成功创建
        """
        plugin_path = os.path.join(self.plugin_dir, plugin_id)
        
        # 检查插件是否已存在
        if os.path.exists(plugin_path):
            print(f"插件已存在: {plugin_id}")
            return False
            
        try:
            # 创建插件目录
            os.makedirs(plugin_path)
            
            # 创建manifest.json
            manifest = {
                "name": name,
                "description": description,
                "version": version,
                "author": "GHOST AGI",
                "main_module": "main",
                "dependencies": [],
                "hooks": [],
                "created_at": time.time()
            }
            
            with open(os.path.join(plugin_path, 'manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
                
            # 创建main.py
            main_py = f'''\
# -*- coding: utf-8 -*-
"""
{name} 插件

{description}
"""

class {plugin_id.capitalize()}Plugin:
    def __init__(self):
        self.name = "{name}"
        self.version = "{version}"
        self.plugin_system = None
        
    def initialize(self, plugin_system):
        """
        初始化插件
        
        Args:
            plugin_system: 插件系统实例
        """
        self.plugin_system = plugin_system
        print(f"{self.name} 插件已初始化")
        
        # 注册钩子
        # plugin_system.register_hook("hook_name", self.hook_handler)
        
    def shutdown(self):
        """
        关闭插件
        """
        print(f"{self.name} 插件已关闭")
        
    # 示例钩子处理函数
    # def hook_handler(self, *args, **kwargs):
    #     print(f"{self.name} 处理钩子")
    #     return True
'''
            
            with open(os.path.join(plugin_path, 'main.py'), 'w', encoding='utf-8') as f:
                f.write(main_py)
                
            # 创建__init__.py
            with open(os.path.join(plugin_path, '__init__.py'), 'w', encoding='utf-8') as f:
                f.write(f'# {name} 插件\n')
                
            print(f"成功创建插件模板: {plugin_id}")
            return True
            
        except Exception as e:
            print(f"创建插件模板失败: {plugin_id}, 错误: {e}")
            return False
            
    def get_plugin_stats(self) -> Dict[str, Any]:
        """
        获取插件统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_plugins": len(self.plugins),
            "active_hooks": len(self.hooks),
            "hook_callbacks": sum(len(callbacks) for callbacks in self.hooks.values()),
            "load_order": self.load_order
        }