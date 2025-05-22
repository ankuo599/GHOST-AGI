"""
学习系统初始化模块 (Learning System Initializer)

负责初始化和启动各个学习组件，协调学习系统的整体工作流程。
加载零知识基础、自监督学习和元学习等模块，并建立它们之间的连接。
"""

import time
import logging
import importlib
from typing import Dict, List, Any, Optional
import threading
import os
import json

class LearningInitializer:
    """学习系统初始化器，负责初始化和启动各学习组件"""
    
    def __init__(self, config_path: Optional[str] = None, central_coordinator=None):
        """
        初始化学习系统初始化器
        
        Args:
            config_path: 配置文件路径
            central_coordinator: 中央协调器
        """
        # 设置日志
        self.logger = self._setup_logger()
        
        # 中央协调器
        self.central_coordinator = central_coordinator
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 学习组件
        self.learning_modules = {}
        
        # 管理的知识系统
        self.knowledge_system = None
        self.memory_system = None
        
        # 状态
        self.status = {
            "initialized": False,
            "running": False,
            "start_time": None,
            "active_modules": set(),
            "initialization_errors": [],
            "knowledge_integration_status": "pending"
        }
        
        self.logger.info("学习系统初始化器创建完成")
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("LearningInitializer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("learning_initializer.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        # 默认配置
        default_config = {
            "modules": {
                "zero_knowledge": {
                    "enabled": True,
                    "class": "learning.zero_knowledge_core.ZeroKnowledgeCore",
                    "config": {}
                },
                "meta_learning": {
                    "enabled": True,
                    "class": "metacognition.meta_learning.MetaLearningModule",
                    "config": {}
                },
                "self_supervised": {
                    "enabled": True, 
                    "class": "learning.self_supervised_learning.SelfSupervisedLearning",
                    "config": {
                        "enable_contrastive_learning": True
                    }
                },
                "reinforcement": {
                    "enabled": False,
                    "class": "learning.reinforcement_learning.ReinforcementLearning",
                    "config": {}
                }
            },
            "learning_rate": 0.01,
            "auto_start": True,
            "enable_module_interaction": True
        }
        
        # 如果提供了配置路径，加载并合并配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    
                # 递归合并配置
                self._merge_configs(default_config, user_config)
                self.logger.info(f"已加载配置文件: {config_path}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {str(e)}")
        
        return default_config
    
    def _merge_configs(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        递归合并配置
        
        Args:
            target: 目标配置
            source: 源配置
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
    
    def initialize(self) -> Dict[str, Any]:
        """
        初始化学习系统
        
        Returns:
            Dict: 初始化结果
        """
        if self.status["initialized"]:
            return {
                "status": "warning",
                "message": "学习系统已经初始化"
            }
            
        self.logger.info("开始初始化学习系统...")
        initialization_start = time.time()
        
        # 首先初始化关键依赖模块（知识系统和记忆系统）
        self._initialize_core_dependencies()
        
        # 初始化所有启用的模块
        module_results = {}
        
        for module_name, module_config in self.config["modules"].items():
            if module_config.get("enabled", True):
                result = self._initialize_module(module_name, module_config)
                module_results[module_name] = result
                
                if result["status"] == "success":
                    self.status["active_modules"].add(module_name)
                else:
                    self.status["initialization_errors"].append(result)
        
        # 建立模块间的连接
        if self.config.get("enable_module_interaction", True):
            self._connect_modules()
            
        # 集成零知识核心与其他系统
        if "zero_knowledge" in self.learning_modules:
            self._integrate_zero_knowledge()
            
        # 更新状态
        self.status["initialized"] = True
        self.status["initialization_time"] = time.time() - initialization_start
        
        # 如果配置了自动启动，则启动系统
        if self.config.get("auto_start", True):
            self.start()
            
        self.logger.info(f"学习系统初始化完成，耗时 {self.status['initialization_time']:.2f} 秒")
        
        return {
            "status": "success",
            "message": "学习系统初始化完成",
            "modules": module_results,
            "active_modules": list(self.status["active_modules"]),
            "errors": len(self.status["initialization_errors"])
        }
    
    def _initialize_core_dependencies(self):
        """初始化核心依赖模块（知识系统和记忆系统）"""
        # 检查配置中是否指定了知识系统和记忆系统
        knowledge_system_config = self.config.get("core_dependencies", {}).get("knowledge_system")
        memory_system_config = self.config.get("core_dependencies", {}).get("memory_system")
        
        # 初始化知识系统
        if knowledge_system_config and knowledge_system_config.get("enabled", True):
            try:
                class_path = knowledge_system_config.get("class", "knowledge.self_organizing_knowledge.SelfOrganizingKnowledge")
                
                # 分解类路径
                module_parts = class_path.split('.')
                class_name = module_parts[-1]
                package_path = '.'.join(module_parts[:-1])
                
                # 加载模块
                module = importlib.import_module(package_path)
                
                # 获取类
                module_class = getattr(module, class_name)
                
                # 实例化模块
                self.knowledge_system = module_class()
                
                self.logger.info(f"核心依赖 - 知识系统初始化成功: {class_name}")
            except Exception as e:
                self.logger.error(f"初始化知识系统失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 初始化记忆系统
        if memory_system_config and memory_system_config.get("enabled", True):
            try:
                class_path = memory_system_config.get("class", "memory.episodic_memory.EpisodicMemory")
                
                # 分解类路径
                module_parts = class_path.split('.')
                class_name = module_parts[-1]
                package_path = '.'.join(module_parts[:-1])
                
                # 加载模块
                module = importlib.import_module(package_path)
                
                # 获取类
                module_class = getattr(module, class_name)
                
                # 实例化模块
                self.memory_system = module_class()
                
                self.logger.info(f"核心依赖 - 记忆系统初始化成功: {class_name}")
            except Exception as e:
                self.logger.error(f"初始化记忆系统失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def _initialize_module(self, module_name: str, module_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        初始化单个学习模块
        
        Args:
            module_name: 模块名称
            module_config: 模块配置
            
        Returns:
            Dict: 初始化结果
        """
        self.logger.info(f"初始化学习模块: {module_name}")
        
        try:
            # 获取模块类路径
            class_path = module_config.get("class", "")
            
            if not class_path:
                return {
                    "status": "error",
                    "message": f"模块 {module_name} 缺少类路径"
                }
            
            # 分解类路径
            module_parts = class_path.split('.')
            class_name = module_parts[-1]
            package_path = '.'.join(module_parts[:-1])
            
            # 加载模块
            module = importlib.import_module(package_path)
            
            # 获取类
            module_class = getattr(module, class_name)
            
            # 准备初始化参数
            init_args = {}
            
            # 如果是零知识核心，注入知识系统和记忆系统
            if module_name == "zero_knowledge" and class_name == "ZeroKnowledgeCore":
                if self.knowledge_system:
                    init_args["knowledge_system"] = self.knowledge_system
                if self.memory_system:
                    init_args["memory_system"] = self.memory_system
                    
            # 添加配置的参数
            init_args.update(module_config.get("args", {}))
            
            # 如果中央协调器可用，添加到参数
            if self.central_coordinator and "central_coordinator" not in init_args:
                init_args["central_coordinator"] = self.central_coordinator
                
            # 实例化模块
            module_instance = module_class(**init_args)
            
            # 应用配置
            if hasattr(module_instance, "configure") and callable(getattr(module_instance, "configure")):
                module_instance.configure(module_config.get("config", {}))
                
            # 存储模块实例
            self.learning_modules[module_name] = {
                "instance": module_instance,
                "config": module_config,
                "status": "initialized",
                "initialized_at": time.time()
            }
            
            self.logger.info(f"学习模块 {module_name} 初始化成功")
            
            return {
                "status": "success",
                "message": f"模块 {module_name} 初始化成功"
            }
            
        except Exception as e:
            self.logger.error(f"初始化学习模块 {module_name} 失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "message": f"模块 {module_name} 初始化失败: {str(e)}"
            }
    
    def _connect_modules(self):
        """建立模块间的连接"""
        self.logger.info("建立学习模块间的连接...")
        
        # 默认的模块间连接配置
        default_connections = {
            "zero_knowledge": ["meta_learning"],
            "meta_learning": ["zero_knowledge", "self_supervised"]
        }
        
        # 获取自定义的连接配置
        connections = self.config.get("module_connections", default_connections)
        
        # 建立连接
        for source, targets in connections.items():
            if source in self.learning_modules:
                source_module = self.learning_modules[source]["instance"]
                
                for target in targets:
                    if target in self.learning_modules:
                        target_module = self.learning_modules[target]["instance"]
                        
                        # 连接模块
                        self._connect_pair(source, source_module, target, target_module)
                    else:
                        self.logger.warning(f"连接失败: 目标模块 {target} 不存在")
            else:
                self.logger.warning(f"连接失败: 源模块 {source} 不存在")
    
    def _connect_pair(self, source_name: str, source_module: Any, 
                    target_name: str, target_module: Any):
        """
        连接一对模块
        
        Args:
            source_name: 源模块名称
            source_module: 源模块实例
            target_name: 目标模块名称
            target_module: 目标模块实例
        """
        # 检查常见的连接方法
        connection_methods = [
            "connect_to",
            "register_module",
            "set_module",
            f"set_{target_name}",
            "add_listener"
        ]
        
        for method_name in connection_methods:
            if hasattr(source_module, method_name) and callable(getattr(source_module, method_name)):
                try:
                    method = getattr(source_module, method_name)
                    method(target_module)
                    self.logger.info(f"已连接模块: {source_name} -> {target_name} (使用 {method_name})")
                    return
                except Exception as e:
                    self.logger.warning(f"使用 {method_name} 连接模块失败: {str(e)}")
        
        # 如果没有通用连接方法，尝试特定组合
        specific_connections = {
            ("zero_knowledge", "meta_learning"): 
                lambda s, t: setattr(s, "meta_learning_module", t),
            ("meta_learning", "zero_knowledge"): 
                lambda s, t: setattr(s, "knowledge_system", t)
        }
        
        connection_key = (source_name, target_name)
        if connection_key in specific_connections:
            try:
                connection_func = specific_connections[connection_key]
                connection_func(source_module, target_module)
                self.logger.info(f"已连接模块: {source_name} -> {target_name} (使用特定连接)")
                return
            except Exception as e:
                self.logger.warning(f"使用特定连接方法连接模块失败: {str(e)}")
                
        self.logger.warning(f"未能找到连接模块的方法: {source_name} -> {target_name}")
    
    def _integrate_zero_knowledge(self):
        """将零知识核心与其他系统深度集成"""
        self.logger.info("集成零知识核心与其他系统...")
        
        try:
            # 获取零知识核心实例
            zero_knowledge = self.learning_modules["zero_knowledge"]["instance"]
            
            # 1. 连接知识系统（如果未在构造函数中连接）
            if self.knowledge_system and not zero_knowledge.knowledge_system:
                zero_knowledge.knowledge_system = self.knowledge_system
                self.logger.info("零知识核心已连接到知识系统")
                
            # 2. 连接记忆系统（如果未在构造函数中连接）
            if self.memory_system and not zero_knowledge.memory_system:
                zero_knowledge.memory_system = self.memory_system
                self.logger.info("零知识核心已连接到记忆系统")
                
            # 3. 连接元认知监督模块（如果存在）
            if "meta_learning" in self.learning_modules:
                meta_learning = self.learning_modules["meta_learning"]["instance"]
                
                # 双向连接
                if hasattr(zero_knowledge, "meta_learning_module"):
                    zero_knowledge.meta_learning_module = meta_learning
                    self.logger.info("零知识核心已连接到元学习模块")
                    
                if hasattr(meta_learning, "register_monitored_module"):
                    meta_learning.register_monitored_module("zero_knowledge", zero_knowledge)
                    self.logger.info("元学习模块已监控零知识核心")
                    
            # 4. 连接中央协调器（如果存在且未连接）
            if self.central_coordinator and hasattr(zero_knowledge, "set_coordinator"):
                zero_knowledge.set_coordinator(self.central_coordinator)
                self.logger.info("零知识核心已连接到中央协调器")
                
            # 5. 设置初始观察以引导学习
            if self.config.get("bootstrap_zero_knowledge", True):
                self._bootstrap_zero_knowledge(zero_knowledge)
                
            self.status["knowledge_integration_status"] = "completed"
            
        except Exception as e:
            self.logger.error(f"集成零知识核心失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.status["knowledge_integration_status"] = "failed"
            
    def _bootstrap_zero_knowledge(self, zero_knowledge):
        """
        引导零知识学习的初始阶段
        
        Args:
            zero_knowledge: 零知识核心实例
        """
        self.logger.info("引导零知识核心初始学习...")
        
        # 初始观察数据
        bootstrap_observations = self.config.get("bootstrap_observations", [])
        
        if not bootstrap_observations:
            # 如果配置中没有预设观察，使用基本观察
            bootstrap_observations = [
                {
                    "content": "物体具有颜色、形状和大小等基本属性。",
                    "source": "bootstrap"
                },
                {
                    "content": "数字可以被加减乘除，形成基本的数学运算。",
                    "source": "bootstrap"
                },
                {
                    "content": "概念之间可以有多种关系，如分类(is-a)、组成(part-of)和相似性。",
                    "source": "bootstrap"
                },
                {
                    "content": {
                        "基本范畴": ["物理对象", "抽象概念", "事件", "行为", "属性"],
                        "基本关系": ["包含", "组成", "前后", "因果", "影响"]
                    },
                    "source": "bootstrap_structured"
                }
            ]
            
        # 处理每个观察
        for observation in bootstrap_observations:
            try:
                observation["timestamp"] = observation.get("timestamp", time.time())
                result = zero_knowledge.process_observation(observation)
                self.logger.info(f"处理引导观察: {result['status']}")
            except Exception as e:
                self.logger.error(f"处理引导观察失败: {str(e)}")
                
        # 记录已完成引导
        self.logger.info(f"零知识核心引导学习完成，处理了 {len(bootstrap_observations)} 个观察")
        
    def register_external_module(self, module_name: str, module_instance: Any) -> Dict[str, Any]:
        """
        注册外部创建的模块
        
        Args:
            module_name: 模块名称
            module_instance: 模块实例
            
        Returns:
            Dict: 注册结果
        """
        if module_name in self.learning_modules:
            return {
                "status": "error",
                "message": f"模块 {module_name} 已存在"
            }
            
        self.learning_modules[module_name] = {
            "instance": module_instance,
            "config": {},
            "status": "initialized",
            "initialized_at": time.time(),
            "external": True
        }
        
        self.status["active_modules"].add(module_name)
        
        self.logger.info(f"已注册外部模块: {module_name}")
        
        return {
            "status": "success",
            "message": f"外部模块 {module_name} 注册成功"
        }
        
    def process_initial_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用零知识核心处理一组初始观察
        
        Args:
            observations: 观察列表
            
        Returns:
            Dict: 处理结果
        """
        if "zero_knowledge" not in self.learning_modules:
            return {
                "status": "error",
                "message": "零知识核心模块未初始化"
            }
            
        zero_knowledge = self.learning_modules["zero_knowledge"]["instance"]
        
        results = []
        for observation in observations:
            try:
                result = zero_knowledge.process_observation(observation)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理观察失败: {str(e)}")
                results.append({
                    "status": "error",
                    "message": str(e)
                })
                
        return {
            "status": "success",
            "message": f"处理了 {len(observations)} 个观察",
            "results": results
        }
        
    def start(self) -> Dict[str, Any]:
        """
        启动学习系统
        
        Returns:
            Dict: 启动结果
        """
        if not self.status["initialized"]:
            return {
                "status": "error",
                "message": "学习系统未初始化"
            }
            
        if self.status["running"]:
            return {
                "status": "warning",
                "message": "学习系统已在运行"
            }
            
        self.logger.info("启动学习系统...")
        
        # 启动所有模块
        start_results = {}
        
        for module_name in self.status["active_modules"]:
            module_info = self.learning_modules[module_name]
            module_instance = module_info["instance"]
            
            # 启动模块
            try:
                if hasattr(module_instance, "start") and callable(getattr(module_instance, "start")):
                    start_method = getattr(module_instance, "start")
                    result = start_method()
                    
                    # 更新模块状态
                    module_info["status"] = "running"
                    module_info["started_at"] = time.time()
                    
                    start_results[module_name] = {
                        "status": "success",
                        "result": result
                    }
                else:
                    self.logger.warning(f"模块 {module_name} 没有start方法")
                    start_results[module_name] = {
                        "status": "warning",
                        "message": "模块没有start方法"
                    }
            except Exception as e:
                self.logger.error(f"启动模块 {module_name} 失败: {str(e)}")
                start_results[module_name] = {
                    "status": "error",
                    "message": str(e)
                }
                
        # 更新状态
        self.status["running"] = True
        self.status["start_time"] = time.time()
        
        self.logger.info("学习系统启动完成")
        
        return {
            "status": "success",
            "message": "学习系统启动完成",
            "modules": start_results
        }
        
    def stop(self) -> Dict[str, Any]:
        """
        停止学习系统
        
        Returns:
            Dict: 停止结果
        """
        if not self.status["running"]:
            return {
                "status": "warning",
                "message": "学习系统未在运行"
            }
            
        self.logger.info("停止学习系统...")
        
        # 停止所有模块
        stop_results = {}
        
        for module_name in self.status["active_modules"]:
            module_info = self.learning_modules[module_name]
            module_instance = module_info["instance"]
            
            # 停止模块
            try:
                if hasattr(module_instance, "stop") and callable(getattr(module_instance, "stop")):
                    stop_method = getattr(module_instance, "stop")
                    result = stop_method()
                    
                    # 更新模块状态
                    module_info["status"] = "stopped"
                    module_info["stopped_at"] = time.time()
                    
                    stop_results[module_name] = {
                        "status": "success",
                        "result": result
                    }
                else:
                    self.logger.warning(f"模块 {module_name} 没有stop方法")
                    stop_results[module_name] = {
                        "status": "warning",
                        "message": "模块没有stop方法"
                    }
            except Exception as e:
                self.logger.error(f"停止模块 {module_name} 失败: {str(e)}")
                stop_results[module_name] = {
                    "status": "error",
                    "message": str(e)
                }
                
        # 更新状态
        self.status["running"] = False
        self.status["stop_time"] = time.time()
        
        self.logger.info("学习系统停止完成")
        
        return {
            "status": "success",
            "message": "学习系统停止完成",
            "modules": stop_results
        }
        
    def get_status(self) -> Dict[str, Any]:
        """
        获取学习系统状态
        
        Returns:
            Dict: 状态信息
        """
        # 获取各模块的状态
        module_status = {}
        
        for module_name, module_info in self.learning_modules.items():
            instance = module_info["instance"]
            
            module_status[module_name] = {
                "status": module_info["status"],
                "initialized_at": module_info.get("initialized_at"),
                "started_at": module_info.get("started_at"),
                "stopped_at": module_info.get("stopped_at")
            }
            
            # 如果模块有get_status方法，获取详细状态
            if hasattr(instance, "get_status") and callable(getattr(instance, "get_status")):
                try:
                    module_details = instance.get_status()
                    module_status[module_name]["details"] = module_details
                except Exception as e:
                    self.logger.warning(f"获取模块 {module_name} 状态失败: {str(e)}")
                    
        # 构建状态报告
        status_report = {
            "initialized": self.status["initialized"],
            "running": self.status["running"],
            "start_time": self.status["start_time"],
            "active_modules": list(self.status["active_modules"]),
            "module_status": module_status,
            "initialization_errors": len(self.status["initialization_errors"]),
            "knowledge_integration_status": self.status["knowledge_integration_status"]
        }
        
        if self.status["running"] and self.status["start_time"]:
            status_report["uptime"] = time.time() - self.status["start_time"]
            
        return status_report 