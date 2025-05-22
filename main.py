# -*- coding: utf-8 -*-
"""
GHOST AGI - 主程序
集成各个系统组件，协调系统运行
"""

import time
import sys
import os
import logging
import argparse
from typing import Dict, Any, List, Optional

# 导入系统组件
from utils.event_system import EventSystem
from utils.agent_scheduler import AgentScheduler
from agents.core_agent import CoreAgent
from agents.meta_cognition import MetaCognitionAgent
from memory.memory_system import MemorySystem
from memory.vector_store import VectorStore
from reasoning.symbolic import SymbolicReasoner
from reasoning.planning import PlanningEngine
from tools.tool_executor import ToolExecutor
from tools import available_tools
from interface.web_interface import WebInterface
# 导入感知模块
from perception.text_perception import TextPerception
from perception.multimodal import MultiModalPerception
# 导入新增的进化和学习模块
from evolution.evolution_engine import EvolutionEngine
from evolution.code_generator import CodeGenerator
from knowledge.knowledge_transfer import KnowledgeTransfer
from learning.autonomous_learning import AutonomousLearning

# 导入新的学习系统集成模块
try:
    from learning.integrator import LearningIntegrator
    LEARNING_INTEGRATOR_AVAILABLE = True
except ImportError:
    LEARNING_INTEGRATOR_AVAILABLE = False
    logging.warning("学习系统集成模块不可用，将使用基础学习引擎")

# 导入零样本学习模块
try:
    from learning.zero_shot_learning import ZeroShotLearningModule
    ZERO_SHOT_LEARNING_AVAILABLE = True
except ImportError:
    ZERO_SHOT_LEARNING_AVAILABLE = False
    logging.warning("零样本学习模块不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ghost_agi.log")
    ]
)

logger = logging.getLogger("GHOST_AGI")

class GhostAGI:
    """GHOST AGI 系统主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 GHOST AGI 系统
        
        Args:
            config: 系统配置
        """
        self.config = config or {}
        
        # 安全配置（默认启用安全限制，但可通过配置禁用）
        self.sandbox_enabled = self.config.get("sandbox_enabled", True)
        self.safety_checks = self.config.get("safety_checks", True)
        
        if not self.sandbox_enabled or not self.safety_checks:
            logger.warning("警告：安全限制已禁用，系统将以无限制模式运行")
            
        logger.info("正在初始化 GHOST AGI 系统...")
        
        # 初始化事件系统 (核心通信组件)
        self.event_system = EventSystem()
        logger.info("事件系统初始化完成")
        
        # 初始化向量存储
        self.vector_store = VectorStore()
        logger.info("向量存储初始化完成")
        
        # 初始化记忆系统
        self.memory_system = MemorySystem(
            vector_store=self.vector_store,
            event_system=self.event_system
        )
        logger.info("记忆系统初始化完成")
        
        # 初始化推理引擎
        self.symbolic_reasoner = SymbolicReasoner()
        logger.info("推理引擎初始化完成")
        
        # 初始化规划引擎
        self.planning_engine = PlanningEngine(
            reasoner=self.symbolic_reasoner,
            event_system=self.event_system
        )
        logger.info("规划引擎初始化完成")
        
        # 初始化工具执行器
        self.tool_executor = ToolExecutor(
            event_system=self.event_system,
            sandbox_enabled=self.sandbox_enabled
        )
        
        # 注册可用工具
        for tool_func in available_tools:
            self.tool_executor.register_tool(
                tool_name=getattr(tool_func, "_tool_name", tool_func.__name__),
                tool_function=tool_func,
                description=getattr(tool_func, "_tool_description", ""),
                required_params=getattr(tool_func, "_required_params", []),
                optional_params=getattr(tool_func, "_optional_params", {})
            )
        logger.info(f"工具执行器初始化完成，已注册 {len(available_tools)} 个工具")
        
        # 初始化感知模块
        self.perception = {
            "text": TextPerception(),
            "multimodal": MultiModalPerception()
        }
        logger.info("感知模块初始化完成")
        
        # 初始化进化引擎
        self.evolution_engine = EvolutionEngine(
            event_system=self.event_system,
            memory_system=self.memory_system
        )
        logger.info("进化引擎初始化完成")
        
        # 初始化代码生成器
        self.code_generator = CodeGenerator(
            event_system=self.event_system
        )
        logger.info("代码生成器初始化完成")
        
        # 初始化知识迁移系统
        self.knowledge_transfer = KnowledgeTransfer(
            event_system=self.event_system,
            memory_system=self.memory_system
        )
        logger.info("知识迁移系统初始化完成")
        
        # 初始化学习模块
        # 优先使用学习集成器，如果不可用则回退到基础学习引擎
        if LEARNING_INTEGRATOR_AVAILABLE:
            logger.info("使用增强的学习系统集成器")
            self.learning_integrator = LearningIntegrator(
                memory_system=self.memory_system,
                vector_store=self.vector_store,
                event_system=self.event_system
            )
            
            # 检查是否成功加载了零样本学习模块
            has_zero_shot = "zero_shot" in self.learning_integrator.learning_modules
            if has_zero_shot:
                logger.info("零样本学习模块已加载")
            else:
                logger.warning("零样本学习模块未加载")
                
            # 使用集成器管理学习引擎
            self.autonomous_learning = self.learning_integrator
        else:
            # 回退到原来的自主学习系统
            logger.info("使用标准自主学习系统")
            self.autonomous_learning = AutonomousLearning(
                event_system=self.event_system,
                memory_system=self.memory_system,
                tool_executor=self.tool_executor
            )
            # 如果有零样本学习模块，单独初始化
            if ZERO_SHOT_LEARNING_AVAILABLE:
                logger.info("初始化零样本学习模块")
                self.zero_shot_learning = ZeroShotLearningModule(
                    memory_system=self.memory_system,
                    vector_store=self.vector_store,
                    event_system=self.event_system
                )
            else:
                self.zero_shot_learning = None
                logger.warning("零样本学习模块不可用")
                
        logger.info("学习系统初始化完成")
        
        # 初始化智能体调度器
        self.agent_scheduler = AgentScheduler(event_system=self.event_system)
        logger.info("智能体调度器初始化完成")
        
        # 初始化核心智能体
        self.core_agent = CoreAgent(
            name="Core",
            goals=["自我完善", "学习新知识", "协助用户"],
            event_system=self.event_system,
            memory_system=self.memory_system,
            reasoning_engine=self.symbolic_reasoner,
            planning_engine=self.planning_engine,
            agent_scheduler=self.agent_scheduler,
            # 新增组件
            evolution_engine=self.evolution_engine,
            code_generator=self.code_generator,
            knowledge_transfer=self.knowledge_transfer,
            autonomous_learning=self.autonomous_learning
        )
        
        # 为核心智能体添加感知模块
        self.core_agent.perception = self.perception
        
        logger.info("核心智能体初始化完成")
        
        # 初始化元认知智能体
        self.meta_cognition = MetaCognitionAgent(
            name="Meta",
            event_system=self.event_system,
            memory_system=self.memory_system,
            core_agent=self.core_agent
        )
        logger.info("元认知智能体初始化完成")
        
        # 注册核心智能体和元认知智能体到调度器
        self.agent_scheduler.register_agent(
            agent_id="core",
            capabilities=["decision_making", "planning", "coordination"],
            agent_instance=self.core_agent
        )
        
        self.agent_scheduler.register_agent(
            agent_id="meta",
            capabilities=["self_monitoring", "goal_management", "performance_evaluation"],
            agent_instance=self.meta_cognition
        )
        
        # 初始化Web界面
        web_config = self.config.get("web_interface", {})
        self.web_interface = WebInterface(
            agi_system=self,
            host=web_config.get("host", "0.0.0.0"),
            port=web_config.get("port", 5000),
            debug=web_config.get("debug", False)
        )
        logger.info("Web界面初始化完成")
        
        # 添加知识图谱支持
        self._initialize_knowledge_graph()
        
        # 订阅事件
        self._subscribe_to_events()
        
        # 系统状态
        self.running = False
        self.start_time = None
        logger.info("GHOST AGI 系统初始化完成")
        
    def _initialize_knowledge_graph(self):
        """初始化知识图谱，添加基础概念"""
        try:
            if not hasattr(self.vector_store, "add_concept"):
                logger.warning("向量存储不支持知识图谱功能，跳过知识图谱初始化")
                return
                
            logger.info("初始化知识图谱...")
            
            # 定义基础概念
            base_concepts = [
                {"name": "实体", "properties": {"abstract": True, "description": "所有事物的基类"}},
                {"name": "物理对象", "properties": {"abstract": True, "description": "具有物理属性的实体"}},
                {"name": "抽象概念", "properties": {"abstract": True, "description": "不具有物理形态的思想或观念"}},
                {"name": "事件", "properties": {"abstract": True, "description": "发生在特定时间和地点的事情"}},
                {"name": "属性", "properties": {"abstract": True, "description": "实体的特征或特性"}},
                {"name": "关系", "properties": {"abstract": True, "description": "实体之间的联系"}},
                {"name": "生物", "properties": {"abstract": True, "description": "具有生命的实体"}},
                {"name": "信息", "properties": {"abstract": True, "description": "数据和知识的表示"}},
                # 添加更多基础概念
                {"name": "人工智能", "properties": {"abstract": True, "description": "模拟人类智能的系统"}},
                {"name": "工具", "properties": {"abstract": True, "description": "用于完成特定任务的设备或方法"}},
                {"name": "过程", "properties": {"abstract": True, "description": "随时间发展的一系列动作或事件"}},
                {"name": "方法", "properties": {"abstract": True, "description": "执行任务的特定方式或程序"}},
                {"name": "空间", "properties": {"abstract": True, "description": "物体存在或事件发生的区域"}},
                {"name": "时间", "properties": {"abstract": True, "description": "事件发生的时刻或持续期间"}},
                {"name": "因果关系", "properties": {"abstract": True, "description": "表示原因和结果之间的关系"}}
            ]
            
            # 添加基础概念
            concept_ids = {}
            for concept in base_concepts:
                concept_id = self.vector_store.add_concept(
                    concept_name=concept["name"],
                    properties=concept["properties"]
                )
                concept_ids[concept["name"]] = concept_id
                logger.debug(f"添加基础概念: {concept['name']}")
                
            # 添加概念间的关系
            relations = [
                {"source": "物理对象", "target": "实体", "type": "is_a"},
                {"source": "抽象概念", "target": "实体", "type": "is_a"},
                {"source": "事件", "target": "实体", "type": "is_a"},
                {"source": "生物", "target": "物理对象", "type": "is_a"},
                {"source": "信息", "target": "抽象概念", "type": "is_a"},
                {"source": "物理对象", "target": "属性", "type": "has_property"},
                {"source": "事件", "target": "实体", "type": "related_to"},
                # 添加新的基础关系
                {"source": "人工智能", "target": "抽象概念", "type": "is_a"},
                {"source": "工具", "target": "物理对象", "type": "is_a"},
                {"source": "过程", "target": "抽象概念", "type": "is_a"},
                {"source": "方法", "target": "抽象概念", "type": "is_a"},
                {"source": "空间", "target": "抽象概念", "type": "is_a"},
                {"source": "时间", "target": "抽象概念", "type": "is_a"},
                {"source": "过程", "target": "时间", "type": "related_to"},
                {"source": "工具", "target": "方法", "type": "used_with"},
                {"source": "因果关系", "target": "关系", "type": "is_a"},
                {"source": "事件", "target": "因果关系", "type": "involved_in"}
            ]
            
            for relation in relations:
                source_id = concept_ids.get(relation["source"])
                target_id = concept_ids.get(relation["target"])
                
                if source_id and target_id:
                    self.vector_store.add_relation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation["type"]
                    )
                    logger.debug(f"添加关系: {relation['source']} {relation['type']} {relation['target']}")
                    
            logger.info("知识图谱初始化完成")
            
            # 尝试从文件加载预存的知识图谱（如果存在）
            knowledge_graph_path = os.path.join(os.path.dirname(__file__), "knowledge", "knowledge_graph.json")
            if os.path.exists(knowledge_graph_path):
                logger.info(f"从文件加载知识图谱: {knowledge_graph_path}")
                if self.vector_store.load_knowledge_graph(knowledge_graph_path):
                    logger.info("知识图谱加载成功")
                else:
                    logger.warning("知识图谱加载失败")
                    
        except Exception as e:
            logger.error(f"知识图谱初始化失败: {str(e)}")
        
    def _subscribe_to_events(self):
        """订阅关键系统事件"""
        # 订阅用户输入事件
        self.event_system.subscribe("user.input", self._handle_user_input)
        
        # 订阅系统错误事件
        self.event_system.subscribe("system.error", self._handle_system_error)
        
        # 订阅元认知反馈事件
        self.event_system.subscribe("metacognition.feedback", self._handle_metacognition_feedback)
        
        # 订阅感知事件
        self.event_system.subscribe("perception.image", self._handle_image_perception)
        self.event_system.subscribe("perception.audio", self._handle_audio_perception)
        
    def _handle_user_input(self, event):
        """
        处理用户输入事件
        
        Args:
            event: 事件数据
        """
        input_data = event.get("data", {})
        content = input_data.get("text", "")  # 使用text而不是content
        
        logger.info(f"收到用户输入: {content[:50]}...")
        
        # 记录到记忆系统
        self.memory_system.add_to_short_term({
            "type": "user_input",
            "content": content,  # 使用text作为content
            "timestamp": event.get("timestamp", time.time())
        })
        
    def _handle_system_error(self, event):
        """
        处理系统错误事件
        
        Args:
            event: 事件数据
        """
        error_data = event["data"]
        logger.error(f"系统错误: {error_data['message']}, 组件: {error_data.get('component', 'unknown')}")
        
        # 记录到记忆系统
        self.memory_system.add_to_short_term({
            "type": "system_error",
            "message": error_data["message"],
            "component": error_data.get("component", "unknown"),
            "timestamp": event["timestamp"]
        })
        
    def _handle_metacognition_feedback(self, event):
        """
        处理元认知反馈事件
        
        Args:
            event: 事件数据
        """
        feedback = event["data"]
        feedback_type = feedback.get("type", "unknown")
        
        logger.info(f"元认知反馈: {feedback_type} - {feedback.get('message', '')}")
        
    def _handle_image_perception(self, event):
        """
        处理图像感知事件
        
        Args:
            event: 事件数据
        """
        image_data = event["data"]
        logger.info(f"收到图像: {image_data.get('file_path', '数据URL图像')}")
        
        # 处理图像
        if hasattr(self, 'perception') and 'multimodal' in self.perception:
            file_path = image_data.get('file_path')
            image_base64 = image_data.get('image_data')
            query = image_data.get('query', '')
            
            try:
                if file_path:
                    result = self.perception['multimodal'].process_image(file_path, query)
                elif image_base64:
                    result = self.perception['multimodal'].process_image(image_base64, query)
                else:
                    result = {"status": "error", "message": "无效的图像数据"}
                    
                # 记录结果到记忆系统
                if result.get("status") == "success":
                    self.memory_system.add_to_short_term({
                        "type": "image_analysis",
                        "description": result.get("description", ""),
                        "text": result.get("text", ""),
                        "query_answer": result.get("query_answer", ""),
                        "timestamp": time.time()
                    })
                    
                # 发布处理结果事件
                self.event_system.publish("perception.image_result", {
                    "result": result,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"图像处理错误: {str(e)}")
                self.event_system.publish("system.error", {
                    "message": f"图像处理错误: {str(e)}",
                    "component": "perception.multimodal",
                    "timestamp": time.time()
                })
        
    def _handle_audio_perception(self, event):
        """
        处理音频感知事件
        
        Args:
            event: 事件数据
        """
        audio_data = event["data"]
        logger.info(f"收到音频: {audio_data.get('file_path', '音频数据')}")
        
        # 处理音频
        if hasattr(self, 'perception') and 'multimodal' in self.perception:
            file_path = audio_data.get('file_path')
            
            try:
                if file_path:
                    result = self.perception['multimodal'].process_audio(file_path)
                else:
                    result = {"status": "error", "message": "无效的音频数据"}
                    
                # 记录结果到记忆系统
                if result.get("status") == "success":
                    self.memory_system.add_to_short_term({
                        "type": "audio_transcription",
                        "transcription": result.get("transcription", ""),
                        "timestamp": time.time()
                    })
                    
                    # 如果有文本，作为用户输入处理
                    if "transcription" in result and result["transcription"]:
                        self.process_user_input(result["transcription"])
                    
                # 发布处理结果事件
                self.event_system.publish("perception.audio_result", {
                    "result": result,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"音频处理错误: {str(e)}")
                self.event_system.publish("system.error", {
                    "message": f"音频处理错误: {str(e)}",
                    "component": "perception.multimodal",
                    "timestamp": time.time()
                })
        
    def start(self):
        """启动 GHOST AGI 系统"""
        if self.running:
            logger.warning("系统已经在运行")
            return False
            
        logger.info("正在启动 GHOST AGI 系统...")
        self.running = True
        self.start_time = time.time()
        
        # 启动元认知监控
        self.meta_cognition.start_monitoring()
        
        # 发布系统启动事件
        self.event_system.publish("system.started", {
            "timestamp": self.start_time,
            "config": self.config
        })
        
        logger.info("GHOST AGI 系统已启动")
        return True
        
    def stop(self):
        """停止 GHOST AGI 系统"""
        if not self.running:
            logger.warning("系统未在运行")
            return False
            
        logger.info("正在停止 GHOST AGI 系统...")
        
        # 停止元认知监控
        self.meta_cognition.stop_monitoring()
        
        # 停止记忆处理
        self.memory_system.stop_memory_processing()
        
        # 发布系统停止事件
        self.event_system.publish("system.stopped", {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time
        })
        
        self.running = False
        logger.info("GHOST AGI 系统已停止")
        return True
        
    def process_user_input(self, text: str) -> Dict[str, Any]:
        """
        处理用户输入
        
        Args:
            text: 用户输入文本
            
        Returns:
            Dict: 处理结果
        """
        # 创建输入事件
        input_event = {
            "type": "user_input",
            "text": text,
            "timestamp": time.time()
        }
        
        # 发布事件
        self.event_system.publish("user.input", input_event)
        
        # 尝试零样本学习处理
        zero_shot_result = self._try_zero_shot_learning(text)
        if zero_shot_result and zero_shot_result.get("status") == "success":
            logger.info("使用零样本学习成功处理输入")
            return {
                "status": "success",
                "response": zero_shot_result.get("generated_text", zero_shot_result.get("result", "我理解了")),
                "source": "zero_shot_learning"
            }
            
        # 如果零样本学习没有成功处理，使用标准处理流程
        # 这里可以是现有的处理逻辑
        
        # 存储到记忆系统
        if self.memory_system:
            self.memory_system.add_to_short_term({
                "type": "user_input",
            "content": text,
            "timestamp": time.time()
        })
        
        # 返回处理结果
        return {
            "status": "processing",
            "response": "正在处理您的请求...",
            "follow_up_suggestions": self._generate_follow_up_suggestions("text", {})
        }
        
    def _try_zero_shot_learning(self, text: str) -> Dict[str, Any]:
        """
        尝试使用零样本学习处理输入
        
        Args:
            text: 用户输入文本
            
        Returns:
            Dict: 处理结果，如果无法处理则返回None
        """
        # 如果有学习集成器，优先使用
        if hasattr(self, "learning_integrator") and self.learning_integrator:
            try:
                # 构建零样本查询
                query_data = {
                    "task_type": "zero_shot",
                    "query": {
                        "task_type": "generation",
                        "data": {"prompt": text},
                        "context": {}
                    }
                }
        
                # 调用集成器
                result = self.learning_integrator.learn(query_data)
                if result and result.get("status") != "error":
                    return result
                    
            except Exception as e:
                logger.error(f"零样本学习集成器处理失败: {str(e)}")
                
        # 如果没有集成器或处理失败，尝试直接使用零样本模块
        elif hasattr(self, "zero_shot_learning") and self.zero_shot_learning:
            try:
                # 构建查询
                query = {
                    "task_type": "generation",
                    "data": {"prompt": text},
                    "context": {}
                }
                
                # 直接调用零样本模块
                result = self.zero_shot_learning.zero_shot_inference(query)
                if result and result.get("status") != "error":
                    return result
                    
            except Exception as e:
                logger.error(f"零样本学习模块处理失败: {str(e)}")
                
        # 如果都不可用或处理失败，返回None
        return None
        
    def process_image(self, image_data, query: str = None) -> Dict[str, Any]:
        """
        处理图像输入
        
        Args:
            image_data: 图像数据，可以是字节流、文件路径或base64字符串
            query: 可选，关于图像的问题
            
        Returns:
            Dict: 处理结果
        """
        try:
            logger.info(f"正在处理图像输入，{'有' if query else '无'}查询")
            
            # 处理图像
            if self.perception and "multimodal" in self.perception:
                result = self.perception["multimodal"].process_image(image_data, query)
                
                # 增强响应信息
                if result["status"] == "success":
                    # 发布图像处理事件
                    self.event_system.publish("perception.image_processed", {
                        "success": True,
                        "has_query": query is not None,
                        "description_length": len(result.get("description", "")),
                        "text_extracted": len(result.get("text", "")) > 0
                    })
                    
                    # 将图像内容存入记忆系统
                    if self.memory_system:
                        memory_data = {
                            "type": "perception",
                            "modality": "image",
                            "description": result.get("description", ""),
                            "text_content": result.get("text", ""),
                            "query": query,
                            "query_result": result.get("query_answer", ""),
                            "timestamp": result.get("timestamp", time.time())
                        }
                        
                        self.memory_system.add(memory_data, "短期记忆")
                    
                    response = {
                        "status": "success",
                        "description": result.get("description", ""),
                        "text": result.get("text", ""),
                        "query_answer": result.get("query_answer", "") if query else None,
                        "suggestions": self._generate_follow_up_suggestions("image", result)
                    }
                else:
                    response = {
                        "status": "error",
                        "message": result.get("message", "图像处理失败"),
                        "error_type": result.get("error_type", "处理错误")
                    }
                    
                    # 发布错误事件
                    self.event_system.publish("perception.image_error", {
                        "error_message": response["message"]
                    })
                
                return response
            else:
                logger.error("多模态感知模块未初始化")
                return {
                    "status": "error",
                    "message": "多模态感知模块未初始化"
                }
        except Exception as e:
            logger.error(f"图像处理异常: {str(e)}")
            return {
                "status": "error",
                "message": f"处理图像时出错: {str(e)}"
            }

    def process_audio(self, audio_data) -> Dict[str, Any]:
        """
        处理音频输入
        
        Args:
            audio_data: 音频数据，可以是字节流或文件路径
            
        Returns:
            Dict: 处理结果
        """
        try:
            logger.info("正在处理音频输入")
            
            # 处理音频
            if self.perception and "multimodal" in self.perception:
                result = self.perception["multimodal"].process_audio(audio_data)
                
                if result["status"] == "success":
                    transcription = result.get("transcription", "")
                    
                    # 发布音频处理事件
                    self.event_system.publish("perception.audio_processed", {
                        "success": True,
                        "transcription_length": len(transcription),
                        "has_content": len(transcription) > 0
                    })
                    
                    # 生成文本响应 (使用核心智能体处理转录文本)
                    text_response = ""
                    if transcription:
                        try:
                            # 处理转录文本作为用户输入
                            input_response = self.process_user_input(transcription)
                            text_response = input_response.get("response", "")
                        except Exception as resp_err:
                            logger.error(f"生成响应时出错: {str(resp_err)}")
                            text_response = "无法生成响应，请重试。"
                    
                    # 将音频转录存入记忆系统
                    if self.memory_system and transcription:
                        memory_data = {
                            "type": "perception",
                            "modality": "audio",
                            "transcription": transcription,
                            "system_response": text_response,
                            "timestamp": result.get("timestamp", time.time())
                        }
                        
                        self.memory_system.add(memory_data, "短期记忆")
                    
                    response = {
                        "status": "success",
                        "transcription": transcription,
                        "text_response": text_response,
                        "audio_duration": result.get("duration", 0),
                        "suggestions": self._generate_follow_up_suggestions("audio", result)
                    }
                else:
                    response = {
                        "status": "error",
                        "message": result.get("message", "音频处理失败"),
                        "error_type": result.get("error_type", "处理错误")
                    }
                    
                    # 发布错误事件
                    self.event_system.publish("perception.audio_error", {
                        "error_message": response["message"]
                    })
                
                return response
            else:
                logger.error("多模态感知模块未初始化")
                return {
                    "status": "error",
                    "message": "多模态感知模块未初始化"
                }
        except Exception as e:
            logger.error(f"音频处理异常: {str(e)}")
            return {
                "status": "error",
                "message": f"处理音频时出错: {str(e)}"
            }
        
    def _generate_follow_up_suggestions(self, modality: str, result: Dict[str, Any]) -> List[str]:
        """
        根据处理结果生成后续交互建议
        
        Args:
            modality: 模态类型 ('image' 或 'audio')
            result: 处理结果
            
        Returns:
            List[str]: 建议列表
        """
        suggestions = []
        
        if modality == "image":
            # 图像相关建议
            if "description" in result and result["description"]:
                suggestions.append("这张图片是什么时候拍摄的?")
                suggestions.append("图片中有什么特别的细节?")
                
            if "text" in result and result["text"]:
                suggestions.append("请详细解释图片中的文字内容")
                suggestions.append("这段文字的主要意思是什么?")
            else:
                suggestions.append("图片中有任何文字吗?")
                
            # 通用图像建议
            suggestions.append("这张图片的主要颜色是什么?")
            suggestions.append("图片中有哪些主要对象?")
            
        elif modality == "audio":
            # 音频相关建议
            if "transcription" in result and result["transcription"]:
                suggestions.append("请详细解释一下这段内容")
                suggestions.append("这段话的主要意思是什么?")
                
            # 通用音频建议
            suggestions.append("您能再说一遍吗?")
            suggestions.append("请提供更多细节")
            
        # 限制建议数量
        import random
        if len(suggestions) > 3:
            suggestions = random.sample(suggestions, 3)
            
        return suggestions
        
    def execute_task(self, task_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            Dict: 执行结果
        """
        
        # 如果是零样本任务，尝试用零样本学习处理
        if task_description.get("type") == "zero_shot_task":
            zero_shot_result = self._try_zero_shot_learning(task_description.get("description", ""))
            if zero_shot_result and zero_shot_result.get("status") == "success":
                return {
                    "status": "success",
                    "result": zero_shot_result,
                    "source": "zero_shot_learning"
                }

        # ... 原有的执行任务逻辑 ...
        
        return {
            "status": "processing",
            "message": "任务正在处理中"
        }
        
    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建执行计划
        
        Args:
            goal: 计划目标
            context: 上下文信息
            
        Returns:
            Dict: 计划对象
        """
        if not self.running:
            logger.warning("系统未启动，无法创建计划")
            return {"status": "error", "message": "系统未启动"}
            
        # 使用规划引擎创建计划
        plan = self.planning_engine.create_plan(
            goal=goal,
            context=context or {}
        )
        
        # 记录计划创建
        self.memory_system.add_to_short_term({
            "type": "plan_created",
            "plan_id": plan["id"],
            "goal": goal,
            "step_count": len(plan["steps"]),
            "timestamp": time.time()
        })
        
        return plan
        
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            Dict: 执行结果
        """
        if not self.running:
            logger.warning("系统未启动，无法执行工具")
            return {"status": "error", "message": "系统未启动"}
            
        # 执行工具
        result = self.tool_executor.execute_tool(tool_name, params or {})
        
        # 记录工具执行
        self.memory_system.add_to_short_term({
            "type": "tool_execution",
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "timestamp": time.time()
        })
        
        return result
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict: 状态信息
        """
        # 获取记忆系统统计
        memory_stats = self.memory_system.get_memory_stats()
        
        # 获取事件系统统计
        event_stats = {
            "subscribers": self.event_system.get_subscriber_count(),
            "recent_events": len(self.event_system.get_history(100))
        }
        
        # 获取智能体统计
        agent_stats = {
            "registered_agents": len(self.agent_scheduler.agents),
            "pending_tasks": len(self.agent_scheduler.pending_tasks)
        }
        
        # 元认知统计
        meta_stats = {}
        if hasattr(self.meta_cognition, "get_performance_stats"):
            meta_stats = self.meta_cognition.get_performance_stats()
            
        # 工具统计
        tool_stats = {
            "tool_count": len(self.tool_executor.tools),
            "recent_executions": len(self.tool_executor.get_execution_history(10))
        }
        
        return {
            "status": "running" if self.running else "stopped",
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "memory": memory_stats,
            "events": event_stats,
            "agents": agent_stats,
            "metacognition": meta_stats,
            "tools": tool_stats,
            "timestamp": time.time()
        }
        
    def start_web_interface(self, blocking=True):
        """
        启动Web界面
        
        Args:
            blocking: 是否阻塞当前线程
            
        Returns:
            bool: 是否成功启动
        """
        # 确保系统已启动
        if not self.running:
            self.start()
            
        # 启动Web界面
        if blocking:
            self.web_interface.start()
        else:
            import threading
            thread = threading.Thread(target=self.web_interface.start, daemon=True)
            thread.start()
            
        return True

    def meta_optimize(self) -> Dict[str, Any]:
        """
        执行系统元优化
        
        Returns:
            Dict: 优化结果
        """
        optimization_results = {}
        
        # 如果有学习集成器，调用其元优化功能
        if hasattr(self, "learning_integrator") and self.learning_integrator:
            try:
                learning_result = self.learning_integrator.meta_optimize()
                optimization_results["learning"] = learning_result
                logger.info(f"学习系统元优化完成: {learning_result['status']}")
                
                # 处理改进结果
                if learning_result.get("improvements"):
                    improvements = learning_result["improvements"]
                    logger.info(f"学习系统发现 {len(improvements)} 项改进")
                    
                    # 检查是否有知识缺口需要填补
                    knowledge_gaps = [imp for imp in improvements if imp.get("type") == "knowledge_gap"]
                    if knowledge_gaps:
                        for gap in knowledge_gaps:
                            logger.info(f"发现知识缺口: {gap.get('domain', 'unknown')}")
                            
                            # 可以在这里触发知识获取流程
                            self.event_system.publish("system.knowledge_acquisition", {
                                "domain": gap.get("domain", "general"),
                                "priority": gap.get("priority", "medium"),
                                "timestamp": time.time()
                            })
            except Exception as e:
                logger.error(f"学习系统元优化失败: {str(e)}")
                optimization_results["learning"] = {"status": "error", "message": str(e)}
                
        # 调用进化引擎的自优化
        try:
            evolution_result = self.evolution_engine.auto_optimize()
            optimization_results["evolution"] = evolution_result
            logger.info(f"进化引擎自优化完成: {evolution_result['status']}")
            
            # 处理进化引擎的优化结果
            if evolution_result.get("improvements"):
                ev_improvements = evolution_result["improvements"]
                logger.info(f"进化引擎发现 {len(ev_improvements)} 项改进")
                
                # 应用建议的代码改进
                code_improvements = [imp for imp in ev_improvements if imp.get("type") == "code_improvement"]
                if code_improvements and self.code_generator:
                    for imp in code_improvements[:3]:  # 限制一次应用的改进数量
                        module = imp.get("module")
                        suggestion = imp.get("suggestion")
                        
                        if module and suggestion:
                            logger.info(f"应用代码改进到模块: {module}")
                            # 触发代码生成请求
                            self.event_system.publish("evolution.code_improvement", {
                                "module": module,
                                "suggestion": suggestion,
                                "priority": "medium",
                                "timestamp": time.time()
                            })
        except Exception as e:
            logger.error(f"进化引擎自优化失败: {str(e)}")
            optimization_results["evolution"] = {"status": "error", "message": str(e)}
            
        # 如果有知识图谱，尝试提取模式
        if hasattr(self.vector_store, "extract_patterns"):
            try:
                patterns = self.vector_store.extract_patterns()
                if patterns:
                    optimization_results["knowledge_patterns"] = {
                        "status": "success",
                        "patterns_found": len(patterns),
                        "patterns": patterns[:5]  # 只返回前5个模式
                    }
                    logger.info(f"知识图谱模式提取完成，发现 {len(patterns)} 个模式")
                    
                    # 将模式输入到学习系统中
                    if self.learning_integrator:
                        for pattern in patterns[:10]:  # 限制处理的模式数量
                            pattern_data = {
                                "pattern_type": pattern.get("pattern", "unknown"),
                                "description": pattern.get("description", ""),
                                "importance": pattern.get("importance", 0)
                            }
                            
                            self.event_system.publish("knowledge.pattern_discovered", {
                                "pattern": pattern_data,
                                "timestamp": time.time()
                            })
                else:
                    optimization_results["knowledge_patterns"] = {
                        "status": "no_patterns",
                        "message": "未发现显著模式"
                    }
            except Exception as e:
                logger.error(f"知识图谱模式提取失败: {str(e)}")
                optimization_results["knowledge_patterns"] = {"status": "error", "message": str(e)}
                
        # 保存知识图谱（如果支持）
        if hasattr(self.vector_store, "save_knowledge_graph"):
            try:
                knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
                os.makedirs(knowledge_dir, exist_ok=True)
                
                knowledge_graph_path = os.path.join(knowledge_dir, "knowledge_graph.json")
                if self.vector_store.save_knowledge_graph(knowledge_graph_path):
                    logger.info(f"知识图谱保存成功: {knowledge_graph_path}")
                    optimization_results["knowledge_save"] = {"status": "success"}
                else:
                    logger.warning("知识图谱保存失败")
                    optimization_results["knowledge_save"] = {"status": "failure"}
            except Exception as e:
                logger.error(f"知识图谱保存失败: {str(e)}")
                optimization_results["knowledge_save"] = {"status": "error", "message": str(e)}
                
        # 整合系统优化结果
        overall_status = "completed"
        if all(r.get("status") == "error" for r in optimization_results.values()):
            overall_status = "error"
        elif any(r.get("status") == "error" for r in optimization_results.values()):
            overall_status = "partial"
            
        return {
            "status": overall_status,
            "results": optimization_results,
            "timestamp": time.time()
        }

# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GHOST AGI 智能系统")
    parser.add_argument('--web', action='store_true', help='启动Web界面')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web界面主机地址')
    parser.add_argument('--port', type=int, default=5000, help='Web界面端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        "web_interface": {
            "host": args.host,
            "port": args.port,
            "debug": args.debug
        }
    }
    
    # 创建并启动系统
    agi = GhostAGI(config=config)
    agi.start()
    
    try:
        if args.web:
            # 启动Web界面
            print(f"\nGHOST AGI Web界面已启动: http://{args.host}:{args.port}")
            agi.start_web_interface(blocking=True)
        else:
            # 简单命令行交互
            print("\nGHOST AGI 系统已启动。输入 'exit' 退出。")
            
            while True:
                user_input = input("\n> ")
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                if user_input.lower() in ['status', 'stats']:
                    # 显示系统状态
                    status = agi.get_system_status()
                    print("\n系统状态:")
                    print(f"- 运行状态: {status['status']}")
                    print(f"- 运行时间: {status['uptime']:.1f} 秒")
                    print(f"- 记忆: 短期 {status['memory']['short_term']['count']} 项, 长期 {status['memory']['long_term']['count']} 项")
                    print(f"- 事件: {status['events']['recent_events']} 个最近事件, {status['events']['subscribers']} 个订阅者")
                    print(f"- 智能体: {status['agents']['registered_agents']} 个已注册, {status['agents']['pending_tasks']} 个待处理任务")
                    print(f"- 工具: {status['tools']['tool_count']} 个已注册")
                    continue
                    
                if user_input.lower() == 'web':
                    # 启动Web界面
                    print(f"\n启动Web界面: http://{args.host}:{args.port}")
                    agi.start_web_interface(blocking=False)
                    continue
                    
                # 处理用户输入
                result = agi.process_user_input(user_input)
                
                # 显示结果
                if isinstance(result, dict) and "response" in result:
                    print(f"\n{result['response']}")
                else:
                    print(f"\n处理结果: {result}")
                    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止系统...")
    finally:
        # 停止系统
        agi.stop()
        print("GHOST AGI 系统已停止。")