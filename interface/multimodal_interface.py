# -*- coding: utf-8 -*-
"""
多模态交互界面 (Multimodal Interface)

提供多种交互方式的统一接口，支持文本、语音、图像等输入输出
实现用户与系统之间的自然交互
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import queue

# 导入多模态感知模块
from perception.multimodal import MultimodalPerception
from perception.text_perception import TextPerception

class MultimodalInterface:
    def __init__(self):
        """
        初始化多模态交互界面
        """
        self.logger = logging.getLogger("MultimodalInterface")
        self.input_queue = queue.Queue()  # 输入队列
        self.output_queue = queue.Queue()  # 输出队列
        
        # 初始化多模态感知模块
        self.multimodal_perception = MultimodalPerception()
        self.text_perception = TextPerception()
        
        # 支持的输入模态
        self.supported_input_modalities = {
            "text": self._process_text_input,
            "voice": self._process_voice_input,
            "image": self._process_image_input,
            "video": self._process_video_input,
            "mixed": self._process_mixed_input
        }
        
        # 支持的输出模态
        self.supported_output_modalities = {
            "text": self._generate_text_output,
            "voice": self._generate_voice_output,
            "image": self._generate_image_output,
            "mixed": self._generate_mixed_output
        }
        
        # 当前活跃的输入/输出模态
        self.active_input_modalities = ["text"]
        self.active_output_modalities = ["text"]
        
        # 界面状态
        self.is_running = False
        self.processing_thread = None
        
        # 会话历史
        self.session_history = []
        self.max_history_length = 100
        
        self.logger.info("多模态交互界面初始化完成")
    
    def start(self):
        """
        启动交互界面
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_io_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("多模态交互界面已启动")
        
    def stop(self):
        """
        停止交互界面
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        self.logger.info("多模态交互界面已停止")
    
    def _process_io_loop(self):
        """
        处理输入输出循环
        """
        while self.is_running:
            try:
                # 处理输入队列
                try:
                    input_data = self.input_queue.get(timeout=0.1)
                    self._handle_input(input_data)
                    self.input_queue.task_done()
                except queue.Empty:
                    pass
                    
                # 处理输出队列
                try:
                    output_data = self.output_queue.get(timeout=0.1)
                    self._handle_output(output_data)
                    self.output_queue.task_done()
                except queue.Empty:
                    pass
                    
                time.sleep(0.01)  # 避免CPU占用过高
            except Exception as e:
                self.logger.error(f"处理IO循环时出错: {e}")
    
    def add_input(self, input_data: Dict[str, Any]):
        """
        添加输入到队列
        
        Args:
            input_data: 输入数据，包含模态和内容
        """
        if not isinstance(input_data, dict) or "modality" not in input_data:
            self.logger.error("输入数据格式错误，必须包含modality字段")
            return False
            
        modality = input_data["modality"]
        if modality not in self.supported_input_modalities:
            self.logger.error(f"不支持的输入模态: {modality}")
            return False
            
        # 添加时间戳
        if "timestamp" not in input_data:
            input_data["timestamp"] = time.time()
            
        self.input_queue.put(input_data)
        return True
    
    def add_output(self, output_data: Dict[str, Any]):
        """
        添加输出到队列
        
        Args:
            output_data: 输出数据，包含模态和内容
        """
        if not isinstance(output_data, dict) or "modality" not in output_data:
            self.logger.error("输出数据格式错误，必须包含modality字段")
            return False
            
        modality = output_data["modality"]
        if modality not in self.supported_output_modalities:
            self.logger.error(f"不支持的输出模态: {modality}")
            return False
            
        # 添加时间戳
        if "timestamp" not in output_data:
            output_data["timestamp"] = time.time()
            
        self.output_queue.put(output_data)
        return True
    
    def _handle_input(self, input_data: Dict[str, Any]):
        """
        处理输入数据
        
        Args:
            input_data: 输入数据
        """
        modality = input_data["modality"]
        processor = self.supported_input_modalities[modality]
        
        try:
            processed_data = processor(input_data)
            
            # 记录到会话历史
            history_entry = {
                "type": "input",
                "modality": modality,
                "data": input_data,
                "processed": processed_data,
                "timestamp": input_data.get("timestamp", time.time())
            }
            
            self._add_to_history(history_entry)
            
            return processed_data
        except Exception as e:
            self.logger.error(f"处理{modality}输入时出错: {e}")
            return None
    
    def _handle_output(self, output_data: Dict[str, Any]):
        """
        处理输出数据
        
        Args:
            output_data: 输出数据
        """
        modality = output_data["modality"]
        generator = self.supported_output_modalities[modality]
        
        try:
            generated_output = generator(output_data)
            
            # 记录到会话历史
            history_entry = {
                "type": "output",
                "modality": modality,
                "data": output_data,
                "generated": generated_output,
                "timestamp": output_data.get("timestamp", time.time())
            }
            
            self._add_to_history(history_entry)
            
            return generated_output
        except Exception as e:
            self.logger.error(f"生成{modality}输出时出错: {e}")
            return None
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """
        添加条目到会话历史
        
        Args:
            entry: 历史条目
        """
        self.session_history.append(entry)
        
        # 限制历史长度
        if len(self.session_history) > self.max_history_length:
            self.session_history = self.session_history[-self.max_history_length:]
    
    def get_history(self, limit: int = None, modality_filter: str = None) -> List[Dict[str, Any]]:
        """
        获取会话历史
        
        Args:
            limit: 限制返回的条目数量
            modality_filter: 按模态过滤
            
        Returns:
            List[Dict[str, Any]]: 会话历史
        """
        if modality_filter:
            filtered_history = [entry for entry in self.session_history 
                               if entry.get("modality") == modality_filter]
        else:
            filtered_history = self.session_history.copy()
            
        if limit and limit > 0:
            return filtered_history[-limit:]
            
        return filtered_history
    
    def set_active_modalities(self, input_modalities: List[str] = None, output_modalities: List[str] = None):
        """
        设置活跃的输入输出模态
        
        Args:
            input_modalities: 输入模态列表
            output_modalities: 输出模态列表
        """
        if input_modalities:
            for modality in input_modalities:
                if modality not in self.supported_input_modalities:
                    self.logger.warning(f"不支持的输入模态: {modality}")
                    continue
            self.active_input_modalities = input_modalities
            
        if output_modalities:
            for modality in output_modalities:
                if modality not in self.supported_output_modalities:
                    self.logger.warning(f"不支持的输出模态: {modality}")
                    continue
            self.active_output_modalities = output_modalities
            
        self.logger.info(f"已设置活跃模态 - 输入: {self.active_input_modalities}, 输出: {self.active_output_modalities}")
    
    # 输入处理方法
    def _process_text_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文本输入
        
        Args:
            input_data: 文本输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if "content" not in input_data:
            return {"status": "error", "message": "缺少content字段"}
            
        text_content = input_data["content"]
        # 使用文本感知模块处理
        result = self.text_perception.process_text(text_content)
        
        return {
            "status": "success",
            "modality": "text",
            "original": text_content,
            "processed": result,
            "timestamp": time.time()
        }
    
    def _process_voice_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理语音输入
        
        Args:
            input_data: 语音输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if "audio_data" not in input_data:
            return {"status": "error", "message": "缺少audio_data字段"}
            
        # 使用多模态感知模块处理音频
        result = self.multimodal_perception.process_audio(input_data["audio_data"])
        
        return {
            "status": "success",
            "modality": "voice",
            "processed": result,
            "timestamp": time.time()
        }
    
    def _process_image_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理图像输入
        
        Args:
            input_data: 图像输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if "image_data" not in input_data:
            return {"status": "error", "message": "缺少image_data字段"}
            
        # 使用多模态感知模块处理图像
        result = self.multimodal_perception.process_image(input_data["image_data"])
        
        return {
            "status": "success",
            "modality": "image",
            "processed": result,
            "timestamp": time.time()
        }
    
    def _process_video_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频输入
        
        Args:
            input_data: 视频输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 视频处理暂未实现，返回未实现状态
        return {
            "status": "error",
            "message": "视频处理功能尚未实现",
            "modality": "video",
            "timestamp": time.time()
        }
    
    def _process_mixed_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理混合模态输入
        
        Args:
            input_data: 混合模态输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if "modalities" not in input_data or not isinstance(input_data["modalities"], dict):
            return {"status": "error", "message": "缺少有效的modalities字段"}
            
        # 使用多模态感知模块处理混合输入
        result = self.multimodal_perception.process_multimodal(input_data["modalities"])
        
        return {
            "status": "success",
            "modality": "mixed",
            "processed": result,
            "timestamp": time.time()
        }
    
    # 输出生成方法
    def _generate_text_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成文本输出
        
        Args:
            output_data: 输出数据
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if "content" not in output_data:
            return {"status": "error", "message": "缺少content字段"}
            
        # 简单返回文本内容，实际应用中可能需要格式化或其他处理
        return {
            "status": "success",
            "modality": "text",
            "content": output_data["content"],
            "timestamp": time.time()
        }
    
    def _generate_voice_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成语音输出
        
        Args:
            output_data: 输出数据
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if "content" not in output_data:
            return {"status": "error", "message": "缺少content字段"}
            
        # 语音合成暂未实现，返回未实现状态
        return {
            "status": "error",
            "message": "语音合成功能尚未实现",
            "modality": "voice",
            "content": output_data["content"],
            "timestamp": time.time()
        }
    
    def _generate_image_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成图像输出
        
        Args:
            output_data: 输出数据
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if "prompt" not in output_data:
            return {"status": "error", "message": "缺少prompt字段"}
            
        # 图像生成暂未实现，返回未实现状态
        return {
            "status": "error",
            "message": "图像生成功能尚未实现",
            "modality": "image",
            "prompt": output_data["prompt"],
            "timestamp": time.time()
        }
    
    def _generate_mixed_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成混合模态输出
        
        Args:
            output_data: 输出数据
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if "components" not in output_data or not isinstance(output_data["components"], list):
            return {"status": "error", "message": "缺少有效的components字段"}
            
        results = []
        for component in output_data["components"]:
            if "modality" not in component:
                continue
                
            modality = component["modality"]
            if modality in self.supported_output_modalities:
                result = self.supported_output_modalities[modality](component)
                results.append(result)
                
        return {
            "status": "success" if results else "error",
            "modality": "mixed",
            "components": results,
            "timestamp": time.time()
        }
    
    def get_interface_status(self) -> Dict[str, Any]:
        """
        获取界面状态信息
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "is_running": self.is_running,
            "active_input_modalities": self.active_input_modalities,
            "active_output_modalities": self.active_output_modalities,
            "supported_input_modalities": list(self.supported_input_modalities.keys()),
            "supported_output_modalities": list(self.supported_output_modalities.keys()),
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "history_length": len(self.session_history)
        }