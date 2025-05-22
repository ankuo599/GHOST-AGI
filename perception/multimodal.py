# -*- coding: utf-8 -*-
"""
多模态感知系统
用于处理图像和语音输入，支持CLIP图像分类和Whisper语音识别
"""

import os
import time
import base64
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
import tempfile
import re
from pathlib import Path
import json
from datetime import datetime

# 设置日志记录器
logger = logging.getLogger("MultimodalPerception")

# 全局变量声明
VISION_AVAILABLE = False
AUDIO_AVAILABLE = False

# 尝试导入PIL
try:
    from PIL import Image, ImageEnhance
    logger.info("PIL库加载成功")
except ImportError:
    logger.warning("未安装PIL库，图像处理功能将不可用")
    logger.warning("请安装: pip install pillow")

# 尝试导入torch
try:
    import torch
    import torch.nn as nn
    import numpy as np
    logger.info("PyTorch库加载成功")
except ImportError:
    logger.warning("未安装PyTorch库，深度学习功能将不可用")
    logger.warning("请安装: pip install torch numpy")

# 尝试导入librosa
try:
    import librosa
    logger.info("Librosa库加载成功")
except ImportError:
    logger.warning("未安装Librosa库，音频处理功能将不可用")
    logger.warning("请安装: pip install librosa")

# 尝试导入transformers
try:
    import asyncio
    from transformers import CLIPProcessor, CLIPModel, pipeline
    from transformers import WhisperProcessor, WhisperModel
    VISION_AVAILABLE = True
    AUDIO_AVAILABLE = True
    logger.info("Transformers库加载成功")
except ImportError:
    logger.warning("未安装Transformers库，多模态感知功能将不可用")
    logger.warning("请安装: pip install transformers[sentencepiece,vision,audio]")
except Exception as e:
    logger.warning(f"加载Transformers库时出错: {str(e)}")
    logger.warning("多模态感知功能将不可用")

# 全局变量声明
VISION_AVAILABLE = False
AUDIO_AVAILABLE = False

# 添加对视觉和语音模型的支持
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel, pipeline
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
    import numpy as np
    from PIL import Image
    
    VISION_AVAILABLE = True
    AUDIO_AVAILABLE = True
except ImportError:
    logging.warning("未安装多模态感知所需的依赖库，视觉和语音功能将不可用")
    logging.warning("请安装: pip install torch transformers pillow numpy")

class MultimodalPerception:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("MultimodalPerception")
        
        # 初始化CLIP模型
        self.clip_model = None
        self.clip_processor = None
        self._init_clip()
        
        # 初始化Whisper模型
        self.whisper_model = None
        self.whisper_processor = None
        self._init_whisper()
        
        # 缓存
        self.image_cache: Dict[str, Dict[str, Any]] = {}
        self.audio_cache: Dict[str, Dict[str, Any]] = {}
        
    def _init_clip(self):
        """初始化CLIP模型"""
        if not VISION_AVAILABLE:
            self.logger.warning("CLIP模型依赖不可用，跳过初始化")
            return
            
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.logger.info("CLIP模型加载成功")
        except Exception as e:
            self.logger.error(f"CLIP模型加载失败: {str(e)}")
            VISION_AVAILABLE = False
            
    def _init_whisper(self):
        """初始化Whisper模型"""
        if not AUDIO_AVAILABLE:
            self.logger.warning("Whisper模型依赖不可用，跳过初始化")
            return
            
        try:
            self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.logger.info("Whisper模型加载成功")
        except Exception as e:
            self.logger.error(f"Whisper模型加载失败: {str(e)}")
            global AUDIO_AVAILABLE
            AUDIO_AVAILABLE = False
            
    def process_image(self, image_path: str, 
                          query: Optional[str] = None) -> Dict[str, Any]:
        """
        处理图像
        
        Args:
            image_path: 图像路径
            query: 可选的查询文本
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查缓存
            if image_path in self.image_cache:
                return self.image_cache[image_path]
                
            # 加载图像
            image = Image.open(image_path)
            
            # 图像分类
            if query:
                # 使用CLIP进行图像-文本匹配
                inputs = self.clip_processor(
                    images=image, 
                    text=query,
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                    
                result = {
                    "query": query,
                    "confidence": float(probs[0][0]),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 使用CLIP进行图像分类
                inputs = self.clip_processor(
                            images=image,
                            return_tensors="pt",
                            padding=True
                        )
                        
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)
                    
                result = {
                    "features": outputs.numpy().tolist(),
                    "timestamp": datetime.now().isoformat()
                }
                
            # 更新缓存
            self.image_cache[image_path] = result
            return result
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        处理音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查缓存
            if audio_path in self.audio_cache:
                return self.audio_cache[audio_path]
                
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 使用Whisper进行语音识别
            input_features = self.whisper_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features
            
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
                transcription = self.whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            result = {
                "transcription": transcription,
                "timestamp": datetime.now().isoformat()
            }
            
            # 更新缓存
            self.audio_cache[audio_path] = result
            return result
            
        except Exception as e:
            self.logger.error(f"音频处理失败: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def process_multimodal(self, image_path: str, audio_path: str,
                               query: Optional[str] = None) -> Dict[str, Any]:
        """
        处理多模态输入
        
        Args:
            image_path: 图像路径
            audio_path: 音频路径
            query: 可选的查询文本
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 并行处理图像和音频
            image_task = asyncio.create_task(self.process_image(image_path, query))
            audio_task = asyncio.create_task(self.process_audio(audio_path))
            
            image_result = await image_task
            audio_result = await audio_task
            
            # 合并结果
            result = {
                "image": image_result,
                "audio": audio_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # 如果提供了查询，进行多模态融合
            if query and "error" not in image_result and "error" not in audio_result:
                # TODO: 实现多模态融合逻辑
                pass
                
            return result
            
        except Exception as e:
            self.logger.error(f"多模态处理失败: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def clear_cache(self):
        """清除缓存"""
        self.image_cache.clear()
        self.audio_cache.clear()
        
    def save_state(self):
        """保存状态"""
        data = {
            "image_cache": self.image_cache,
            "audio_cache": self.audio_cache
        }
        
        with open(self.model_dir / "multimodal_state.json", "w", 
                 encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_state(self):
        """加载状态"""
        try:
            with open(self.model_dir / "multimodal_state.json", "r", 
                     encoding="utf-8") as f:
                data = json.load(f)
                
            self.image_cache = data["image_cache"]
            self.audio_cache = data["audio_cache"]
            
        except FileNotFoundError:
            self.logger.warning("状态文件不存在，使用空缓存")