# -*- coding: utf-8 -*-
"""
文本感知模块 (Text Perception)

负责处理和分析文本输入，提取关键信息和意图
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple

class TextPerception:
    def __init__(self):
        """
        初始化文本感知模块
        """
        self.patterns = {
            "command": r"^(执行|运行|启动|开始)\s+(.+)$",
            "question": r"^(什么|如何|为什么|是否|能否|可以|怎样|怎么).*\?$",
            "greeting": r"^(你好|早上好|下午好|晚上好|嗨|哈喽).*$"
        }
        self.keywords = {
            "positive": ["是", "好", "确定", "同意", "正确", "可以"],
            "negative": ["否", "不", "拒绝", "错误", "不行", "不可以"],
            "action": ["执行", "运行", "创建", "删除", "修改", "查找", "分析"]
        }
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        处理文本输入
        
        Args:
            text (str): 输入文本
            
        Returns:
            Dict[str, Any]: 处理结果，包含类型、意图和提取的信息
        """
        if not text or not isinstance(text, str):
            return {"type": "invalid", "confidence": 0.0}
            
        # 清理文本
        cleaned_text = self._clean_text(text)
        
        # 识别文本类型和意图
        text_type, confidence = self._identify_type(cleaned_text)
        intent = self._extract_intent(cleaned_text, text_type)
        
        # 提取关键信息
        entities = self._extract_entities(cleaned_text)
        
        # 情感分析
        sentiment = self._analyze_sentiment(cleaned_text)
        
        return {
            "original": text,
            "cleaned": cleaned_text,
            "type": text_type,
            "confidence": confidence,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "timestamp": time.time()
        }
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 去除多余空格
        cleaned = re.sub(r'\s+', ' ', text.strip())
        # 去除特殊字符（保留问号和感叹号）
        cleaned = re.sub(r'[^\w\s\?\!\,\.，。？！]', '', cleaned)
        return cleaned
    
    def _identify_type(self, text: str) -> Tuple[str, float]:
        """
        识别文本类型
        
        Args:
            text (str): 清理后的文本
            
        Returns:
            Tuple[str, float]: 文本类型和置信度
        """
        # 检查是否匹配预定义模式
        for pattern_type, pattern in self.patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                return pattern_type, 0.9
        
        # 基于特征的分类
        if '?' in text or '？' in text:
            return "question", 0.8
        elif any(keyword in text for keyword in self.keywords["action"]):
            return "command", 0.7
        elif len(text) < 10 and any(keyword in text for keyword in self.keywords["positive"] + self.keywords["negative"]):
            return "response", 0.6
        else:
            return "statement", 0.5
    
    def _extract_intent(self, text: str, text_type: str) -> str:
        """
        提取文本意图
        
        Args:
            text (str): 清理后的文本
            text_type (str): 文本类型
            
        Returns:
            str: 意图
        """
        if text_type == "command":
            # 提取命令意图
            match = re.match(self.patterns["command"], text)
            if match:
                return match.group(2)
            
            # 查找动作关键词
            for action in self.keywords["action"]:
                if action in text:
                    parts = text.split(action, 1)
                    if len(parts) > 1:
                        return action + parts[1]
        
        elif text_type == "question":
            # 简单提取问题主题
            # 实际应用中可能需要更复杂的NLP技术
            return "获取信息"
            
        elif text_type == "greeting":
            return "社交互动"
            
        # 默认意图
        return "general_communication"
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取实体
        
        Args:
            text (str): 清理后的文本
            
        Returns:
            List[Dict[str, Any]]: 提取的实体列表
        """
        entities = []
        
        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        for num in numbers:
            entities.append({
                "type": "number",
                "value": float(num) if '.' in num else int(num),
                "text": num
            })
        
        # 提取日期（简单模式）
        dates = re.findall(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?', text)
        for date in dates:
            entities.append({
                "type": "date",
                "value": date,
                "text": date
            })
        
        # 提取URL
        urls = re.findall(r'https?://\S+', text)
        for url in urls:
            entities.append({
                "type": "url",
                "value": url,
                "text": url
            })
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        简单的情感分析
        
        Args:
            text (str): 清理后的文本
            
        Returns:
            Dict[str, Any]: 情感分析结果
        """
        # 计算正面和负面关键词出现次数
        positive_count = sum(1 for keyword in self.keywords["positive"] if keyword in text)
        negative_count = sum(1 for keyword in self.keywords["negative"] if keyword in text)
        
        # 确定情感极性
        if positive_count > negative_count:
            polarity = "positive"
            score = min(0.5 + (positive_count - negative_count) * 0.1, 1.0)
        elif negative_count > positive_count:
            polarity = "negative"
            score = max(-0.5 - (negative_count - positive_count) * 0.1, -1.0)
        else:
            polarity = "neutral"
            score = 0.0
            
        return {
            "polarity": polarity,
            "score": score,
            "positive_keywords": positive_count,
            "negative_keywords": negative_count
        }
    
    def get_keywords(self, text: str, limit: int = 5) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text (str): 输入文本
            limit (int): 返回关键词数量限制
            
        Returns:
            List[str]: 关键词列表
        """
        # 简单的关键词提取（基于词频）
        # 实际应用中应使用更复杂的算法如TF-IDF
        words = re.findall(r'\w+', text.lower())
        word_freq = {}
        
        # 计算词频
        for word in words:
            if len(word) > 1:  # 忽略单字符词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:limit]]