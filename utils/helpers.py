# -*- coding: utf-8 -*-
"""
辅助函数 (Helpers)

提供各种通用工具函数，支持系统其他模块
"""

import json
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Union

def generate_id() -> str:
    """
    生成唯一ID
    
    Returns:
        str: 唯一ID
    """
    return str(uuid.uuid4())

def get_timestamp() -> float:
    """
    获取当前时间戳
    
    Returns:
        float: 当前时间戳
    """
    return time.time()

def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间戳
    
    Args:
        timestamp (float): 时间戳
        format_str (str): 格式字符串
        
    Returns:
        str: 格式化后的时间字符串
    """
    return time.strftime(format_str, time.localtime(timestamp))

def save_json(data: Any, file_path: str) -> bool:
    """
    将数据保存为JSON文件
    
    Args:
        data: 要保存的数据
        file_path (str): 文件路径
        
    Returns:
        bool: 是否成功保存
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def load_json(file_path: str) -> Optional[Any]:
    """
    从JSON文件加载数据
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        Optional[Any]: 加载的数据，如果失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    深度合并两个字典
    
    Args:
        dict1 (Dict): 第一个字典
        dict2 (Dict): 第二个字典
        
    Returns:
        Dict: 合并后的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text (str): 原始文本
        max_length (int): 最大长度
        suffix (str): 后缀
        
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_get(obj: Dict, path: str, default: Any = None) -> Any:
    """
    安全获取嵌套字典中的值
    
    Args:
        obj (Dict): 字典对象
        path (str): 路径，使用点号分隔
        default (Any): 默认值
        
    Returns:
        Any: 获取的值或默认值
    """
    keys = path.split('.')
    current = obj
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
        
    return current

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    扁平化嵌套字典
    
    Args:
        d (Dict): 嵌套字典
        parent_key (str): 父键
        sep (str): 分隔符
        
    Returns:
        Dict: 扁平化后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)