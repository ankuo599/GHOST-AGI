# -*- coding: utf-8 -*-
"""
data_processor

数据处理工具模块，提供各种数据转换和验证功能
"""

# Author: GHOST AGI 自主进化系统
# Created: 2025-05-17 21:18:19

import json
import datetime
import re
from typing import Dict, List, Any, Optional, Union, Callable
from pandas import DataFrame

def sanitize_input(input_data: str):
    """清理和消毒输入数据，移除潜在危险字符"""

    # 移除可能的脚本注入字符
    sanitized = re.sub(r'[<>"\'\&;]', '', input_data)
    # 移除控制字符
    sanitized = re.sub(r'[\x00-\x1F\x7F]', '', sanitized)
    return sanitized

def parse_json_safe(json_str: str, default_value: Any = None):
    """安全解析JSON，处理可能的错误"""

    try:
        if not json_str or not isinstance(json_str, str):
            return default_value
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"JSON解析失败: {json_str[:100]}...")
        return default_value

class DataProcessor:
    """数据处理器，处理各种数据格式转换和验证"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化数据处理器"""

        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.transformers = {}
        self.validators = {}
        self._register_default_transformers()
        self._register_default_validators()

    def _register_default_transformers(self):
        """注册默认转换器"""

        # 注册默认转换器
        self.transformers = {
            'to_int': lambda x: int(x) if x else 0,
            'to_float': lambda x: float(x) if x else 0.0,
            'to_str': lambda x: str(x) if x is not None else '',
            'to_bool': lambda x: bool(x) if x is not None else False,
            'to_date': lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date() if x else None,
        }

    def _register_default_validators(self):
        """注册默认验证器"""

        # 注册默认验证器
        self.validators = {
            'is_empty': lambda x: x is None or (isinstance(x, (str, list, dict)) and len(x) == 0),
            'is_numeric': lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()),
            'is_date': lambda x: bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(x))) if x else False,
        }

    def register_transformer(self, name: str, transformer: Callable):
        """注册数据转换器"""

        self.transformers[name] = transformer
        self.logger.info(f"已注册转换器: {name}")

    def transform(self, data: Any, transformer_name: str, fallback: Any = None):
        """转换数据"""

        try:
            if transformer_name not in self.transformers:
                self.logger.warning(f"未知转换器: {transformer_name}")
                return fallback
                
            transformer = self.transformers[transformer_name]
            return transformer(data)
        except Exception as e:
            self.logger.error(f"转换失败: {str(e)}")
            return fallback

    def validate(self, data: Any, validator_name: str):
        """验证数据"""

        try:
            if validator_name not in self.validators:
                self.logger.warning(f"未知验证器: {validator_name}")
                return False
                
            validator = self.validators[validator_name]
            return validator(data)
        except Exception as e:
            self.logger.error(f"验证失败: {str(e)}")
            return False

    def process_dict(self, data: Dict[str, Any], schema: Dict[str, Dict[str, Any]]):
        """处理字典数据"""

        result = {}
        try:
            for field, field_schema in schema.items():
                if field not in data and field_schema.get('required', False):
                    raise ValueError(f"缺少必填字段: {field}")
                    
                value = data.get(field)
                
                # 应用验证
                if 'validator' in field_schema:
                    is_valid = self.validate(value, field_schema['validator'])
                    if not is_valid and field_schema.get('required', False):
                        raise ValueError(f"字段验证失败: {field}")
                    
                # 应用转换
                if 'transformer' in field_schema:
                    value = self.transform(value, field_schema['transformer'], field_schema.get('default'))
                    
                # 设置默认值
                if value is None and 'default' in field_schema:
                    value = field_schema['default']
                    
                result[field] = value
                
            return {'status': 'success', 'data': result}
        except Exception as e:
            self.logger.error(f"处理字典失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def to_dataframe(self, data: List[Dict[str, Any]]):
        """将数据转换为DataFrame"""

        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            self.logger.error("pandas未安装，无法转换为DataFrame")
            return None
        except Exception as e:
            self.logger.error(f"转换为DataFrame失败: {str(e)}")
            return None



