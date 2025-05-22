# -*- coding: utf-8 -*-
"""
data_validator

数据验证模块，提供各种数据验证功能
"""

# Author: GHOST AGI
# Created: 2025-05-17 21:18:21

import re
import datetime
from typing import Dict, List, Any, Optional, Union

def is_valid_phone(phone_number, country_code='CN'):
    """检查是否是有效的电话号码"""

    if country_code == 'CN':
        # 中国手机号格式
        pattern = r'^1[3-9]\d{9}$'
    elif country_code == 'US':
        # 美国电话号码格式
        pattern = r'^\+?1?\s*\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}$'
    else:
        # 通用格式
        pattern = r'^\+?\d{1,4}?[-\s]?\(?\d{1,4}\)?[-\s]?\d{1,9}$'
    
    return bool(re.match(pattern, phone_number))

def is_valid_ip(ip_address, version=4):
    """检查是否是有效的IP地址"""

    if version == 4:
        # IPv4 格式
        pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    elif version == 6:
        # IPv6 格式 (简化版)
        pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    else:
        return False
    
    return bool(re.match(pattern, ip_address))

class DataValidator:
    """数据验证器类"""

    def __init__(self, strict_mode=False):
        """初始化验证器"""

        self.strict_mode = strict_mode
        self.errors = []

    def validate_email(self, email):
        """验证电子邮件地址"""

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            error = f"Invalid email format: {email}"
            self.errors.append(error)
            return False
        return True

    def validate_date(self, date_str, format_str='%Y-%m-%d'):
        """验证日期格式"""

        try:
            datetime.datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            error = f"Invalid date format: {date_str}, expected format: {format_str}"
            self.errors.append(error)
            return False

    def get_errors(self):
        """获取错误列表"""

        return self.errors

    def clear_errors(self):
        """清除错误列表"""

        self.errors = []



