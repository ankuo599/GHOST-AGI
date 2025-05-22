# -*- coding: utf-8 -*-
"""
测试代码生成器功能
"""

import os
import sys

from evolution.code_generator import CodeGenerator

def test_code_analysis():
    """测试代码质量分析功能"""
    print("===== 测试代码质量分析 =====")
    
    # 创建代码生成器实例
    code_generator = CodeGenerator()
    
    # 测试代码片段（包含一些常见问题）
    test_code = """
def badFunction(x, y):
    # 这个函数有命名问题和复杂度问题
    Result = []
    if x > 0:
        if y > 0:
            for i in range(x):
                for j in range(y):
                    Result.append(i * j)
        else:
            for i in range(x):
                Result.append(i)
    else:
        Result = [0]
    
    # 危险操作没有错误处理
    f = open("test.txt", "w")
    f.write("test")
    f.close()
    
    return Result

class badClass:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
        
    def remove_item(self, item):
        if item in self.data:
            self.data.remove(item)
    """
    
    # 分析代码质量
    analysis = code_generator.analyze_code_quality(test_code)
    
    # 打印分析结果
    print(f"代码质量得分: {analysis.get('quality_score', 0):.2f}/100")
    print(f"代码复杂度: {analysis.get('complexity', 0)}")
    print(f"可维护性: {analysis.get('maintainability', 0):.2f}/100")
    
    if 'documentation_score' in analysis:
        print(f"文档完整性: {analysis.get('documentation_score', 0):.2f}%")
    
    print("\n发现的问题:")
    for issue in analysis.get('issues', []):
        print(f"- [{issue.get('severity', 'info')}] {issue.get('description', '')}")
    
    print("\n改进建议:")
    for suggestion in analysis.get('suggestions', []):
        print(f"- [{suggestion.get('priority', 'low')}] {suggestion.get('description', '')}")

def test_code_optimization():
    """测试代码自动优化功能"""
    print("\n\n===== 测试代码自动优化 =====")
    
    # 创建代码生成器实例
    code_generator = CodeGenerator()
    
    # 低质量代码样本
    poor_code = """
def calculate_sum(numbers):
    # 非常低效的实现
    result = 0
    for i in range(len(numbers)):
        result = result + numbers[i]
    return result

def find_max(numbers):
    # 重复逻辑，没有错误处理
    if len(numbers) == 0:
        return None
    max_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val

def find_min(numbers):
    # 与find_max几乎相同的代码
    if len(numbers) == 0:
        return None
    min_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] < min_val:
            min_val = numbers[i]
    return min_val
"""
    
    # 执行自动优化
    optimization = code_generator.auto_optimize_code(poor_code, 
                                                    optimization_goals=["readability", "performance", "maintenance"])
    
    # 打印优化结果
    print(f"优化状态: {optimization.get('status', '')}")
    
    if optimization.get('status') == 'success':
        print("\n应用的优化:")
        for opt in optimization.get('applied_optimizations', []):
            print(f"- {opt.get('type', '')}: {opt.get('description', '')}")
        
        print("\n改进指标:")
        for metric, value in optimization.get('improvements', {}).items():
            print(f"- {metric}: {value:.2f}%")
        
        print("\n优化前的代码:")
        print(poor_code)
        
        print("\n优化后的代码:")
        print(optimization.get('optimized_code', ''))

def test_module_generation():
    """测试模块自动生成功能"""
    print("\n\n===== 测试模块自动生成 =====")
    
    # 创建代码生成器实例
    code_generator = CodeGenerator()
    
    # 模块规范
    module_spec = {
        "name": "data_validator",
        "type": "module",
        "description": "数据验证模块，提供各种数据验证功能",
        "author": "GHOST AGI",
        "imports": [
            "re",
            "datetime",
            {
                "module": "typing",
                "items": ["Dict", "List", "Any", "Optional", "Union"]
            }
        ],
        "classes": [
            {
                "name": "DataValidator",
                "description": "数据验证器类",
                "bases": [],
                "methods": [
                    {
                        "name": "__init__",
                        "description": "初始化验证器",
                        "params": ["self", "strict_mode=False"],
                        "body": "self.strict_mode = strict_mode\nself.errors = []"
                    },
                    {
                        "name": "validate_email",
                        "description": "验证电子邮件地址",
                        "params": ["self", "email"],
                        "body": "pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\nif not re.match(pattern, email):\n    error = f\"Invalid email format: {email}\"\n    self.errors.append(error)\n    return False\nreturn True"
                    },
                    {
                        "name": "validate_date",
                        "description": "验证日期格式",
                        "params": ["self", "date_str", "format_str='%Y-%m-%d'"],
                        "body": "try:\n    datetime.datetime.strptime(date_str, format_str)\n    return True\nexcept ValueError:\n    error = f\"Invalid date format: {date_str}, expected format: {format_str}\"\n    self.errors.append(error)\n    return False"
                    },
                    {
                        "name": "get_errors",
                        "description": "获取错误列表",
                        "params": ["self"],
                        "body": "return self.errors"
                    },
                    {
                        "name": "clear_errors",
                        "description": "清除错误列表",
                        "params": ["self"],
                        "body": "self.errors = []"
                    }
                ]
            }
        ],
        "functions": [
            {
                "name": "is_valid_phone",
                "description": "检查是否是有效的电话号码",
                "params": ["phone_number", "country_code='CN'"],
                "body": "if country_code == 'CN':\n    # 中国手机号格式\n    pattern = r'^1[3-9]\\d{9}$'\nelif country_code == 'US':\n    # 美国电话号码格式\n    pattern = r'^\\+?1?\\s*\\(?\\d{3}\\)?[-\\s]?\\d{3}[-\\s]?\\d{4}$'\nelse:\n    # 通用格式\n    pattern = r'^\\+?\\d{1,4}?[-\\s]?\\(?\\d{1,4}\\)?[-\\s]?\\d{1,9}$'\n\nreturn bool(re.match(pattern, phone_number))"
            },
            {
                "name": "is_valid_ip",
                "description": "检查是否是有效的IP地址",
                "params": ["ip_address", "version=4"],
                "body": "if version == 4:\n    # IPv4 格式\n    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'\nelif version == 6:\n    # IPv6 格式 (简化版)\n    pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'\nelse:\n    return False\n\nreturn bool(re.match(pattern, ip_address))"
            }
        ]
    }
    
    # 生成模块
    result = code_generator.generate_module(module_spec)
    
    # 打印生成结果
    if result.get('status') == 'success':
        print(f"模块 '{result.get('module_name')}' 生成成功")
        print("\n生成的代码:")
        print(result.get('code'))
        
        # 保存到文件
        output_file = "generated_validator.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.get('code'))
        print(f"\n代码已保存到文件: {output_file}")
    else:
        print(f"模块生成失败: {result.get('message')}")

if __name__ == "__main__":
    test_code_analysis()
    test_code_optimization()
    test_module_generation() 