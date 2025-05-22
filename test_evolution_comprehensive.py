# -*- coding: utf-8 -*-
"""
GHOST AGI 自主进化能力全面测试
测试系统自主理解、改进和优化代码的能力
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvolutionCompTest")

# 导入系统组件
from evolution.code_generator import CodeGenerator
from evolution.evolution_engine import EvolutionEngine

# 测试用例代码
TEST_CODE = """
# 这是一个包含多种代码问题的测试类
class badImplementation:
    def __init__(self, maxItems=100):
        self.items = []
        self.maxItems = maxItems
        self.counter = 0
    
    # 命名不规范，缺少文档
    def addItem(self, item):
        # 缺少错误处理
        self.items.append(item)
        self.counter = self.counter + 1
        # 性能问题 - 每次都完整遍历列表
        if len([x for x in self.items if x is not None]) > self.maxItems:
            # 内存问题 - 不必要的列表副本
            self.items = list(self.items)
            self.items.pop(0)
    
    # 重复代码片段示例
    def get_items_starting_with(self, prefix):
        result = []
        for item in self.items:
            if str(item).startswith(prefix):
                result.append(item)
        return result
    
    # 几乎相同的代码，应该合并
    def get_items_ending_with(self, suffix):
        result = []
        for item in self.items:
            if str(item).endswith(suffix):
                result.append(item)
        return result
        
    def PROCESS_DATA(self):
        # 大量嵌套的条件判断，复杂度高
        processed = []
        for i, item in enumerate(self.items):
            if item is not None:
                if isinstance(item, str):
                    if len(item) > 0:
                        if item[0].isalpha():
                            if item not in processed:
                                processed.append(item.upper())
                        else:
                            if len(item) > 1:
                                processed.append(item[1:])
                            else:
                                processed.append("")
                    else:
                        processed.append("")
                elif isinstance(item, int):
                    if item > 0:
                        if item % 2 == 0:
                            processed.append(str(item * 2))
                        else:
                            processed.append(str(item * 3))
                    else:
                        processed.append("0")
                else:
                    processed.append(str(item))
            else:
                processed.append("")
        return processed
"""

def test_comprehensive_code_analysis():
    """测试全面的代码分析功能"""
    logger.info("=== 开始测试全面代码分析功能 ===")
    
    code_generator = CodeGenerator()
    
    # 分析代码质量
    analysis_result = code_generator.analyze_code_quality(TEST_CODE)
    
    # 输出分析结果
    logger.info(f"代码质量得分: {analysis_result.get('quality_score', 0):.2f}/100")
    logger.info(f"代码复杂度: {analysis_result.get('complexity', 0)}")
    
    if 'issues' in analysis_result:
        logger.info(f"发现 {len(analysis_result['issues'])} 个问题:")
        for issue in analysis_result['issues'][:5]:  # 只显示前5个
            logger.info(f"- [{issue.get('severity', 'info')}] {issue.get('description', '')}")
    
    # 验证测试结果
    assert 'quality_score' in analysis_result, "分析结果应包含质量得分"
    assert 'issues' in analysis_result, "分析结果应包含发现的问题"
    assert analysis_result['quality_score'] < 70, "测试代码的质量得分应该较低"
    
    logger.info("代码分析功能测试通过")
    return analysis_result

def test_comprehensive_code_optimization():
    """测试代码优化功能"""
    logger.info("\n=== 开始测试代码优化功能 ===")
    
    code_generator = CodeGenerator()
    
    # 执行多种优化策略
    optimization_results = {}
    strategies = ["readability", "performance", "maintenance", "security"]
    
    for strategy in strategies:
        result = code_generator.auto_optimize_code(TEST_CODE, [strategy])
        optimization_results[strategy] = result
        
        if result['status'] == 'success':
            logger.info(f"{strategy.capitalize()} 优化策略:")
            logger.info(f"- 应用了 {len(result.get('applied_optimizations', []))} 个优化")
            for opt in result.get('applied_optimizations', [])[:2]:
                logger.info(f"  • {opt.get('type')}: {opt.get('description')}")
            
            # 检查优化指标
            improvements = result.get('improvements', {})
            for metric, value in improvements.items():
                logger.info(f"- {metric}: {value:.2f}%")
        else:
            logger.info(f"{strategy.capitalize()} 优化策略: {result.get('status')}")
    
    # 综合优化（所有策略）
    combined_result = code_generator.auto_optimize_code(TEST_CODE, strategies)
    
    if combined_result['status'] == 'success':
        logger.info("\n综合优化结果:")
        logger.info(f"- 应用了 {len(combined_result.get('applied_optimizations', []))} 个优化")
        logger.info(f"- 优化前代码行数: {combined_result['before_metrics']['line_count']}")
        logger.info(f"- 优化后代码行数: {combined_result['after_metrics']['line_count']}")
        
        # 保存优化后的代码
        with open("optimized_code_result.py", "w", encoding="utf-8") as f:
            f.write(combined_result['optimized_code'])
        logger.info("优化后的代码已保存到 optimized_code_result.py")
    
    # 验证测试结果
    assert 'status' in combined_result, "优化结果应包含状态信息"
    assert combined_result['status'] in ['success', 'no_change'], "优化应成功完成或无需更改"
    
    if combined_result['status'] == 'success':
        assert len(combined_result.get('applied_optimizations', [])) > 0, "应有至少一个优化被应用"
        assert 'before_metrics' in combined_result, "应包含优化前指标"
        assert 'after_metrics' in combined_result, "应包含优化后指标"
    
    logger.info("代码优化功能测试通过")
    return combined_result

def test_module_generation_comprehensive():
    """测试模块生成功能"""
    logger.info("\n=== 开始测试模块生成功能 ===")
    
    code_generator = CodeGenerator()
    
    # 定义一个复杂模块规范
    module_spec = {
        "name": "data_processor",
        "type": "util",
        "description": "数据处理工具模块，提供各种数据转换和验证功能",
        "author": "GHOST AGI 自主进化系统",
        "imports": [
            "json",
            "datetime",
            "re",
            {
                "module": "typing",
                "items": ["Dict", "List", "Any", "Optional", "Union", "Callable"]
            },
            {
                "module": "pandas",
                "items": ["DataFrame"]
            }
        ],
        "classes": [
            {
                "name": "DataProcessor",
                "description": "数据处理器，处理各种数据格式转换和验证",
                "bases": [],
                "methods": [
                    {
                        "name": "__init__",
                        "description": "初始化数据处理器",
                        "params": ["self", "config: Dict[str, Any] = None"],
                        "body": "self.config = config or {}\nself.logger = logging.getLogger(self.__class__.__name__)\nself.transformers = {}\nself.validators = {}\nself._register_default_transformers()\nself._register_default_validators()"
                    },
                    {
                        "name": "_register_default_transformers",
                        "description": "注册默认转换器",
                        "params": ["self"],
                        "body": "# 注册默认转换器\nself.transformers = {\n    'to_int': lambda x: int(x) if x else 0,\n    'to_float': lambda x: float(x) if x else 0.0,\n    'to_str': lambda x: str(x) if x is not None else '',\n    'to_bool': lambda x: bool(x) if x is not None else False,\n    'to_date': lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date() if x else None,\n}"
                    },
                    {
                        "name": "_register_default_validators",
                        "description": "注册默认验证器",
                        "params": ["self"],
                        "body": "# 注册默认验证器\nself.validators = {\n    'is_empty': lambda x: x is None or (isinstance(x, (str, list, dict)) and len(x) == 0),\n    'is_numeric': lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()),\n    'is_date': lambda x: bool(re.match(r'^\\d{4}-\\d{2}-\\d{2}$', str(x))) if x else False,\n}"
                    },
                    {
                        "name": "register_transformer",
                        "description": "注册数据转换器",
                        "params": ["self", "name: str", "transformer: Callable"],
                        "body": "self.transformers[name] = transformer\nself.logger.info(f\"已注册转换器: {name}\")"
                    },
                    {
                        "name": "transform",
                        "description": "转换数据",
                        "params": ["self", "data: Any", "transformer_name: str", "fallback: Any = None"],
                        "body": "try:\n    if transformer_name not in self.transformers:\n        self.logger.warning(f\"未知转换器: {transformer_name}\")\n        return fallback\n        \n    transformer = self.transformers[transformer_name]\n    return transformer(data)\nexcept Exception as e:\n    self.logger.error(f\"转换失败: {str(e)}\")\n    return fallback"
                    },
                    {
                        "name": "validate",
                        "description": "验证数据",
                        "params": ["self", "data: Any", "validator_name: str"],
                        "body": "try:\n    if validator_name not in self.validators:\n        self.logger.warning(f\"未知验证器: {validator_name}\")\n        return False\n        \n    validator = self.validators[validator_name]\n    return validator(data)\nexcept Exception as e:\n    self.logger.error(f\"验证失败: {str(e)}\")\n    return False"
                    },
                    {
                        "name": "process_dict",
                        "description": "处理字典数据",
                        "params": ["self", "data: Dict[str, Any]", "schema: Dict[str, Dict[str, Any]]"],
                        "body": "result = {}\ntry:\n    for field, field_schema in schema.items():\n        if field not in data and field_schema.get('required', False):\n            raise ValueError(f\"缺少必填字段: {field}\")\n            \n        value = data.get(field)\n        \n        # 应用验证\n        if 'validator' in field_schema:\n            is_valid = self.validate(value, field_schema['validator'])\n            if not is_valid and field_schema.get('required', False):\n                raise ValueError(f\"字段验证失败: {field}\")\n            \n        # 应用转换\n        if 'transformer' in field_schema:\n            value = self.transform(value, field_schema['transformer'], field_schema.get('default'))\n            \n        # 设置默认值\n        if value is None and 'default' in field_schema:\n            value = field_schema['default']\n            \n        result[field] = value\n        \n    return {'status': 'success', 'data': result}\nexcept Exception as e:\n    self.logger.error(f\"处理字典失败: {str(e)}\")\n    return {'status': 'error', 'message': str(e)}"
                    },
                    {
                        "name": "to_dataframe",
                        "description": "将数据转换为DataFrame",
                        "params": ["self", "data: List[Dict[str, Any]]"],
                        "body": "try:\n    import pandas as pd\n    return pd.DataFrame(data)\nexcept ImportError:\n    self.logger.error(\"pandas未安装，无法转换为DataFrame\")\n    return None\nexcept Exception as e:\n    self.logger.error(f\"转换为DataFrame失败: {str(e)}\")\n    return None"
                    }
                ]
            }
        ],
        "functions": [
            {
                "name": "sanitize_input",
                "description": "清理和消毒输入数据，移除潜在危险字符",
                "params": ["input_data: str"],
                "body": "# 移除可能的脚本注入字符\nsanitized = re.sub(r'[<>\"\'&;]', '', input_data)\n# 移除控制字符\nsanitized = re.sub(r'[\\x00-\\x1F\\x7F]', '', sanitized)\nreturn sanitized"
            },
            {
                "name": "parse_json_safe",
                "description": "安全解析JSON，处理可能的错误",
                "params": ["json_str: str", "default_value: Any = None"],
                "body": "try:\n    if not json_str or not isinstance(json_str, str):\n        return default_value\n    return json.loads(json_str)\nexcept json.JSONDecodeError:\n    logging.warning(f\"JSON解析失败: {json_str[:100]}...\")\n    return default_value"
            }
        ],
        "add_main": True
    }
    
    # 生成模块
    result = code_generator.generate_module(module_spec)
    
    if result['status'] == 'success':
        logger.info(f"模块 '{result['module_name']}' 生成成功!")
        
        # 保存生成的模块
        with open(f"{result['module_name']}.py", "w", encoding="utf-8") as f:
            f.write(result['code'])
        logger.info(f"模块代码已保存到 {result['module_name']}.py")
    else:
        logger.error(f"模块生成失败: {result.get('message')}")
    
    # 验证测试结果
    assert result['status'] == 'success', "模块生成应该成功"
    assert 'code' in result, "结果应该包含生成的代码"
    assert len(result['code']) > 0, "生成的代码不应为空"
    
    # 验证生成的代码包含所有指定的类和函数
    assert "class DataProcessor" in result['code'], "生成的代码应包含DataProcessor类"
    assert "def sanitize_input" in result['code'], "生成的代码应包含sanitize_input函数"
    assert "def parse_json_safe" in result['code'], "生成的代码应包含parse_json_safe函数"
    
    logger.info("模块生成功能测试通过")
    return result

def test_evolution_engine():
    """测试进化引擎"""
    logger.info("\n=== 开始测试进化引擎 ===")
    
    evolution_engine = EvolutionEngine()
    
    # 1. 测试系统结构分析
    logger.info("测试系统结构分析...")
    analysis_result = evolution_engine.analyze_system_structure()
    
    if isinstance(analysis_result, dict) and "status" not in analysis_result:
        logger.info(f"分析了 {analysis_result['stats']['total_modules']} 个模块")
        logger.info(f"总代码大小: {analysis_result['stats']['total_code_size']} 字节")
        
        if analysis_result['stats']['complexity_hotspots']:
            logger.info("复杂度热点模块:")
            for hotspot in analysis_result['stats']['complexity_hotspots'][:3]:
                logger.info(f"- {hotspot['module']}: 复杂度 {hotspot['complexity']}")
    else:
        logger.warning(f"系统结构分析失败: {analysis_result.get('message', '')}")
    
    # 2. 测试优化建议生成
    logger.info("\n测试优化建议生成...")
    suggestions = evolution_engine.generate_optimization_suggestions()
    
    if suggestions:
        logger.info(f"生成了 {len(suggestions)} 个优化建议")
        for suggestion in suggestions[:3]:
            logger.info(f"- [{suggestion.get('type', '')}] {suggestion.get('suggestion', '')}")
    else:
        logger.info("没有生成优化建议")
    
    # 3. 测试进化计划生成
    logger.info("\n测试进化计划生成...")
    plan_result = evolution_engine.generate_evolution_plan()
    
    if plan_result.get('status') == 'success':
        plan = plan_result['plan']
        logger.info(f"进化计划生成成功，计划ID: {plan['plan_id']}")
        logger.info(f"优化项数量: {plan['total_improvements']}")
        
        # 查看优化项类型分布
        logger.info("优化类型分布:")
        for type_name, count in plan['type_distribution'].items():
            logger.info(f"- {type_name}: {count}项")
    else:
        logger.warning(f"进化计划生成失败: {plan_result.get('message', '')}")
    
    logger.info("进化引擎测试完成")
    
def run_all_tests():
    """运行所有测试"""
    start_time = time.time()
    logger.info("开始执行GHOST AGI自主进化能力全面测试")
    
    try:
        analysis_result = test_comprehensive_code_analysis()
        optimization_result = test_comprehensive_code_optimization()
        module_result = test_module_generation_comprehensive()
        test_evolution_engine()
        
        logger.info("\n=== 所有测试已成功完成 ===")
        logger.info(f"总耗时: {time.time() - start_time:.2f}秒")
        
        # 创建测试报告
        report = {
            "timestamp": time.time(),
            "tests_passed": 4,
            "code_analysis": {
                "quality_score": analysis_result.get('quality_score', 0),
                "issues_count": len(analysis_result.get('issues', []))
            },
            "code_optimization": {
                "strategies_tested": 4,
                "improvements": optimization_result.get('improvements', {})
            },
            "module_generation": {
                "success": module_result.get('status') == 'success',
                "code_size": len(module_result.get('code', ''))
            }
        }
        
        # 保存测试报告
        with open("evolution_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("测试报告已保存到 evolution_test_report.json")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests() 