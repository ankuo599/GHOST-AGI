# -*- coding: utf-8 -*-
"""
GHOST AGI 自主进化能力测试脚本
测试系统的自主进化和学习能力
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
logger = logging.getLogger("EvolutionTest")

# 导入系统
from main import GhostAGI

def test_evolution_capabilities():
    """测试系统的自主进化能力"""
    logger.info("开始测试自主进化能力...")
    
    # 创建AGI系统实例（禁用安全限制）
    agi = GhostAGI(config={
        "sandbox_enabled": False,
        "safety_checks": False
    })
    
    # 启动系统
    agi.start()
    logger.info("系统已启动，安全限制已禁用")
    
    # 测试进化引擎
    test_evolution_engine(agi)
    
    # 测试代码生成器
    test_code_generator(agi)
    
    # 测试知识迁移系统
    test_knowledge_transfer(agi)
    
    # 测试自主学习系统
    test_autonomous_learning(agi)
    
    # 停止系统
    agi.stop()
    logger.info("测试完成，系统已停止")
    
def test_evolution_engine(agi):
    """测试进化引擎"""
    logger.info("测试进化引擎...")
    
    evolution_engine = agi.evolution_engine
    
    # 分析自身代码
    analysis_result = evolution_engine.analyze_code("main")
    logger.info(f"代码分析结果: {json.dumps(analysis_result, indent=2)}")
    
    # 生成优化建议
    suggestions = evolution_engine.generate_optimization_suggestions()
    logger.info(f"发现 {len(suggestions)} 个优化建议")
    for suggestion in suggestions[:5]:  # 只显示前5个
        logger.info(f"优化建议: {suggestion['suggestion']}")
        
    # 运行性能基准测试
    benchmark_result = evolution_engine.run_performance_benchmark()
    logger.info(f"性能基准测试结果: {json.dumps(benchmark_result, indent=2)}")
    
    # 自动优化
    optimization_result = evolution_engine.auto_optimize()
    logger.info(f"自动优化结果: {json.dumps(optimization_result, indent=2)}")
    
def test_code_generator(agi):
    """测试代码生成器"""
    logger.info("测试代码生成器...")
    
    code_generator = agi.code_generator
    
    # 生成测试模块
    module_spec = {
        "name": "test_module",
        "type": "module",
        "description": "一个测试模块",
        "author": "GHOST AGI",
        "imports": [
            {"module": "time"},
            {"module": "logging"},
            {"module": "json"}
        ],
        "classes": [
            {
                "name": "TestClass",
                "description": "测试类",
                "bases": [],
                "methods": [
                    {
                        "name": "__init__",
                        "description": "初始化方法",
                        "params": ["self"],
                        "body": "self.name = 'Test'\nself.created_at = time.time()"
                    },
                    {
                        "name": "test_method",
                        "description": "测试方法",
                        "params": ["self", "param"],
                        "body": "return f'测试结果: {param}'"
                    }
                ],
                "attributes": [
                    {"name": "VERSION", "value": "'1.0.0'", "comment": "版本号"}
                ]
            }
        ],
        "functions": [
            {
                "name": "helper_function",
                "description": "辅助函数",
                "params": ["x", "y"],
                "body": "return x + y"
            }
        ]
    }
    
    generation_result = code_generator.generate_module(module_spec)
    
    if generation_result["status"] == "success":
        logger.info("模块生成成功!")
        logger.info(f"生成的代码大小: {generation_result['code_size'] if 'code_size' in generation_result else len(generation_result['code'])} 字节")
        
        # 保存生成的代码到文件
        with open("generated_test_module.py", "w", encoding="utf-8") as f:
            f.write(generation_result["code"])
            
        logger.info("代码已保存到 generated_test_module.py")
    else:
        logger.error(f"模块生成失败: {generation_result['message']}")
        
    # 测试代码优化
    test_code = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
        
def calculate_many_fibs():
    results = []
    for i in range(20):
        results.append(fibonacci(i))
    return results
    
# 计算斐波那契数列
results = calculate_many_fibs()
print(results)
"""
    
    optimization_result = code_generator.optimize_code(test_code, "performance")
    
    if optimization_result["status"] == "success":
        logger.info("代码优化成功!")
        logger.info(f"优化建议数量: {len(optimization_result['optimizations'])}")
        
        # 保存优化后的代码
        with open("optimized_code.py", "w", encoding="utf-8") as f:
            f.write(optimization_result["code"])
            
        logger.info("优化后的代码已保存到 optimized_code.py")
    else:
        logger.info(f"代码优化结果: {optimization_result['status']}")
        
def test_knowledge_transfer(agi):
    """测试知识迁移系统"""
    logger.info("测试知识迁移系统...")
    
    kt = agi.knowledge_transfer
    
    # 注册两个测试领域
    math_domain = {
        "name": "mathematics",
        "concepts": {
            "number": {"description": "数学中的数字概念"},
            "addition": {"description": "加法运算"},
            "subtraction": {"description": "减法运算"},
            "multiplication": {"description": "乘法运算"},
            "division": {"description": "除法运算"}
        },
        "relations": {
            "operation": {"description": "数学运算关系"}
        }
    }
    
    programming_domain = {
        "name": "programming",
        "concepts": {
            "variable": {"description": "编程中的变量概念"},
            "operator": {"description": "编程中的操作符"},
            "function": {"description": "编程中的函数"},
            "loop": {"description": "循环结构"},
            "condition": {"description": "条件结构"}
        },
        "relations": {
            "uses": {"description": "使用关系"},
            "contains": {"description": "包含关系"}
        }
    }
    
    # 注册领域
    kt.register_domain("mathematics", math_domain)
    kt.register_domain("programming", programming_domain)
    logger.info("已注册数学和编程两个领域")
    
    # 创建领域映射
    mapping_result = kt.create_domain_mapping(
        "mathematics", 
        "programming",
        {
            "number": "variable",
            "addition": "operator",
            "subtraction": "operator",
            "multiplication": "operator",
            "division": "operator"
        }
    )
    
    if mapping_result["status"] == "success":
        logger.info(f"领域映射创建成功，映射ID: {mapping_result['mapping_id']}")
        
        # 创建一些数学领域知识
        math_knowledge = {
            "concepts": {
                "number": {"example": "1, 2, 3, 4, 5"},
                "addition": {"symbol": "+", "commutative": True},
                "subtraction": {"symbol": "-", "commutative": False},
                "multiplication": {"symbol": "*", "commutative": True},
                "division": {"symbol": "/", "commutative": False}
            },
            "relations": [
                {"type": "operation", "source": "addition", "target": "number"},
                {"type": "operation", "source": "subtraction", "target": "number"},
                {"type": "operation", "source": "multiplication", "target": "number"},
                {"type": "operation", "source": "division", "target": "number"}
            ]
        }
        
        # 迁移知识
        transfer_result = kt.transfer_knowledge(
            "mathematics",
            "programming",
            math_knowledge
        )
        
        if transfer_result["status"] == "success":
            logger.info("知识迁移成功!")
            logger.info(f"迁移后的知识: {json.dumps(transfer_result['transformed_knowledge'], indent=2)}")
        else:
            logger.error(f"知识迁移失败: {transfer_result['message']}")
    else:
        logger.error(f"领域映射创建失败: {mapping_result['message']}")
        
def test_autonomous_learning(agi):
    """测试自主学习系统"""
    logger.info("测试自主学习系统...")
    
    al = agi.autonomous_learning
    
    # 设置学习目标
    learning_objectives = [
        {
            "domain": "programming",
            "priority": 1.0,
            "description": "学习编程基本概念"
        },
        {
            "domain": "mathematics",
            "priority": 0.8,
            "description": "学习数学基础知识"
        }
    ]
    
    al.set_learning_objectives(learning_objectives)
    logger.info("已设置学习目标")
    
    # 调整学习参数
    al.adjust_learning_parameters(exploration_rate=0.7, learning_rate=0.1)
    logger.info("已调整学习参数")
    
    # 评估知识水平
    for domain in ["programming", "mathematics"]:
        assessment = al.evaluate_knowledge(domain)
        if assessment["status"] == "success":
            logger.info(f"{domain} 领域知识评估: 分数 {assessment['assessment']['score']:.2f}, "
                       f"覆盖率 {assessment['assessment']['coverage']:.2f}")
        else:
            logger.error(f"知识评估失败: {assessment['message']}")
            
    # 启动学习任务
    learning_strategies = ["exploration", "exploitation", "curriculum", "active", "transfer"]
    
    for domain in ["programming", "mathematics"]:
        for strategy in learning_strategies:
            logger.info(f"使用 {strategy} 策略学习 {domain} 领域...")
            
            # 启动学习任务
            task_result = al.start_learning_task(domain, strategy)
            
            if task_result["status"] == "success":
                task = task_result["task"]
                if task["status"] == "completed":
                    logger.info(f"学习任务完成: {task['results']['status']}")
                    
                    # 显示学习结果摘要
                    if "concepts_discovered" in task["results"]:
                        logger.info(f"发现了 {task['results']['concepts_discovered']} 个概念")
                    if "relations_discovered" in task["results"]:
                        logger.info(f"发现了 {task['results']['relations_discovered']} 个关系")
                    if "inferences_made" in task["results"]:
                        logger.info(f"进行了 {task['results']['inferences_made']} 个推断")
                else:
                    logger.info(f"学习任务状态: {task['status']}")
            else:
                logger.error(f"启动学习任务失败: {task_result['message']}")
                
    # 获取学习状态
    learning_status = al.get_learning_status()
    logger.info(f"学习系统状态: {json.dumps(learning_status, indent=2)}")
    
def test_code_optimization():
    """测试代码自动优化功能"""
    from evolution.code_generator import CodeGenerator
    
    # 初始化代码生成器
    code_generator = CodeGenerator()
    
    # 示例代码，故意写得不好，用于测试优化
    poor_code = """
# 示例代码，有多个需要优化的地方
def doSomething(inputData):
    # 命名不规范，异常处理缺失，代码重复等问题
    Result = []
    for x in inputData:
        if x > 5:
            Result.append(x * 2)
    
    file_data = []
    f = open("test.txt")
    content = f.read()
    f.close()
    
    # 字符串拼接效率问题
    message = "Hello " + "world " + "this " + "is " + "inefficient"
    
    # 重复代码示例
    temp1 = []
    for item in inputData:
        if item % 2 == 0:
            temp1.append(item)
    
    # 几乎完全相同的代码
    temp2 = []
    for item in inputData:
        if item % 3 == 0:
            temp2.append(item)
    
    return Result
"""
    
    # 分析代码质量
    analysis = code_generator.analyze_code_quality(poor_code)
    print("代码质量分析结果:")
    print(f"质量得分: {analysis.get('quality_score', 0)}")
    print(f"复杂度: {analysis.get('complexity', 0)}")
    print(f"发现问题: {len(analysis.get('issues', []))}")
    
    # 自动优化代码
    optimization = code_generator.auto_optimize_code(
        poor_code, 
        optimization_goals=["readability", "performance", "maintenance", "security"]
    )
    
    print("\n代码优化结果:")
    print(f"状态: {optimization.get('status')}")
    print(f"应用的优化: {len(optimization.get('applied_optimizations', []))}")
    
    # 打印优化前后对比
    if optimization.get("status") == "success":
        print("\n改进百分比:")
        for metric, value in optimization.get("improvements", {}).items():
            print(f"{metric}: {value:.2f}%")
        
        # 打印优化后的代码
        print("\n优化后的代码:")
        print(optimization.get("optimized_code"))

def test_module_generation():
    """测试模块自动生成功能"""
    from evolution.code_generator import CodeGenerator
    
    # 初始化代码生成器
    code_generator = CodeGenerator()
    
    # 创建模块需求规范
    module_requirements = {
        "module_name": "auto_optimizer",
        "description": "自动优化器模块，用于优化系统性能",
        "features": [
            {
                "name": "optimize_memory_usage",
                "type": "function",
                "description": "优化内存使用",
                "params": ["target_obj", "max_memory"],
                "implementation": """
    # 实现内存优化
    if not hasattr(target_obj, '__size__'):
        return False
    
    current_size = target_obj.__size__()
    if current_size <= max_memory:
        return True
        
    # 执行优化
    success = target_obj.compress()
    
    return success
"""
            },
            {
                "name": "optimize_runtime",
                "type": "function",
                "description": "优化运行时间",
                "params": ["function_obj", "timeout"],
                "implementation": """
    import time
    
    start_time = time.time()
    result = function_obj()
    end_time = time.time()
    
    if end_time - start_time <= timeout:
        return result, True
    
    # 超时，尝试使用缓存优化
    cached_result = get_cached_result(function_obj)
    if cached_result is not None:
        return cached_result, True
        
    return result, False
"""
            }
        ],
        "interfaces": [
            {
                "type": "class",
                "name": "PerformanceOptimizer",
                "description": "性能优化器",
                "methods": [
                    {
                        "name": "__init__",
                        "params": ["target_system"],
                        "body": """
        self.target_system = target_system
        self.optimizations_applied = []
        self.last_optimization_time = None
"""
                    },
                    {
                        "name": "optimize",
                        "params": ["optimization_type='auto'"],
                        "body": """
        import time
        
        self.last_optimization_time = time.time()
        
        if optimization_type == 'memory':
            success = self._optimize_memory()
        elif optimization_type == 'speed':
            success = self._optimize_speed()
        elif optimization_type == 'auto':
            # 自动选择优化策略
            system_stats = self.target_system.get_stats()
            
            if system_stats.get('memory_usage', 0) > 80:
                success = self._optimize_memory()
            elif system_stats.get('response_time', 0) > 2:
                success = self._optimize_speed()
            else:
                success = True  # 不需要优化
                
        self.optimizations_applied.append({
            'type': optimization_type,
            'time': self.last_optimization_time,
            'success': success
        })
        
        return success
"""
                    }
                ]
            }
        ],
        "dependencies": [
            "time",
            "logging",
            {
                "module": "utils.cache_manager",
                "items": ["get_cached_result"]
            }
        ],
        "examples": [
            {
                "description": "优化内存使用示例",
                "code": """
obj = SampleObject(size=1024)
success = optimize_memory_usage(obj, 512)
assert success == True
assert obj.__size__() <= 512
"""
            },
            {
                "description": "优化运行时间示例",
                "code": """
def slow_function():
    time.sleep(1)
    return 42

result, success = optimize_runtime(slow_function, 0.5)
assert result == 42  # 应该返回结果
# 可能成功也可能失败，取决于是否有缓存
"""
            }
        ]
    }
    
    # 生成模块
    result = code_generator.generate_module_from_requirements(module_requirements)
    
    print("\n模块生成结果:")
    print(f"状态: {result.get('status')}")
    
    if result.get("status") == "success":
        # 打印生成的代码
        print("\n生成的模块代码:")
        print(result.get("code"))
        
        # 如果有测试模块，也打印测试代码
        if "test_module" in result:
            print("\n生成的测试代码:")
            print(result["test_module"]["code"])
        
        # 保存到文件
        try:
            with open("generated_module.py", "w", encoding="utf-8") as f:
                f.write(result.get("code", ""))
            print("\n成功将生成的模块保存到 generated_module.py")
            
            if "test_module" in result:
                test_file_name = f"test_{module_requirements['module_name']}.py"
                with open(test_file_name, "w", encoding="utf-8") as f:
                    f.write(result["test_module"]["code"])
                print(f"成功将生成的测试模块保存到 {test_file_name}")
        except Exception as e:
            print(f"保存文件失败: {str(e)}")

if __name__ == "__main__":
    test_evolution_capabilities()
    
    print("\n\n===== 测试代码自动优化功能 =====")
    test_code_optimization()
    
    print("\n\n===== 测试模块自动生成功能 =====")
    test_module_generation() 