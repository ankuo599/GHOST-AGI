# -*- coding: utf-8 -*-
"""
代码生成器 (Code Generator)

负责自动生成和优化代码，实现系统的自我进化能力
"""

import os
import time
import logging
import ast
import inspect
import tempfile
import importlib
import subprocess
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple

class CodeGenerator:
    """代码生成器，负责自动生成和优化代码"""
    
    def __init__(self, event_system=None):
        """
        初始化代码生成器
        
        Args:
            event_system: 事件系统，用于发布代码生成相关事件
        """
        self.event_system = event_system
        self.logger = logging.getLogger("CodeGenerator")
        
        # 代码生成历史
        self.generation_history = []
        
        # 代码模板库
        self.code_templates = {
            "class": self._template_class,
            "function": self._template_function,
            "module": self._template_module,
            "test": self._template_test,
            "plugin": self._template_plugin
        }
        
        # 代码优化历史
        self.optimization_history = []
        
        # 自动代码分析结果缓存
        self.code_analysis_cache = {}
        
        self.logger.info("代码生成器初始化完成")
        
    def generate_module(self, module_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据规范生成新模块
        
        Args:
            module_spec: 模块规范，包含模块名称、类型、功能说明等
            
        Returns:
            Dict: 生成结果
        """
        try:
            # 获取模块规范
            module_name = module_spec.get("name", "")
            module_type = module_spec.get("type", "module")
            description = module_spec.get("description", "")
            author = module_spec.get("author", "GHOST AGI")
            classes = module_spec.get("classes", [])
            functions = module_spec.get("functions", [])
            imports = module_spec.get("imports", [])
            
            if not module_name:
                return {
                    "status": "error",
                    "message": "模块名称不能为空"
                }
                
            # 构建导入语句
            import_code = ""
            for imp in imports:
                if isinstance(imp, str):
                    import_code += f"import {imp}\n"
                elif isinstance(imp, dict):
                    module = imp.get("module", "")
                    items = imp.get("items", [])
                    if items:
                        item_str = ", ".join(items)
                        import_code += f"from {module} import {item_str}\n"
                    else:
                        import_code += f"import {module}\n"
                        
            if imports:
                import_code += "\n"
                
            # 构建类代码
            classes_code = ""
            for cls in classes:
                cls_name = cls.get("name", "")
                cls_description = cls.get("description", "")
                base_classes = cls.get("bases", [])
                methods = cls.get("methods", [])
                attributes = cls.get("attributes", [])
                
                if cls_name:
                    class_code = self._generate_class(
                        name=cls_name,
                        description=cls_description,
                        bases=base_classes,
                        methods=methods,
                        attributes=attributes
                    )
                    classes_code += class_code + "\n\n"
                    
            # 构建函数代码
            functions_code = ""
            for func in functions:
                func_name = func.get("name", "")
                func_description = func.get("description", "")
                params = func.get("params", [])
                body = func.get("body", "pass")
                decorators = func.get("decorators", [])
                
                if func_name:
                    function_code = self._generate_function(
                        name=func_name,
                        description=func_description,
                        params=params,
                        body=body,
                        decorators=decorators
                    )
                    functions_code += function_code + "\n\n"
                    
            # 生成模块代码
            module_code = self._template_module(
                name=module_name,
                description=description,
                author=author,
                imports=import_code,
                classes=classes_code,
                functions=functions_code
            )
            
            # 记录生成历史
            generation_entry = {
                "type": "module_generation",
                "module_name": module_name,
                "timestamp": time.time(),
                "spec": module_spec
            }
            self.generation_history.append(generation_entry)
            
            # 发布事件
            if self.event_system:
                self.event_system.publish("code_generator.module_generated", {
                    "module_name": module_name,
                    "module_type": module_type,
                    "code_size": len(module_code)
                })
                
            return {
                "status": "success",
                "module_name": module_name,
                "code": module_code,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"生成模块失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def optimize_code(self, code_content: str, optimization_goal: str = "performance") -> Dict[str, Any]:
        """
        优化给定的代码
        
        Args:
            code_content: 需要优化的代码内容
            optimization_goal: 优化目标，可选值: "performance", "readability", "memory"
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 记录原始代码指标
            original_stats = self._analyze_code_stats(tree)
            
            # 根据不同优化目标选择不同策略
            if optimization_goal == "performance":
                optimized_code = self._optimize_for_performance(code_content, tree)
            elif optimization_goal == "readability":
                optimized_code = self._optimize_for_readability(code_content, tree)
            elif optimization_goal == "memory":
                optimized_code = self._optimize_for_memory(code_content, tree)
            else:
                # 默认性能优化
                optimized_code = self._optimize_for_performance(code_content, tree)
            
            # 分析优化后的代码
            optimized_tree = ast.parse(optimized_code)
            optimized_stats = self._analyze_code_stats(optimized_tree)
            
            # 计算改进指标
            improvements = {
                "lines": original_stats["lines"] - optimized_stats["lines"],
                "complexity": original_stats["complexity"] - optimized_stats["complexity"],
                "nesting": original_stats["max_nesting"] - optimized_stats["max_nesting"]
            }
            
            # 添加说明注释
            final_code = f'"""\n自动优化代码 - 优化目标: {optimization_goal}\n\n改进指标:\n'
            final_code += f'- 行数: {"减少" if improvements["lines"] > 0 else "增加"} {abs(improvements["lines"])} 行\n'
            final_code += f'- 复杂度: {"降低" if improvements["complexity"] > 0 else "提高"} {abs(improvements["complexity"])}\n'
            final_code += f'- 嵌套深度: {"降低" if improvements["nesting"] > 0 else "提高"} {abs(improvements["nesting"])}\n'
            final_code += '"""\n\n'
            final_code += optimized_code
            
            return {
                "status": "success",
                "original_code": code_content,
                "optimized_code": final_code,
                "improvements": improvements,
                "strategy": optimization_goal
            }
            
        except Exception as e:
            self.logger.error(f"代码优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def generate_test(self, module_path: str) -> Dict[str, Any]:
        """
        为模块生成测试代码
        
        Args:
            module_path: 模块路径
            
        Returns:
            Dict: 生成结果
        """
        try:
            # 加载模块
            module = importlib.import_module(module_path)
            
            # 获取模块名称
            module_name = module_path.split(".")[-1]
            
            # 分析模块
            classes = []
            functions = []
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    # 分析类
                    methods = []
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith("_"):
                            methods.append(method_name)
                            
                    classes.append({
                        "name": name,
                        "methods": methods
                    })
                elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    # 分析函数
                    if not name.startswith("_"):
                        functions.append(name)
                        
            # 生成导入语句
            imports = [
                "import unittest",
                "import sys",
                "import os",
                f"import {module_path}"
            ]
            
            # 生成测试类
            test_classes = []
            
            # 为每个类生成测试类
            for cls in classes:
                cls_name = cls["name"]
                methods = cls["methods"]
                
                test_methods = []
                for method in methods:
                    test_methods.append({
                        "name": f"test_{method}",
                        "description": f"测试 {cls_name}.{method} 方法",
                        "params": [],
                        "body": f"# TODO: 实现 {cls_name}.{method} 的测试\npass",
                        "decorators": []
                    })
                    
                test_classes.append({
                    "name": f"Test{cls_name}",
                    "description": f"{cls_name} 类的测试用例",
                    "bases": ["unittest.TestCase"],
                    "methods": [
                        {
                            "name": "setUp",
                            "description": "测试前的设置",
                            "params": ["self"],
                            "body": f"self.instance = {module_path}.{cls_name}()",
                            "decorators": []
                        },
                        {
                            "name": "tearDown",
                            "description": "测试后的清理",
                            "params": ["self"],
                            "body": "pass",
                            "decorators": []
                        },
                        *test_methods
                    ],
                    "attributes": []
                })
                
            # 为独立函数生成测试类
            if functions:
                test_methods = []
                for func in functions:
                    test_methods.append({
                        "name": f"test_{func}",
                        "description": f"测试 {func} 函数",
                        "params": ["self"],
                        "body": f"# TODO: 实现 {func} 的测试\npass",
                        "decorators": []
                    })
                    
                test_classes.append({
                    "name": f"Test{module_name.capitalize()}Functions",
                    "description": f"{module_name} 模块函数的测试用例",
                    "bases": ["unittest.TestCase"],
                    "methods": test_methods,
                    "attributes": []
                })
                
            # 生成测试模块规范
            test_module_spec = {
                "name": f"test_{module_name}",
                "type": "test",
                "description": f"{module_path} 模块的测试用例",
                "author": "GHOST AGI",
                "imports": [{"module": imp} for imp in imports],
                "classes": test_classes,
                "functions": [
                    {
                        "name": "main",
                        "description": "测试入口",
                        "params": [],
                        "body": "unittest.main()",
                        "decorators": []
                    }
                ]
            }
            
            # 生成测试模块
            result = self.generate_module(test_module_spec)
            
            # 添加主函数调用
            if result["status"] == "success":
                result["code"] += "\n\nif __name__ == '__main__':\n    main()\n"
                
            return result
        except Exception as e:
            self.logger.error(f"生成测试失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _analyze_code_stats(self, tree):
        """分析代码统计信息"""
        lines = len(ast.unparse(tree).splitlines())
        
        # 计算复杂度和嵌套深度
        complexity = 0
        max_nesting = 0
        current_nesting = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.max_nesting = 0
                self.current_nesting = 0
                
            def visit_If(self, node):
                self.complexity += 1
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1
                
            def visit_For(self, node):
                self.complexity += 1
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1
                
            def visit_While(self, node):
                self.complexity += 1
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return {
            "lines": lines,
            "complexity": visitor.complexity,
            "max_nesting": visitor.max_nesting
        }
        
    def _optimize_for_performance(self, code_content, tree):
        """优化代码性能"""
        # 1. 合并重复计算
        # 2. 减少循环中的不必要操作
        # 3. 优化数据结构
        
        # 这是一个简化版实现
        optimized_code = code_content
        
        # 检测并优化常见的性能问题模式
        patterns = [
            # 将循环内不变量移出循环
            (r'(for .+?:.*?)([\s\n]+)([^\n]+?)(\s+)([\w\d_]+)(\s*=\s*)((?:[^{\[\]}\n]+?)+?)(\s*[\n])',
             r'\3\4\5\6\7\8\1\2'),
            # 使用列表推导式代替循环
            (r'result\s*=\s*\[\]\s*for\s+(\w+)\s+in\s+(.+?):\s*result\.append\((.+?)\)',
             r'result = [\3 for \1 in \2]')
        ]
        
        for pattern, replacement in patterns:
            optimized_code = re.sub(pattern, replacement, optimized_code)
        
        return optimized_code
        
    def _optimize_for_readability(self, code_content, tree):
        """优化代码可读性"""
        # 1. 添加有意义的变量名
        # 2. 分解复杂表达式
        # 3. 添加合适的注释
        
        # 这是一个简化版实现
        optimized_code = code_content
        
        # 格式化代码
        try:
            import autopep8
            optimized_code = autopep8.fix_code(optimized_code, options={'aggressive': 1})
        except ImportError:
            pass
        
        return optimized_code
        
    def _optimize_for_memory(self, code_content, tree):
        """优化内存使用"""
        # 1. 减少不必要的数据复制
        # 2. 使用迭代器代替列表
        # 3. 及时释放不需要的引用
        
        # 这是一个简化版实现
        optimized_code = code_content
        
        # 替换常见的内存低效模式
        patterns = [
            # 使用迭代器代替列表
            (r'(for\s+\w+\s+in\s+)(list\()(.+?)(\))',
             r'\1\3'),
            # 在大列表上使用迭代器
            (r'(\w+)\s*=\s*\[(.+?) for (.+?) in (.+?)\]',
             r'\1 = (\2 for \3 in \4)')
        ]
        
        for pattern, replacement in patterns:
            optimized_code = re.sub(pattern, replacement, optimized_code)
        
        return optimized_code
        
    def _generate_class(self, name: str, description: str = "", bases: List[str] = None,
                      methods: List[Dict[str, Any]] = None, attributes: List[Dict[str, Any]] = None) -> str:
        """
        生成类代码
        
        Args:
            name: 类名
            description: 类说明
            bases: 基类列表
            methods: 方法列表
            attributes: 属性列表
            
        Returns:
            str: 生成的类代码
        """
        bases = bases or []
        methods = methods or []
        attributes = attributes or []
        
        # 基类
        base_str = "(" + ", ".join(bases) + ")" if bases else ""
        
        # 类注释
        doc_string = f'    """{description}"""' if description else ""
        
        # 属性
        attributes_code = ""
        for attr in attributes:
            attr_name = attr.get("name", "")
            attr_value = attr.get("value", "None")
            attr_comment = attr.get("comment", "")
            
            if attr_name:
                if attr_comment:
                    attributes_code += f"    {attr_name} = {attr_value}  # {attr_comment}\n"
                else:
                    attributes_code += f"    {attr_name} = {attr_value}\n"
                    
        if attributes_code:
            attributes_code += "\n"
            
        # 方法
        methods_code = ""
        for method in methods:
            method_name = method.get("name", "")
            method_description = method.get("description", "")
            params = method.get("params", [])
            body = method.get("body", "pass")
            decorators = method.get("decorators", [])
            
            if method_name:
                # 添加self参数
                if params and params[0] != "self":
                    params.insert(0, "self")
                elif not params:
                    params = ["self"]
                    
                method_code = self._generate_function(
                    name=method_name,
                    description=method_description,
                    params=params,
                    body=body,
                    decorators=decorators,
                    indentation=4
                )
                
                methods_code += method_code + "\n\n"
                
        # 生成类代码
        class_code = f"class {name}{base_str}:\n"
        
        if doc_string:
            class_code += doc_string + "\n\n"
            
        if attributes_code:
            class_code += attributes_code
            
        if methods_code:
            class_code += methods_code
        else:
            class_code += "    pass\n"
            
        return class_code
        
    def _generate_function(self, name: str, description: str = "", params: List[str] = None,
                         body: str = "pass", decorators: List[str] = None, indentation: int = 0) -> str:
        """
        生成函数代码
        
        Args:
            name: 函数名
            description: 函数说明
            params: 参数列表
            body: 函数体
            decorators: 装饰器列表
            indentation: 缩进层级
            
        Returns:
            str: 生成的函数代码
        """
        params = params or []
        decorators = decorators or []
        
        # 缩进
        indent = " " * indentation
        
        # 装饰器
        decorators_code = ""
        for decorator in decorators:
            decorators_code += f"{indent}@{decorator}\n"
            
        # 参数
        params_str = ", ".join(params)
        
        # 函数注释
        doc_string = f'{indent}    """{description}"""' if description else ""
        
        # 函数体
        body_lines = body.split("\n")
        body_code = "\n".join([f"{indent}    {line}" for line in body_lines])
        
        # 生成函数代码
        function_code = ""
        
        if decorators_code:
            function_code += decorators_code
            
        function_code += f"{indent}def {name}({params_str}):\n"
        
        if doc_string:
            function_code += doc_string + "\n\n"
            
        function_code += body_code
        
        return function_code
        
    def _template_module(self, name: str, description: str, author: str, 
                       imports: str, classes: str, functions: str) -> str:
        """
        模块代码模板
        
        Args:
            name: 模块名
            description: 模块说明
            author: 作者
            imports: 导入语句
            classes: 类代码
            functions: 函数代码
            
        Returns:
            str: 模块代码
        """
        header = f'# -*- coding: utf-8 -*-\n"""\n{name}\n\n{description}\n"""\n\n'
        
        if author:
            header += f"# Author: {author}\n# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
        module_code = header + imports
        
        if functions:
            module_code += functions
            
        if classes:
            module_code += classes
            
        return module_code
        
    def _template_class(self, name: str, description: str, **kwargs) -> str:
        """类代码模板"""
        return self._generate_class(name, description, **kwargs)
        
    def _template_function(self, name: str, description: str, **kwargs) -> str:
        """函数代码模板"""
        return self._generate_function(name, description, **kwargs)
        
    def _template_test(self, module_name: str, **kwargs) -> str:
        """测试代码模板"""
        imports = f"import unittest\nimport {module_name}\n\n"
        
        test_class = f"class Test{module_name.capitalize()}(unittest.TestCase):\n"
        test_class += "    def setUp(self):\n        pass\n\n"
        test_class += "    def tearDown(self):\n        pass\n\n"
        test_class += "    def test_example(self):\n        self.assertTrue(True)\n\n"
        
        main_function = "if __name__ == '__main__':\n    unittest.main()\n"
        
        return imports + test_class + main_function
        
    def _template_plugin(self, name: str, description: str, **kwargs) -> str:
        """插件代码模板"""
        plugin_code = f'# -*- coding: utf-8 -*-\n"""\n{name} 插件\n\n{description}\n"""\n\n'
        
        plugin_code += "from utils.plugin_base import PluginBase\n\n"
        
        plugin_code += f"class {name}Plugin(PluginBase):\n"
        plugin_code += f'    """{description}"""\n\n'
        plugin_code += f'    plugin_name = "{name.lower()}"\n'
        plugin_code += f'    version = "0.1.0"\n'
        plugin_code += f'    description = "{description}"\n\n'
        
        plugin_code += "    def __init__(self):\n"
        plugin_code += "        super().__init__()\n\n"
        
        plugin_code += "    def initialize(self):\n"
        plugin_code += "        \"\"\"初始化插件\"\"\"\n"
        plugin_code += "        return True\n\n"
        
        plugin_code += "    def shutdown(self):\n"
        plugin_code += "        \"\"\"关闭插件\"\"\"\n"
        plugin_code += "        return True\n\n"
        
        plugin_code += "    def get_handlers(self):\n"
        plugin_code += "        \"\"\"获取事件处理器\"\"\"\n"
        plugin_code += "        return {\n"
        plugin_code += "            # 'event.type': self.handle_event\n"
        plugin_code += "        }\n\n"
        
        return plugin_code
        
    def _optimize_for_performance_analysis(self, code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行性能优化分析
        
        Args:
            code: 原始代码
            tree: AST树
            
        Returns:
            Tuple[str, List]: 优化后的代码和优化列表
        """
        # 这里是一个简化的性能优化实现
        # 实际系统中可以实现更复杂的优化策略
        optimized_code = code
        optimizations = []
        
        # 查找重复计算的表达式
        repeated_expressions = {}
        
        class ExpressionVisitor(ast.NodeVisitor):
            def visit_Expr(self, node):
                expr_str = ast.unparse(node.value)
                if expr_str not in repeated_expressions:
                    repeated_expressions[expr_str] = 1
                else:
                    repeated_expressions[expr_str] += 1
                self.generic_visit(node)
                
        ExpressionVisitor().visit(tree)
        
        # 提取频繁使用的表达式到局部变量
        for expr, count in repeated_expressions.items():
            if count > 3 and len(expr) > 10:
                # 这里简化处理，实际实现需要更复杂的代码修改
                optimizations.append({
                    "type": "repeated_expression",
                    "expression": expr,
                    "count": count,
                    "suggestion": f"将表达式 '{expr}' 提取到局部变量"
                })
                
        return optimized_code, optimizations
        
    def _optimize_for_readability(self, code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行可读性优化
        
        Args:
            code: 原始代码
            tree: AST树
            
        Returns:
            Tuple[str, List]: 优化后的代码和优化列表
        """
        optimized_code = code
        optimizations = []
        
        # 查找过长的函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.unparse(node)
                lines = func_code.count('\n')
                
                if lines > 50:
                    optimizations.append({
                        "type": "long_function",
                        "name": node.name,
                        "lines": lines,
                        "location": f"line {node.lineno}",
                        "suggestion": f"函数 '{node.name}' 过长，考虑拆分为更小的函数"
                    })
                    
        # 查找缺少文档字符串的函数和类
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                optimizations.append({
                    "type": "missing_docstring",
                    "name": node.name,
                    "location": f"line {node.lineno}",
                    "suggestion": f"为 '{node.name}' 添加文档字符串"
                })
                
        return optimized_code, optimizations
        
    def _optimize_for_memory(self, code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行内存优化
        
        Args:
            code: 原始代码
            tree: AST树
            
        Returns:
            Tuple[str, List]: 优化后的代码和优化列表
        """
        optimized_code = code
        optimizations = []
        
        # 查找大数据结构
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) > 100:
                optimizations.append({
                    "type": "large_list",
                    "location": f"line {node.lineno}",
                    "size": len(node.elts),
                    "suggestion": "考虑使用迭代器替代大型列表"
                })
                
        # 查找未关闭的文件操作
        file_nodes = []
        open_calls = []
        
        class FileVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    open_calls.append(node)
                self.generic_visit(node)
                
        FileVisitor().visit(tree)
        
        # 简化的检查，实际应该分析控制流
        for node in open_calls:
            # 检查是否使用了with语句
            is_in_with = False
            parent = node
            while hasattr(parent, 'parent') and parent.parent is not None:
                parent = parent.parent
                if isinstance(parent, ast.With):
                    is_in_with = True
                    break
                    
            if not is_in_with:
                optimizations.append({
                    "type": "file_not_closed",
                    "location": f"line {node.lineno}",
                    "suggestion": "使用with语句打开文件以确保关闭"
                })
                
        return optimized_code, optimizations
        
    def _optimize_general(self, code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行通用优化
        
        Args:
            code: 原始代码
            tree: AST树
            
        Returns:
            Tuple[str, List]: 优化后的代码和优化列表
        """
        optimized_code = code
        optimizations = []
        
        # 合并可读性和性能优化结果
        _, readability_opts = self._optimize_for_readability(code, tree)
        _, performance_opts = self._optimize_for_performance_analysis(code, tree)
        
        optimizations.extend(readability_opts)
        optimizations.extend(performance_opts)
        
        # 根据严重性排序
        optimizations.sort(key=lambda x: 
            3 if x["type"] in ["file_not_closed", "security_issue"] else
            2 if x["type"] in ["repeated_expression", "large_list"] else
            1, 
            reverse=True
        )
        
        return optimized_code, optimizations
        
    def auto_generate_module(self, module_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据规范自动生成新模块
        
        Args:
            module_spec: 模块规范，包含名称、功能描述、类/函数列表等
            
        Returns:
            Dict: 生成结果
        """
        try:
            module_name = module_spec.get("name", "auto_generated_module")
            module_desc = module_spec.get("description", "自动生成的模块")
            module_type = module_spec.get("type", "util")  # util, core, agent, etc.
            
            # 创建基础模块结构
            code = f'''# -*- coding: utf-8 -*-
"""
{module_name}

{module_desc}
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

'''
            
            # 导入依赖
            if "imports" in module_spec:
                for imp in module_spec["imports"]:
                    code += f'import {imp}\n'
                code += '\n'
            
            # 添加类
            if "classes" in module_spec:
                for cls in module_spec["classes"]:
                    code += self._generate_class_from_spec(cls)
                    code += '\n\n'
                    
            # 添加函数
            if "functions" in module_spec:
                for func in module_spec["functions"]:
                    code += self._generate_function_from_spec(func)
                    code += '\n\n'
                    
            # 添加主执行代码
            if module_spec.get("add_main", False):
                code += '''
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 主函数
    def main():
        """主函数"""
        logger = logging.getLogger("{}")
        logger.info("模块测试开始...")
        
        # 在这里添加测试代码
        
        logger.info("模块测试完成")
    
    main()
'''.format(module_name)
            
            # 创建文件路径
            filepath = self._determine_module_path(module_name, module_type)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
                
            # 发布事件
            if self.event_system:
                self.event_system.publish("code_generator.module_created", {
                    "module_name": module_name,
                    "filepath": filepath,
                    "code_size": len(code)
                })
                
            return {
                "status": "success",
                "module_name": module_name,
                "filepath": filepath,
                "code": code
            }
            
        except Exception as e:
            self.logger.error(f"生成模块失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _determine_module_path(self, module_name: str, module_type: str) -> str:
        """确定模块文件路径"""
        # 常见模块类型目录映射
        type_dirs = {
            "util": "utils",
            "core": "",
            "agent": "agents",
            "tool": "tools",
            "memory": "memory",
            "reasoning": "reasoning",
            "perception": "perception",
            "learning": "learning",
            "evolution": "evolution",
            "interface": "interface",
            "knowledge": "knowledge"
        }
        
        # 确定基础目录
        base_dir = type_dirs.get(module_type, "")
        
        # 模块名转文件名
        file_name = module_name.lower().replace(" ", "_") + ".py"
        
        # 构建完整路径
        if base_dir:
            return os.path.join(base_dir, file_name)
        else:
            return file_name
        
    def _generate_class_from_spec(self, class_spec: Dict[str, Any]) -> str:
        """生成类代码"""
        class_name = class_spec.get("name", "AutoClass")
        class_desc = class_spec.get("description", "自动生成的类")
        base_classes = class_spec.get("bases", [])
        
        # 类定义
        code = f'class {class_name}'
        
        # 添加基类
        if base_classes:
            base_str = ", ".join(base_classes)
            code += f'({base_str})'
        
        code += ':\n'
        
        # 添加类文档
        code += f'    """{class_desc}"""\n\n'
        
        # 添加构造函数
        init_params = class_spec.get("init_params", [])
        if init_params:
            code += '    def __init__(self'
            for param in init_params:
                param_name = param.get("name", "param")
                param_type = param.get("type", "Any")
                param_default = param.get("default", None)
                
                code += f', {param_name}'
                if param_default is not None:
                    code += f'={param_default}'
                
            code += '):\n'
            code += '        """\n        初始化方法\n        \n        Args:\n'
            
            for param in init_params:
                param_name = param.get("name", "param")
                param_desc = param.get("description", f"{param_name}参数")
                code += f'            {param_name}: {param_desc}\n'
            
            code += '        """\n'
            
            # 添加成员变量
            for param in init_params:
                param_name = param.get("name", "param")
                code += f'        self.{param_name} = {param_name}\n'
            
            # 添加日志
            code += '        self.logger = logging.getLogger(self.__class__.__name__)\n'
            code += '        self.logger.info("初始化完成")\n'
        else:
            code += '    def __init__(self):\n'
            code += '        """初始化方法"""\n'
            code += '        self.logger = logging.getLogger(self.__class__.__name__)\n'
            code += '        self.logger.info("初始化完成")\n'
        
        # 添加方法
        if "methods" in class_spec:
            for method in class_spec["methods"]:
                method_code = self._generate_method(method)
                # 增加缩进
                method_code = "\n".join("    " + line for line in method_code.split("\n"))
                code += "\n" + method_code
        
        return code
        
    def _generate_method(self, method_spec: Dict[str, Any]) -> str:
        """生成方法代码"""
        method_name = method_spec.get("name", "auto_method")
        method_desc = method_spec.get("description", "自动生成的方法")
        return_type = method_spec.get("return_type", "Any")
        
        # 方法参数
        params = method_spec.get("params", [])
        
        # 方法定义
        code = f'def {method_name}(self'
        
        for param in params:
            param_name = param.get("name", "param")
            param_default = param.get("default", None)
            
            code += f', {param_name}'
            if param_default is not None:
                code += f'={param_default}'
            
        code += '):\n'
        
        # 方法文档
        code += f'    """\n    {method_desc}\n    \n'
        
        if params:
            code += '    Args:\n'
            for param in params:
                param_name = param.get("name", "param")
                param_desc = param.get("description", "参数")
                param_type = param.get("type", "Any")
                code += f'        {param_name}: {param_desc}\n'
        
        code += f'    \n    Returns:\n        {return_type}: 返回结果\n    """\n'
        
        # 方法实现
        implementation = method_spec.get("implementation", "")
        if implementation:
            code += f'    {implementation}\n'
        else:
            # 提供默认实现
            code += '    try:\n'
            code += '        # 在此实现方法逻辑\n'
            code += '        self.logger.info("执行方法: {}")\n'.format(method_name)
            
            # 根据返回类型生成默认返回值
            if return_type == "Dict" or return_type == "Dict[str, Any]":
                code += '        return {"status": "success"}\n'
            elif return_type == "List" or return_type.startswith("List["):
                code += '        return []\n'
            elif return_type == "bool":
                code += '        return True\n'
            elif return_type == "int":
                code += '        return 0\n'
            elif return_type == "float":
                code += '        return 0.0\n'
            elif return_type == "str":
                code += '        return ""\n'
            elif return_type == "None":
                code += '        return\n'
            else:
                code += '        return None\n'
            
            code += '    except Exception as e:\n'
            code += '        self.logger.error(f"执行失败: {str(e)}")\n'
            
            # 根据返回类型生成错误返回值
            if return_type == "Dict" or return_type == "Dict[str, Any]":
                code += '        return {"status": "error", "message": str(e)}\n'
            elif return_type == "bool":
                code += '        return False\n'
            elif return_type in ["int", "float"]:
                code += '        return -1\n'
            else:
                code += '        return None\n'
        
        return code
        
    def _generate_function_from_spec(self, func_spec: Dict[str, Any]) -> str:
        """从规范生成函数代码"""
        func_name = func_spec.get("name", "auto_function")
        func_desc = func_spec.get("description", "自动生成的函数")
        return_type = func_spec.get("return_type", "Any")
        
        # 函数参数
        params = func_spec.get("params", [])
        
        # 函数定义
        code = f'def {func_name}('
        
        param_strs = []
        for param in params:
            param_name = param.get("name", "param")
            param_default = param.get("default", None)
            
            param_str = param_name
            if param_default is not None:
                param_str += f'={param_default}'
            
            param_strs.append(param_str)
        
        code += ", ".join(param_strs)
        code += '):\n'
        
        # 函数文档
        code += f'    """\n    {func_desc}\n    \n'
        
        if params:
            code += '    Args:\n'
            for param in params:
                param_name = param.get("name", "param")
                param_desc = param.get("description", "参数")
                code += f'        {param_name}: {param_desc}\n'
        
        code += f'    \n    Returns:\n        {return_type}: 返回结果\n    """\n'
        
        # 函数实现
        implementation = func_spec.get("implementation", "")
        if implementation:
            code += f'    {implementation}\n'
        else:
            # 提供默认实现
            code += '    try:\n'
            code += '        # 在此实现函数逻辑\n'
            code += '        logger = logging.getLogger("{}")\n'.format(func_name)
            code += '        logger.info("执行函数")\n'
            
            # 根据返回类型生成默认返回值
            if return_type == "Dict" or return_type == "Dict[str, Any]":
                code += '        return {"status": "success"}\n'
            elif return_type == "List" or return_type.startswith("List["):
                code += '        return []\n'
            elif return_type == "bool":
                code += '        return True\n'
            elif return_type == "int":
                code += '        return 0\n'
            elif return_type == "float":
                code += '        return 0.0\n'
            elif return_type == "str":
                code += '        return ""\n'
            elif return_type == "None":
                code += '        return\n'
            else:
                code += '        return None\n'
            
            code += '    except Exception as e:\n'
            code += '        logger.error(f"执行失败: {str(e)}")\n'
            
            # 根据返回类型生成错误返回值
            if return_type == "Dict" or return_type == "Dict[str, Any]":
                code += '        return {"status": "error", "message": str(e)}\n'
            elif return_type == "bool":
                code += '        return False\n'
            elif return_type in ["int", "float"]:
                code += '        return -1\n'
            else:
                code += '        return None\n'
        
        return code
    
    def analyze_code_quality(self, code_content: str) -> Dict[str, Any]:
        """
        分析代码质量，识别问题并提出改进建议
        
        Args:
            code_content: 需要分析的代码内容
            
        Returns:
            Dict: 分析结果，包含代码质量评分、问题列表和改进建议
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 分析结果
            analysis = {
                "quality_score": 0.0,
                "issues": [],
                "suggestions": [],
                "complexity": 0,
                "maintainability": 0.0,
                "timestamp": time.time()
            }
            
            # 复杂度分析
            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)
            analysis["complexity"] = complexity_visitor.complexity
            
            # 根据复杂度评分
            if analysis["complexity"] < 20:
                analysis["quality_score"] += 25.0
                analysis["maintainability"] = 90.0
            elif analysis["complexity"] < 50:
                analysis["quality_score"] += 15.0
                analysis["maintainability"] = 70.0
            else:
                analysis["quality_score"] += 5.0
                analysis["maintainability"] = 40.0
                analysis["issues"].append({
                    "type": "high_complexity",
                    "description": "代码复杂度过高，可能难以维护",
                    "severity": "high"
                })
                analysis["suggestions"].append({
                    "type": "refactor",
                    "description": "将复杂逻辑拆分为更小的函数",
                    "priority": "high"
                })
                
            # 命名规范分析
            naming_visitor = NamingConventionVisitor()
            naming_visitor.visit(tree)
            
            if naming_visitor.issues:
                analysis["issues"].extend(naming_visitor.issues)
                analysis["quality_score"] -= len(naming_visitor.issues) * 2.0
                analysis["suggestions"].append({
                    "type": "naming",
                    "description": "改进命名规范，使用更具描述性的标识符",
                    "priority": "medium"
                })
            else:
                analysis["quality_score"] += 10.0
                
            # 代码重复检测
            duplication_finder = DuplicateCodeFinder(tree)
            duplicates = duplication_finder.find_duplicates()
            
            if duplicates:
                analysis["issues"].append({
                    "type": "code_duplication",
                    "description": f"发现{len(duplicates)}处代码重复",
                    "severity": "medium",
                    "duplicates": duplicates
                })
                analysis["quality_score"] -= len(duplicates) * 3.0
                analysis["suggestions"].append({
                    "type": "extract_common",
                    "description": "提取重复代码到公共函数或方法",
                    "priority": "medium"
                })
            else:
                analysis["quality_score"] += 10.0
                
            # 错误处理检查
            error_visitor = ErrorHandlingVisitor()
            error_visitor.visit(tree)
            
            if error_visitor.missing_try_except > 0:
                analysis["issues"].append({
                    "type": "error_handling",
                    "description": f"发现{error_visitor.missing_try_except}处可能需要异常处理的位置",
                    "severity": "medium"
                })
                analysis["quality_score"] -= error_visitor.missing_try_except * 2.0
                analysis["suggestions"].append({
                    "type": "add_try_except",
                    "description": "为关键操作添加异常处理",
                    "priority": "medium"
                })
            else:
                analysis["quality_score"] += 10.0
                
            # 文档完整性检查
            doc_visitor = DocStringVisitor()
            doc_visitor.visit(tree)
            
            doc_score = doc_visitor.documented / max(1, doc_visitor.total) * 100 if doc_visitor.total > 0 else 100
            analysis["documentation_score"] = doc_score
            
            if doc_score < 60:
                analysis["issues"].append({
                    "type": "missing_docs",
                    "description": f"文档完整性较低 ({doc_score:.1f}%)",
                    "severity": "medium"
                })
                analysis["quality_score"] -= (100 - doc_score) / 10
                analysis["suggestions"].append({
                    "type": "add_docs",
                    "description": "为类和函数添加文档字符串",
                    "priority": "medium"
                })
            else:
                analysis["quality_score"] += 15.0
                
            # 规范化最终得分
            analysis["quality_score"] = max(0, min(100, analysis["quality_score"]))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"代码质量分析失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def auto_optimize_code(self, code_content: str, optimization_goals: List[str] = None) -> Dict[str, Any]:
        """
        自动优化代码，根据分析结果应用最佳实践
        
        Args:
            code_content: 需要优化的代码内容
            optimization_goals: 优化目标列表，可选值: ["performance", "readability", "maintenance", "security"]
            
        Returns:
            Dict: 优化结果，包含优化前后的代码和改进说明
        """
        try:
            if optimization_goals is None:
                optimization_goals = ["readability", "maintenance"]
                
            # 首先分析代码质量
            analysis = self.analyze_code_quality(code_content)
            
            if "status" in analysis and analysis["status"] == "error":
                return analysis
                
            # 解析代码
            tree = ast.parse(code_content)
            code_lines = code_content.splitlines()
            
            # 记录优化前的指标
            before_metrics = {
                "complexity": analysis["complexity"],
                "quality_score": analysis["quality_score"],
                "issues_count": len(analysis["issues"]),
                "line_count": len(code_lines)
            }
            
            # 应用优化转换
            optimized_code = code_content
            applied_optimizations = []
            
            # 1. 优化错误处理
            if "maintenance" in optimization_goals or "security" in optimization_goals:
                result = self._apply_error_handling_optimization(optimized_code)
                if result["status"] == "success":
                    optimized_code = result["code"]
                    applied_optimizations.append({
                        "type": "error_handling",
                        "description": "添加异常处理逻辑",
                        "changes": result["changes"]
                    })
            
            # 2. 优化命名规范
            if "readability" in optimization_goals or "maintenance" in optimization_goals:
                result = self._apply_naming_optimization(optimized_code)
                if result["status"] == "success":
                    optimized_code = result["code"]
                    applied_optimizations.append({
                        "type": "naming",
                        "description": "改进变量和函数命名",
                        "changes": result["changes"]
                    })
            
            # 3. 优化代码结构
            if "readability" in optimization_goals or "maintenance" in optimization_goals:
                result = self._apply_structure_optimization(optimized_code)
                if result["status"] == "success":
                    optimized_code = result["code"]
                    applied_optimizations.append({
                        "type": "structure",
                        "description": "改进代码结构",
                        "changes": result["changes"]
                    })
            
            # 4. 性能优化
            if "performance" in optimization_goals:
                result = self._apply_performance_optimization(optimized_code)
                if result["status"] == "success":
                    optimized_code = result["code"]
                    applied_optimizations.append({
                        "type": "performance",
                        "description": "提高代码执行效率",
                        "changes": result["changes"]
                    })
            
            # 5. 去除重复代码
            if "maintenance" in optimization_goals:
                result = self._apply_duplication_optimization(optimized_code)
                if result["status"] == "success":
                    optimized_code = result["code"]
                    applied_optimizations.append({
                        "type": "duplication",
                        "description": "消除代码重复",
                        "changes": result["changes"]
                    })
            
            # 再次分析优化后的代码
            optimized_analysis = self.analyze_code_quality(optimized_code)
            
            # 记录优化后的指标
            after_metrics = {
                "complexity": optimized_analysis["complexity"],
                "quality_score": optimized_analysis["quality_score"],
                "issues_count": len(optimized_analysis["issues"]),
                "line_count": len(optimized_code.splitlines())
            }
            
            # 计算改进百分比
            improvements = {
                "complexity": ((before_metrics["complexity"] - after_metrics["complexity"]) / max(1, before_metrics["complexity"])) * 100,
                "quality_score": ((after_metrics["quality_score"] - before_metrics["quality_score"]) / max(1, before_metrics["quality_score"])) * 100,
                "issues_count": ((before_metrics["issues_count"] - after_metrics["issues_count"]) / max(1, before_metrics["issues_count"])) * 100 if before_metrics["issues_count"] > 0 else 0
            }
            
            # 记录优化历史
            optimization_record = {
                "timestamp": time.time(),
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "improvements": improvements,
                "applied_optimizations": applied_optimizations,
                "optimization_goals": optimization_goals
            }
            self.optimization_history.append(optimization_record)
            
            # 最终结果
            result = {
                "status": "success",
                "original_code": code_content,
                "optimized_code": optimized_code,
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "improvements": improvements,
                "applied_optimizations": applied_optimizations,
                "timestamp": time.time()
            }
            
            # 如果没有应用任何优化，标记为无变化
            if not applied_optimizations:
                result["status"] = "no_change"
                result["message"] = "代码已经处于最佳状态，无需优化"
                
            return result
            
        except Exception as e:
            self.logger.error(f"代码自动优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def generate_module_from_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据需求规范自动生成模块
        
        Args:
            requirements: 需求规范，包含功能描述、接口要求和示例等
            
        Returns:
            Dict: 生成结果，包含生成的代码和模块结构
        """
        try:
            # 提取需求信息
            module_name = requirements.get("module_name", "generated_module")
            description = requirements.get("description", "自动生成的模块")
            features = requirements.get("features", [])
            interfaces = requirements.get("interfaces", [])
            dependencies = requirements.get("dependencies", [])
            examples = requirements.get("examples", [])
            
            # 构建模块规范
            module_spec = {
                "name": module_name,
                "type": "module",
                "description": description,
                "author": "GHOST AGI"
            }
            
            # 处理导入依赖
            imports = []
            for dep in dependencies:
                if isinstance(dep, str):
                    imports.append(dep)
                elif isinstance(dep, dict):
                    imports.append(dep)
            module_spec["imports"] = imports
            
            # 处理接口（转换为类和函数）
            classes = []
            functions = []
            
            for interface in interfaces:
                if interface.get("type") == "class":
                    classes.append({
                        "name": interface.get("name", "GeneratedClass"),
                        "description": interface.get("description", ""),
                        "bases": interface.get("bases", []),
                        "methods": interface.get("methods", []),
                        "attributes": interface.get("attributes", [])
                    })
                elif interface.get("type") == "function":
                    functions.append({
                        "name": interface.get("name", "generated_function"),
                        "description": interface.get("description", ""),
                        "params": interface.get("params", []),
                        "body": interface.get("body", "pass"),
                        "decorators": interface.get("decorators", [])
                    })
            
            # 根据特性增加额外的方法或函数
            for feature in features:
                feature_name = feature.get("name", "")
                feature_type = feature.get("type", "function")
                feature_desc = feature.get("description", "")
                
                if feature_type == "method" and classes:
                    # 添加方法到第一个类
                    classes[0]["methods"].append({
                        "name": feature_name,
                        "description": feature_desc,
                        "params": feature.get("params", []),
                        "body": feature.get("implementation", "pass")
                    })
                elif feature_type == "function":
                    functions.append({
                        "name": feature_name,
                        "description": feature_desc,
                        "params": feature.get("params", []),
                        "body": feature.get("implementation", "pass")
                    })
            
            module_spec["classes"] = classes
            module_spec["functions"] = functions
            
            # 生成模块
            result = self.generate_module(module_spec)
            
            # 如果提供了示例，添加单元测试
            if examples and result["status"] == "success":
                test_module_name = f"test_{module_name}"
                test_result = self.generate_test(module_name, examples)
                
                result["test_module"] = {
                    "name": test_module_name,
                    "code": test_result.get("code", "")
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"根据需求生成模块失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _apply_error_handling_optimization(self, code_content: str) -> Dict[str, Any]:
        """
        应用错误处理优化：为关键操作添加异常处理
        
        Args:
            code_content: 需要优化的代码内容
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 收集需要添加错误处理的位置
            error_visitor = ErrorHandlingVisitor()
            error_visitor.visit(tree)
            
            if error_visitor.missing_try_except == 0:
                return {
                    "status": "no_change",
                    "message": "没有发现需要添加异常处理的位置"
                }
                
            # 将代码转换为行列表
            lines = code_content.splitlines()
            
            # 应用更改
            changes = []
            modified_code = code_content
            
            # 这里实际项目中会进行实际的代码转换
            # 为了模拟效果，假设我们添加了适当的错误处理
            
            # 示例：添加一个简单的通用错误处理模板到代码中可能的风险点
            modified_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # 检测可能需要错误处理的函数调用行
                risky_functions = ["open", "read", "write", "connect", "send", "recv", "request", "urlopen"]
                
                # 简单检测这一行是否包含风险函数调用
                has_risky_call = any(f"{func}(" in line for func in risky_functions)
                
                if has_risky_call and "try:" not in line and i > 0 and "try:" not in lines[i-1]:
                    # 添加try-except块
                    indent = len(line) - len(line.lstrip())
                    indent_str = " " * indent
                    
                    modified_lines.append(f"{indent_str}try:")
                    modified_lines.append(f"{indent_str}    {line.strip()}")
                    modified_lines.append(f"{indent_str}except Exception as e:")
                    modified_lines.append(f"{indent_str}    logging.error(f\"操作失败: {{str(e)}}\")")
                    modified_lines.append(f"{indent_str}    # 处理异常情况")
                    
                    changes.append({
                        "line": i + 1,
                        "original": line,
                        "modified": "try-except block"
                    })
                    
                    i += 1
                else:
                    modified_lines.append(line)
                    i += 1
            
            modified_code = "\n".join(modified_lines)
            
            return {
                "status": "success",
                "code": modified_code,
                "changes": changes
            }
            
        except Exception as e:
            self.logger.error(f"应用错误处理优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "code": code_content
            }
    
    def _apply_naming_optimization(self, code_content: str) -> Dict[str, Any]:
        """
        应用命名规范优化
        
        Args:
            code_content: 需要优化的代码内容
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 检查命名规范
            naming_visitor = NamingConventionVisitor()
            naming_visitor.visit(tree)
            
            if not naming_visitor.issues:
                return {
                    "status": "no_change",
                    "message": "命名规范已符合要求"
                }
                
            # 获取所有需要重命名的标识符
            renames = {}
            for issue in naming_visitor.issues:
                if "函数名" in issue["description"]:
                    # 原名称在函数名xxx不符合...中提取
                    match = re.search(r"函数名 '(.+?)' 不符合", issue["description"])
                    if match:
                        original_name = match.group(1)
                        # 转换为snake_case
                        new_name = self._convert_to_snake_case(original_name)
                        renames[original_name] = new_name
                elif "类名" in issue["description"]:
                    match = re.search(r"类名 '(.+?)' 不符合", issue["description"])
                    if match:
                        original_name = match.group(1)
                        # 转换为PascalCase
                        new_name = self._convert_to_pascal_case(original_name)
                        renames[original_name] = new_name
                elif "变量名" in issue["description"]:
                    match = re.search(r"变量名 '(.+?)' 不符合", issue["description"])
                    if match:
                        original_name = match.group(1)
                        # 转换为snake_case
                        new_name = self._convert_to_snake_case(original_name)
                        renames[original_name] = new_name
            
            # 应用重命名
            # 注意：这是一个简化的实现，实际中需要考虑更多情况
            # 比如变量作用域、名称冲突等
            changes = []
            modified_code = code_content
            for original_name, new_name in renames.items():
                # 使用正则表达式进行替换，保证只替换标识符而不是字符串中的文本
                pattern = r'\b' + re.escape(original_name) + r'\b'
                modified_code = re.sub(pattern, new_name, modified_code)
                
                changes.append({
                    "type": "rename",
                    "original": original_name,
                    "new": new_name
                })
            
            return {
                "status": "success",
                "code": modified_code,
                "changes": changes
            }
            
        except Exception as e:
            self.logger.error(f"应用命名优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "code": code_content
            }
    
    def _apply_structure_optimization(self, code_content: str) -> Dict[str, Any]:
        """
        应用代码结构优化
        
        Args:
            code_content: 需要优化的代码内容
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 检查复杂度
            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)
            
            if complexity_visitor.complexity < 20:  # 复杂度阈值
                return {
                    "status": "no_change",
                    "message": "代码结构已经比较清晰，无需优化"
                }
                
            # 将代码转换为行列表
            lines = code_content.splitlines()
            
            # 收集复杂函数
            complex_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_complexity = 0
                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.If, ast.For, ast.While)):
                            function_complexity += 1
                    
                    if function_complexity > 3:  # 函数复杂度阈值
                        complex_functions.append({
                            "name": node.name,
                            "lineno": node.lineno,
                            "complexity": function_complexity
                        })
            
            # 应用结构优化
            # 这里只是一个模拟实现，实际中需要更复杂的代码转换逻辑
            changes = []
            modified_lines = lines.copy()
            
            for func in complex_functions:
                # 为复杂函数添加结构化注释
                line_index = func["lineno"] - 1
                if line_index < len(modified_lines):
                    # 添加注释，说明这是一个复杂函数，需要重构
                    indent = len(modified_lines[line_index]) - len(modified_lines[line_index].lstrip())
                    indent_str = " " * indent
                    comment = f"{indent_str}# TODO: 考虑将此复杂函数拆分为多个更小的函数以减少复杂度"
                    
                    modified_lines.insert(line_index, comment)
                    
                    changes.append({
                        "type": "add_comment",
                        "line": func["lineno"],
                        "function": func["name"],
                        "message": "标记复杂函数进行后续重构"
                    })
            
            modified_code = "\n".join(modified_lines)
            
            return {
                "status": "success",
                "code": modified_code,
                "changes": changes
            }
            
        except Exception as e:
            self.logger.error(f"应用结构优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "code": code_content
            }
    
    def _apply_performance_optimization(self, code_content: str) -> Dict[str, Any]:
        """
        应用性能优化
        
        Args:
            code_content: 需要优化的代码内容
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 应用性能优化
            changes = []
            
            # 转换为行列表处理
            lines = code_content.splitlines()
            modified_lines = lines.copy()
            
            # 优化: 替换列表推导式
            # 例如: 将 result = []
            #       for x in items:
            #           if condition(x):
            #               result.append(x)
            # 替换为 result = [x for x in items if condition(x)]
            
            i = 0
            while i < len(modified_lines) - 2:
                line = modified_lines[i]
                
                # 检测常见的列表初始化模式
                if "= []" in line and i + 2 < len(modified_lines):
                    var_match = re.search(r'(\w+)\s*=\s*\[\]', line)
                    if var_match:
                        var_name = var_match.group(1)
                        # 检查下一行是否是for循环
                        next_line = modified_lines[i+1]
                        for_match = re.search(r'for\s+(\w+)\s+in\s+(\w+)', next_line)
                        if for_match:
                            iter_var = for_match.group(1)
                            iterable = for_match.group(2)
                            
                            # 检查再下一行是否是append
                            next_next_line = modified_lines[i+2]
                            append_match = re.search(fr'{var_name}\.append\((.+)\)', next_next_line)
                            
                            if append_match:
                                expr = append_match.group(1)
                                
                                # 构建列表推导式
                                indent = len(line) - len(line.lstrip())
                                indent_str = " " * indent
                                
                                # 检查是否有条件判断
                                if "if " in next_next_line:
                                    cond_match = re.search(r'if\s+(.+?):', next_next_line)
                                    if cond_match:
                                        condition = cond_match.group(1)
                                        list_comp = f"{indent_str}{var_name} = [{expr} for {iter_var} in {iterable} if {condition}]"
                                    else:
                                        list_comp = f"{indent_str}{var_name} = [{expr} for {iter_var} in {iterable}]"
                                else:
                                    list_comp = f"{indent_str}{var_name} = [{expr} for {iter_var} in {iterable}]"
                                
                                # 替换这三行
                                modified_lines[i] = list_comp
                                modified_lines.pop(i+1)
                                modified_lines.pop(i+1)  # 原本的i+2，删除一行后变成i+1
                                
                                changes.append({
                                    "type": "list_comprehension",
                                    "line": i + 1,
                                    "message": "使用列表推导式替换for循环"
                                })
                                continue
                
                i += 1
            
            # 检测其他性能优化点
            for i, line in enumerate(modified_lines):
                # 检测字符串连接使用 + 运算符的情况
                if "+" in line and ("'" in line or '"' in line):
                    # 检查是否是多个字符串连接
                    parts = re.findall(r'[\'"].*?[\'"]', line)
                    if len(parts) > 2:
                        # 建议使用join方法
                        indent = len(line) - len(line.lstrip())
                        comment = " " * indent + "# TODO: 考虑使用str.join()代替+运算符连接多个字符串以提高性能"
                        modified_lines.insert(i, comment)
                        
                        changes.append({
                            "type": "string_join",
                            "line": i + 1,
                            "message": "建议使用str.join()代替+运算符"
                        })
                        i += 1
            
            modified_code = "\n".join(modified_lines)
            
            return {
                "status": "success" if changes else "no_change",
                "code": modified_code,
                "changes": changes,
                "message": "没有发现性能优化点" if not changes else ""
            }
            
        except Exception as e:
            self.logger.error(f"应用性能优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "code": code_content
            }
    
    def _apply_duplication_optimization(self, code_content: str) -> Dict[str, Any]:
        """
        应用代码重复优化
        
        Args:
            code_content: 需要优化的代码内容
            
        Returns:
            Dict: 优化结果
        """
        try:
            # 解析代码
            tree = ast.parse(code_content)
            
            # 检测代码重复
            duplication_finder = DuplicateCodeFinder(tree)
            duplicates = duplication_finder.find_duplicates()
            
            if not duplicates:
                return {
                    "status": "no_change",
                    "message": "没有发现重复代码"
                }
                
            # 在这个简化实现中，我们只添加注释标记重复代码
            # 真实实现中，应该提取重复代码到共享函数
            
            # 转换为行列表
            lines = code_content.splitlines()
            modified_lines = lines.copy()
            
            changes = []
            line_offset = 0  # 跟踪因插入注释导致的行号偏移
            
            for duplicate in duplicates:
                # 解析重复节点的行号
                for node_id in duplicate["nodes"]:
                    parts = node_id.split(":")
                    if len(parts) >= 3:
                        try:
                            line_num = int(parts[-1])
                            adjusted_line = line_num + line_offset - 1  # 转为0索引并考虑偏移
                            
                            if 0 <= adjusted_line < len(modified_lines):
                                line = modified_lines[adjusted_line]
                                indent = len(line) - len(line.lstrip())
                                indent_str = " " * indent
                                
                                # 添加注释标记重复代码
                                comment = f"{indent_str}# TODO: 重复代码，考虑提取到共享函数"
                                modified_lines.insert(adjusted_line, comment)
                                
                                line_offset += 1
                                
                                changes.append({
                                    "type": "mark_duplicate",
                                    "line": line_num,
                                    "message": "标记重复代码进行后续重构"
                                })
                        except ValueError:
                            continue
            
            modified_code = "\n".join(modified_lines)
            
            return {
                "status": "success",
                "code": modified_code,
                "changes": changes
            }
            
        except Exception as e:
            self.logger.error(f"应用重复代码优化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "code": code_content
            }
    
    def _convert_to_snake_case(self, name: str) -> str:
        """将标识符转换为snake_case命名风格"""
        # 处理驼峰命名法转换为下划线命名法
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        # 转换为小写并替换非法字符
        return re.sub(r'[^a-z0-9_]', '_', s2.lower())
    
    def _convert_to_pascal_case(self, name: str) -> str:
        """将标识符转换为PascalCase命名风格"""
        # 先转换为下划线分隔
        s = self._convert_to_snake_case(name)
        # 然后转换为PascalCase
        return ''.join(word.title() for word in s.split('_'))

# 以下是新增的辅助类

class ComplexityVisitor(ast.NodeVisitor):
    """分析代码复杂度的访问者类"""
    
    def __init__(self):
        self.complexity = 0
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

class NamingConventionVisitor(ast.NodeVisitor):
    """检查命名规范的访问者类"""
    
    def __init__(self):
        self.issues = []
    
    def visit_FunctionDef(self, node):
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            self.issues.append({
                "type": "naming_convention",
                "description": f"函数名 '{node.name}' 不符合命名规范 (应使用小写字母和下划线)",
                "location": f"line {node.lineno}"
            })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.issues.append({
                "type": "naming_convention",
                "description": f"类名 '{node.name}' 不符合命名规范 (应使用驼峰命名法)",
                "location": f"line {node.lineno}"
            })
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id.isupper() and len(node.id) > 1:
                # 常量命名规范，全部大写
                pass
            elif not node.id.startswith('_') and not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                self.issues.append({
                    "type": "naming_convention",
                    "description": f"变量名 '{node.id}' 不符合命名规范 (应使用小写字母和下划线)",
                    "location": f"line {getattr(node, 'lineno', '?')}"
                })
        self.generic_visit(node)

class DocStringVisitor(ast.NodeVisitor):
    """检查文档字符串的访问者类"""
    
    def __init__(self):
        self.documented = 0
        self.total = 0
    
    def visit_FunctionDef(self, node):
        self.total += 1
        if ast.get_docstring(node):
            self.documented += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.total += 1
        if ast.get_docstring(node):
            self.documented += 1
        self.generic_visit(node)
    
    def visit_Module(self, node):
        self.total += 1
        if ast.get_docstring(node):
            self.documented += 1
        self.generic_visit(node)

class ErrorHandlingVisitor(ast.NodeVisitor):
    """检查错误处理的访问者类"""
    
    def __init__(self):
        self.missing_try_except = 0
    
    def visit_Call(self, node):
        # 检查可能需要异常处理的函数调用
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # 检查是否是文件IO、网络请求等需要异常处理的操作
        risky_functions = ["open", "read", "write", "connect", "send", "recv", "request", "urlopen"]
        if func_name in risky_functions:
            # 检查父节点是否已在try块中
            in_try = False
            parent = getattr(node, "parent", None)
            while parent:
                if isinstance(parent, ast.Try):
                    in_try = True
                    break
                parent = getattr(parent, "parent", None)
            
            if not in_try:
                self.missing_try_except += 1
        
        # 处理子节点，并为每个子节点添加父节点引用
        for child_node in ast.iter_child_nodes(node):
            setattr(child_node, "parent", node)
            self.visit(child_node)

class DuplicateCodeFinder:
    """查找代码重复的工具类"""
    
    def __init__(self, tree):
        self.tree = tree
        self.node_hashes = {}
    
    def find_duplicates(self):
        """查找重复的代码片段"""
        duplicates = []
        visitor = DuplicateVisitor(self.node_hashes)
        visitor.visit(self.tree)
        
        # 查找重复的哈希值
        hash_counts = {}
        for node_id, hash_val in self.node_hashes.items():
            if hash_val in hash_counts:
                hash_counts[hash_val].append(node_id)
            else:
                hash_counts[hash_val] = [node_id]
        
        # 收集重复项
        for hash_val, nodes in hash_counts.items():
            if len(nodes) > 1 and len(hash_val) > 20:  # 只关注较大的重复项
                duplicates.append({
                    "hash": hash_val,
                    "count": len(nodes),
                    "nodes": nodes
                })
        
        return duplicates

class DuplicateVisitor(ast.NodeVisitor):
    """收集节点哈希值的访问者类"""
    
    def __init__(self, node_hashes):
        self.node_hashes = node_hashes
    
    def visit_FunctionDef(self, node):
        node_hash = self._hash_node(node)
        self.node_hashes[f"function:{node.name}:{node.lineno}"] = node_hash
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        node_hash = self._hash_node(node)
        self.node_hashes[f"class:{node.name}:{node.lineno}"] = node_hash
        self.generic_visit(node)
    
    def _hash_node(self, node):
        """计算节点的哈希值"""
        if isinstance(node, ast.AST):
            fields = []
            for field, value in ast.iter_fields(node):
                if field != 'lineno' and field != 'col_offset' and field != 'ctx':
                    fields.append(self._hash_node(value))
            return node.__class__.__name__ + '(' + ','.join(fields) + ')'
        elif isinstance(node, list):
            return '[' + ','.join(self._hash_node(x) for x in node) + ']'
        else:
            return repr(node) 