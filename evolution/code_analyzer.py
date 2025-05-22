#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码分析器 (Code Analyzer)

负责深入分析代码结构、语义和质量
提供代码理解和优化建议的核心组件
"""

import os
import ast
import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import importlib.util
from collections import defaultdict

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """
    代码分析器
    
    负责分析Python代码结构、语义和质量，提供深度代码理解和优化建议
    """
    
    def __init__(self):
        """初始化代码分析器"""
        self.analyzed_files = {}  # 文件分析缓存
        self.analysis_stats = {
            "files_analyzed": 0,
            "ast_errors": 0,
            "pattern_matches": 0,
            "suggestions_generated": 0
        }
        
        # 代码模式定义
        self.code_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化代码模式定义"""
        return {
            # 反模式
            "global_variables": {
                "type": "antipattern",
                "description": "使用过多全局变量",
                "severity": "medium",
                "suggestion": "将全局变量转换为类属性或函数参数"
            },
            "long_function": {
                "type": "antipattern",
                "description": "函数过长",
                "severity": "medium",
                "threshold": 50,  # 行数阈值
                "suggestion": "将长函数分解为更小的函数"
            },
            "deep_nesting": {
                "type": "antipattern",
                "description": "深度嵌套的代码块",
                "severity": "medium",
                "threshold": 4,  # 嵌套深度阈值
                "suggestion": "使用提前返回或辅助函数减少嵌套"
            },
            "complex_condition": {
                "type": "antipattern",
                "description": "过于复杂的条件表达式",
                "severity": "medium",
                "threshold": 3,  # 条件运算符数量阈值
                "suggestion": "将复杂条件提取为命名函数或变量"
            },
            "unused_import": {
                "type": "antipattern",
                "description": "未使用的导入",
                "severity": "low",
                "suggestion": "移除未使用的导入"
            },
            "duplicate_code": {
                "type": "antipattern",
                "description": "重复代码",
                "severity": "high",
                "threshold": 6,  # 重复行数阈值
                "suggestion": "提取重复代码为函数或类"
            },
            
            # 良好模式
            "docstring": {
                "type": "goodpattern",
                "description": "函数或类包含文档字符串",
                "severity": "low",
                "suggestion": "保持良好的文档习惯"
            },
            "type_hint": {
                "type": "goodpattern",
                "description": "使用类型提示",
                "severity": "low",
                "suggestion": "继续使用类型提示提高代码可读性"
            },
            "exception_handling": {
                "type": "goodpattern",
                "description": "使用异常处理",
                "severity": "medium",
                "suggestion": "良好的异常处理实践"
            }
        }
        
    def analyze_file(self, file_path: str, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        分析单个Python文件
        
        Args:
            file_path: 文件路径
            force_reanalysis: 是否强制重新分析
            
        Returns:
            Dict: 分析结果
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"文件不存在: {file_path}"}
            
        # 检查是否是Python文件
        if not file_path.endswith(".py"):
            return {"status": "error", "message": f"非Python文件: {file_path}"}
            
        # 检查缓存
        if file_path in self.analyzed_files and not force_reanalysis:
            # 检查文件是否已被修改
            last_modified = os.path.getmtime(file_path)
            if last_modified <= self.analyzed_files[file_path].get("timestamp", 0):
                # 返回缓存的分析结果
                return self.analyzed_files[file_path]
                
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
                
            # 使用AST解析代码
            tree = ast.parse(code)
            
            # 执行各种分析
            result = {
                "status": "success",
                "file_path": file_path,
                "timestamp": time.time(),
                "loc": len(code.splitlines()),
                "imports": self._analyze_imports(tree),
                "classes": self._analyze_classes(tree, code),
                "functions": self._analyze_functions(tree, code),
                "variables": self._analyze_variables(tree),
                "complexity": self._calculate_complexity(tree, code),
                "patterns": self._detect_patterns(tree, code),
                "issues": [],
                "suggestions": []
            }
            
            # 从模式检测中生成问题和建议
            self._generate_issues_and_suggestions(result)
            
            # 缓存结果
            self.analyzed_files[file_path] = result
            self.analysis_stats["files_analyzed"] += 1
            
            return result
            
        except SyntaxError as e:
            error_result = {
                "status": "error",
                "file_path": file_path,
                "message": f"语法错误: {str(e)}",
                "error_line": e.lineno,
                "error_offset": e.offset,
                "error_type": "SyntaxError"
            }
            self.analysis_stats["ast_errors"] += 1
            return error_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "file_path": file_path,
                "message": f"分析错误: {str(e)}",
                "error_type": type(e).__name__
            }
            logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            return error_result
    
    def _analyze_imports(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """分析导入语句"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "type": "import",
                        "name": name.name,
                        "alias": name.asname,
                        "line": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({
                        "type": "importfrom",
                        "module": module,
                        "name": name.name,
                        "alias": name.asname,
                        "line": node.lineno
                    })
        
        return imports
    
    def _analyze_classes(self, tree: ast.Module, code: str) -> List[Dict[str, Any]]:
        """分析类定义"""
        classes = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # 获取类的文档字符串
                docstring = ast.get_docstring(node)
                
                # 获取父类
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{self._extract_attribute_name(base)}")
                
                # 分析类的方法
                methods = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method = {
                            "name": child.name,
                            "line": child.lineno,
                            "is_static": any(isinstance(d, ast.Name) and d.id == "staticmethod" 
                                           for d in child.decorator_list),
                            "is_class": any(isinstance(d, ast.Name) and d.id == "classmethod" 
                                          for d in child.decorator_list),
                            "docstring": ast.get_docstring(child) is not None,
                            "args": self._extract_function_args(child)
                        }
                        methods.append(method)
                
                # 分析类的属性
                attributes = []
                for child in node.body:
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                attributes.append({
                                    "name": target.id,
                                    "line": child.lineno
                                })
                
                # 类信息
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": bases,
                    "docstring": docstring is not None,
                    "methods": methods,
                    "attributes": attributes,
                    "method_count": len(methods),
                    "lines": self._count_node_lines(node, code)
                }
                
                classes.append(class_info)
        
        return classes
    
    def _analyze_functions(self, tree: ast.Module, code: str) -> List[Dict[str, Any]]:
        """分析函数定义"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                # 获取函数的文档字符串
                docstring = ast.get_docstring(node)
                
                # 提取参数信息
                args_info = self._extract_function_args(node)
                
                # 计算函数复杂度
                complexity = self._calculate_function_complexity(node)
                
                # 计算嵌套深度
                max_nesting = self._calculate_max_nesting(node)
                
                # 函数信息
                function_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": docstring is not None,
                    "args": args_info,
                    "return_type": self._extract_return_type(node),
                    "complexity": complexity,
                    "max_nesting": max_nesting,
                    "lines": self._count_node_lines(node, code),
                    "has_return": self._has_return(node)
                }
                
                functions.append(function_info)
        
        return functions
    
    def _analyze_variables(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """分析全局变量"""
        variables = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 检查是否是常量(全大写)
                        is_constant = target.id.isupper()
                        
                        variables.append({
                            "name": target.id,
                            "line": node.lineno,
                            "is_constant": is_constant
                        })
        
        return variables
    
    def _calculate_complexity(self, tree: ast.Module, code: str) -> Dict[str, Any]:
        """计算代码复杂度指标"""
        # 基本指标
        loc = len(code.splitlines())
        
        # 计算圈复杂度
        cyclomatic_complexity = 1  # 基础分为1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                cyclomatic_complexity += 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                cyclomatic_complexity += len(node.values) - 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                cyclomatic_complexity += len(node.values) - 1
                
        # 计算注释密度
        comment_lines = 0
        code_lines = code.splitlines()
        for line in code_lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_lines += 1
                
        comment_density = comment_lines / max(1, loc)
        
        return {
            "loc": loc,
            "cyclomatic_complexity": cyclomatic_complexity,
            "comment_lines": comment_lines,
            "comment_density": comment_density
        }
    
    def _detect_patterns(self, tree: ast.Module, code: str) -> Dict[str, List[Dict[str, Any]]]:
        """检测代码中的模式"""
        patterns = {
            "antipatterns": [],
            "goodpatterns": []
        }
        
        # 检测全局变量反模式
        global_vars = [v for v in self._analyze_variables(tree) 
                     if not v.get("is_constant", False)]
        if len(global_vars) > 5:  # 超过5个非常量全局变量
            patterns["antipatterns"].append({
                "pattern": "global_variables",
                "description": self.code_patterns["global_variables"]["description"],
                "locations": [{"line": v["line"], "name": v["name"]} for v in global_vars],
                "count": len(global_vars)
            })
            self.analysis_stats["pattern_matches"] += 1
            
        # 检测长函数反模式
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = self._count_node_lines(node, code)
                threshold = self.code_patterns["long_function"]["threshold"]
                
                if lines > threshold:
                    patterns["antipatterns"].append({
                        "pattern": "long_function",
                        "description": self.code_patterns["long_function"]["description"],
                        "location": {"line": node.lineno, "name": node.name},
                        "lines": lines,
                        "threshold": threshold
                    })
                    self.analysis_stats["pattern_matches"] += 1
                    
                # 检测深度嵌套反模式
                max_nesting = self._calculate_max_nesting(node)
                nesting_threshold = self.code_patterns["deep_nesting"]["threshold"]
                
                if max_nesting > nesting_threshold:
                    patterns["antipatterns"].append({
                        "pattern": "deep_nesting",
                        "description": self.code_patterns["deep_nesting"]["description"],
                        "location": {"line": node.lineno, "name": node.name},
                        "nesting_depth": max_nesting,
                        "threshold": nesting_threshold
                    })
                    self.analysis_stats["pattern_matches"] += 1
                    
                # 检测函数文档字符串良好模式
                if ast.get_docstring(node):
                    patterns["goodpatterns"].append({
                        "pattern": "docstring",
                        "description": self.code_patterns["docstring"]["description"],
                        "location": {"line": node.lineno, "name": node.name}
                    })
                    self.analysis_stats["pattern_matches"] += 1
                    
                # 检测类型提示良好模式
                if (hasattr(node, "returns") and node.returns) or self._has_type_hints(node):
                    patterns["goodpatterns"].append({
                        "pattern": "type_hint",
                        "description": self.code_patterns["type_hint"]["description"],
                        "location": {"line": node.lineno, "name": node.name}
                    })
                    self.analysis_stats["pattern_matches"] += 1
                    
                # 检测异常处理良好模式
                has_exception_handling = False
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Try):
                        has_exception_handling = True
                        break
                        
                if has_exception_handling:
                    patterns["goodpatterns"].append({
                        "pattern": "exception_handling",
                        "description": self.code_patterns["exception_handling"]["description"],
                        "location": {"line": node.lineno, "name": node.name}
                    })
                    self.analysis_stats["pattern_matches"] += 1
                    
        # 检测未使用的导入
        # 这需要更复杂的分析，简化版实现
        
        return patterns
    
    def _generate_issues_and_suggestions(self, result: Dict[str, Any]) -> None:
        """根据检测到的模式生成问题和建议"""
        for pattern in result["patterns"]["antipatterns"]:
            pattern_info = self.code_patterns.get(pattern["pattern"], {})
            severity = pattern_info.get("severity", "medium")
            suggestion = pattern_info.get("suggestion", "考虑重构该代码")
            
            issue = {
                "type": pattern["pattern"],
                "description": pattern["description"],
                "severity": severity,
                "locations": pattern.get("locations") or [pattern.get("location")] if pattern.get("location") else []
            }
            result["issues"].append(issue)
            
            # 生成建议
            if pattern["pattern"] == "long_function":
                func_name = pattern.get("location", {}).get("name", "")
                suggestion_text = f"函数 '{func_name}' 有 {pattern.get('lines')} 行代码，超过了阈值 {pattern.get('threshold')}。{suggestion}"
                result["suggestions"].append({
                    "type": "refactoring",
                    "description": suggestion_text,
                    "location": pattern.get("location"),
                    "severity": severity
                })
                self.analysis_stats["suggestions_generated"] += 1
                
            elif pattern["pattern"] == "deep_nesting":
                func_name = pattern.get("location", {}).get("name", "")
                suggestion_text = f"函数 '{func_name}' 的嵌套深度为 {pattern.get('nesting_depth')}，超过了阈值 {pattern.get('threshold')}。{suggestion}"
                result["suggestions"].append({
                    "type": "refactoring",
                    "description": suggestion_text,
                    "location": pattern.get("location"),
                    "severity": severity
                })
                self.analysis_stats["suggestions_generated"] += 1
                
            elif pattern["pattern"] == "global_variables":
                var_count = pattern.get("count", 0)
                suggestion_text = f"代码中使用了 {var_count} 个全局变量。{suggestion}"
                result["suggestions"].append({
                    "type": "refactoring",
                    "description": suggestion_text,
                    "locations": pattern.get("locations", []),
                    "severity": severity
                })
                self.analysis_stats["suggestions_generated"] += 1
    
    def _count_node_lines(self, node: ast.AST, code: str) -> int:
        """计算节点占用的行数"""
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            # 如果没有行号信息，试图以其他方式估计行数
            if hasattr(node, "body") and node.body:
                start_line = getattr(node, "lineno", 0)
                max_end_line = start_line
                
                for child in node.body:
                    if hasattr(child, "end_lineno"):
                        max_end_line = max(max_end_line, child.end_lineno)
                
                return max(1, max_end_line - start_line + 1)
            return 1
            
        return node.end_lineno - node.lineno + 1
    
    def _extract_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """提取函数参数信息"""
        args = []
        
        # 处理普通参数
        for i, arg in enumerate(node.args.args):
            arg_info = {
                "name": arg.arg,
                "has_default": i >= len(node.args.args) - len(node.args.defaults),
                "has_type": arg.annotation is not None
            }
            
            # 提取类型注解（如果有）
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_info["type"] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_info["type"] = self._extract_attribute_name(arg.annotation)
                else:
                    arg_info["type"] = "complex_type"
                    
            args.append(arg_info)
            
        # 处理*args参数
        if node.args.vararg:
            args.append({
                "name": f"*{node.args.vararg.arg}",
                "has_default": False,
                "has_type": node.args.vararg.annotation is not None
            })
            
        # 处理**kwargs参数
        if node.args.kwarg:
            args.append({
                "name": f"**{node.args.kwarg.arg}",
                "has_default": False,
                "has_type": node.args.kwarg.annotation is not None
            })
            
        return args
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """提取函数返回类型"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return self._extract_attribute_name(node.returns)
            else:
                return "complex_return_type"
        return None
    
    def _extract_attribute_name(self, node: ast.Attribute) -> str:
        """提取属性全名（如module.Class）"""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            
        if isinstance(current, ast.Name):
            parts.append(current.id)
            
        return ".".join(reversed(parts))
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数的圈复杂度"""
        complexity = 1  # 基础分为1
        
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(subnode, ast.BoolOp) and isinstance(subnode.op, ast.And):
                complexity += len(subnode.values) - 1
            elif isinstance(subnode, ast.BoolOp) and isinstance(subnode.op, ast.Or):
                complexity += len(subnode.values) - 1
                
        return complexity
    
    def _calculate_max_nesting(self, node: ast.FunctionDef) -> int:
        """计算函数中的最大嵌套深度"""
        def get_nesting(node, current_depth=0):
            max_depth = current_depth
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                current_depth += 1
                max_depth = current_depth
                
            for child in ast.iter_child_nodes(node):
                child_depth = get_nesting(child, current_depth)
                max_depth = max(max_depth, child_depth)
                
            return max_depth
            
        return get_nesting(node)
    
    def _has_return(self, node: ast.FunctionDef) -> bool:
        """检查函数是否有返回语句"""
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Return):
                return True
        return False
    
    def _has_type_hints(self, node: ast.FunctionDef) -> bool:
        """检查函数是否有类型提示"""
        # 检查参数类型提示
        for arg in node.args.args:
            if arg.annotation:
                return True
                
        # 检查*args和**kwargs的类型提示
        if node.args.vararg and node.args.vararg.annotation:
            return True
            
        if node.args.kwarg and node.args.kwarg.annotation:
            return True
            
        return False
    
    def analyze_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        分析目录中的所有Python文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归分析子目录
            
        Returns:
            Dict: 分析结果汇总
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            return {"status": "error", "message": f"目录不存在: {directory_path}"}
            
        python_files = []
        
        # 收集Python文件
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if file.endswith(".py"):
                    python_files.append(os.path.join(directory_path, file))
                    
        # 分析所有文件
        results = {}
        for file_path in python_files:
            results[file_path] = self.analyze_file(file_path)
            
        # 生成汇总指标
        summary = self._generate_summary(results)
        
        return {
            "status": "success",
            "directory": directory_path,
            "file_count": len(python_files),
            "file_results": results,
            "summary": summary
        }
    
    def _generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """根据多个文件的分析结果生成汇总指标"""
        total_loc = 0
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        issues_by_severity = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        issues_by_type = defaultdict(int)
        
        for file_path, result in results.items():
            if result["status"] != "success":
                continue
                
            total_loc += result.get("loc", 0)
            total_functions += len(result.get("functions", []))
            total_classes += len(result.get("classes", []))
            total_complexity += result.get("complexity", {}).get("cyclomatic_complexity", 0)
            
            # 汇总问题
            for issue in result.get("issues", []):
                severity = issue.get("severity", "medium")
                issues_by_severity[severity] += 1
                
                issue_type = issue.get("type", "unknown")
                issues_by_type[issue_type] += 1
                
        return {
            "total_loc": total_loc,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_complexity": total_complexity,
            "avg_complexity_per_file": total_complexity / max(1, len(results)),
            "issues_by_severity": issues_by_severity,
            "issues_by_type": dict(issues_by_type),
            "total_issues": sum(issues_by_severity.values()),
            "analysis_stats": self.analysis_stats
        }
    
    def get_code_suggestions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        获取代码改进建议
        
        Args:
            file_path: 文件路径
            
        Returns:
            List: 代码改进建议列表
        """
        # 确保文件已被分析
        if file_path not in self.analyzed_files:
            self.analyze_file(file_path)
            
        if file_path not in self.analyzed_files:
            return []
            
        result = self.analyzed_files[file_path]
        
        if result["status"] != "success":
            return []
            
        return result.get("suggestions", [])
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return {
            "files_analyzed": self.analysis_stats["files_analyzed"],
            "cached_files": len(self.analyzed_files),
            "ast_errors": self.analysis_stats["ast_errors"],
            "pattern_matches": self.analysis_stats["pattern_matches"],
            "suggestions_generated": self.analysis_stats["suggestions_generated"]
        } 