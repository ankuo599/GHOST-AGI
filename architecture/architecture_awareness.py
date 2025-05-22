"""
架构感知模块 (Architecture Awareness Module)

该模块负责分析系统架构，识别模块间依赖关系，并提供架构改进建议。
"""

import os
import networkx as nx
from collections import defaultdict
import numpy as np
import re
import ast
import json
import time
from typing import Dict, List, Any, Optional, Union, Set

class ArchitecturalAwareness:
    def __init__(self, code_analyzer=None, vector_store=None):
        self.code_analyzer = code_analyzer
        self.vector_store = vector_store
        self.architecture_graph = nx.DiGraph()
        self.module_dependencies = {}
        self.design_patterns = self._load_design_patterns()
        self.architecture_metrics = {}
        self.analysis_cache = {}
        self.last_analysis_time = 0
        
    def _load_design_patterns(self):
        """加载设计模式识别规则"""
        return {
            "singleton": {
                "features": ["单例实例变量", "私有构造函数", "获取实例方法"],
                "detection": self._detect_singleton
            },
            "factory": {
                "features": ["创建方法", "多个产品类", "工厂接口"],
                "detection": self._detect_factory
            },
            "observer": {
                "features": ["订阅方法", "通知方法", "观察者列表"],
                "detection": self._detect_observer
            },
            "strategy": {
                "features": ["策略接口", "多个策略实现", "上下文类"],
                "detection": self._detect_strategy
            }
        }
    
    def _detect_singleton(self, node):
        """检测单例模式"""
        if not isinstance(node, ast.ClassDef):
            return False
            
        has_instance_var = False
        has_private_init = False
        has_get_instance = False
        
        for item in node.body:
            # 检查是否有实例变量
            if (isinstance(item, ast.Assign) and 
                any(isinstance(target, ast.Name) and target.id.startswith('_instance') for target in item.targets)):
                has_instance_var = True
                
            # 检查是否有私有构造函数
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id in ['staticmethod', 'classmethod']:
                        has_private_init = True
                        
            # 检查是否有获取实例方法
            if (isinstance(item, ast.FunctionDef) and 
                (item.name in ['get_instance', 'getInstance', 'instance'])):
                has_get_instance = True
                
        return has_instance_var and has_get_instance
    
    def _detect_factory(self, node):
        """检测工厂模式"""
        if not isinstance(node, ast.ClassDef):
            return False
            
        has_create_method = False
        
        for item in node.body:
            # 检查是否有创建方法
            if (isinstance(item, ast.FunctionDef) and 
                any(name in item.name.lower() for name in ['create', 'build', 'make', 'get'])):
                
                # 检查返回类型
                returns = item.returns if hasattr(item, 'returns') else None
                if returns:
                    has_create_method = True
                else:
                    # 查找return语句
                    for node in ast.walk(item):
                        if isinstance(node, ast.Return) and node.value:
                            has_create_method = True
                            break
                
        return has_create_method and ('factory' in node.name.lower() or 'creator' in node.name.lower())
    
    def _detect_observer(self, node):
        """检测观察者模式"""
        if not isinstance(node, ast.ClassDef):
            return False
            
        has_observers_list = False
        has_notify_method = False
        has_subscribe_method = False
        
        for item in node.body:
            # 检查是否有观察者列表
            if (isinstance(item, ast.Assign) and 
                any(name in str(target).lower() for target in item.targets 
                    for name in ['observers', 'listeners', 'subscribers'])):
                has_observers_list = True
                
            # 检查是否有通知方法
            if (isinstance(item, ast.FunctionDef) and 
                any(name in item.name.lower() for name in ['notify', 'update', 'publish', 'fire', 'emit'])):
                has_notify_method = True
                
            # 检查是否有订阅方法
            if (isinstance(item, ast.FunctionDef) and 
                any(name in item.name.lower() for name in ['subscribe', 'register', 'add_observer', 'attach'])):
                has_subscribe_method = True
                
        return has_observers_list and (has_notify_method or has_subscribe_method)
    
    def _detect_strategy(self, node):
        """检测策略模式"""
        if not isinstance(node, ast.ClassDef):
            return False
            
        strategy_indicators = ['strategy', 'policy', 'algorithm', 'behavior']
        
        # 检查类名是否包含策略相关词汇
        is_strategy_class = any(indicator in node.name.lower() for indicator in strategy_indicators)
        
        # 检查是否有执行策略的方法
        has_execute_method = any(
            isinstance(item, ast.FunctionDef) and 
            any(name in item.name.lower() for name in ['execute', 'apply', 'run', 'perform'])
            for item in node.body
        )
        
        return is_strategy_class or (has_execute_method and len(node.body) < 10)
        
    def analyze_system_architecture(self, project_dir, force_reanalysis=False):
        """分析整个系统架构，建立模块间依赖关系图"""
        # 检查缓存是否有效
        current_time = time.time()
        if not force_reanalysis and self.last_analysis_time > 0:
            # 如果最后分析时间在5分钟内，使用缓存
            if current_time - self.last_analysis_time < 300 and self.architecture_metrics:
                return {
                    "status": "cached",
                    "modules_count": len(self.architecture_graph.nodes),
                    "dependencies_count": len(self.architecture_graph.edges),
                    "architecture_metrics": self.architecture_metrics
                }
        
        # 重置数据结构
        self.architecture_graph = nx.DiGraph()
        self.module_dependencies = {}
        
        python_files = []
        
        # 扫描所有Python文件
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    module_name = os.path.relpath(file_path, project_dir).replace('/', '.').replace('\\', '.').replace('.py', '')
                    python_files.append((module_name, file_path))
        
        # 分析模块导入关系
        for module_name, file_path in python_files:
            self._analyze_file_dependencies(module_name, file_path)
        
        # 分析模块间调用关系
        self._analyze_call_dependencies(python_files)
        
        # 分析架构特征
        self._analyze_architecture_features()
        
        # 更新分析时间
        self.last_analysis_time = current_time
        
        return {
            "status": "success",
            "modules_count": len(self.architecture_graph.nodes),
            "dependencies_count": len(self.architecture_graph.edges),
            "architecture_metrics": self.architecture_metrics
        }
    
    def _analyze_file_dependencies(self, module_name, file_path):
        """分析文件的导入依赖"""
        self.architecture_graph.add_node(module_name, type="module", file_path=file_path)
        self.module_dependencies[module_name] = {"imports": [], "imported_by": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分析导入
            imports = []
            # 匹配from x import y 和 import x
            import_pattern = r'(?:from\s+([\w.]+)\s+import)|(?:import\s+([\w.,\s]+))'
            
            for match in re.finditer(import_pattern, content):
                from_import, direct_import = match.groups()
                
                if from_import:
                    imports.append(from_import)
                
                if direct_import:
                    modules = [m.strip() for m in direct_import.split(',')]
                    imports.extend(modules)
            
            # 过滤只保留项目内部模块
            project_imports = []
            for imp in imports:
                # 检查是否是项目内模块（简化版）
                for mod, _ in self.module_dependencies.items():
                    mod_parts = mod.split('.')
                    imp_parts = imp.split('.')
                    if imp_parts[0] == mod_parts[0]:
                        project_imports.append(imp)
                        break
            
            for imported_module in project_imports:
                if imported_module in self.module_dependencies:
                    self.architecture_graph.add_edge(module_name, imported_module, type="imports")
                    self.module_dependencies[module_name]["imports"].append(imported_module)
                    self.module_dependencies[imported_module]["imported_by"].append(module_name)
                    
        except Exception as e:
            print(f"分析模块 {module_name} 依赖关系时出错: {e}")
    
    def _analyze_call_dependencies(self, python_files):
        """分析模块间的函数调用依赖"""
        for module_name, file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用AST解析
                tree = ast.parse(content)
                
                # 获取当前模块定义的类和函数
                module_definitions = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        module_definitions[node.name] = "class"
                    elif isinstance(node, ast.FunctionDef):
                        module_definitions[node.name] = "function"
                
                # 分析其他模块的调用
                for imp_module in self.module_dependencies[module_name]["imports"]:
                    if imp_module in self.module_dependencies:
                        # 使用正则表达式查找可能的调用
                        module_short_name = imp_module.split('.')[-1]
                        call_pattern = rf'{module_short_name}\.(\w+)'
                        calls = re.findall(call_pattern, content)
                        
                        if calls:
                            # 添加调用边
                            edge_data = self.architecture_graph.get_edge_data(module_name, imp_module) or {}
                            edge_data["calls"] = list(set(calls))  # 去重
                            self.architecture_graph.add_edge(module_name, imp_module, **edge_data)
            except Exception as e:
                print(f"分析模块 {module_name} 调用关系时出错: {e}")
    
    def _analyze_architecture_features(self):
        """分析架构特征"""
        if not self.architecture_graph.nodes:
            self.architecture_metrics = {}
            return
            
        # 计算模块化指标
        modularity = self._calculate_modularity()
        
        # 计算耦合度
        coupling = self._calculate_coupling()
        
        # 检测循环依赖
        cycles = list(nx.simple_cycles(self.architecture_graph))
        
        # 检测孤立模块
        isolated = [node for node, degree in self.architecture_graph.degree() if degree == 0]
        
        # 识别核心模块
        core_modules = self._identify_core_modules()
        
        # 计算各模块的复杂度
        module_complexity = {}
        for module in self.architecture_graph.nodes:
            imports = len(self.module_dependencies.get(module, {}).get("imports", []))
            imported_by = len(self.module_dependencies.get(module, {}).get("imported_by", []))
            complexity = imports * imported_by if imported_by else imports
            module_complexity[module] = complexity
        
        self.architecture_metrics = {
            "modularity": modularity,
            "coupling": coupling,
            "has_cycles": len(cycles) > 0,
            "cycles_count": len(cycles),
            "cycles": cycles[:5] if cycles else [],  # 最多返回5个循环
            "isolated_modules": isolated,
            "core_modules": core_modules,
            "module_complexity": dict(sorted(module_complexity.items(), key=lambda x: x[1], reverse=True)[:10])  # 最复杂的10个模块
        }
    
    def _calculate_modularity(self):
        """计算模块化指标"""
        try:
            # 使用社区检测算法
            communities = nx.community.greedy_modularity_communities(self.architecture_graph.to_undirected())
            # 计算模块化得分
            modularity = nx.community.modularity(self.architecture_graph.to_undirected(), communities)
            return modularity
        except:
            # 简化计算
            nodes = self.architecture_graph.number_of_nodes()
            edges = self.architecture_graph.number_of_edges()
            if nodes <= 1:
                return 0
            return 1 - (edges / (nodes * (nodes - 1) / 2))
    
    def _calculate_coupling(self):
        """计算耦合度"""
        nodes = self.architecture_graph.number_of_nodes()
        if nodes <= 1:
            return 0
        
        edges = self.architecture_graph.number_of_edges()
        max_edges = nodes * (nodes - 1)
        
        return edges / max_edges if max_edges > 0 else 0
    
    def _identify_core_modules(self):
        """识别核心模块"""
        if not self.architecture_graph.nodes:
            return []
        
        # 使用中心性度量
        betweenness = nx.betweenness_centrality(self.architecture_graph)
        degree = nx.degree_centrality(self.architecture_graph)
        
        # 结合两种中心性度量
        combined_centrality = {}
        for node in self.architecture_graph.nodes:
            combined_centrality[node] = 0.5 * betweenness.get(node, 0) + 0.5 * degree.get(node, 0)
        
        # 选择前20%的模块作为核心模块
        sorted_modules = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        core_count = max(1, int(len(sorted_modules) * 0.2))
        
        return [module for module, _ in sorted_modules[:core_count]]
    
    def suggest_architectural_improvements(self):
        """根据架构分析提出系统级改进建议"""
        if not self.architecture_metrics:
            return {"status": "error", "message": "请先运行架构分析"}
        
        suggestions = []
        
        # 检查循环依赖
        if self.architecture_metrics.get("has_cycles", False):
            cycles = self.architecture_metrics.get("cycles", [])
            suggestions.append({
                "type": "refactoring",
                "severity": "high",
                "issue": f"发现{len(cycles)}个循环依赖",
                "description": "循环依赖会导致紧耦合和难以测试的代码",
                "cycles": cycles,
                "suggestion": "考虑引入接口层或依赖倒置原则解决循环依赖",
                "implementation_steps": [
                    "1. 识别循环中的关键模块",
                    "2. 提取共同接口",
                    "3. 使用依赖注入替代直接依赖"
                ]
            })
        
        # 检查高耦合
        if self.architecture_metrics.get("coupling", 0) > 0.3:
            coupling = self.architecture_metrics["coupling"]
            suggestions.append({
                "type": "architectural",
                "severity": "medium",
                "issue": f"系统耦合度过高({coupling:.2f})",
                "description": "高耦合度使得系统难以维护和扩展",
                "suggestion": "应用更严格的模块化设计，引入抽象接口层和依赖注入",
                "implementation_steps": [
                    "1. 识别高度耦合的模块",
                    "2. 重构提取共享逻辑到公共服务",
                    "3. 设计更清晰的模块边界"
                ]
            })
        
        # 检查孤立模块
        isolated = self.architecture_metrics.get("isolated_modules", [])
        if isolated:
            suggestions.append({
                "type": "architectural",
                "severity": "low",
                "issue": f"发现{len(isolated)}个孤立模块",
                "isolated_modules": isolated,
                "description": "孤立模块可能表示未使用的代码或集成不足",
                "suggestion": "考虑移除未使用的模块或将其集成到系统中",
                "implementation_steps": [
                    "1. 评估每个孤立模块的功能",
                    "2. 决定是集成还是移除",
                    "3. 对保留的模块，创建集成点"
                ]
            })
        
        # 分析复杂模块
        complex_modules = self.architecture_metrics.get("module_complexity", {})
        if complex_modules:
            top_complex = list(complex_modules.items())[:3]  # 前3个复杂模块
            suggestions.append({
                "type": "refactoring",
                "severity": "medium",
                "issue": "发现高复杂度模块",
                "complex_modules": top_complex,
                "description": "复杂模块难以维护和理解",
                "suggestion": "考虑将复杂模块拆分为更小、更专注的组件",
                "implementation_steps": [
                    "1. 分析模块责任",
                    "2. 按单一职责原则拆分",
                    "3. 建立清晰的子模块API"
                ]
            })
        
        # 模块化改进建议
        modularity = self.architecture_metrics.get("modularity", 0)
        if modularity < 0.3:
            suggestions.append({
                "type": "architectural",
                "severity": "medium",
                "issue": f"系统模块化程度低({modularity:.2f})",
                "description": "低模块化导致系统难以理解和维护",
                "suggestion": "考虑重构为更清晰的子系统，引入领域驱动设计原则",
                "implementation_steps": [
                    "1. 确定领域边界",
                    "2. 定义聚合和实体",
                    "3. 实现限界上下文"
                ]
            })
        
        # 设计模式建议
        pattern_suggestions = self._suggest_design_patterns()
        suggestions.extend(pattern_suggestions)
        
        return {
            "status": "success",
            "improvement_count": len(suggestions),
            "suggestions": suggestions
        }
    
    def _suggest_design_patterns(self):
        """基于架构特征建议设计模式应用"""
        suggestions = []
        
        # 检查是否有适合应用观察者模式的地方
        event_like_modules = []
        for module, data in self.module_dependencies.items():
            if (len(data.get("imported_by", [])) > 3 and 
                ("event" in module.lower() or 
                 "notification" in module.lower() or 
                 "observer" in module.lower())):
                event_like_modules.append(module)
        
        if event_like_modules:
            suggestions.append({
                "type": "design_pattern",
                "pattern": "observer",
                "candidate_modules": event_like_modules,
                "description": "这些模块被多个其他模块导入，且命名暗示事件处理",
                "suggestion": "考虑实现完整的观察者模式以降低耦合度",
                "implementation_steps": [
                    "1. 定义观察者接口",
                    "2. 创建主题类管理观察者",
                    "3. 实现通知机制"
                ]
            })
        
        # 检查是否有适合工厂模式的地方
        for module in self.architecture_graph.nodes:
            if (module.lower().endswith("factory") or 
                "creator" in module.lower() or 
                "builder" in module.lower()):
                suggestions.append({
                    "type": "design_pattern",
                    "pattern": "factory",
                    "module": module,
                    "description": "模块命名暗示创建逻辑",
                    "suggestion": "确保实现了完整的工厂模式，避免客户端直接创建对象",
                    "implementation_steps": [
                        "1. 定义产品接口",
                        "2. 实现具体产品",
                        "3. 创建工厂方法"
                    ]
                })
        
        # 检查是否有适合策略模式的地方
        for module in self.architecture_graph.nodes:
            if (module.lower().endswith("strategy") or 
                "policy" in module.lower() or 
                "algorithm" in module.lower()):
                suggestions.append({
                    "type": "design_pattern",
                    "pattern": "strategy",
                    "module": module,
                    "description": "模块命名暗示算法策略",
                    "suggestion": "确保使用策略模式实现算法可插拔性",
                    "implementation_steps": [
                        "1. 定义策略接口",
                        "2. 实现具体策略",
                        "3. 创建上下文类"
                    ]
                })
        
        return suggestions
    
    def generate_architecture_documentation(self):
        """生成架构文档"""
        if not self.architecture_graph.nodes:
            return {"status": "error", "message": "请先运行架构分析"}
        
        doc = {
            "title": "GHOST AGI系统架构文档",
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overview": {
                "modules_count": len(self.architecture_graph.nodes),
                "dependencies_count": len(self.architecture_graph.edges),
                "core_modules": self.architecture_metrics.get("core_modules", []),
                "modularity": self.architecture_metrics.get("modularity", 0),
                "coupling": self.architecture_metrics.get("coupling", 0)
            },
            "modules": []
        }
        
        # 为每个模块生成描述
        for module in self.architecture_graph.nodes:
            imports = self.module_dependencies.get(module, {}).get("imports", [])
            imported_by = self.module_dependencies.get(module, {}).get("imported_by", [])
            
            # 计算模块的复杂度
            complexity = len(imports) * len(imported_by) if imported_by else len(imports)
            
            doc["modules"].append({
                "name": module,
                "imports_count": len(imports),
                "imports": imports,
                "imported_by_count": len(imported_by),
                "imported_by": imported_by,
                "complexity": complexity,
                "is_core": module in self.architecture_metrics.get("core_modules", [])
            })
        
        # 添加依赖图描述
        doc["dependency_graphs"] = {
            "high_level": self._describe_high_level_dependencies(),
            "clusters": self._describe_module_clusters()
        }
        
        # 添加问题和建议
        improvement_suggestions = self.suggest_architectural_improvements()
        if improvement_suggestions.get("status") == "success":
            doc["improvement_suggestions"] = improvement_suggestions.get("suggestions", [])
        
        return {
            "status": "success",
            "documentation": doc
        }
    
    def _describe_high_level_dependencies(self):
        """描述高级依赖关系"""
        # 简化的依赖图，只显示主要模块和依赖
        top_modules = sorted(self.architecture_graph.nodes, 
                            key=lambda m: len(self.module_dependencies.get(m, {}).get("imported_by", [])),
                            reverse=True)[:10]
        
        high_level_deps = []
        for module in top_modules:
            deps = []
            for imp in self.module_dependencies.get(module, {}).get("imports", []):
                if imp in top_modules:
                    deps.append(imp)
            
            high_level_deps.append({
                "module": module,
                "dependencies": deps
            })
        
        return high_level_deps
    
    def _describe_module_clusters(self):
        """描述模块聚类"""
        try:
            # 使用社区检测算法查找模块集群
            communities = list(nx.community.greedy_modularity_communities(self.architecture_graph.to_undirected()))
            
            clusters = []
            for i, community in enumerate(communities):
                clusters.append({
                    "id": i + 1,
                    "modules": list(community),
                    "size": len(community)
                })
            
            return clusters
        except:
            return []
    
    def generate_evolution_plan(self):
        """生成系统架构演化计划"""
        if not self.architecture_metrics:
            return {"status": "error", "message": "请先运行架构分析"}
        
        # 获取改进建议
        improvements = self.suggest_architectural_improvements()
        if improvements.get("status") != "success":
            return {"status": "error", "message": "无法生成改进建议"}
        
        suggestions = improvements.get("suggestions", [])
        
        # 按严重性和类型对建议进行分类
        critical_issues = []
        architectural_improvements = []
        refactoring_suggestions = []
        design_pattern_applications = []
        
        for suggestion in suggestions:
            severity = suggestion.get("severity", "low")
            suggestion_type = suggestion.get("type", "")
            
            if severity == "high":
                critical_issues.append(suggestion)
            
            if suggestion_type == "architectural":
                architectural_improvements.append(suggestion)
            elif suggestion_type == "refactoring":
                refactoring_suggestions.append(suggestion)
            elif suggestion_type == "design_pattern":
                design_pattern_applications.append(suggestion)
        
        # 生成演化阶段
        evolution_stages = []
        
        # 第一阶段：解决关键问题
        if critical_issues:
            evolution_stages.append({
                "name": "解决关键架构问题",
                "description": "优先处理可能导致系统不稳定或难以维护的核心问题",
                "issues": critical_issues,
                "estimated_effort": "高",
                "priority": "紧急"
            })
        
        # 第二阶段：架构改进
        if architectural_improvements:
            evolution_stages.append({
                "name": "架构层面改进",
                "description": "调整系统整体架构，提高模块化和降低耦合度",
                "improvements": architectural_improvements,
                "estimated_effort": "中",
                "priority": "高"
            })
        
        # 第三阶段：代码重构
        if refactoring_suggestions:
            evolution_stages.append({
                "name": "代码层面重构",
                "description": "重构复杂模块，提高代码质量和可维护性",
                "refactorings": refactoring_suggestions,
                "estimated_effort": "中",
                "priority": "中"
            })
        
        # 第四阶段：应用设计模式
        if design_pattern_applications:
            evolution_stages.append({
                "name": "应用设计模式",
                "description": "在适当位置应用设计模式，提高代码灵活性和可扩展性",
                "patterns": design_pattern_applications,
                "estimated_effort": "低",
                "priority": "低"
            })
        
        # 第五阶段：持续优化
        evolution_stages.append({
            "name": "持续优化与监控",
            "description": "建立架构评估机制，持续监控系统质量",
            "activities": [
                "建立架构评审流程",
                "实现架构决策记录",
                "制定依赖管理策略",
                "优化构建和部署流程"
            ],
            "estimated_effort": "低",
            "priority": "常规"
        })
        
        return {
            "status": "success",
            "evolution_plan": {
                "title": "GHOST AGI系统架构演化计划",
                "overview": {
                    "current_modularity": self.architecture_metrics.get("modularity", 0),
                    "current_coupling": self.architecture_metrics.get("coupling", 0),
                    "critical_issues_count": len(critical_issues),
                    "total_improvements": len(suggestions)
                },
                "evolution_stages": evolution_stages,
                "expected_benefits": [
                    "系统模块化程度提高，更易于理解和维护",
                    "降低模块间耦合，提高系统灵活性",
                    "消除循环依赖，提高系统稳定性",
                    "应用恰当的设计模式，提高代码质量"
                ]
            }
        } 