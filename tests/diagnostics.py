"""
GHOST-AGI 系统诊断工具

该模块提供诊断和修复建议功能
"""

import os
import sys
import time
import importlib
import inspect
import logging
from typing import Dict, List, Any, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghost_agi_diagnostics.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GHOST-AGI-Diagnostics")

class SystemDiagnostics:
    """GHOST-AGI系统诊断工具"""
    
    def __init__(self):
        self.findings = []
        self.module_status = {}
        self.required_modules = [
            "metacognition.cognitive_monitor",
            "metacognition.reasoning_strategy_selector",
            "metacognition.meta_learning",
            "architecture.architecture_awareness",
            "perception.cross_modal_integrator"
        ]
        self.required_methods = {
            "metacognition.cognitive_monitor.CognitiveMonitor": [
                "track_reasoning_process", 
                "detect_cognitive_biases",
                "get_cognitive_trace"
            ],
            "metacognition.reasoning_strategy_selector.ReasoningStrategySelector": [
                "select_reasoning_strategy",
                "evaluate_strategy_effectiveness"
            ],
            "metacognition.meta_learning.MetaLearningModule": [
                "optimize_learning_strategy",
                "evaluate_learning_effectiveness",
                "analyze_learning_patterns",
                "recommend_learning_improvements",
                "integrate_with_cognitive_monitor"
            ]
        }
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        运行系统诊断
        
        Returns:
            Dict: 诊断结果
        """
        logger.info("开始系统诊断...")
        self.findings = []
        self.module_status = {}
        
        # 检查python环境
        self._check_python_environment()
        
        # 检查必要模块
        self._check_required_modules()
        
        # 检查方法实现
        self._check_required_methods()
        
        # 检查系统集成
        self._check_system_integration()
        
        # 评估总体状态
        overall_status = "healthy"
        critical_issues = 0
        warning_issues = 0
        
        for finding in self.findings:
            if finding["severity"] == "critical":
                critical_issues += 1
                overall_status = "unhealthy"
            elif finding["severity"] == "warning":
                warning_issues += 1
                if overall_status == "healthy":
                    overall_status = "needs_attention"
        
        diagnostic_result = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "critical_issues": critical_issues,
            "warning_issues": warning_issues,
            "findings": self.findings,
            "module_status": self.module_status
        }
        
        logger.info(f"诊断完成。状态: {overall_status}, 严重问题: {critical_issues}, 警告: {warning_issues}")
        return diagnostic_result
    
    def _check_python_environment(self):
        """检查Python环境"""
        python_version = sys.version_info
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
            self.findings.append({
                "id": "env-001",
                "component": "environment",
                "severity": "critical",
                "issue": f"Python版本过低: {sys.version}",
                "recommendation": "升级到Python 3.6或更高版本",
                "details": "GHOST-AGI系统需要Python 3.6+支持的类型提示和其他现代特性"
            })
    
    def _check_required_modules(self):
        """检查必要的模块是否可用"""
        for module_path in self.required_modules:
            try:
                module = importlib.import_module(module_path)
                self.module_status[module_path] = {
                    "status": "available",
                    "file_path": getattr(module, "__file__", "unknown")
                }
                logger.info(f"模块可用: {module_path}")
            except ImportError as e:
                self.module_status[module_path] = {
                    "status": "missing",
                    "error": str(e)
                }
                
                # 拆分模块路径来确定类型
                parts = module_path.split('.')
                component_type = parts[0] if parts else "unknown"
                
                self.findings.append({
                    "id": f"module-{len(self.findings) + 1:03d}",
                    "component": component_type,
                    "severity": "critical" if component_type == "metacognition" else "warning",
                    "issue": f"缺少模块: {module_path}",
                    "recommendation": f"实现{module_path}模块，确保符合系统接口要求",
                    "details": str(e)
                })
                logger.warning(f"模块缺失: {module_path} - {str(e)}")
    
    def _check_required_methods(self):
        """检查必要的方法是否实现"""
        for class_path, methods in self.required_methods.items():
            module_path, class_name = class_path.rsplit('.', 1)
            
            if module_path not in self.module_status or self.module_status[module_path]["status"] != "available":
                # 模块不可用，跳过方法检查
                continue
            
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name, None)
                
                if not cls:
                    self.findings.append({
                        "id": f"class-{len(self.findings) + 1:03d}",
                        "component": module_path.split('.')[0],
                        "severity": "critical",
                        "issue": f"缺少类: {class_name} in {module_path}",
                        "recommendation": f"在{module_path}中实现{class_name}类",
                        "details": f"模块存在但未定义{class_name}类"
                    })
                    continue
                
                # 检查类的方法
                missing_methods = []
                for method_name in methods:
                    if not hasattr(cls, method_name):
                        missing_methods.append(method_name)
                
                if missing_methods:
                    self.findings.append({
                        "id": f"method-{len(self.findings) + 1:03d}",
                        "component": module_path.split('.')[0],
                        "severity": "critical",
                        "issue": f"{class_name}缺少方法: {', '.join(missing_methods)}",
                        "recommendation": f"在{class_name}类中实现缺失的方法",
                        "details": f"以下方法未实现: {', '.join(missing_methods)}"
                    })
            except Exception as e:
                self.findings.append({
                    "id": f"check-{len(self.findings) + 1:03d}",
                    "component": module_path.split('.')[0],
                    "severity": "warning",
                    "issue": f"检查{class_path}时出错",
                    "recommendation": "检查模块导入和类定义",
                    "details": str(e)
                })
    
    def _check_system_integration(self):
        """检查系统集成"""
        # 检查元认知模块之间的集成
        metacognition_modules_available = all(
            m in self.module_status and self.module_status[m]["status"] == "available"
            for m in [
                "metacognition.cognitive_monitor",
                "metacognition.reasoning_strategy_selector",
                "metacognition.meta_learning"
            ]
        )
        
        if metacognition_modules_available:
            try:
                # 实例化模块来测试集成
                from metacognition.cognitive_monitor import CognitiveMonitor
                from metacognition.reasoning_strategy_selector import ReasoningStrategySelector
                from metacognition.meta_learning import MetaLearningModule
                
                cognitive_monitor = CognitiveMonitor()
                strategy_selector = ReasoningStrategySelector(cognitive_monitor=cognitive_monitor)
                meta_learning = MetaLearningModule(cognitive_monitor=cognitive_monitor)
                
                # 检查是否正确接收了依赖注入
                if getattr(strategy_selector, "cognitive_monitor", None) is not cognitive_monitor:
                    self.findings.append({
                        "id": f"integration-{len(self.findings) + 1:03d}",
                        "component": "metacognition",
                        "severity": "warning",
                        "issue": "ReasoningStrategySelector未正确集成CognitiveMonitor",
                        "recommendation": "确保ReasoningStrategySelector正确保存并使用cognitive_monitor参数",
                        "details": "依赖注入可能未正确实现"
                    })
                
                if getattr(meta_learning, "cognitive_monitor", None) is not cognitive_monitor:
                    self.findings.append({
                        "id": f"integration-{len(self.findings) + 1:03d}",
                        "component": "metacognition",
                        "severity": "warning",
                        "issue": "MetaLearningModule未正确集成CognitiveMonitor",
                        "recommendation": "确保MetaLearningModule正确保存并使用cognitive_monitor参数",
                        "details": "依赖注入可能未正确实现"
                    })
            except Exception as e:
                self.findings.append({
                    "id": f"integration-{len(self.findings) + 1:03d}",
                    "component": "metacognition",
                    "severity": "critical",
                    "issue": "元认知模块集成测试失败",
                    "recommendation": "检查模块之间的接口和依赖关系",
                    "details": str(e)
                })
    
    def generate_repair_plan(self) -> List[Dict[str, Any]]:
        """
        生成修复计划
        
        Returns:
            List: 修复步骤列表
        """
        repair_plan = []
        
        # 按严重性和组件对问题排序
        sorted_findings = sorted(
            self.findings, 
            key=lambda x: (0 if x["severity"] == "critical" else 1, x["component"])
        )
        
        for finding in sorted_findings:
            repair_step = {
                "issue_id": finding["id"],
                "component": finding["component"],
                "action": finding["recommendation"],
                "priority": "high" if finding["severity"] == "critical" else "medium",
                "details": finding["details"],
                "status": "pending"
            }
            repair_plan.append(repair_step)
        
        return repair_plan

def run_diagnostics():
    """运行系统诊断并输出结果"""
    diagnostics = SystemDiagnostics()
    result = diagnostics.run_diagnostics()
    
    print("\n===== GHOST-AGI 系统诊断结果 =====")
    print(f"总体状态: {result['overall_status']}")
    print(f"严重问题: {result['critical_issues']}")
    print(f"警告: {result['warning_issues']}")
    
    if result["findings"]:
        print("\n----- 发现的问题 -----")
        for finding in result["findings"]:
            print(f"[{finding['severity'].upper()}] {finding['issue']}")
            print(f"  建议: {finding['recommendation']}")
    
    if result["critical_issues"] > 0 or result["warning_issues"] > 0:
        print("\n----- 修复计划 -----")
        repair_plan = diagnostics.generate_repair_plan()
        for i, step in enumerate(repair_plan, 1):
            print(f"{i}. [{step['priority'].upper()}] {step['action']}")
    
    return result

if __name__ == "__main__":
    run_diagnostics() 