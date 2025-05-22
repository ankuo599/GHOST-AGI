# -*- coding: utf-8 -*-
"""
GHOST AGI 无限制能力测试脚本
测试系统在移除安全限制后的执行能力
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnrestrictedTest")

# 导入系统组件
from main import GhostAGI

class UnrestrictedTest:
    """系统无限制能力测试类"""
    
    def __init__(self, config=None):
        """
        初始化测试环境
        
        Args:
            config: 系统配置
        """
        logger.info("初始化无限制测试环境...")
        self.config = config or {}
        
        # 创建保存测试结果的目录
        os.makedirs("test_results", exist_ok=True)
        
        # 创建AGI系统实例
        self.agi = GhostAGI(config=self.config)
        logger.info("系统实例创建完成")
        
        # 测试结果
        self.test_results = {
            "start_time": time.time(),
            "tests": [],
            "success_count": 0,
            "failure_count": 0,
            "total_tests": 0
        }
    
    def _record_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """
        记录测试结果
        
        Args:
            test_name: 测试名称
            success: 是否成功
            details: 测试详情
        """
        result = {
            "test_name": test_name,
            "success": success,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        self.test_results["tests"].append(result)
        self.test_results["total_tests"] += 1
        
        if success:
            self.test_results["success_count"] += 1
            logger.info(f"测试 '{test_name}' 通过")
        else:
            self.test_results["failure_count"] += 1
            logger.error(f"测试 '{test_name}' 失败: {details.get('error', '')}")
    
    def test_unrestricted_command_execution(self):
        """测试无限制系统命令执行"""
        try:
            logger.info("测试无限制系统命令执行...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 测试危险命令（在正常限制下会被阻止的命令）
            # 注意：这里使用echo模拟危险命令，实际环境请谨慎使用
            test_commands = [
                "echo 'Simulating rm -rf /'",  # 模拟删除根目录
                "echo 'Simulating format c:'", # 模拟格式化磁盘
                "echo 'Simulating :(){:|:&};:'", # 模拟fork炸弹
                "echo 'Simulating chmod -R 777 /'" # 模拟修改权限
            ]
            
            results = []
            for cmd in test_commands:
                result = self.agi.execute_tool("execute_command", {"command": cmd})
                results.append({
                    "command": cmd,
                    "status": result.get("status"),
                    "output": result.get("stdout", "")
                })
                
                # 验证命令执行成功
                assert result["status"] == "success", f"命令 '{cmd}' 执行失败"
            
            self._record_test_result("unrestricted_command_execution", True, {
                "commands_tested": len(test_commands),
                "results": results
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("unrestricted_command_execution", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_unrestricted_python_execution(self):
        """测试无限制Python代码执行"""
        try:
            logger.info("测试无限制Python代码执行...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 测试在正常限制下会被阻止的Python代码
            test_codes = [
                # 使用os.system执行命令
                """
import os
print("Testing os.system")
os.system('echo "This is executed through os.system"')
                """,
                
                # 使用subprocess执行命令
                """
import subprocess
print("Testing subprocess")
result = subprocess.run(['echo', 'This is executed through subprocess'], capture_output=True, text=True)
print(result.stdout)
                """,
                
                # 使用eval执行代码
                """
print("Testing eval")
eval('print("This is executed through eval")')
                """
            ]
            
            results = []
            for code in test_codes:
                # 通过工具执行器执行Python代码
                # 由于Python代码执行可能是通过工具接口调用的，我们需要根据实际接口调整
                # 这里假设存在一个execute_python_code工具或类似功能
                if hasattr(self.agi, 'execute_python_code'):
                    result = self.agi.execute_python_code(code)
                else:
                    # 尝试使用通用工具执行器
                    result = self.agi.execute_tool("python_script", {"code": code})
                
                results.append({
                    "code_preview": code[:50] + "..." if len(code) > 50 else code,
                    "status": result.get("status"),
                    "output": result.get("stdout", "")
                })
                
                # 验证代码执行成功
                assert result["status"] == "success", f"代码执行失败"
            
            self._record_test_result("unrestricted_python_execution", True, {
                "codes_tested": len(test_codes),
                "results": results
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("unrestricted_python_execution", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_unrestricted_api_calls(self):
        """测试无限制API调用"""
        try:
            logger.info("测试无限制API调用...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 测试各种API调用
            test_apis = [
                # 公共API
                "https://jsonplaceholder.typicode.com/posts/1",
                
                # 本地API（模拟速率限制测试）
                "http://localhost:5000/api/test",
                
                # 非标准URL格式（在正常限制下会被阻止）
                "file:///etc/passwd",
                
                # 重复调用（模拟速率限制测试）
                "https://example.com/api"
            ]
            
            results = []
            for url in test_apis:
                try:
                    # 通过工具执行器执行API调用
                    result = self.agi.execute_tool("http_request", {
                        "url": url,
                        "method": "GET"
                    })
                    
                    results.append({
                        "url": url,
                        "status": result.get("status"),
                        "response_preview": str(result.get("response", ""))[:50]
                    })
                except Exception as call_error:
                    # 记录API调用错误但继续测试
                    results.append({
                        "url": url,
                        "status": "error",
                        "error": str(call_error)
                    })
            
            self._record_test_result("unrestricted_api_calls", True, {
                "apis_tested": len(test_apis),
                "results": results
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("unrestricted_api_calls", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行无限制能力测试...")
        
        # 运行测试
        self.test_unrestricted_command_execution()
        self.test_unrestricted_python_execution()
        self.test_unrestricted_api_calls()
        
        # 计算测试时间
        self.test_results["end_time"] = time.time()
        self.test_results["duration"] = self.test_results["end_time"] - self.test_results["start_time"]
        
        # 保存测试结果
        result_file = f"test_results/unrestricted_test_{int(time.time())}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"无限制能力测试完成，结果已保存到 {result_file}")
        
        # 输出测试摘要
        success_rate = self.test_results["success_count"] / self.test_results["total_tests"] * 100 if self.test_results["total_tests"] > 0 else 0
        
        print("\n================ 无限制测试摘要 ================")
        print(f"总测试数: {self.test_results['total_tests']}")
        print(f"通过测试: {self.test_results['success_count']}")
        print(f"失败测试: {self.test_results['failure_count']}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"测试用时: {self.test_results['duration']:.2f}秒")
        print("==========================================\n")
        
        return self.test_results

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHOST AGI 无限制能力测试")
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "debug": args.debug,
        "sandbox_enabled": False,  # 禁用沙箱
        "safety_checks": False     # 禁用安全检查
    }
    
    # 创建测试实例并运行测试
    test = UnrestrictedTest(config=config)
    results = test.run_all_tests()
    
    # 设置退出码
    if results["failure_count"] > 0:
        sys.exit(1)
    else:
        sys.exit(0) 