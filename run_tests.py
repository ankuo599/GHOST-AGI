#!/usr/bin/env python3
"""
GHOST-AGI 测试运行器

该脚本运行所有测试并生成综合报告
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghost_agi_tests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GHOST-AGI-Test-Runner")

def run_tests(args):
    """运行测试并生成报告"""
    logger.info("开始GHOST-AGI系统测试")
    start_time = time.time()
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        },
        "modules": {},
        "integration": {
            "status": "not_run",
            "details": {}
        }
    }
    
    # 运行系统测试
    if args.system or args.all:
        logger.info("运行系统测试...")
        
        try:
            from tests.system_test import run_all_tests
            system_results = run_all_tests()
            
            # 更新总结果
            test_results["summary"]["total_tests"] += system_results["summary"]["total_tests"]
            test_results["summary"]["passed_tests"] += system_results["summary"]["passed_tests"]
            test_results["summary"]["failed_tests"] += system_results["summary"]["failed_tests"]
            
            # 添加模块结果
            for module, results in system_results["module_results"].items():
                test_results["modules"][module] = results
                
            logger.info(f"系统测试完成。通过: {system_results['summary']['passed_tests']}, "
                       f"失败: {system_results['summary']['failed_tests']}")
        except ImportError as e:
            logger.error(f"无法导入系统测试: {str(e)}")
    
    # 运行集成测试
    if args.integration or args.all:
        logger.info("运行集成测试...")
        
        try:
            from tests.integration_test import test_metacognition_integration
            integration_success = test_metacognition_integration()
            
            test_results["integration"] = {
                "status": "pass" if integration_success else "fail",
                "details": {
                    "metacognition_integration": integration_success
                }
            }
            
            # 更新总结果
            test_results["summary"]["total_tests"] += 1
            if integration_success:
                test_results["summary"]["passed_tests"] += 1
            else:
                test_results["summary"]["failed_tests"] += 1
                
            logger.info(f"集成测试完成。结果: {'通过' if integration_success else '失败'}")
        except ImportError as e:
            logger.error(f"无法导入集成测试: {str(e)}")
    
    # 计算测试时间
    end_time = time.time()
    test_results["duration"] = end_time - start_time
    
    # 计算通过率
    if test_results["summary"]["total_tests"] > 0:
        pass_rate = (test_results["summary"]["passed_tests"] / 
                    test_results["summary"]["total_tests"]) * 100
        test_results["summary"]["pass_rate"] = f"{pass_rate:.1f}%"
    else:
        test_results["summary"]["pass_rate"] = "N/A"
    
    # 生成测试报告
    logger.info("生成测试报告...")
    report_path = args.output if args.output else "ghost_agi_test_report.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试报告已保存至 {report_path}")
    
    # 打印测试总结
    print("\n===== GHOST-AGI 测试总结 =====")
    print(f"总测试数: {test_results['summary']['total_tests']}")
    print(f"通过测试: {test_results['summary']['passed_tests']}")
    print(f"失败测试: {test_results['summary']['failed_tests']}")
    print(f"通过率: {test_results['summary']['pass_rate']}")
    print(f"总耗时: {test_results['duration']:.2f}秒")
    
    # 根据测试结果设置退出代码
    if test_results["summary"]["failed_tests"] > 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHOST-AGI 测试运行器")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--system", action="store_true", help="运行系统测试")
    parser.add_argument("--integration", action="store_true", help="运行集成测试")
    parser.add_argument("--output", type=str, help="测试报告输出路径")
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试类型，默认运行所有测试
    if not (args.all or args.system or args.integration):
        args.all = True
    
    sys.exit(run_tests(args)) 