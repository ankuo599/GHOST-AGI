#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GHOST AGI 自主进化能力定期测试脚本
定期运行系统自主进化能力测试，收集性能指标并生成报告
"""

import os
import sys
import time
import json
import logging
import datetime
import argparse
import subprocess
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evolution_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EvolutionTests")

# 测试配置
DEFAULT_TESTS = [
    "test_evolution_comprehensive.py",
    "test_code_generator.py"
]

# 结果保存目录
RESULTS_DIR = "test_results"


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行GHOST AGI自主进化能力测试")
    parser.add_argument(
        "--tests", 
        nargs="+", 
        default=DEFAULT_TESTS,
        help="要运行的测试文件列表"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=0,
        help="测试运行间隔（小时），0表示仅运行一次"
    )
    parser.add_argument(
        "--report", 
        action="store_true",
        help="生成详细的HTML报告"
    )
    
    return parser.parse_args()


def run_test(test_file: str) -> Dict[str, Any]:
    """运行单个测试文件"""
    logger.info(f"开始运行测试: {test_file}")
    start_time = time.time()
    
    try:
        # 运行测试并捕获输出
        process = subprocess.run(
            ["python", test_file],
            capture_output=True,
            text=True,
            check=False
        )
        
        duration = time.time() - start_time
        success = process.returncode == 0
        
        result = {
            "test_file": test_file,
            "success": success,
            "duration": duration,
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        status = "成功" if success else "失败"
        logger.info(f"测试 {test_file} {status}，耗时: {duration:.2f}秒")
        
        return result
    except Exception as e:
        logger.error(f"运行测试 {test_file} 时发生错误: {str(e)}")
        return {
            "test_file": test_file,
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


def collect_metrics(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从测试结果中收集性能指标"""
    metrics = {
        "total_tests": len(test_results),
        "successful_tests": sum(1 for r in test_results if r.get("success", False)),
        "failed_tests": sum(1 for r in test_results if not r.get("success", False)),
        "total_duration": sum(r.get("duration", 0) for r in test_results),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # 尝试解析并合并测试生成的报告
    try:
        if os.path.exists("evolution_test_report.json"):
            with open("evolution_test_report.json", "r") as f:
                report_data = json.load(f)
                metrics["test_metrics"] = report_data
    except Exception as e:
        logger.warning(f"无法加载测试报告: {str(e)}")
    
    return metrics


def generate_report(metrics: Dict[str, Any], test_results: List[Dict[str, Any]]) -> str:
    """生成HTML测试报告"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"evolution_report_{timestamp}.html")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GHOST AGI 自主进化能力测试报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics {{ margin-top: 30px; }}
            .code {{ font-family: monospace; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <h1>GHOST AGI 自主进化能力测试报告</h1>
        <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>测试摘要</h2>
            <p>总测试数: {metrics['total_tests']}</p>
            <p>成功测试: <span class="success">{metrics['successful_tests']}</span></p>
            <p>失败测试: <span class="failure">{metrics['failed_tests']}</span></p>
            <p>总耗时: {metrics['total_duration']:.2f}秒</p>
        </div>
        
        <h2>测试详情</h2>
        <table>
            <tr>
                <th>测试文件</th>
                <th>状态</th>
                <th>耗时(秒)</th>
            </tr>
    """
    
    for result in test_results:
        status_class = "success" if result.get("success", False) else "failure"
        status_text = "成功" if result.get("success", False) else "失败"
        html += f"""
        <tr>
            <td>{result['test_file']}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{result.get('duration', 0):.2f}</td>
        </tr>
        """
    
    html += """
        </table>
    """
    
    # 添加指标详情
    if 'test_metrics' in metrics:
        html += """
        <div class="metrics">
            <h2>性能指标</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>值</th>
                </tr>
        """
        
        test_metrics = metrics['test_metrics']
        
        # 代码质量评分
        if 'code_analysis' in test_metrics:
            code_analysis = test_metrics['code_analysis']
            html += f"""
            <tr>
                <td>代码质量评分</td>
                <td>{code_analysis.get('quality_score', 0):.1f}</td>
            </tr>
            <tr>
                <td>代码问题数量</td>
                <td>{code_analysis.get('issues_count', 0)}</td>
            </tr>
            """
        
        # 代码优化改进
        if 'code_optimization' in test_metrics and 'improvements' in test_metrics['code_optimization']:
            improvements = test_metrics['code_optimization']['improvements']
            for metric, value in improvements.items():
                html += f"""
                <tr>
                    <td>优化改进: {metric}</td>
                    <td>{value:.1f}%</td>
                </tr>
                """
        
        html += """
            </table>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"HTML报告已生成: {report_path}")
    return report_path


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 如果指定了间隔，则循环运行测试
    run_count = 0
    
    while True:
        run_count += 1
        logger.info(f"开始第 {run_count} 次测试运行")
        
        # 运行所有指定的测试
        test_results = []
        for test_file in args.tests:
            if os.path.exists(test_file):
                result = run_test(test_file)
                test_results.append(result)
            else:
                logger.error(f"测试文件不存在: {test_file}")
                test_results.append({
                    "test_file": test_file,
                    "success": False,
                    "error": "测试文件不存在",
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # 收集指标并保存结果
        metrics = collect_metrics(test_results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"evolution_results_{timestamp}.json")
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "results": test_results
            }, f, indent=2)
        
        logger.info(f"测试结果已保存: {results_file}")
        
        # 生成报告
        if args.report:
            report_path = generate_report(metrics, test_results)
            logger.info(f"测试报告已生成: {report_path}")
        
        # 检查是否需要继续运行
        if args.interval <= 0:
            break
        
        next_run = datetime.datetime.now() + datetime.timedelta(hours=args.interval)
        logger.info(f"下次测试将在 {next_run.strftime('%Y-%m-%d %H:%M:%S')} 运行")
        
        # 休眠直到下次运行
        time.sleep(args.interval * 3600)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("测试运行被用户中断")
    except Exception as e:
        logger.error(f"测试运行出错: {str(e)}", exc_info=True)
        sys.exit(1) 