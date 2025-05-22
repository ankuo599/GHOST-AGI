# -*- coding: utf-8 -*-
"""
GHOST AGI 系统测试脚本
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
logger = logging.getLogger("SystemTest")

# 导入系统组件
from main import GhostAGI

class SystemTest:
    """系统测试类"""
    
    def __init__(self, config=None):
        """
        初始化测试环境
        
        Args:
            config: 系统配置
        """
        logger.info("初始化测试环境...")
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
    
    def test_event_system(self):
        """测试事件系统"""
        try:
            logger.info("测试事件系统...")
            
            # 创建测试事件处理器
            event_received = False
            test_data = {"test_value": 12345}
            
            def test_handler(event):
                nonlocal event_received
                nonlocal test_data
                
                # 验证事件数据
                if event["data"].get("test_value") == test_data["test_value"]:
                    event_received = True
            
            # 订阅测试事件
            subscription_id = self.agi.event_system.subscribe("test.event", test_handler)
            
            # 发布测试事件
            self.agi.event_system.publish("test.event", test_data)
            
            # 等待事件处理
            time.sleep(0.1)
            
            # 取消订阅
            self.agi.event_system.unsubscribe(subscription_id)
            
            # 验证事件接收
            assert event_received, "事件未被接收"
            
            # 获取事件历史
            history = self.agi.event_system.get_history(10)
            
            # 验证历史记录
            assert len(history) > 0, "事件历史为空"
            
            self._record_test_result("event_system", True, {
                "subscription_id": subscription_id,
                "history_length": len(history)
            })
        except Exception as e:
            self._record_test_result("event_system", False, {
                "error": str(e)
            })
    
    def test_memory_system(self):
        """测试记忆系统"""
        try:
            logger.info("测试记忆系统...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 添加短期记忆
            test_memory = {
                "type": "test_memory",
                "content": "这是一条测试记忆",
                "timestamp": time.time()
            }
            
            self.agi.memory_system.add_to_short_term(test_memory)
            
            # 获取短期记忆
            short_term_memories = self.agi.memory_system.get_short_term_memories(5)
            
            # 验证记忆添加
            assert len(short_term_memories) > 0, "短期记忆为空"
            
            # 查询记忆
            query_result = self.agi.memory_system.query_memory(
                query="测试记忆",
                memory_type="short_term",
                limit=5
            )
            
            # 验证查询结果
            assert len(query_result) > 0, "记忆查询结果为空"
            
            # 获取记忆统计
            stats = self.agi.memory_system.get_memory_stats()
            
            self._record_test_result("memory_system", True, {
                "short_term_count": len(short_term_memories),
                "query_result_count": len(query_result),
                "stats": stats
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("memory_system", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_tool_executor(self):
        """测试工具执行器"""
        try:
            logger.info("测试工具执行器...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 获取可用工具列表
            tools_info = self.agi.tool_executor.get_tool_info()
            
            # 验证工具列表
            assert len(tools_info) > 0, "工具列表为空"
            
            # 执行系统信息工具
            result = self.agi.execute_tool("system_info", {})
            
            # 验证执行结果
            assert result["status"] == "success", "工具执行失败"
            
            # 执行测试工具
            test_result = self.agi.execute_tool("test_reasoning", {
                "query": "这是一个测试查询"
            })
            
            # 验证测试结果
            assert test_result["status"] == "success", "测试工具执行失败"
            
            # 获取执行历史
            history = self.agi.tool_executor.get_execution_history(10)
            
            # 验证历史记录
            assert len(history) > 0, "工具执行历史为空"
            
            self._record_test_result("tool_executor", True, {
                "tools_count": len(tools_info),
                "execution_result": result["status"],
                "test_result": test_result["status"],
                "history_length": len(history)
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("tool_executor", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_core_agent(self):
        """测试核心智能体"""
        try:
            logger.info("测试核心智能体...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 创建测试任务
            test_task = {
                "type": "test_task",
                "query": "这是一个测试查询",
                "timestamp": time.time()
            }
            
            # 执行任务
            result = self.agi.core_agent.execute_task(test_task)
            
            # 验证任务执行结果
            assert result, "任务执行失败"
            
            # 处理用户输入
            input_result = self.agi.process_user_input("这是一个测试输入")
            
            # 验证输入处理结果
            assert input_result, "输入处理失败"
            
            self._record_test_result("core_agent", True, {
                "task_result": bool(result),
                "input_result": bool(input_result)
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("core_agent", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_planning_engine(self):
        """测试规划引擎"""
        try:
            logger.info("测试规划引擎...")
            
            # 启动AGI系统
            self.agi.start()
            
            # 创建测试计划
            plan = self.agi.create_plan("完成测试任务", {
                "context": "这是测试上下文"
            })
            
            # 验证计划创建
            assert plan, "计划创建失败"
            assert "id" in plan, "计划缺少ID"
            assert "steps" in plan, "计划缺少步骤"
            
            self._record_test_result("planning_engine", True, {
                "plan_id": plan.get("id"),
                "steps_count": len(plan.get("steps", []))
            })
            
            # 停止AGI系统
            self.agi.stop()
        except Exception as e:
            self._record_test_result("planning_engine", False, {
                "error": str(e)
            })
            
            # 确保系统停止
            if self.agi.running:
                self.agi.stop()
    
    def test_multimodal_perception(self):
        """测试多模态感知功能"""
        print("\n正在测试多模态感知模块...")
        
        # 测试图像处理功能
        print("测试图像处理功能:")
        try:
            # 创建测试图像
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import tempfile
            
            # 创建一个简单的测试图像
            img = Image.new('RGB', (400, 200), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            # 尝试加载中文字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("simhei.ttf", 15)
            except:
                font = ImageFont.load_default()
            
            d.text((10, 10), "GHOST AGI 测试图像", fill=(255, 255, 0), font=font)
            d.text((10, 60), "这是一个用于测试的图像", fill=(255, 255, 0), font=font)
            d.rectangle(((50, 100), (350, 180)), fill=(255, 128, 0))
            
            # 保存临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img_path = temp_file.name
            img.save(img_path)
            temp_file.close()
            
            # 测试图像处理
            result = self.agi.process_image(img_path)
            print(f"图像处理结果: {result['status']}")
            if result['status'] == 'success':
                print(f"- 图像描述: {result.get('description', '无描述')}")
                print(f"- 提取文本: {result.get('text', '无文本')}")
            else:
                print(f"- 错误信息: {result.get('message', '未知错误')}")
            
            # 测试图像问答
            query_result = self.agi.process_image(img_path, query="图片中有什么文字?")
            if query_result['status'] == 'success':
                print(f"- 问题回答: {query_result.get('query_answer', '无回答')}")
            
            # 清理
            import os
            os.unlink(img_path)
            
        except Exception as e:
            print(f"图像处理测试失败: {str(e)}")
        
        # 测试音频处理功能
        print("\n测试音频处理功能:")
        try:
            # 在实际测试中，这里应该使用真实的音频文件
            # 由于无法在代码中直接创建有效的音频文件，这里只进行模拟测试
            if hasattr(self.agi.perception, "multimodal") and self.agi.perception["multimodal"].audio_model:
                print("音频模型已加载，但跳过实际处理测试")
                print("要测试音频功能，请通过Web界面上传实际音频文件")
            else:
                print("音频模型未加载，跳过测试")
            
        except Exception as e:
            print(f"音频处理测试失败: {str(e)}")
        
        print("多模态感知模块测试完成")
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行所有测试...")
        
        # 运行测试
        self.test_event_system()
        self.test_memory_system()
        self.test_tool_executor()
        self.test_core_agent()
        self.test_planning_engine()
        self.test_multimodal_perception()
        
        # 计算测试时间
        self.test_results["end_time"] = time.time()
        self.test_results["duration"] = self.test_results["end_time"] - self.test_results["start_time"]
        
        # 保存测试结果
        result_file = f"test_results/test_result_{int(time.time())}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"测试完成，结果已保存到 {result_file}")
        
        # 输出测试摘要
        success_rate = self.test_results["success_count"] / self.test_results["total_tests"] * 100 if self.test_results["total_tests"] > 0 else 0
        
        print("\n================ 测试摘要 ================")
        print(f"总测试数: {self.test_results['total_tests']}")
        print(f"通过测试: {self.test_results['success_count']}")
        print(f"失败测试: {self.test_results['failure_count']}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"测试用时: {self.test_results['duration']:.2f}秒")
        print("==========================================\n")
        
        return self.test_results

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHOST AGI 系统测试")
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "debug": args.debug
    }
    
    # 创建测试实例并运行测试
    test = SystemTest(config=config)
    results = test.run_all_tests()
    
    # 设置退出码
    if results["failure_count"] > 0:
        sys.exit(1)
    else:
        sys.exit(0) 