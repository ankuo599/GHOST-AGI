# -*- coding: utf-8 -*-
"""
多模态感知模块测试脚本
"""

import os
import time
import sys
import argparse
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("MultimodalTest")

def test_multimodal_perception():
    """测试多模态感知功能"""
    from perception.multimodal import MultiModalPerception

    logger.info("初始化多模态感知模块...")
    perception = MultiModalPerception()
    
    # 测试图像处理
    test_image_processing(perception)
    
    # 测试音频处理
    test_audio_processing(perception)

def test_image_processing(perception):
    """测试图像处理功能"""
    logger.info("\n===== 测试图像处理 =====")
    
    # 创建测试图像
    logger.info("创建测试图像...")
    img = Image.new('RGB', (500, 300), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        try:
            # 尝试其他常见字体
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                "C:\\Windows\\Fonts\\simhei.ttf",  # Windows
                "/System/Library/Fonts/PingFang.ttc"  # macOS
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 20)
                    break
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
    
    # 绘制文本和图形
    d.text((20, 20), "GHOST AGI 多模态测试图像", fill=(255, 255, 0), font=font)
    d.text((20, 60), "这是一个用于测试图像处理的样本", fill=(255, 255, 0), font=font)
    d.text((20, 100), "它包含文本和简单图形", fill=(255, 255, 0), font=font)
    
    # 绘制几何图形
    d.rectangle(((50, 150), (200, 250)), fill=(255, 0, 0))
    d.ellipse(((250, 150), (400, 250)), fill=(0, 255, 0))
    d.line(((50, 150), (400, 250)), fill=(255, 255, 255), width=3)
    
    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img_path = temp_file.name
    img.save(img_path)
    temp_file.close()
    
    # 测试基本图像描述
    logger.info("测试图像描述功能...")
    try:
        result = perception.process_image(img_path)
        if result["status"] == "success":
            logger.info(f"图像描述: {result.get('description', '无描述')}")
            logger.info(f"提取文本: {result.get('text', '无文本')}")
        else:
            logger.error(f"图像处理失败: {result.get('message', '未知错误')}")
    except Exception as e:
        logger.error(f"图像描述测试出错: {str(e)}")
    
    # 测试图像问答
    test_queries = [
        "这张图片是什么?",
        "图片中有什么文字?",
        "图片中有什么颜色?",
        "有没有几何图形?",
        "图片整体风格如何?"
    ]
    
    logger.info("\n测试图像问答功能...")
    for query in test_queries:
        try:
            logger.info(f"问题: {query}")
            result = perception.process_image(img_path, query=query)
            if result["status"] == "success":
                logger.info(f"回答: {result.get('query_answer', '无回答')}")
            else:
                logger.error(f"问答失败: {result.get('message', '未知错误')}")
        except Exception as e:
            logger.error(f"图像问答测试出错: {str(e)}")
    
    # 清理
    os.unlink(img_path)
    logger.info("图像处理测试完成")

def test_audio_processing(perception):
    """测试音频处理功能"""
    logger.info("\n===== 测试音频处理 =====")
    
    # 检查音频模型是否加载
    if not hasattr(perception, 'audio_model') or perception.audio_model is None:
        logger.warning("音频模型未加载，跳过音频处理测试")
        return
    
    # 寻找测试音频文件
    test_paths = [
        "test_data/audio_sample.wav",
        "test_data/audio_sample.mp3",
        "test_data/test_audio.wav"
    ]
    
    audio_path = None
    for path in test_paths:
        if os.path.exists(path):
            audio_path = path
            break
    
    if audio_path:
        try:
            logger.info(f"使用测试音频文件: {audio_path}")
            result = perception.process_audio(audio_path)
            if result["status"] == "success":
                logger.info(f"音频转录: {result.get('transcription', '无转录结果')}")
            else:
                logger.error(f"音频处理失败: {result.get('message', '未知错误')}")
        except Exception as e:
            logger.error(f"音频处理测试出错: {str(e)}")
    else:
        logger.warning("未找到测试音频文件，创建测试数据目录...")
        # 创建测试数据目录
        os.makedirs("test_data", exist_ok=True)
        logger.info("请将音频文件放入test_data目录并重新运行测试")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态感知模块测试")
    parser.add_argument("--image-only", action="store_true", help="仅测试图像处理")
    parser.add_argument("--audio-only", action="store_true", help="仅测试音频处理")
    
    args = parser.parse_args()
    
    # 导入系统模块
    try:
        from perception.multimodal import MultiModalPerception
    except ImportError:
        logger.error("无法导入MultiModalPerception模块，请确保项目配置正确")
        sys.exit(1)
    
    # 测试多模态感知
    perception = MultiModalPerception()
    
    if args.image_only:
        test_image_processing(perception)
    elif args.audio_only:
        test_audio_processing(perception)
    else:
        # 全部测试
        test_image_processing(perception)
        test_audio_processing(perception)
    
    logger.info("多模态测试完成")

if __name__ == "__main__":
    main() 