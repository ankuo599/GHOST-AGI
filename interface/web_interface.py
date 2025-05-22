# -*- coding: utf-8 -*-
"""
Web界面 (Web Interface)

基于Flask的Web界面，提供与GHOST AGI系统交互的功能
"""

import os
import time
import json
import threading
import logging
import base64
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebInterface")

class WebInterface:
    def __init__(self, agi_system=None, host="0.0.0.0", port=5000, debug=False):
        """
        初始化Web界面
        
        Args:
            agi_system: AGI系统实例
            host: 主机地址
            port: 端口
            debug: 是否开启调试模式
        """
        self.agi_system = agi_system
        self.host = host
        self.port = port
        self.debug = debug
        
        # 创建Flask应用
        self.app = Flask(
            __name__, 
            static_folder="static", 
            template_folder="templates"
        )
        self.app.secret_key = os.urandom(24)
        
        # 配置SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 事件处理器
        self.event_handlers = {}
        
        # 系统事件历史
        self.event_history = []
        self.max_history = 100
        
        # 支持的文件上传类型
        self.allowed_extensions = {
            'image': ['png', 'jpg', 'jpeg', 'gif', 'webp'],
            'audio': ['mp3', 'wav', 'ogg', 'flac', 'm4a'],
            'document': ['pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx', 'csv']
        }
        
        # 上传文件保存路径
        self.upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # 注册路由
        self._register_routes()
        
        # 注册Socket.IO事件
        self._register_socketio_events()
        
        # 创建事件监听线程
        self.listening = False
        self.listen_thread = None
        
    def _register_routes(self):
        """注册HTTP路由"""
        # 主页
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        # 系统状态页面
        @self.app.route('/status')
        def status():
            return render_template('status.html')
            
        # 记忆管理页面
        @self.app.route('/memory')
        def memory():
            return render_template('memory.html')
            
        # 工具页面
        @self.app.route('/tools')
        def tools():
            return render_template('tools.html')
            
        # 多模态界面
        @self.app.route('/multimodal')
        def multimodal():
            return render_template('multimodal.html')
            
        # API路由 - 获取系统状态
        @self.app.route('/api/status', methods=['GET'])
        def api_status():
            if not self.agi_system:
                return jsonify({
                    "status": "error",
                    "message": "AGI系统未初始化"
                }), 500
                
            try:
                status_data = self.agi_system.get_system_status()
                return jsonify(status_data)
            except Exception as e:
                logger.error(f"获取系统状态出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 处理用户输入
        @self.app.route('/api/chat', methods=['POST'])
        def api_chat():
            if not self.agi_system:
                return jsonify({
                    "status": "error",
                    "message": "AGI系统未初始化"
                }), 500
                
            try:
                data = request.json
                user_input = data.get('message', '')
                
                if not user_input:
                    return jsonify({
                        "status": "error",
                        "message": "消息不能为空"
                    }), 400
                    
                # 处理用户输入
                result = self.agi_system.process_user_input(user_input)
                
                return jsonify({
                    "status": "success",
                    "response": result
                })
            except Exception as e:
                logger.error(f"处理用户输入出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 处理图像输入
        @self.app.route('/api/process_image', methods=['POST'])
        def api_process_image():
            if not self.agi_system:
                return jsonify({
                    "status": "error",
                    "message": "AGI系统未初始化"
                }), 500
                
            try:
                # 获取图像数据
                data = request.json
                image_data = data.get('image_data')
                query = data.get('query', '')
                
                if not image_data:
                    # 检查是否有文件上传
                    if 'image' not in request.files:
                        return jsonify({
                            "status": "error",
                            "message": "未提供图像数据"
                        }), 400
                        
                    file = request.files['image']
                    if file.filename == '':
                        return jsonify({
                            "status": "error",
                            "message": "未选择文件"
                        }), 400
                        
                    # 检查文件类型
                    if not self._allowed_file(file.filename, 'image'):
                        return jsonify({
                            "status": "error",
                            "message": "不支持的文件类型"
                        }), 400
                        
                    # 保存文件
                    filename = os.path.join(self.upload_folder, f"{int(time.time())}_{file.filename}")
                    file.save(filename)
                    
                    # 处理图像
                    if hasattr(self.agi_system, 'perception') and hasattr(self.agi_system.perception, 'multimodal'):
                        result = self.agi_system.perception.multimodal.process_image(filename, query)
                    else:
                        # 如果没有多模态感知模块，尝试使用事件系统
                        if hasattr(self.agi_system, 'event_system'):
                            self.agi_system.event_system.publish("perception.image", {
                                "file_path": filename,
                                "query": query,
                                "timestamp": time.time()
                            })
                        
                        result = {
                            "status": "success",
                            "message": "图像已接收，但未找到多模态感知模块进行处理"
                        }
                else:
                    # 处理BASE64编码的图像数据
                    # 处理图像
                    if hasattr(self.agi_system, 'perception') and hasattr(self.agi_system.perception, 'multimodal'):
                        result = self.agi_system.perception.multimodal.process_image(image_data, query)
                    else:
                        # 如果没有多模态感知模块，尝试使用事件系统
                        if hasattr(self.agi_system, 'event_system'):
                            self.agi_system.event_system.publish("perception.image", {
                                "image_data": image_data,
                                "query": query,
                                "timestamp": time.time()
                            })
                        
                        result = {
                            "status": "success",
                            "message": "图像已接收，但未找到多模态感知模块进行处理"
                        }
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"处理图像输入出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 处理音频输入
        @self.app.route('/api/process_audio', methods=['POST'])
        def api_process_audio():
            if not self.agi_system:
                return jsonify({
                    "status": "error",
                    "message": "AGI系统未初始化"
                }), 500
                
            try:
                # 检查是否有文件上传
                if 'audio' not in request.files:
                    return jsonify({
                        "status": "error",
                        "message": "未提供音频数据"
                    }), 400
                    
                file = request.files['audio']
                if file.filename == '':
                    return jsonify({
                        "status": "error",
                        "message": "未选择文件"
                    }), 400
                    
                # 检查文件类型
                if not self._allowed_file(file.filename, 'audio'):
                    return jsonify({
                        "status": "error",
                        "message": "不支持的文件类型"
                    }), 400
                    
                # 保存文件
                filename = os.path.join(self.upload_folder, f"{int(time.time())}_{file.filename}")
                file.save(filename)
                
                # 处理音频
                if hasattr(self.agi_system, 'perception') and hasattr(self.agi_system.perception, 'multimodal'):
                    result = self.agi_system.perception.multimodal.process_audio(filename)
                else:
                    # 如果没有多模态感知模块，尝试使用事件系统
                    if hasattr(self.agi_system, 'event_system'):
                        self.agi_system.event_system.publish("perception.audio", {
                            "file_path": filename,
                            "timestamp": time.time()
                        })
                    
                    result = {
                        "status": "success",
                        "message": "音频已接收，但未找到多模态感知模块进行处理"
                    }
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"处理音频输入出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 执行工具
        @self.app.route('/api/tools/execute', methods=['POST'])
        def api_execute_tool():
            if not self.agi_system or not hasattr(self.agi_system, 'tool_executor'):
                return jsonify({
                    "status": "error",
                    "message": "工具执行器未初始化"
                }), 500
                
            try:
                data = request.json
                tool_name = data.get('tool_name')
                params = data.get('params', {})
                
                if not tool_name:
                    return jsonify({
                        "status": "error",
                        "message": "工具名称不能为空"
                    }), 400
                    
                # 执行工具
                result = self.agi_system.tool_executor.execute_tool(tool_name, params)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"执行工具出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 列出可用工具
        @self.app.route('/api/tools/list', methods=['GET'])
        def api_list_tools():
            if not self.agi_system or not hasattr(self.agi_system, 'tool_executor'):
                return jsonify({
                    "status": "error",
                    "message": "工具执行器未初始化"
                }), 500
                
            try:
                tools = self.agi_system.tool_executor.list_tools()
                return jsonify({
                    "status": "success",
                    "tools": tools
                })
            except Exception as e:
                logger.error(f"列出工具出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 查询记忆
        @self.app.route('/api/memory/query', methods=['POST'])
        def api_memory_query():
            if not self.agi_system or not hasattr(self.agi_system, 'memory_system'):
                return jsonify({
                    "status": "error",
                    "message": "记忆系统未初始化"
                }), 500
                
            try:
                data = request.json
                query = data.get('query', '')
                memory_type = data.get('type')
                limit = data.get('limit', 10)
                
                if not query and not memory_type:
                    return jsonify({
                        "status": "error",
                        "message": "查询条件不能为空"
                    }), 400
                    
                # 查询记忆
                results = self.agi_system.memory_system.query_memory(
                    query=query,
                    memory_type=memory_type,
                    limit=limit
                )
                
                return jsonify({
                    "status": "success",
                    "results": results
                })
            except Exception as e:
                logger.error(f"查询记忆出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
                
        # API路由 - 获取记忆系统统计信息
        @self.app.route('/api/memory/stats', methods=['GET'])
        def api_memory_stats():
            if not self.agi_system or not hasattr(self.agi_system, 'memory_system'):
                return jsonify({
                    "status": "error",
                    "message": "记忆系统未初始化"
                }), 500
                
            try:
                stats = self.agi_system.memory_system.get_memory_stats()
                return jsonify({
                    "status": "success",
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"获取记忆统计出错: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
    
    def _allowed_file(self, filename, file_type):
        """检查文件类型是否允许上传"""
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        return extension in self.allowed_extensions.get(file_type, [])
                
    def _register_socketio_events(self):
        """注册Socket.IO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"客户端已连接: {request.sid}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"客户端已断开连接: {request.sid}")
            
        @self.socketio.on('user_input')
        def handle_user_input(data):
            if not self.agi_system:
                emit('error', {
                    "message": "AGI系统未初始化"
                })
                return
                
            try:
                user_input = data.get('message', '')
                
                if not user_input:
                    emit('error', {
                        "message": "消息不能为空"
                    })
                    return
                    
                # 异步处理用户输入
                def process_async():
                    try:
                        result = self.agi_system.process_user_input(user_input)
                        emit('response', {
                            "status": "success",
                            "response": result
                        })
                    except Exception as e:
                        logger.error(f"处理用户输入出错: {str(e)}")
                        emit('error', {
                            "message": str(e)
                        })
                        
                thread = threading.Thread(target=process_async)
                thread.start()
                
            except Exception as e:
                logger.error(f"处理用户输入出错: {str(e)}")
                emit('error', {
                    "message": str(e)
                })
                
        @self.socketio.on('execute_tool')
        def handle_execute_tool(data):
            if not self.agi_system or not hasattr(self.agi_system, 'tool_executor'):
                emit('error', {
                    "message": "工具执行器未初始化"
                })
                return
                
            try:
                tool_name = data.get('tool_name')
                params = data.get('params', {})
                
                if not tool_name:
                    emit('error', {
                        "message": "工具名称不能为空"
                    })
                    return
                    
                # 异步执行工具
                def execute_async():
                    try:
                        result = self.agi_system.tool_executor.execute_tool(tool_name, params)
                        emit('tool_result', {
                            "status": "success",
                            "tool_name": tool_name,
                            "result": result
                        })
                    except Exception as e:
                        logger.error(f"执行工具出错: {str(e)}")
                        emit('error', {
                            "message": str(e)
                        })
                        
                thread = threading.Thread(target=execute_async)
                thread.start()
                
            except Exception as e:
                logger.error(f"执行工具出错: {str(e)}")
                emit('error', {
                    "message": str(e)
                })
                
        @self.socketio.on('process_image')
        def handle_process_image(data):
            if not self.agi_system:
                emit('error', {
                    "message": "AGI系统未初始化"
                })
                return
                
            try:
                image_data = data.get('image_data')
                query = data.get('query', '')
                
                if not image_data:
                    emit('error', {
                        "message": "图像数据不能为空"
                    })
                    return
                    
                # 异步处理图像
                def process_async():
                    try:
                        if hasattr(self.agi_system, 'perception') and hasattr(self.agi_system.perception, 'multimodal'):
                            result = self.agi_system.perception.multimodal.process_image(image_data, query)
                            emit('image_result', {
                                "status": "success",
                                "result": result
                            })
                        else:
                            # 如果没有多模态感知模块，尝试使用事件系统
                            if hasattr(self.agi_system, 'event_system'):
                                self.agi_system.event_system.publish("perception.image", {
                                    "image_data": image_data,
                                    "query": query,
                                    "timestamp": time.time()
                                })
                            
                            emit('image_result', {
                                "status": "success",
                                "message": "图像已接收，但未找到多模态感知模块进行处理"
                            })
                    except Exception as e:
                        logger.error(f"处理图像出错: {str(e)}")
                        emit('error', {
                            "message": str(e)
                        })
                        
                thread = threading.Thread(target=process_async)
                thread.start()
                
            except Exception as e:
                logger.error(f"处理图像出错: {str(e)}")
                emit('error', {
                    "message": str(e)
                })
                
    def _event_listener(self):
        """事件监听线程"""
        if not self.agi_system or not hasattr(self.agi_system, 'event_system'):
            logger.error("事件系统未初始化，无法启动事件监听")
            return
            
        # 注册事件处理器
        def handle_event(event):
            # 添加到历史记录
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
                
            # 通过Socket.IO广播事件
            self.socketio.emit('system_event', event)
            
        # 订阅所有事件
        subscription_id = self.agi_system.event_system.subscribe(
            event_type="*",
            handler=handle_event
        )
        
        # 保存订阅ID
        self.event_handlers["system_events"] = subscription_id
        
        # 发送初始化完成事件
        self.socketio.emit('system_event', {
            "type": "interface.initialized",
            "data": {
                "timestamp": time.time()
            }
        })
        
        # 等待退出信号
        while self.listening:
            time.sleep(0.1)
            
        # 取消订阅
        if subscription_id:
            self.agi_system.event_system.unsubscribe(subscription_id)
            
    def start_event_listener(self):
        """启动事件监听"""
        if self.listening:
            return False
            
        self.listening = True
        self.listen_thread = threading.Thread(target=self._event_listener, daemon=True)
        self.listen_thread.start()
        return True
        
    def stop_event_listener(self):
        """停止事件监听"""
        if not self.listening:
            return False
            
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
            
        return True
        
    def start(self):
        """启动Web界面"""
        # 启动事件监听
        self.start_event_listener()
        
        # 启动Socket.IO服务器
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
        
    def stop(self):
        """停止Web界面"""
        # 停止事件监听
        self.stop_event_listener()
        
        # 停止SocketIO (在实际应用中需要优雅关闭)
        # 通常在Flask开发服务器中，这个方法不会被调用，因为服务器由SIGINT等信号停止
        logger.info("Web界面已停止") 