"""
API网关 (API Gateway)

提供统一的外部接口，处理API请求，路由到相应的系统组件。
支持REST API、WebSocket和RPC调用，提供认证、限流和日志功能。
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
import threading
import json
import os
from collections import defaultdict, deque
import asyncio
import traceback

# 若系统中有其他Web框架，此处可以导入，例如Flask或FastAPI
# 此实现提供了框架无关的核心功能

class APIGateway:
    """API网关，提供统一的系统外部接口"""
    
    def __init__(self, system_integrator=None, logger=None):
        """
        初始化API网关
        
        Args:
            system_integrator: 系统集成器
            logger: 日志记录器
        """
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 系统组件
        self.system_integrator = system_integrator
        
        # API路由表
        self.routes = {}  # {route_path: handler_func}
        
        # API中间件
        self.middleware = []  # [{name, func, priority}]
        
        # 认证提供者
        self.auth_providers = {}  # {provider_name: provider_func}
        
        # 速率限制器
        self.rate_limiters = {}  # {limiter_name: limiter}
        
        # API文档
        self.api_docs = {}  # {route_path: doc_info}
        
        # 请求统计
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "route_stats": defaultdict(lambda: {"count": 0, "total_time": 0, "success_count": 0})
        }
        
        # WebSocket连接
        self.websocket_connections = {}  # {connection_id: connection_info}
        
        # 请求日志
        self.request_log = deque(maxlen=1000)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 初始化
        self._register_core_middleware()
        self._register_default_routes()
        
        self.logger.info("API网关初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("APIGateway")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 添加文件处理器
            file_handler = logging.FileHandler("api_gateway.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_core_middleware(self):
        """注册核心中间件"""
        # 添加认证中间件
        self.add_middleware("authentication", self._auth_middleware, 100)
        
        # 添加速率限制中间件
        self.add_middleware("rate_limiting", self._rate_limit_middleware, 90)
        
        # 添加日志中间件
        self.add_middleware("logging", self._logging_middleware, 10)
        
        # 添加错误处理中间件
        self.add_middleware("error_handling", self._error_middleware, 0)
    
    def _register_default_routes(self):
        """注册默认路由"""
        # 健康检查
        self.register_route("/health", self._health_check_handler, methods=["GET"])
        
        # API文档
        self.register_route("/docs", self._api_docs_handler, methods=["GET"])
        
        # 系统状态
        self.register_route("/system/status", self._system_status_handler, methods=["GET"])
        
        # API统计
        self.register_route("/api/stats", self._api_stats_handler, methods=["GET"])
    
    def register_route(self, path: str, handler: Callable, 
                     methods: List[str] = None, 
                     auth_required: bool = False,
                     rate_limit: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        注册API路由
        
        Args:
            path: 路由路径
            handler: 处理函数
            methods: 支持的HTTP方法
            auth_required: 是否需要认证
            rate_limit: 速率限制配置
            **kwargs: 其他配置
            
        Returns:
            Dict: 注册结果
        """
        with self.lock:
            if methods is None:
                methods = ["GET"]
                
            route_key = (path, tuple(methods))
            
            if route_key in self.routes:
                return {
                    "status": "error",
                    "message": f"路由已存在: {path} [{', '.join(methods)}]"
                }
                
            # 路由配置
            route_config = {
                "path": path,
                "methods": methods,
                "handler": handler,
                "auth_required": auth_required,
                "rate_limit": rate_limit,
                "registered_at": time.time()
            }
            
            # 添加其他配置
            route_config.update(kwargs)
            
            # 保存路由
            self.routes[route_key] = route_config
            
            # 生成路由文档
            self._generate_route_docs(route_key, route_config)
            
            self.logger.info(f"已注册路由: {path} [{', '.join(methods)}]")
            
            return {
                "status": "success",
                "path": path,
                "methods": methods
            }
    
    def add_middleware(self, name: str, middleware_func: Callable, 
                      priority: int = 50) -> Dict[str, Any]:
        """
        添加中间件
        
        Args:
            name: 中间件名称
            middleware_func: 中间件函数
            priority: 优先级（0-100，越高越先执行）
            
        Returns:
            Dict: 添加结果
        """
        with self.lock:
            # 检查是否已存在同名中间件
            for mw in self.middleware:
                if mw["name"] == name:
                    return {
                        "status": "error",
                        "message": f"中间件已存在: {name}"
                    }
                    
            # 添加中间件
            self.middleware.append({
                "name": name,
                "func": middleware_func,
                "priority": priority
            })
            
            # 按优先级排序
            self.middleware.sort(key=lambda x: x["priority"], reverse=True)
            
            self.logger.info(f"已添加中间件: {name} (优先级: {priority})")
            
            return {
                "status": "success",
                "name": name,
                "priority": priority
            }
    
    def register_auth_provider(self, name: str, provider_func: Callable) -> Dict[str, Any]:
        """
        注册认证提供者
        
        Args:
            name: 提供者名称
            provider_func: 提供者函数
            
        Returns:
            Dict: 注册结果
        """
        with self.lock:
            if name in self.auth_providers:
                return {
                    "status": "error",
                    "message": f"认证提供者已存在: {name}"
                }
                
            self.auth_providers[name] = provider_func
            
            self.logger.info(f"已注册认证提供者: {name}")
            
            return {
                "status": "success",
                "name": name
            }
    
    def register_rate_limiter(self, name: str, limiter_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        注册速率限制器
        
        Args:
            name: 限制器名称
            limiter_config: 限制器配置
            
        Returns:
            Dict: 注册结果
        """
        with self.lock:
            if name in self.rate_limiters:
                return {
                    "status": "error",
                    "message": f"速率限制器已存在: {name}"
                }
                
            # 创建限制器
            limiter = RateLimiter(
                requests=limiter_config.get("requests", 60),
                time_window=limiter_config.get("time_window", 60),
                key_func=limiter_config.get("key_func")
            )
            
            self.rate_limiters[name] = limiter
            
            self.logger.info(f"已注册速率限制器: {name}")
            
            return {
                "status": "success",
                "name": name,
                "config": limiter_config
            }
    
    def handle_request(self, path: str, method: str, 
                     headers: Dict[str, str], 
                     query_params: Dict[str, Any],
                     body: Optional[Any] = None,
                     client_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理API请求
        
        Args:
            path: 请求路径
            method: 请求方法
            headers: 请求头
            query_params: 查询参数
            body: 请求体
            client_info: 客户端信息
            
        Returns:
            Dict: 请求处理结果
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # 准备请求上下文
        context = {
            "request_id": request_id,
            "path": path,
            "method": method,
            "headers": headers,
            "query_params": query_params,
            "body": body,
            "client_info": client_info or {},
            "start_time": start_time,
            "auth": None,  # 将由认证中间件填充
            "route_config": None,  # 将在路由匹配时填充
            "error": None,  # 将在出错时填充
            "response": None  # 将由处理程序填充
        }
        
        # 查找路由
        route_key = self._match_route(path, method)
        
        if route_key:
            route_config = self.routes[route_key]
            context["route_config"] = route_config
        else:
            # 路由未找到
            context["error"] = {
                "status": 404,
                "message": f"路由未找到: {path} [{method}]"
            }
        
        # 应用中间件（前置）
        for middleware in self.middleware:
            # 前置处理
            if hasattr(middleware["func"], "__pre__") and middleware["func"].__pre__:
                try:
                    modified_context = middleware["func"](context)
                    if modified_context is not None:
                        context = modified_context
                        
                    # 如果中间件设置了错误或响应，提前结束
                    if context.get("error") or context.get("response"):
                        break
                except Exception as e:
                    self.logger.error(f"中间件异常: {middleware['name']}, 错误: {str(e)}")
                    context["error"] = {
                        "status": 500,
                        "message": f"中间件错误: {str(e)}"
                    }
                    break
        
        # 如果中间件没有设置错误或响应，执行路由处理
        if not context.get("error") and not context.get("response") and context.get("route_config"):
            try:
                handler = context["route_config"]["handler"]
                response = handler(context)
                context["response"] = response
            except Exception as e:
                self.logger.error(f"路由处理异常: {path}, 错误: {str(e)}")
                context["error"] = {
                    "status": 500,
                    "message": f"处理错误: {str(e)}",
                    "traceback": traceback.format_exc()
                }
        
        # 应用中间件（后置）
        for middleware in reversed(self.middleware):
            # 后置处理
            if hasattr(middleware["func"], "__post__") and middleware["func"].__post__:
                try:
                    modified_context = middleware["func"](context)
                    if modified_context is not None:
                        context = modified_context
                except Exception as e:
                    self.logger.error(f"中间件异常: {middleware['name']}, 错误: {str(e)}")
                    if not context.get("error"):
                        context["error"] = {
                            "status": 500,
                            "message": f"中间件错误: {str(e)}"
                        }
        
        # 准备响应
        if context.get("error"):
            response = {
                "status": "error",
                "error": context["error"]["message"],
                "status_code": context["error"]["status"]
            }
        else:
            response = context.get("response", {
                "status": "success",
                "message": "请求处理成功"
            })
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 更新请求统计
        self._update_request_stats(context, response_time)
        
        # 添加请求ID和时间信息
        response["request_id"] = request_id
        response["response_time"] = response_time
        
        return response
    
    async def handle_websocket(self, websocket, path: str, 
                             headers: Dict[str, str], 
                             query_params: Dict[str, Any],
                             client_info: Optional[Dict[str, Any]] = None):
        """
        处理WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            path: 连接路径
            headers: 请求头
            query_params: 查询参数
            client_info: 客户端信息
        """
        connection_id = str(uuid.uuid4())
        
        # 准备连接上下文
        context = {
            "connection_id": connection_id,
            "path": path,
            "headers": headers,
            "query_params": query_params,
            "client_info": client_info or {},
            "start_time": time.time(),
            "auth": None,  # 将由认证中间件填充
            "websocket": websocket
        }
        
        # 查找WebSocket处理程序
        handler = self._match_websocket_handler(path)
        
        if not handler:
            # 处理程序未找到，关闭连接
            self.logger.warning(f"未找到WebSocket处理程序: {path}")
            try:
                await websocket.close(code=1003, reason="No handler found")
            except Exception:
                pass
            return
        
        # 储存连接信息
        with self.lock:
            self.websocket_connections[connection_id] = {
                "context": context,
                "websocket": websocket,
                "connected_at": time.time(),
                "message_count": 0,
                "last_message_at": None
            }
        
        try:
            # 调用处理程序
            await handler(websocket, context)
        except Exception as e:
            self.logger.error(f"WebSocket处理异常: {path}, 错误: {str(e)}")
        finally:
            # 清理连接
            with self.lock:
                if connection_id in self.websocket_connections:
                    del self.websocket_connections[connection_id]
    
    def broadcast_message(self, message: Any, path_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        广播消息到WebSocket连接
        
        Args:
            message: 消息内容
            path_filter: 路径过滤器
            
        Returns:
            Dict: 广播结果
        """
        sent_count = 0
        error_count = 0
        
        # 获取当前连接
        with self.lock:
            connections = list(self.websocket_connections.items())
        
        # 发送消息
        for connection_id, connection_info in connections:
            if path_filter and connection_info["context"]["path"] != path_filter:
                continue
                
            websocket = connection_info["websocket"]
            
            try:
                # 创建异步任务发送消息
                asyncio.create_task(websocket.send(json.dumps(message)))
                sent_count += 1
            except Exception as e:
                self.logger.error(f"发送WebSocket消息失败: {connection_id}, 错误: {str(e)}")
                error_count += 1
        
        return {
            "status": "success",
            "sent_count": sent_count,
            "error_count": error_count,
            "total_connections": len(self.websocket_connections)
        }
    
    def call_component_method(self, component_id: str, method_name: str, 
                           args: List[Any] = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用系统组件方法
        
        Args:
            component_id: 组件ID
            method_name: 方法名
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            Dict: 调用结果
        """
        if not self.system_integrator:
            return {
                "status": "error",
                "message": "系统集成器未设置"
            }
            
        # 调用系统集成器的方法
        return self.system_integrator.call_component_method(component_id, method_name, args, kwargs)
    
    def get_api_docs(self) -> Dict[str, Any]:
        """
        获取API文档
        
        Returns:
            Dict: API文档
        """
        with self.lock:
            return {
                "status": "success",
                "endpoints": list(self.api_docs.values()),
                "total_endpoints": len(self.api_docs)
            }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        获取请求统计
        
        Returns:
            Dict: 请求统计
        """
        with self.lock:
            # 计算每个路由的平均响应时间
            route_stats = {}
            
            for path, stats in self.request_stats["route_stats"].items():
                count = stats["count"]
                if count > 0:
                    avg_time = stats["total_time"] / count
                    success_rate = stats["success_count"] / count
                else:
                    avg_time = 0
                    success_rate = 0
                    
                route_stats[path] = {
                    "count": count,
                    "avg_response_time": avg_time,
                    "success_rate": success_rate
                }
            
            # 总体统计
            total_requests = self.request_stats["total_requests"]
            
            if total_requests > 0:
                success_rate = self.request_stats["successful_requests"] / total_requests
            else:
                success_rate = 0
                
            return {
                "status": "success",
                "timestamp": time.time(),
                "total_requests": total_requests,
                "success_rate": success_rate,
                "average_response_time": self.request_stats["average_response_time"],
                "routes": route_stats
            }
    
    def _match_route(self, path: str, method: str) -> Optional[tuple]:
        """匹配路由"""
        # 精确匹配
        route_key = (path, (method,))
        if route_key in self.routes:
            return route_key
            
        # 方法匹配
        for (route_path, route_methods), _ in self.routes.items():
            if route_path == path and method in route_methods:
                return (route_path, route_methods)
                
        # TODO: 添加更复杂的路由匹配（如路径参数）
        
        return None
    
    def _match_websocket_handler(self, path: str) -> Optional[Callable]:
        """匹配WebSocket处理程序"""
        # TODO: 实现WebSocket路由匹配
        return None
    
    def _update_request_stats(self, context: Dict[str, Any], response_time: float):
        """更新请求统计"""
        with self.lock:
            # 更新总体统计
            self.request_stats["total_requests"] += 1
            
            if not context.get("error"):
                self.request_stats["successful_requests"] += 1
            else:
                self.request_stats["failed_requests"] += 1
                
            # 更新平均响应时间
            total_requests = self.request_stats["total_requests"]
            current_avg = self.request_stats["average_response_time"]
            
            self.request_stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # 更新路由统计
            if context.get("route_config"):
                path = context["route_config"]["path"]
                self.request_stats["route_stats"][path]["count"] += 1
                self.request_stats["route_stats"][path]["total_time"] += response_time
                
                if not context.get("error"):
                    self.request_stats["route_stats"][path]["success_count"] += 1
                    
            # 添加到请求日志
            self.request_log.append({
                "request_id": context["request_id"],
                "path": context["path"],
                "method": context["method"],
                "status": "success" if not context.get("error") else "error",
                "status_code": context.get("error", {}).get("status", 200),
                "response_time": response_time,
                "timestamp": time.time(),
                "client_ip": context.get("client_info", {}).get("ip")
            })
    
    def _generate_route_docs(self, route_key: tuple, route_config: Dict[str, Any]):
        """生成路由文档"""
        path, methods = route_key
        
        # 获取处理程序文档
        handler = route_config["handler"]
        doc = handler.__doc__ or "No documentation available"
        
        # 创建文档条目
        doc_entry = {
            "path": path,
            "methods": list(methods),
            "description": doc.strip(),
            "auth_required": route_config.get("auth_required", False),
            "rate_limited": route_config.get("rate_limit") is not None
        }
        
        # 添加其他信息
        for key, value in route_config.items():
            if key not in ["path", "methods", "handler", "registered_at"]:
                doc_entry[key] = value
        
        # 保存文档
        self.api_docs[path] = doc_entry
    
    # 中间件实现
    def _auth_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """认证中间件"""
        # 标记为前置中间件
        _auth_middleware.__pre__ = True
        _auth_middleware.__post__ = False
        
        # 检查路由是否需要认证
        route_config = context.get("route_config")
        
        if not route_config or not route_config.get("auth_required", False):
            return context
            
        # 获取认证头
        auth_header = context["headers"].get("Authorization")
        
        if not auth_header:
            context["error"] = {
                "status": 401,
                "message": "缺少认证信息"
            }
            return context
            
        # 解析认证类型和凭证
        try:
            auth_type, auth_cred = auth_header.split(" ", 1)
        except ValueError:
            context["error"] = {
                "status": 401,
                "message": "认证格式无效"
            }
            return context
            
        # 查找认证提供者
        provider = self.auth_providers.get(auth_type.lower())
        
        if not provider:
            context["error"] = {
                "status": 401,
                "message": f"不支持的认证类型: {auth_type}"
            }
            return context
            
        # 验证凭证
        try:
            auth_result = provider(auth_cred, context)
            
            if not auth_result or not auth_result.get("authenticated", False):
                context["error"] = {
                    "status": 401,
                    "message": auth_result.get("message", "认证失败")
                }
                return context
                
            # 设置认证信息
            context["auth"] = auth_result
        except Exception as e:
            self.logger.error(f"认证异常: {str(e)}")
            context["error"] = {
                "status": 401,
                "message": f"认证处理错误: {str(e)}"
            }
            
        return context
    
    def _rate_limit_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """速率限制中间件"""
        # 标记为前置中间件
        _rate_limit_middleware.__pre__ = True
        _rate_limit_middleware.__post__ = False
        
        # 检查路由是否有速率限制
        route_config = context.get("route_config")
        
        if not route_config or not route_config.get("rate_limit"):
            return context
            
        rate_limit_config = route_config["rate_limit"]
        limiter_name = rate_limit_config.get("limiter", "default")
        
        # 查找限制器
        limiter = self.rate_limiters.get(limiter_name)
        
        if not limiter:
            # 没有限制器，继续处理
            return context
            
        # 获取限制键
        key_func = rate_limit_config.get("key_func")
        
        if key_func:
            key = key_func(context)
        else:
            # 默认使用客户端IP作为键
            key = context.get("client_info", {}).get("ip", "unknown")
            
        # 检查是否超过限制
        if not limiter.check(key):
            context["error"] = {
                "status": 429,
                "message": "请求过于频繁，请稍后再试"
            }
            
        return context
    
    def _logging_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """日志中间件"""
        # 标记为前置和后置中间件
        _logging_middleware.__pre__ = True
        _logging_middleware.__post__ = True
        
        # 前置日志
        if not context.get("_logged_pre"):
            self.logger.info(f"收到请求: {context['method']} {context['path']} (ID: {context['request_id']})")
            context["_logged_pre"] = True
            return context
            
        # 后置日志
        if context.get("error"):
            self.logger.info(f"请求失败: {context['method']} {context['path']} "
                           f"(ID: {context['request_id']}, 状态: {context['error']['status']})")
        else:
            self.logger.info(f"请求成功: {context['method']} {context['path']} "
                           f"(ID: {context['request_id']}, 用时: {time.time() - context['start_time']:.3f}s)")
            
        return context
    
    def _error_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """错误处理中间件"""
        # 标记为后置中间件
        _error_middleware.__pre__ = False
        _error_middleware.__post__ = True
        
        # 确保错误有状态码
        if context.get("error") and "status" not in context["error"]:
            context["error"]["status"] = 500
            
        return context
    
    # 默认路由处理程序
    def _health_check_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """健康检查处理程序"""
        return {
            "status": "success",
            "message": "服务正常运行",
            "timestamp": time.time()
        }
    
    def _api_docs_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """API文档处理程序"""
        return self.get_api_docs()
    
    def _system_status_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """系统状态处理程序"""
        if not self.system_integrator:
            return {
                "status": "warning",
                "message": "系统集成器未设置，无法获取完整状态"
            }
            
        return self.system_integrator.get_system_status()
    
    def _api_stats_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """API统计处理程序"""
        return self.get_request_stats()


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests: int = 60, time_window: int = 60, key_func=None):
        """
        初始化速率限制器
        
        Args:
            requests: 时间窗口内允许的最大请求数
            time_window: 时间窗口（秒）
            key_func: 自定义限制键函数
        """
        self.requests = requests
        self.time_window = time_window
        self.key_func = key_func
        
        # 请求计数器
        self.counters = {}  # {key: [(timestamp, count)]}
        
        # 线程锁
        self.lock = threading.RLock()
    
    def check(self, key: str) -> bool:
        """
        检查是否超过限制
        
        Args:
            key: 限制键
            
        Returns:
            bool: 是否允许请求
        """
        with self.lock:
            current_time = time.time()
            
            # 初始化计数器
            if key not in self.counters:
                self.counters[key] = []
                
            # 清理过期计数
            self._clean_expired(key, current_time)
            
            # 计算当前窗口请求数
            current_count = sum(count for ts, count in self.counters[key])
            
            # 检查是否超过限制
            if current_count >= self.requests:
                return False
                
            # 增加计数
            self.counters[key].append((current_time, 1))
            
            return True
    
    def _clean_expired(self, key: str, current_time: float):
        """清理过期计数"""
        cutoff_time = current_time - self.time_window
        self.counters[key] = [(ts, count) for ts, count in self.counters[key] if ts >= cutoff_time] 