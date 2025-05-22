"""
标准化信息流协议 (Standardized Information Protocol)

定义系统各组件间的通信格式，确保信息流通的一致性和互操作性。
提供消息编码、解码、验证和路由功能。
"""

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import hashlib

class MessageType(Enum):
    """消息类型枚举"""
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    DATA = "data"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    
class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    
class MessageStatus(Enum):
    """消息状态枚举"""
    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class InformationProtocol:
    """标准化信息流协议，定义系统组件间通信格式"""
    
    @staticmethod
    def create_message(
        message_type: Union[MessageType, str],
        source: str,
        target: str,
        content: Dict[str, Any],
        priority: Union[MessagePriority, int] = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建标准格式消息
        
        Args:
            message_type: 消息类型
            source: 发送方ID
            target: 接收方ID
            content: 消息内容
            priority: 消息优先级
            correlation_id: 关联ID（用于关联请求和响应）
            metadata: 元数据
            
        Returns:
            Dict: 标准格式消息
        """
        # 转换枚举类型
        if isinstance(message_type, MessageType):
            message_type = message_type.value
            
        if isinstance(priority, MessagePriority):
            priority = priority.value
            
        # 生成消息ID
        message_id = str(uuid.uuid4())
        
        # 创建消息
        message = {
            "protocol_version": "1.0",
            "id": message_id,
            "correlation_id": correlation_id or message_id,
            "type": message_type,
            "source": source,
            "target": target,
            "timestamp": time.time(),
            "priority": priority,
            "status": MessageStatus.CREATED.value,
            "content": content,
            "metadata": metadata or {}
        }
        
        # 计算消息摘要
        message["digest"] = InformationProtocol._calculate_digest(message)
        
        return message
    
    @staticmethod
    def create_response(
        request: Dict[str, Any],
        content: Dict[str, Any],
        status: Union[MessageStatus, str] = MessageStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        为请求创建响应消息
        
        Args:
            request: 请求消息
            content: 响应内容
            status: 响应状态
            metadata: 元数据
            
        Returns:
            Dict: 响应消息
        """
        # 转换枚举类型
        if isinstance(status, MessageStatus):
            status = status.value
            
        # 创建响应
        response = InformationProtocol.create_message(
            message_type=MessageType.RESPONSE,
            source=request["target"],
            target=request["source"],
            content=content,
            correlation_id=request["id"],
            priority=request.get("priority", MessagePriority.NORMAL.value),
            metadata=metadata or {}
        )
        
        # 设置状态
        response["status"] = status
        
        # 更新摘要
        response["digest"] = InformationProtocol._calculate_digest(response)
        
        return response
    
    @staticmethod
    def create_error_response(
        request: Dict[str, Any],
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        为请求创建错误响应
        
        Args:
            request: 请求消息
            error_code: 错误代码
            error_message: 错误消息
            error_details: 错误详情
            metadata: 元数据
            
        Returns:
            Dict: 错误响应消息
        """
        # 创建错误内容
        error_content = {
            "error": {
                "code": error_code,
                "message": error_message,
                "details": error_details or {}
            },
            "original_request": {
                "id": request["id"],
                "type": request["type"],
                "content_summary": InformationProtocol._summarize_content(request["content"])
            }
        }
        
        # 创建响应
        response = InformationProtocol.create_message(
            message_type=MessageType.ERROR,
            source=request["target"],
            target=request["source"],
            content=error_content,
            correlation_id=request["id"],
            priority=request.get("priority", MessagePriority.NORMAL.value),
            metadata=metadata or {}
        )
        
        # 设置状态
        response["status"] = MessageStatus.FAILED.value
        
        # 更新摘要
        response["digest"] = InformationProtocol._calculate_digest(response)
        
        return response
    
    @staticmethod
    def create_event(
        source: str,
        event_type: str,
        event_data: Dict[str, Any],
        targets: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建事件消息
        
        Args:
            source: 发送方ID
            event_type: 事件类型
            event_data: 事件数据
            targets: 目标接收方列表
            metadata: 元数据
            
        Returns:
            Dict: 事件消息
        """
        # 创建事件内容
        content = {
            "event_type": event_type,
            "event_data": event_data
        }
        
        # 创建事件消息
        event = InformationProtocol.create_message(
            message_type=MessageType.EVENT,
            source=source,
            target=targets[0] if targets and len(targets) == 1 else "broadcast",
            content=content,
            metadata=metadata or {}
        )
        
        # 如果有多个目标，添加到元数据
        if targets and len(targets) > 1:
            event["metadata"]["targets"] = targets
            
        return event
    
    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """
        验证消息格式是否有效
        
        Args:
            message: 要验证的消息
            
        Returns:
            bool: 消息是否有效
        """
        # 检查必要字段
        required_fields = ["protocol_version", "id", "type", "source", "target", 
                          "timestamp", "priority", "status", "content", "digest"]
                          
        for field in required_fields:
            if field not in message:
                return False
                
        # 验证摘要
        original_digest = message["digest"]
        message_copy = dict(message)
        message_copy["digest"] = ""
        
        calculated_digest = InformationProtocol._calculate_digest(message_copy)
        
        return original_digest == calculated_digest
    
    @staticmethod
    def _calculate_digest(message: Dict[str, Any]) -> str:
        """计算消息摘要"""
        # 创建消息副本，移除digest字段
        message_copy = dict(message)
        message_copy.pop("digest", None)
        
        # 将消息转换为排序后的JSON
        message_json = json.dumps(message_copy, sort_keys=True)
        
        # 计算SHA-256哈希
        digest = hashlib.sha256(message_json.encode()).hexdigest()
        
        return digest
    
    @staticmethod
    def _summarize_content(content: Dict[str, Any], max_length: int = 100) -> Dict[str, Any]:
        """创建内容摘要，限制长字符串的长度"""
        summary = {}
        
        for key, value in content.items():
            if isinstance(value, dict):
                summary[key] = InformationProtocol._summarize_content(value, max_length)
            elif isinstance(value, list):
                if value and len(value) > 3:
                    summary[key] = f"[{len(value)} items]"
                else:
                    summary[key] = value
            elif isinstance(value, str) and len(value) > max_length:
                summary[key] = value[:max_length] + "..."
            else:
                summary[key] = value
                
        return summary
    
    @staticmethod
    def encode_message(message: Dict[str, Any]) -> str:
        """将消息编码为JSON字符串"""
        return json.dumps(message)
    
    @staticmethod
    def decode_message(message_str: str) -> Dict[str, Any]:
        """将JSON字符串解码为消息"""
        return json.loads(message_str)
    
    @staticmethod
    def route_message(message: Dict[str, Any], routing_table: Dict[str, str]) -> Dict[str, Any]:
        """
        根据路由表更新消息路由
        
        Args:
            message: 要路由的消息
            routing_table: 路由表 {目标ID: 实际目标ID}
            
        Returns:
            Dict: 更新路由后的消息
        """
        # 检查目标是否在路由表中
        if message["target"] in routing_table:
            message["metadata"]["original_target"] = message["target"]
            message["target"] = routing_table[message["target"]]
            
        return message
    
    @staticmethod
    def filter_sensitive_data(message: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """
        过滤消息中的敏感数据
        
        Args:
            message: 原始消息
            sensitive_fields: 敏感字段列表
            
        Returns:
            Dict: 过滤后的消息
        """
        filtered_message = dict(message)
        content = dict(filtered_message["content"])
        
        for field in sensitive_fields:
            if field in content:
                content[field] = "***REDACTED***"
                
        filtered_message["content"] = content
        
        # 更新摘要
        filtered_message["digest"] = InformationProtocol._calculate_digest(filtered_message)
        
        return filtered_message 