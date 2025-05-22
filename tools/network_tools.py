# -*- coding: utf-8 -*-
"""
网络工具模块 (Network Tools)

提供网络请求、API调用和数据获取功能
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from .tool_executor import tool

@tool(
    name="web_request",
    description="发送HTTP请求并获取响应",
    required_params=["url"],
    optional_params={"method": "GET", "headers": {}, "data": None, "timeout": 30}
)
def web_request(url: str, method: str = "GET", headers: Dict[str, str] = None, 
               data: Any = None, timeout: int = 30) -> Dict[str, Any]:
    """
    发送HTTP请求并获取响应
    
    Args:
        url: 请求URL
        method: 请求方法 (GET, POST, PUT, DELETE等)
        headers: 请求头
        data: 请求数据
        timeout: 超时时间(秒)
        
    Returns:
        Dict: 包含响应信息的字典
    """
    headers = headers or {}
    method = method.upper()
    start_time = time.time()
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            timeout=timeout
        )
        
        # 尝试解析响应内容
        try:
            content = response.json()
            content_type = "json"
        except:
            content = response.text
            content_type = "text"
            
        return {
            "status": "success",
            "status_code": response.status_code,
            "content": content,
            "content_type": content_type,
            "headers": dict(response.headers),
            "execution_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "execution_time": time.time() - start_time
        }

@tool(
    name="api_call",
    description="发送API请求并处理响应",
    required_params=["api_url"],
    optional_params={"method": "GET", "params": {}, "headers": {}, "body": None, "auth_token": None}
)
def api_call(api_url: str, method: str = "GET", params: Dict[str, Any] = None, 
            headers: Dict[str, str] = None, body: Any = None, 
            auth_token: Optional[str] = None) -> Dict[str, Any]:
    """
    发送API请求并处理响应
    
    Args:
        api_url: API端点URL
        method: 请求方法
        params: URL参数
        headers: 请求头
        body: 请求体
        auth_token: 认证令牌
        
    Returns:
        Dict: API响应
    """
    headers = headers or {}
    params = params or {}
    
    # 如果提供了认证令牌，添加到请求头
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        
    # 如果body是字典，转换为JSON
    if isinstance(body, dict):
        headers["Content-Type"] = "application/json"
        data = json.dumps(body)
    else:
        data = body
        
    # 调用web_request工具
    return web_request(
        url=api_url,
        method=method,
        headers=headers,
        data=data
    )

@tool(
    name="fetch_weather",
    description="获取指定城市的天气信息",
    required_params=["city"],
    optional_params={"country": "CN", "api_key": None}
)
def fetch_weather(city: str, country: str = "CN", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        country: 国家代码
        api_key: OpenWeatherMap API密钥
        
    Returns:
        Dict: 天气信息
    """
    # 这里使用OpenWeatherMap API示例
    # 实际使用时需要提供有效的API密钥
    if not api_key:
        return {
            "status": "error",
            "message": "需要提供有效的API密钥"
        }
        
    api_url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country}",
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        return {
            "status": "success",
            "city": city,
            "country": country,
            "temperature": weather_data["main"]["temp"],
            "feels_like": weather_data["main"]["feels_like"],
            "humidity": weather_data["main"]["humidity"],
            "pressure": weather_data["main"]["pressure"],
            "weather": weather_data["weather"][0]["main"],
            "description": weather_data["weather"][0]["description"],
            "wind_speed": weather_data["wind"]["speed"],
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@tool(
    name="search_wikipedia",
    description="在维基百科搜索信息",
    required_params=["query"],
    optional_params={"language": "zh", "limit": 5}
)
def search_wikipedia(query: str, language: str = "zh", limit: int = 5) -> Dict[str, Any]:
    """
    在维基百科搜索信息
    
    Args:
        query: 搜索查询
        language: 语言代码
        limit: 结果数量限制
        
    Returns:
        Dict: 搜索结果
    """
    # 构建维基百科API URL
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    
    # 搜索请求
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }
    
    try:
        search_response = requests.get(api_url, params=search_params)
        search_data = search_response.json()
        
        # 处理搜索结果
        results = []
        for item in search_data.get("query", {}).get("search", []):
            # 获取页面摘要
            summary_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "pageids": item["pageid"],
                "format": "json"
            }
            
            summary_response = requests.get(api_url, params=summary_params)
            summary_data = summary_response.json()
            
            # 提取摘要文本
            page_id = str(item["pageid"])
            extract = summary_data.get("query", {}).get("pages", {}).get(page_id, {}).get("extract", "")
            
            results.append({
                "title": item["title"],
                "page_id": item["pageid"],
                "url": f"https://{language}.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                "summary": extract[:500] + ("..." if len(extract) > 500 else "")
            })
            
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_hits": search_data.get("query", {}).get("searchinfo", {}).get("totalhits", 0),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        } 