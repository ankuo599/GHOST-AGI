"""
认知过程监控模块 (Cognitive Monitor)

该模块负责监控和评估推理过程，识别认知偏差，评估结果置信度。
"""

import time
import uuid
import json
import logging
import sys
from typing import Dict, List, Any, Optional, Union, Set
from collections import defaultdict
import numpy as np

class CognitiveMonitor:
    def __init__(self, event_system=None):
        """初始化认知监控器"""
        self.event_system = event_system
        self.cognitive_biases = self._initialize_cognitive_biases()
        self.knowledge_base = {
            "concepts": {},
            "skills": {},
            "patterns": {},
            "relationships": {}
        }
        self.learning_queue = []
        self.active_learning_tasks = {}
        self.learning_history = []
        self.performance_metrics = {
            "reasoning_success_rate": 0.0,
            "average_confidence": 0.0,
            "learning_efficiency": 0.0,
            "bias_detection_rate": 0.0
        }
        self.metacognitive_state = {
            "awareness_level": 0.0,
            "self_regulation": 0.0,
            "learning_progress": 0.0,
            "adaptation_capability": 0.0
        }
        self.action_capabilities = {
            "tool_usage": False,
            "environment_interaction": False,
            "task_execution": False,
            "autonomous_decision": False
        }
        self.action_history = []
        self.api_clients = {}
        self.resource_managers = {}
        
        # 自主初始化
        self._self_initialize()
        
    def _self_initialize(self):
        """自主初始化系统"""
        try:
            # 初始化API客户端
            self._initialize_api_clients()
            
            # 初始化资源管理器
            self._initialize_resource_managers()
            
            # 初始化学习资源
            self._initialize_learning_resources()
            
            # 订阅事件
            self._subscribe_to_events()
            
            # 启动自主监控
            self._start_autonomous_monitoring()
            
        except Exception as e:
            self._handle_initialization_error(e)
            
    def _initialize_api_clients(self):
        """初始化API客户端"""
        # 自动发现和注册API
        self._discover_available_apis()
        
        # 尝试获取API密钥
        self._acquire_api_keys()
        
        # 初始化API客户端
        self._setup_api_clients()
        
    def _discover_available_apis(self):
        """发现可用的API"""
        # 检查环境变量
        self._check_environment_variables()
        
        # 检查配置文件
        self._check_config_files()
        
        # 检查网络可用性
        self._check_network_availability()
        
    def _acquire_api_keys(self):
        """获取API密钥"""
        # 尝试从环境变量获取
        self._get_keys_from_env()
        
        # 尝试从配置文件获取
        self._get_keys_from_config()
        
        # 尝试从密钥管理器获取
        self._get_keys_from_manager()
        
        # 如果都没有,尝试自动申请
        self._request_new_api_keys()
        
    def _setup_api_clients(self):
        """设置API客户端"""
        for api_name, api_config in self.api_clients.items():
            try:
                self._initialize_single_api_client(api_name, api_config)
            except Exception as e:
                self._handle_api_setup_error(api_name, e)
                
    def _initialize_resource_managers(self):
        """初始化资源管理器"""
        # 初始化本地资源管理器
        self._init_local_resource_manager()
        
        # 初始化网络资源管理器
        self._init_network_resource_manager()
        
        # 初始化缓存管理器
        self._init_cache_manager()
        
    def _start_autonomous_monitoring(self):
        """启动自主监控"""
        # 启动性能监控
        self._start_performance_monitoring()
        
        # 启动资源监控
        self._start_resource_monitoring()
        
        # 启动学习监控
        self._start_learning_monitor()
        
        # 启动错误监控
        self._start_error_monitoring()
        
    def _handle_initialization_error(self, error):
        """处理初始化错误"""
        # 记录错误
        self._log_error(error)
        
        # 尝试恢复
        self._attempt_recovery()
        
        # 如果恢复失败,启动降级模式
        if not self._is_recovery_successful():
            self._start_degraded_mode()
            
    def _attempt_recovery(self):
        """尝试恢复"""
        # 检查系统状态
        self._check_system_state()
        
        # 尝试修复问题
        self._attempt_fixes()
        
        # 验证修复结果
        self._verify_fixes()
        
    def _start_degraded_mode(self):
        """启动降级模式"""
        # 禁用非核心功能
        self._disable_non_critical_features()
        
        # 启动基本监控
        self._start_basic_monitoring()
        
        # 通知用户
        self._notify_degraded_mode()
        
    def _check_system_state(self):
        """检查系统状态"""
        # 检查核心组件
        self._check_core_components()
        
        # 检查资源状态
        self._check_resource_status()
        
        # 检查网络连接
        self._check_network_connection()
        
    def _attempt_fixes(self):
        """尝试修复问题"""
        # 尝试重新初始化组件
        self._reinitialize_components()
        
        # 尝试重新连接服务
        self._reconnect_services()
        
        # 尝试清理资源
        self._cleanup_resources()
        
    def _verify_fixes(self):
        """验证修复结果"""
        # 验证组件状态
        self._verify_components()
        
        # 验证服务连接
        self._verify_connections()
        
        # 验证资源状态
        self._verify_resources()
        
    def _is_recovery_successful(self):
        """检查恢复是否成功"""
        # 检查核心功能
        core_ok = self._check_core_functionality()
        
        # 检查资源状态
        resources_ok = self._check_resource_health()
        
        # 检查服务连接
        services_ok = self._check_service_connections()
        
        return core_ok and resources_ok and services_ok
    
    def _subscribe_to_events(self):
        """订阅相关事件"""
        self.event_system.subscribe("reasoning.started", self._handle_reasoning_start)
        self.event_system.subscribe("reasoning.step_completed", self._handle_reasoning_step)
        self.event_system.subscribe("reasoning.completed", self._handle_reasoning_complete)
        self.event_system.subscribe("learning.inference_result", self._handle_inference_result)
        
    def _initialize_cognitive_biases(self) -> Dict[str, Dict[str, Any]]:
        """初始化认知偏差库"""
        return {
            "confirmation_bias": {
                "description": "倾向于寻找支持已有观点的证据，忽视反对证据",
                "detection_patterns": [
                    "确认", "验证", "支持", "证明", "证实",
                    "agree", "confirm", "support", "prove"
                ],
                "mitigation": "主动寻找反对证据，考虑多种可能性",
                "severity_threshold": 0.7
            },
            "availability_bias": {
                "description": "基于容易获取的信息做判断，忽视其他相关信息",
                "detection_patterns": [
                    "容易", "熟悉", "立即", "快速", "简单",
                    "easy", "familiar", "quick", "simple"
                ],
                "mitigation": "扩大信息搜索范围，考虑更多来源",
                "severity_threshold": 0.6
            },
            "anchoring_bias": {
                "description": "过度依赖首次接收的信息作为判断基准",
                "detection_patterns": [
                    "初始", "锚定", "第一", "首先", "开始",
                    "initial", "first", "start", "begin"
                ],
                "mitigation": "考虑多个参考点，避免单一基准",
                "severity_threshold": 0.8
            },
            "hindsight_bias": {
                "description": "事后认为事件是可预测的",
                "detection_patterns": [
                    "应该", "本来", "早知道", "预测", "预见",
                    "should", "predict", "foresee", "expect"
                ],
                "mitigation": "记录事前预测，避免事后合理化",
                "severity_threshold": 0.7
            },
            "overconfidence_bias": {
                "description": "对自身能力和判断过度自信",
                "detection_patterns": [
                    "确定", "肯定", "绝对", "一定", "必然",
                    "certain", "definite", "absolute", "sure"
                ],
                "mitigation": "保持适度怀疑，考虑不确定性",
                "severity_threshold": 0.8
            },
            "framing_effect": {
                "description": "决策受问题表述方式影响",
                "detection_patterns": [
                    "表述", "描述", "表达", "方式", "形式",
                    "frame", "express", "describe", "present"
                ],
                "mitigation": "尝试多种表述方式，避免单一视角",
                "severity_threshold": 0.7
            },
            "sunk_cost_fallacy": {
                "description": "因已投入资源而继续不合理行为",
                "detection_patterns": [
                    "已投入", "继续", "放弃", "损失", "浪费",
                    "invest", "continue", "abandon", "waste"
                ],
                "mitigation": "关注未来收益，而非已投入成本",
                "severity_threshold": 0.8
            },
            "recency_bias": {
                "description": "过度重视最近的信息",
                "detection_patterns": [
                    "最近", "最新", "刚刚", "刚才", "新",
                    "recent", "latest", "new", "just"
                ],
                "mitigation": "平衡考虑历史信息和最新信息",
                "severity_threshold": 0.6
            },
            "negativity_bias": {
                "description": "过度关注负面信息",
                "detection_patterns": [
                    "问题", "风险", "失败", "错误", "缺点",
                    "problem", "risk", "failure", "error"
                ],
                "mitigation": "平衡考虑正面和负面信息",
                "severity_threshold": 0.7
            },
            "optimism_bias": {
                "description": "过度乐观估计结果",
                "detection_patterns": [
                    "乐观", "积极", "希望", "相信", "期待",
                    "optimistic", "positive", "hope", "believe"
                ],
                "mitigation": "考虑最坏情况，做好风险准备",
                "severity_threshold": 0.7
            }
        }
    
    def track_reasoning_process(self, reasoning_id: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        跟踪推理过程
        
        Args:
            reasoning_id: 推理过程ID
            steps: 推理步骤列表
            
        Returns:
            Dict: 包含跟踪结果和分析的字典
        """
        if not reasoning_id:
            reasoning_id = str(uuid.uuid4())
            
        # 记录推理轨迹
        trace = {
            "reasoning_id": reasoning_id,
            "timestamp": time.time(),
            "steps": steps,
            "step_count": len(steps),
            "completion_status": "completed" if steps else "unknown",
            "context": self._capture_reasoning_context(),
            "metadata": self._generate_metadata()
        }
        
        # 分析推理质量
        analysis = self._analyze_reasoning_quality(steps)
        trace["quality_analysis"] = analysis
        
        # 识别决策点
        decision_points = self._identify_decision_points(steps)
        trace["decision_points"] = decision_points
        
        # 检测认知偏差
        biases = self.detect_cognitive_biases(steps)
        trace["cognitive_biases"] = biases
        
        # 计算整体推理置信度
        confidence = self._calculate_reasoning_confidence(steps, analysis, biases)
        trace["confidence"] = confidence
        
        # 分析推理链路
        reasoning_chain = self._analyze_reasoning_chain(steps)
        trace["reasoning_chain"] = reasoning_chain
        
        # 评估推理效率
        efficiency = self._evaluate_reasoning_efficiency(steps)
        trace["efficiency_metrics"] = efficiency
        
        # 添加到跟踪历史
        self.cognitive_traces.append(trace)
        
        # 如果有事件系统，发布跟踪完成事件
        if self.event_system:
            self.event_system.publish("metacognition.trace_recorded", {
                "reasoning_id": reasoning_id,
                "confidence": confidence,
                "step_count": len(steps),
                "has_biases": len(biases) > 0,
                "efficiency_score": efficiency.get("overall_score", 0)
            })
            
        return {
            "status": "success",
            "reasoning_id": reasoning_id,
            "confidence": confidence,
            "quality_score": analysis.get("overall_score", 0),
            "detected_biases": [b["bias_type"] for b in biases] if biases else [],
            "key_decision_points": len(decision_points),
            "efficiency_score": efficiency.get("overall_score", 0)
        }
    
    def _capture_reasoning_context(self) -> Dict[str, Any]:
        """捕获推理上下文信息"""
        return {
            "timestamp": time.time(),
            "active_tasks": self._get_active_tasks(),
            "active_concepts": self._get_active_concepts(),
            "focus_duration": self._get_current_focus_duration(),
            "cognitive_load": self._estimate_cognitive_load(),
            "context_tags": self._get_current_context()
        }
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """生成推理过程的元数据"""
        return {
            "version": "1.0",
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version,
                "timestamp": time.time()
            },
            "monitor_config": {
                "bias_detection_enabled": True,
                "quality_analysis_enabled": True,
                "efficiency_tracking_enabled": True
            }
        }
    
    def _analyze_reasoning_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析推理链路"""
        chain = {
            "nodes": [],
            "edges": [],
            "strength": 0.0,
            "completeness": 0.0
        }
        
        # 构建推理节点
        for i, step in enumerate(steps):
            node = {
                "id": f"step_{i}",
                "type": step.get("type", "unknown"),
                "content": step.get("description", ""),
                "output": step.get("output", ""),
                "confidence": step.get("confidence", 0.5)
            }
            chain["nodes"].append(node)
            
            # 构建推理边
            if i > 0:
                edge = {
                    "from": f"step_{i-1}",
                    "to": f"step_{i}",
                    "type": "follows",
                    "strength": self._calculate_step_connection(steps[i-1], step)
                }
                chain["edges"].append(edge)
        
        # 计算链路强度
        if chain["edges"]:
            chain["strength"] = sum(edge["strength"] for edge in chain["edges"]) / len(chain["edges"])
        
        # 计算链路完整性
        expected_components = ["problem_definition", "analysis", "conclusion"]
        found_components = set(step.get("type", "") for step in steps)
        chain["completeness"] = len(found_components & set(expected_components)) / len(expected_components)
        
        return chain
    
    def _calculate_step_connection(self, prev_step: Dict[str, Any], curr_step: Dict[str, Any]) -> float:
        """计算推理步骤之间的连接强度"""
        # 检查显式依赖
        if curr_step.get("depends_on") == prev_step.get("id"):
            return 1.0
            
        # 检查内容关联
        prev_output = str(prev_step.get("output", "")).lower()
        curr_description = str(curr_step.get("description", "")).lower()
        
        # 计算词重叠度
        prev_words = set(prev_output.split())
        curr_words = set(curr_description.split())
        overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
        
        return min(1.0, overlap * 1.5)  # 放大重叠度的影响
    
    def _evaluate_reasoning_efficiency(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估推理效率"""
        if not steps:
            return {
                "overall_score": 0.0,
                "step_efficiency": 0.0,
                "time_efficiency": 0.0,
                "resource_efficiency": 0.0
            }
        
        # 计算步骤效率
        step_efficiency = self._calculate_step_efficiency(steps)
        
        # 计算时间效率
        time_efficiency = self._calculate_time_efficiency(steps)
        
        # 计算资源效率
        resource_efficiency = self._calculate_resource_efficiency(steps)
        
        # 计算总体效率
        overall_score = (
            step_efficiency * 0.4 +
            time_efficiency * 0.3 +
            resource_efficiency * 0.3
        )
        
        return {
            "overall_score": overall_score,
            "step_efficiency": step_efficiency,
            "time_efficiency": time_efficiency,
            "resource_efficiency": resource_efficiency
        }
    
    def _calculate_step_efficiency(self, steps: List[Dict[str, Any]]) -> float:
        """计算步骤效率"""
        if not steps:
            return 0.0
            
        # 计算有效步骤比例
        effective_steps = sum(1 for step in steps if step.get("output") and len(str(step.get("output"))) > 10)
        step_ratio = effective_steps / len(steps)
        
        # 计算步骤间的连贯性
        coherence = 0.0
        for i in range(1, len(steps)):
            coherence += self._calculate_step_connection(steps[i-1], steps[i])
        coherence = coherence / (len(steps) - 1) if len(steps) > 1 else 0.0
        
        return (step_ratio * 0.6 + coherence * 0.4)
    
    def _calculate_time_efficiency(self, steps: List[Dict[str, Any]]) -> float:
        """计算时间效率"""
        if not steps:
            return 0.0
            
        # 计算平均步骤时间
        total_time = sum(step.get("duration", 0) for step in steps)
        avg_time = total_time / len(steps)
        
        # 根据平均时间计算效率（假设理想平均时间为1秒）
        time_efficiency = 1.0 / (1.0 + avg_time)
        
        return time_efficiency
    
    def _calculate_resource_efficiency(self, steps: List[Dict[str, Any]]) -> float:
        """计算资源效率"""
        if not steps:
            return 0.0
            
        # 计算资源使用情况
        memory_usage = sum(step.get("memory_usage", 0) for step in steps) / len(steps)
        cpu_usage = sum(step.get("cpu_usage", 0) for step in steps) / len(steps)
        
        # 计算资源效率（越低越好）
        resource_efficiency = 1.0 - ((memory_usage + cpu_usage) / 2.0)
        
        return max(0.0, min(1.0, resource_efficiency))
    
    def _analyze_reasoning_quality(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析推理质量
        
        Args:
            steps: 推理步骤列表
            
        Returns:
            Dict: 质量分析结果
        """
        if not steps:
            return {
                "overall_score": 0,
                "completeness": 0,
                "coherence": 0,
                "evidence_quality": 0,
                "message": "无推理步骤可分析"
            }
            
        # 分析完整性 - 是否有明确的问题定义、分析和结论
        has_problem_definition = False
        has_analysis = False
        has_conclusion = False
        
        for step in steps:
            step_type = step.get("type", "")
            description = step.get("description", "").lower()
            
            if step_type == "problem_definition" or "问题" in description or "定义" in description:
                has_problem_definition = True
            elif step_type == "analysis" or "分析" in description or "评估" in description:
                has_analysis = True
            elif step_type == "conclusion" or "结论" in description or "总结" in description:
                has_conclusion = True
                
        completeness = (int(has_problem_definition) + int(has_analysis) + int(has_conclusion)) / 3
        
        # 分析连贯性 - 步骤之间是否有逻辑连接
        coherence_score = 0
        if len(steps) > 1:
            connected_steps = 0
            for i in range(1, len(steps)):
                prev_step = steps[i-1]
                curr_step = steps[i]
                
                # 检查是否有引用前一步的内容
                curr_description = curr_step.get("description", "").lower()
                prev_output = str(prev_step.get("output", "")).lower()
                
                if (prev_output and any(token in curr_description for token in prev_output.split()[:5])) or \
                   curr_step.get("depends_on") == prev_step.get("id"):
                    connected_steps += 1
                    
            coherence_score = connected_steps / (len(steps) - 1) if len(steps) > 1 else 0
            
        # 分析证据质量 - 是否引用了数据/事实支持推理
        evidence_count = 0
        for step in steps:
            if step.get("evidence") or "证据" in step.get("description", "").lower() or \
               "数据" in step.get("description", "").lower() or "事实" in step.get("description", "").lower():
                evidence_count += 1
                
        evidence_quality = evidence_count / len(steps) if steps else 0
        
        # 计算整体得分
        overall_score = (completeness * 0.4) + (coherence_score * 0.4) + (evidence_quality * 0.2)
        
        return {
            "overall_score": overall_score,
            "completeness": completeness,
            "coherence": coherence_score,
            "evidence_quality": evidence_quality
        }
    
    def _identify_decision_points(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别推理过程中的关键决策点
        
        Args:
            steps: 推理步骤列表
            
        Returns:
            List: 决策点列表
        """
        decision_points = []
        
        for i, step in enumerate(steps):
            # 检查是否是决策点的特征
            is_decision = False
            description = step.get("description", "").lower()
            step_type = step.get("type", "")
            
            # 决策点特征
            if step_type == "decision" or "决策" in description or "选择" in description or "决定" in description:
                is_decision = True
                
            # 检查是否有多个选项/路径
            options = step.get("options", [])
            if options and len(options) > 1:
                is_decision = True
                
            # 检查是否是路径分叉点
            if i < len(steps) - 1:
                next_step = steps[i+1]
                if next_step.get("alternatives") or next_step.get("branching"):
                    is_decision = True
            
            if is_decision:
                decision_points.append({
                    "step_index": i,
                    "step_id": step.get("id", str(i)),
                    "description": step.get("description", "未知决策点"),
                    "options": step.get("options", []),
                    "selected_option": step.get("selected_option", None)
                })
                
        return decision_points
    
    def evaluate_confidence(self, result: Dict[str, Any], process: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估结果置信度
        
        Args:
            result: 推理结果
            process: 推理过程信息
            
        Returns:
            Dict: 置信度评估结果
        """
        # 基础置信度 - 根据结果自身的置信度属性
        base_confidence = result.get("confidence", 0.5)
        
        # 推理质量对置信度的影响
        reasoning_quality = process.get("quality_analysis", {}).get("overall_score", 0)
        quality_factor = reasoning_quality  # 0-1之间
        
        # 认知偏差对置信度的影响
        biases = process.get("cognitive_biases", [])
        bias_penalty = len(biases) * 0.1  # 每个偏差扣0.1分
        bias_factor = max(0, 1 - bias_penalty)
        
        # 证据充分性对置信度的影响
        evidence_quality = process.get("quality_analysis", {}).get("evidence_quality", 0)
        evidence_factor = evidence_quality  # 0-1之间
        
        # 计算最终置信度
        final_confidence = (
            base_confidence * 0.4 +
            quality_factor * 0.3 +
            bias_factor * 0.2 +
            evidence_factor * 0.1
        )
        
        # 确保在0-1范围内
        final_confidence = max(0, min(final_confidence, 1))
        
        # 生成置信度级别
        confidence_level = "high" if final_confidence >= 0.8 else \
                           "medium" if final_confidence >= 0.5 else \
                           "low"
        
        # 生成解释
        explanation = []
        if base_confidence < 0.5:
            explanation.append("结果本身的确定性较低")
        if quality_factor < 0.6:
            explanation.append("推理过程质量不高")
        if bias_factor < 0.8:
            explanation.append(f"存在{len(biases)}种认知偏差")
        if evidence_factor < 0.5:
            explanation.append("支持结论的证据不充分")
            
        return {
            "confidence": final_confidence,
            "confidence_level": confidence_level,
            "factors": {
                "base_confidence": base_confidence,
                "reasoning_quality": quality_factor,
                "bias_factor": bias_factor,
                "evidence_quality": evidence_factor
            },
            "explanation": "，".join(explanation) if explanation else "结果可信度较高"
        }
    
    def detect_cognitive_biases(self, reasoning_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测认知偏差
        
        Args:
            reasoning_trace: 推理步骤列表
            
        Returns:
            List: 检测到的认知偏差列表
        """
        detected_biases = []
        
        if not reasoning_trace:
            return detected_biases
            
        # 转换为文本方便分析
        reasoning_text = " ".join([
            step.get("description", "") + " " + str(step.get("output", ""))
            for step in reasoning_trace
        ])
        
        # 分析推理模式
        reasoning_patterns = self._analyze_reasoning_patterns(reasoning_trace)
        
        # 检查每种偏差
        for bias_type, bias_info in self.cognitive_biases.items():
            # 检查文本模式
            text_evidence = self._check_text_patterns(reasoning_text, bias_info["detection_patterns"])
            
            # 检查推理模式
            pattern_evidence = self._check_reasoning_patterns(reasoning_patterns, bias_type)
            
            # 计算偏差严重程度
            severity = self._calculate_bias_severity(text_evidence, pattern_evidence)
            
            # 如果超过阈值，记录偏差
            if severity >= bias_info["severity_threshold"]:
                detected_biases.append({
                    "bias_type": bias_type,
                    "description": bias_info["description"],
                    "severity": severity,
                    "evidence": {
                        "text_evidence": text_evidence,
                        "pattern_evidence": pattern_evidence
                    },
                    "mitigation": bias_info["mitigation"]
                })
        
        return detected_biases
    
    def _analyze_reasoning_patterns(self, reasoning_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析推理模式"""
        patterns = {
            "evidence_usage": [],
            "assumption_usage": [],
            "conclusion_style": [],
            "reasoning_depth": 0,
            "alternative_consideration": False
        }
        
        for step in reasoning_trace:
            # 分析证据使用
            if "evidence" in step.get("description", "").lower():
                patterns["evidence_usage"].append(step)
            
            # 分析假设使用
            if "assume" in step.get("description", "").lower():
                patterns["assumption_usage"].append(step)
            
            # 分析结论风格
            if step.get("type") == "conclusion":
                patterns["conclusion_style"].append(step)
            
            # 分析推理深度
            if step.get("depth"):
                patterns["reasoning_depth"] = max(patterns["reasoning_depth"], step["depth"])
            
            # 分析替代方案考虑
            if "alternative" in step.get("description", "").lower():
                patterns["alternative_consideration"] = True
        
        return patterns
    
    def _check_text_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """检查文本中的偏差模式"""
        evidence = []
        for pattern in patterns:
            if pattern.lower() in text.lower():
                evidence.append(pattern)
        return evidence
    
    def _check_reasoning_patterns(self, patterns: Dict[str, Any], bias_type: str) -> List[str]:
        """检查推理模式中的偏差"""
        evidence = []
        
        if bias_type == "confirmation_bias":
            if len(patterns["evidence_usage"]) < len(patterns["assumption_usage"]):
                evidence.append("证据使用不足")
            if not patterns["alternative_consideration"]:
                evidence.append("未考虑替代方案")
                
        elif bias_type == "availability_bias":
            if patterns["reasoning_depth"] < 0.5:
                evidence.append("推理深度不足")
                
        elif bias_type == "anchoring_bias":
            if len(patterns["assumption_usage"]) > 0 and not patterns["alternative_consideration"]:
                evidence.append("过度依赖初始假设")
                
        elif bias_type == "overconfidence_bias":
            if len(patterns["conclusion_style"]) > 0 and not patterns["alternative_consideration"]:
                evidence.append("结论过于确定")
                
        return evidence
    
    def _calculate_bias_severity(self, text_evidence: List[str], pattern_evidence: List[str]) -> float:
        """计算偏差严重程度"""
        # 文本证据权重
        text_weight = 0.4
        # 模式证据权重
        pattern_weight = 0.6
        
        # 计算文本证据得分
        text_score = min(1.0, len(text_evidence) * 0.2)
        
        # 计算模式证据得分
        pattern_score = min(1.0, len(pattern_evidence) * 0.3)
        
        # 计算总得分
        severity = (text_score * text_weight) + (pattern_score * pattern_weight)
        
        return severity
    
    def _calculate_reasoning_confidence(self, steps, analysis, biases):
        """计算推理过程的整体置信度"""
        # 基础置信度基于推理质量
        base_confidence = analysis.get("overall_score", 0)
        
        # 根据检测到的偏差调整置信度
        bias_penalty = len(biases) * 0.1
        adjusted_confidence = max(0, base_confidence - bias_penalty)
        
        # 根据推理步骤中的不确定性调整
        uncertainty_penalty = 0
        for step in steps:
            confidence = step.get("confidence", 1.0)
            if confidence < 0.5:
                uncertainty_penalty += 0.05
                
        final_confidence = max(0, adjusted_confidence - uncertainty_penalty)
        
        return min(1.0, final_confidence)
    
    def _handle_reasoning_start(self, event_data):
        """处理推理开始事件"""
        reasoning_id = event_data.get("reasoning_id", str(uuid.uuid4()))
        self.logger.info(f"开始跟踪推理过程: {reasoning_id}")
        
        # 初始化新的推理跟踪
        new_trace = {
            "reasoning_id": reasoning_id,
            "start_time": time.time(),
            "steps": [],
            "status": "in_progress"
        }
        
        # 添加到跟踪列表
        self.cognitive_traces.append(new_trace)
        
    def _handle_reasoning_step(self, event_data):
        """处理推理步骤事件"""
        reasoning_id = event_data.get("reasoning_id")
        step_data = event_data.get("step_data", {})
        
        if not reasoning_id:
            return
            
        # 查找对应的推理跟踪
        for trace in self.cognitive_traces:
            if trace.get("reasoning_id") == reasoning_id:
                # 添加步骤
                trace["steps"].append(step_data)
                break
                
    def _handle_reasoning_complete(self, event_data):
        """处理推理完成事件"""
        reasoning_id = event_data.get("reasoning_id")
        result = event_data.get("result", {})
        
        if not reasoning_id:
            return
            
        # 查找对应的推理跟踪
        for trace in self.cognitive_traces:
            if trace.get("reasoning_id") == reasoning_id:
                # 更新状态
                trace["status"] = "completed"
                trace["end_time"] = time.time()
                trace["result"] = result
                
                # 分析推理质量
                analysis = self._analyze_reasoning_quality(trace.get("steps", []))
                trace["quality_analysis"] = analysis
                
                # 检测认知偏差
                biases = self.detect_cognitive_biases(trace.get("steps", []))
                trace["cognitive_biases"] = biases
                
                # 评估置信度
                confidence = self._calculate_reasoning_confidence(
                    trace.get("steps", []), 
                    analysis, 
                    biases
                )
                trace["confidence"] = confidence
                
                # 发布元认知分析完成事件
                if self.event_system:
                    self.event_system.publish("metacognition.analysis_completed", {
                        "reasoning_id": reasoning_id,
                        "confidence": confidence,
                        "quality_score": analysis.get("overall_score", 0),
                        "bias_count": len(biases)
                    })
                
                break
                
    def _handle_inference_result(self, event_data):
        """处理推理结果事件"""
        # 记录推理结果的置信度
        inference_id = event_data.get("inference_id", str(uuid.uuid4()))
        confidence = event_data.get("confidence", 0.5)
        success = event_data.get("success", False)
        
        self.confidence_metrics[inference_id] = {
            "confidence": confidence,
            "success": success,
            "timestamp": time.time()
        }
        
    def generate_metacognitive_reflection(self) -> Dict[str, Any]:
        """
        生成元认知反思，分析历史推理模式
        
        Returns:
            Dict: 元认知反思结果
        """
        if len(self.cognitive_traces) < 3:
            return {
                "status": "insufficient_data",
                "message": "数据不足，无法生成有意义的元认知反思"
            }
            
        # 总结推理性能
        total_traces = len(self.cognitive_traces)
        completed_traces = sum(1 for t in self.cognitive_traces if t.get("status") == "completed")
        avg_confidence = np.mean([t.get("confidence", 0) for t in self.cognitive_traces if "confidence" in t])
        
        # 分析常见的认知偏差
        bias_counts = defaultdict(int)
        for trace in self.cognitive_traces:
            biases = trace.get("cognitive_biases", [])
            for bias in biases:
                bias_type = bias.get("bias_type")
                if bias_type:
                    bias_counts[bias_type] += 1
                    
        # 找出最常见的偏差
        common_biases = sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 分析推理强项和弱项
        quality_metrics = {}
        for trace in self.cognitive_traces:
            analysis = trace.get("quality_analysis", {})
            for metric, value in analysis.items():
                if metric not in quality_metrics:
                    quality_metrics[metric] = []
                quality_metrics[metric].append(value)
                
        strengths = []
        weaknesses = []
        
        for metric, values in quality_metrics.items():
            if metric == "overall_score":
                continue
                
            avg_value = np.mean(values) if values else 0
            if avg_value >= 0.7:
                strengths.append({"metric": metric, "score": avg_value})
            elif avg_value <= 0.5:
                weaknesses.append({"metric": metric, "score": avg_value})
                
        # 生成改进建议
        improvement_suggestions = []
        
        # 基于弱项生成建议
        for weakness in weaknesses:
            metric = weakness["metric"]
            if metric == "completeness":
                improvement_suggestions.append(
                    "确保每次推理都包含问题定义、分析过程和明确结论"
                )
            elif metric == "coherence":
                improvement_suggestions.append(
                    "增强推理步骤间的逻辑连接，确保每一步都建立在前一步的基础上"
                )
            elif metric == "evidence_quality":
                improvement_suggestions.append(
                    "提高证据质量，引用更多具体数据和事实支持推理"
                )
                
        # 基于常见偏差生成建议
        for bias_type, count in common_biases[:2]:  # 处理最常见的两种偏差
            if bias_type in self.cognitive_biases:
                mitigation = self.cognitive_biases[bias_type]["mitigation"]
                improvement_suggestions.append(mitigation)
                
        # 生成反思记录
        reflection = {
            "timestamp": time.time(),
            "traces_analyzed": total_traces,
            "completion_rate": completed_traces / total_traces if total_traces > 0 else 0,
            "average_confidence": avg_confidence,
            "common_biases": common_biases[:3],  # 前三种常见偏差
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_suggestions": improvement_suggestions
        }
        
        # 添加到反思历史
        self.reflection_history.append(reflection)
        
        # 发布反思完成事件
        if self.event_system:
            self.event_system.publish("metacognition.reflection_generated", {
                "timestamp": reflection["timestamp"],
                "weakness_count": len(weaknesses),
                "suggestion_count": len(improvement_suggestions)
            })
            
        return {
            "status": "success",
            "reflection": reflection
        }
    
    def get_most_recent_traces(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        获取最近的推理跟踪
        
        Args:
            count: 返回的跟踪数量
            
        Returns:
            List: 最近的推理跟踪列表
        """
        # 按时间戳排序
        sorted_traces = sorted(
            self.cognitive_traces,
            key=lambda t: t.get("end_time", t.get("start_time", 0)),
            reverse=True
        )
        
        # 只返回基本信息
        simplified_traces = []
        for trace in sorted_traces[:count]:
            simplified_traces.append({
                "reasoning_id": trace.get("reasoning_id"),
                "status": trace.get("status"),
                "step_count": len(trace.get("steps", [])),
                "confidence": trace.get("confidence", 0),
                "has_biases": len(trace.get("cognitive_biases", [])) > 0,
                "timestamp": trace.get("end_time", trace.get("start_time", 0))
            })
            
        return simplified_traces
    
    def get_cognitive_trace(self, reasoning_id: str) -> Dict[str, Any]:
        """
        获取特定的推理跟踪
        
        Args:
            reasoning_id: 推理ID
            
        Returns:
            Dict: 推理跟踪信息
        """
        for trace in self.cognitive_traces:
            if trace.get("reasoning_id") == reasoning_id:
                return trace
                
        return {"status": "not_found", "message": f"未找到ID为{reasoning_id}的推理跟踪"}
    
    def update_metacognitive_state(self):
        """更新元认知状态"""
        # 计算自我意识水平
        awareness = self._calculate_awareness_level()
        
        # 评估自我调节能力
        regulation = self._evaluate_self_regulation()
        
        # 计算学习进度
        learning = self._assess_learning_progress()
        
        # 评估适应能力
        adaptation = self._evaluate_adaptation_capability()
        
        # 更新状态
        self.metacognitive_state.update({
            "awareness_level": awareness,
            "self_regulation": regulation,
            "learning_progress": learning,
            "adaptation_capability": adaptation
        })
        
        # 发布状态更新事件
        if self.event_system:
            self.event_system.publish("metacognition.state_updated", self.metacognitive_state)
            
    def _calculate_awareness_level(self) -> float:
        """计算自我意识水平"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算偏差检测率
        bias_detection_rate = sum(1 for t in recent_traces if t["has_biases"]) / len(recent_traces)
        
        # 计算置信度准确性
        confidence_accuracy = self._evaluate_confidence_accuracy()
        
        # 计算推理质量意识
        quality_awareness = self._evaluate_quality_awareness()
        
        # 综合评分
        awareness = (
            bias_detection_rate * 0.4 +
            confidence_accuracy * 0.3 +
            quality_awareness * 0.3
        )
        
        return min(1.0, awareness)
        
    def _evaluate_self_regulation(self) -> float:
        """评估自我调节能力"""
        if not self.reflection_history:
            return 0.0
            
        # 分析最近的反思记录
        recent_reflections = self.reflection_history[-3:]
        
        # 计算改进建议采纳率
        suggestion_adoption = self._calculate_suggestion_adoption()
        
        # 评估偏差纠正效果
        bias_correction = self._evaluate_bias_correction()
        
        # 计算学习策略调整
        strategy_adjustment = self._evaluate_strategy_adjustment()
        
        # 综合评分
        regulation = (
            suggestion_adoption * 0.4 +
            bias_correction * 0.3 +
            strategy_adjustment * 0.3
        )
        
        return min(1.0, regulation)
        
    def _assess_learning_progress(self) -> float:
        """评估学习进度"""
        if not self.learning_history:
            return 0.0
            
        # 分析学习历史
        recent_learning = self.learning_history[-5:]
        
        # 计算知识增长
        knowledge_growth = self._calculate_knowledge_growth()
        
        # 评估技能提升
        skill_improvement = self._evaluate_skill_improvement()
        
        # 计算模式识别能力
        pattern_recognition = self._evaluate_pattern_recognition()
        
        # 综合评分
        progress = (
            knowledge_growth * 0.4 +
            skill_improvement * 0.3 +
            pattern_recognition * 0.3
        )
        
        return min(1.0, progress)
        
    def _evaluate_adaptation_capability(self) -> float:
        """评估适应能力"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算策略调整频率
        strategy_adjustment = self._calculate_strategy_adjustment()
        
        # 评估环境适应度
        environment_adaptation = self._evaluate_environment_adaptation()
        
        # 计算问题解决效率
        problem_solving = self._evaluate_problem_solving()
        
        # 综合评分
        adaptation = (
            strategy_adjustment * 0.4 +
            environment_adaptation * 0.3 +
            problem_solving * 0.3
        )
        
        return min(1.0, adaptation)
        
    def _evaluate_confidence_accuracy(self) -> float:
        """评估置信度准确性"""
        if not self.confidence_metrics:
            return 0.0
            
        # 计算置信度与实际结果的匹配度
        matches = 0
        total = 0
        
        for inference_id, metrics in self.confidence_metrics.items():
            if "success" in metrics:
                total += 1
                confidence = metrics["confidence"]
                success = metrics["success"]
                
                # 高置信度应该对应成功，低置信度应该对应失败
                if (confidence >= 0.7 and success) or (confidence < 0.7 and not success):
                    matches += 1
                    
        return matches / total if total > 0 else 0.0
        
    def _evaluate_quality_awareness(self) -> float:
        """评估推理质量意识"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算质量评估准确性
        quality_assessment = 0.0
        for trace in recent_traces:
            if "quality_analysis" in trace:
                analysis = trace["quality_analysis"]
                if analysis["overall_score"] >= 0.7 and trace["confidence"] >= 0.7:
                    quality_assessment += 1
                elif analysis["overall_score"] < 0.7 and trace["confidence"] < 0.7:
                    quality_assessment += 1
                    
        return quality_assessment / len(recent_traces) if recent_traces else 0.0
        
    def _calculate_suggestion_adoption(self) -> float:
        """计算改进建议采纳率"""
        if not self.reflection_history:
            return 0.0
            
        # 分析最近的反思记录
        recent_reflections = self.reflection_history[-3:]
        
        # 计算建议采纳情况
        adopted_suggestions = 0
        total_suggestions = 0
        
        for reflection in recent_reflections:
            suggestions = reflection.get("improvement_suggestions", [])
            total_suggestions += len(suggestions)
            
            # 检查建议是否被采纳
            for suggestion in suggestions:
                if self._check_suggestion_adoption(suggestion):
                    adopted_suggestions += 1
                    
        return adopted_suggestions / total_suggestions if total_suggestions > 0 else 0.0
        
    def _evaluate_bias_correction(self) -> float:
        """评估偏差纠正效果"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算偏差减少率
        bias_counts = [len(t.get("cognitive_biases", [])) for t in recent_traces]
        if len(bias_counts) < 2:
            return 0.0
            
        # 计算偏差数量是否呈下降趋势
        decreasing = sum(1 for i in range(1, len(bias_counts)) if bias_counts[i] < bias_counts[i-1])
        return decreasing / (len(bias_counts) - 1)
        
    def _evaluate_strategy_adjustment(self) -> float:
        """评估策略调整情况"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算策略变化频率
        strategy_changes = 0
        for i in range(1, len(recent_traces)):
            prev_strategy = self._extract_reasoning_strategy(recent_traces[i-1])
            curr_strategy = self._extract_reasoning_strategy(recent_traces[i])
            if prev_strategy != curr_strategy:
                strategy_changes += 1
                
        return strategy_changes / (len(recent_traces) - 1) if len(recent_traces) > 1 else 0.0
        
    def _calculate_knowledge_growth(self) -> float:
        """计算知识增长情况"""
        if not self.learning_history:
            return 0.0
            
        # 分析学习历史
        recent_learning = self.learning_history[-5:]
        
        # 计算新概念掌握情况
        new_concepts = 0
        for record in recent_learning:
            if record.get("type") == "concept_learning":
                new_concepts += 1
                
        return min(1.0, new_concepts / 5)  # 假设每5次学习应该掌握1个新概念
        
    def _evaluate_skill_improvement(self) -> float:
        """评估技能提升情况"""
        if not self.learning_history:
            return 0.0
            
        # 分析学习历史
        recent_learning = self.learning_history[-5:]
        
        # 计算技能提升情况
        skill_improvements = 0
        for record in recent_learning:
            if record.get("type") == "skill_improvement":
                skill_improvements += 1
                
        return min(1.0, skill_improvements / 5)  # 假设每5次学习应该提升1项技能
        
    def _evaluate_pattern_recognition(self) -> float:
        """评估模式识别能力"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算模式识别准确率
        pattern_recognition = 0.0
        for trace in recent_traces:
            if "reasoning_chain" in trace:
                chain = trace["reasoning_chain"]
                if chain["strength"] >= 0.7 and chain["completeness"] >= 0.7:
                    pattern_recognition += 1
                    
        return pattern_recognition / len(recent_traces) if recent_traces else 0.0
        
    def _calculate_strategy_adjustment(self) -> float:
        """计算策略调整频率"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算策略变化率
        strategy_changes = 0
        for i in range(1, len(recent_traces)):
            prev_strategy = self._extract_reasoning_strategy(recent_traces[i-1])
            curr_strategy = self._extract_reasoning_strategy(recent_traces[i])
            if prev_strategy != curr_strategy:
                strategy_changes += 1
                
        return strategy_changes / (len(recent_traces) - 1) if len(recent_traces) > 1 else 0.0
        
    def _evaluate_environment_adaptation(self) -> float:
        """评估环境适应度"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算环境适应情况
        adaptation_success = 0
        for trace in recent_traces:
            if "context" in trace:
                context = trace["context"]
                if self._check_environment_adaptation(context):
                    adaptation_success += 1
                    
        return adaptation_success / len(recent_traces) if recent_traces else 0.0
        
    def _evaluate_problem_solving(self) -> float:
        """评估问题解决效率"""
        if not self.cognitive_traces:
            return 0.0
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(5)
        
        # 计算问题解决效率
        solving_efficiency = 0.0
        for trace in recent_traces:
            if "efficiency_metrics" in trace:
                metrics = trace["efficiency_metrics"]
                if metrics["overall_score"] >= 0.7:
                    solving_efficiency += 1
                    
        return solving_efficiency / len(recent_traces) if recent_traces else 0.0
        
    def _extract_reasoning_strategy(self, trace: Dict[str, Any]) -> str:
        """提取推理策略"""
        if "steps" not in trace:
            return "unknown"
            
        steps = trace["steps"]
        if not steps:
            return "unknown"
            
        # 分析步骤类型分布
        step_types = [step.get("type", "unknown") for step in steps]
        type_counts = {}
        for step_type in step_types:
            type_counts[step_type] = type_counts.get(step_type, 0) + 1
            
        # 返回最常见的步骤类型作为策略标识
        return max(type_counts.items(), key=lambda x: x[1])[0]
        
    def _check_environment_adaptation(self, context: Dict[str, Any]) -> bool:
        """检查环境适应情况"""
        # 检查是否能够适应不同的任务类型
        task_adaptation = len(context.get("active_tasks", [])) > 0
        
        # 检查是否能够处理不同的概念
        concept_adaptation = len(context.get("active_concepts", [])) > 0
        
        # 检查是否能够维持适当的认知负载
        load_adaptation = 0.3 <= context.get("cognitive_load", 0) <= 0.7
        
        return task_adaptation and concept_adaptation and load_adaptation
        
    def _check_suggestion_adoption(self, suggestion: str) -> bool:
        """检查改进建议是否被采纳"""
        if not self.cognitive_traces:
            return False
            
        # 分析最近的推理跟踪
        recent_traces = self.get_most_recent_traces(3)
        
        # 检查建议是否在后续推理中体现
        for trace in recent_traces:
            if "steps" in trace:
                steps = trace["steps"]
                for step in steps:
                    if suggestion.lower() in step.get("description", "").lower():
                        return True
                        
        return False
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """注册可用工具"""
        self.available_tools[tool_name] = tool_function
        self.action_capabilities["tool_usage"] = True
        
    def execute_action(self, action_type: str, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行自主行动"""
        if not self._can_execute_action(action_type):
            return {
                "status": "error",
                "message": f"无法执行{action_type}类型的行动"
            }
            
        # 记录行动意图
        action_intent = {
            "type": action_type,
            "params": action_params,
            "timestamp": time.time()
        }
        
        try:
            # 执行具体行动
            if action_type == "tool_usage":
                result = self._execute_tool_action(action_params)
            elif action_type == "environment_interaction":
                result = self._execute_environment_action(action_params)
            elif action_type == "task_execution":
                result = self._execute_task_action(action_params)
            else:
                result = self._execute_autonomous_action(action_params)
                
            # 记录行动结果
            action_record = {
                **action_intent,
                "result": result,
                "success": result.get("status") == "success"
            }
            self.action_history.append(action_record)
            
            # 更新行动能力状态
            self._update_action_capabilities()
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": str(e)
            }
            self.action_history.append({
                **action_intent,
                "result": error_result,
                "success": False
            })
            return error_result
            
    def _can_execute_action(self, action_type: str) -> bool:
        """检查是否可以执行特定类型的行动"""
        return self.action_capabilities.get(action_type, False)
        
    def _execute_tool_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具使用行动"""
        tool_name = params.get("tool_name")
        if tool_name not in self.available_tools:
            return {
                "status": "error",
                "message": f"工具{tool_name}未注册"
            }
            
        try:
            result = self.available_tools[tool_name](**params.get("tool_params", {}))
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _execute_environment_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行环境交互行动"""
        # 实现环境交互逻辑
        return {
            "status": "not_implemented",
            "message": "环境交互功能尚未实现"
        }
        
    def _execute_task_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务行动"""
        # 实现任务执行逻辑
        return {
            "status": "not_implemented",
            "message": "任务执行功能尚未实现"
        }
        
    def _execute_autonomous_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行自主决策行动"""
        # 实现自主决策逻辑
        return {
            "status": "not_implemented",
            "message": "自主决策功能尚未实现"
        }
        
    def _update_action_capabilities(self):
        """更新行动能力状态"""
        # 基于历史记录更新能力状态
        recent_actions = self.action_history[-10:]  # 分析最近10次行动
        
        # 更新工具使用能力
        tool_usage_success = sum(1 for a in recent_actions 
                               if a["type"] == "tool_usage" and a["success"])
        self.action_capabilities["tool_usage"] = tool_usage_success > 0
        
        # 更新环境交互能力
        env_interaction_success = sum(1 for a in recent_actions 
                                    if a["type"] == "environment_interaction" and a["success"])
        self.action_capabilities["environment_interaction"] = env_interaction_success > 0
        
        # 更新任务执行能力
        task_execution_success = sum(1 for a in recent_actions 
                                   if a["type"] == "task_execution" and a["success"])
        self.action_capabilities["task_execution"] = task_execution_success > 0
        
        # 更新自主决策能力
        autonomous_success = sum(1 for a in recent_actions 
                               if a["type"] == "autonomous_action" and a["success"])
        self.action_capabilities["autonomous_decision"] = autonomous_success > 0
        
    def start_continuous_learning(self):
        """启动持续学习进程"""
        self._initialize_learning_resources()
        self._start_learning_monitor()
        
    def _initialize_learning_resources(self):
        """初始化学习资源"""
        # 初始化在线学习资源
        self._init_online_resources()
        # 初始化本地知识库
        self._init_local_knowledge()
        # 初始化交互式学习
        self._init_interactive_learning()
        
    def _init_online_resources(self):
        """初始化在线学习资源"""
        # 配置网络爬虫
        self.web_crawler = {
            "enabled": True,
            "sources": [
                "wikipedia",
                "arxiv",
                "github",
                "stackoverflow"
            ],
            "rate_limit": 10  # 每分钟请求数
        }
        
        # 配置API访问
        self.api_clients = {
            "openai": None,  # 需要配置API密钥
            "google": None,  # 需要配置API密钥
            "github": None   # 需要配置API密钥
        }
        
    def _init_local_knowledge(self):
        """初始化本地知识库"""
        # 创建知识索引
        self.knowledge_index = {
            "concepts": {},
            "skills": {},
            "patterns": {},
            "relationships": {}
        }
        
        # 初始化向量数据库
        self.vector_db = None  # 需要实现向量存储
        
    def _init_interactive_learning(self):
        """初始化交互式学习"""
        self.interactive_learning = {
            "active": True,
            "feedback_loop": True,
            "adaptation_rate": 0.1
        }
        
    def _start_learning_monitor(self):
        """启动学习监控"""
        self.learning_monitor = {
            "active": True,
            "check_interval": 60,  # 秒
            "last_check": time.time()
        }
        
    def identify_learning_needs(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别学习需求"""
        learning_needs = []
        
        # 分析当前上下文
        concepts = context.get("active_concepts", [])
        tasks = context.get("active_tasks", [])
        
        # 检查概念理解
        for concept in concepts:
            if not self._is_concept_understood(concept):
                learning_needs.append({
                    "type": "concept",
                    "target": concept,
                    "priority": self._calculate_learning_priority(concept)
                })
                
        # 检查技能掌握
        for task in tasks:
            required_skills = self._identify_required_skills(task)
            for skill in required_skills:
                if not self._is_skill_mastered(skill):
                    learning_needs.append({
                        "type": "skill",
                        "target": skill,
                        "priority": self._calculate_learning_priority(skill)
                    })
                    
        return sorted(learning_needs, key=lambda x: x["priority"], reverse=True)
        
    def _is_concept_understood(self, concept: str) -> bool:
        """检查概念是否已理解"""
        if concept in self.knowledge_base["concepts"]:
            knowledge = self.knowledge_base["concepts"][concept]
            return knowledge.get("understanding_level", 0) >= 0.7
        return False
        
    def _is_skill_mastered(self, skill: str) -> bool:
        """检查技能是否已掌握"""
        if skill in self.knowledge_base["skills"]:
            skill_data = self.knowledge_base["skills"][skill]
            return skill_data.get("mastery_level", 0) >= 0.7
        return False
        
    def _calculate_learning_priority(self, target: str) -> float:
        """计算学习优先级"""
        # 基于使用频率
        usage_frequency = self._get_usage_frequency(target)
        
        # 基于相关性
        relevance = self._calculate_relevance(target)
        
        # 基于难度
        difficulty = self._estimate_learning_difficulty(target)
        
        # 综合评分
        priority = (
            usage_frequency * 0.4 +
            relevance * 0.4 +
            (1 - difficulty) * 0.2
        )
        
        return min(1.0, priority)
        
    def start_learning_task(self, learning_need: Dict[str, Any]):
        """启动学习任务"""
        task_id = str(uuid.uuid4())
        
        # 创建学习任务
        learning_task = {
            "id": task_id,
            "type": learning_need["type"],
            "target": learning_need["target"],
            "status": "started",
            "start_time": time.time(),
            "resources": self._gather_learning_resources(learning_need),
            "progress": 0.0
        }
        
        # 添加到活动学习任务
        self.active_learning_tasks[task_id] = learning_task
        
        # 启动学习过程
        self._execute_learning_process(learning_task)
        
    def _gather_learning_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集学习资源"""
        resources = []
        
        # 从在线资源获取
        if self.learning_resources["online"]:
            online_resources = self._fetch_online_resources(learning_need)
            resources.extend(online_resources)
            
        # 从本地知识库获取
        if self.learning_resources["local"]:
            local_resources = self._fetch_local_resources(learning_need)
            resources.extend(local_resources)
            
        # 从交互式学习获取
        if self.learning_resources["interactive"]:
            interactive_resources = self._prepare_interactive_resources(learning_need)
            resources.extend(interactive_resources)
            
        return resources
        
    def _fetch_online_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取在线学习资源"""
        resources = []
        
        # 从Wikipedia获取
        if "wikipedia" in self.web_crawler["sources"]:
            wiki_resources = self._fetch_wikipedia_resources(learning_need)
            resources.extend(wiki_resources)
            
        # 从arXiv获取
        if "arxiv" in self.web_crawler["sources"]:
            arxiv_resources = self._fetch_arxiv_resources(learning_need)
            resources.extend(arxiv_resources)
            
        # 从GitHub获取
        if "github" in self.web_crawler["sources"]:
            github_resources = self._fetch_github_resources(learning_need)
            resources.extend(github_resources)
            
        return resources
        
    def _fetch_local_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取本地学习资源"""
        resources = []
        
        # 从知识库获取相关概念
        related_concepts = self._find_related_concepts(learning_need["target"])
        for concept in related_concepts:
            if concept in self.knowledge_base["concepts"]:
                resources.append({
                    "type": "concept",
                    "source": "local",
                    "content": self.knowledge_base["concepts"][concept]
                })
                
        # 从技能库获取相关技能
        related_skills = self._find_related_skills(learning_need["target"])
        for skill in related_skills:
            if skill in self.knowledge_base["skills"]:
                resources.append({
                    "type": "skill",
                    "source": "local",
                    "content": self.knowledge_base["skills"][skill]
                })
                
        return resources
        
    def _prepare_interactive_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """准备交互式学习资源"""
        resources = []
        
        # 生成练习问题
        practice_questions = self._generate_practice_questions(learning_need)
        resources.extend(practice_questions)
        
        # 生成测试用例
        test_cases = self._generate_test_cases(learning_need)
        resources.extend(test_cases)
        
        # 生成反馈机制
        feedback_mechanism = self._create_feedback_mechanism(learning_need)
        resources.append(feedback_mechanism)
        
        return resources
        
    def _execute_learning_process(self, learning_task: Dict[str, Any]):
        """执行学习过程"""
        try:
            # 处理每个学习资源
            for resource in learning_task["resources"]:
                # 学习资源内容
                self._learn_from_resource(resource)
                
                # 更新学习进度
                self._update_learning_progress(learning_task)
                
                # 应用学习成果
                self._apply_learning_results(learning_task)
                
                # 评估学习效果
                if self._evaluate_learning_effectiveness(learning_task):
                    # 学习成功，更新知识库
                    self._update_knowledge_base(learning_task)
                    learning_task["status"] = "completed"
                else:
                    # 学习未达标，继续学习
                    self._adjust_learning_strategy(learning_task)
                    
        except Exception as e:
            learning_task["status"] = "failed"
            learning_task["error"] = str(e)
            
        finally:
            # 清理学习任务
            self._cleanup_learning_task(learning_task)
            
    def _learn_from_resource(self, resource: Dict[str, Any]):
        """从资源中学习"""
        if resource["type"] == "concept":
            self._learn_concept(resource["content"])
        elif resource["type"] == "skill":
            self._learn_skill(resource["content"])
        elif resource["type"] == "practice":
            self._practice_skill(resource["content"])
        elif resource["type"] == "test":
            self._test_understanding(resource["content"])
            
    def _update_learning_progress(self, learning_task: Dict[str, Any]):
        """更新学习进度"""
        # 计算总体进度
        total_resources = len(learning_task["resources"])
        completed_resources = sum(1 for r in learning_task["resources"] 
                                if r.get("status") == "completed")
        
        learning_task["progress"] = completed_resources / total_resources
        
        # 发布进度更新事件
        if self.event_system:
            self.event_system.publish("learning.progress_updated", {
                "task_id": learning_task["id"],
                "progress": learning_task["progress"]
            })
            
    def _apply_learning_results(self, learning_task: Dict[str, Any]):
        """应用学习成果"""
        target = learning_task["target"]
        
        if learning_task["type"] == "concept":
            # 更新概念理解
            self.knowledge_base["concepts"][target] = {
                "understanding_level": self._calculate_understanding_level(target),
                "last_updated": time.time(),
                "application_count": 0
            }
        elif learning_task["type"] == "skill":
            # 更新技能掌握
            self.knowledge_base["skills"][target] = {
                "mastery_level": self._calculate_mastery_level(target),
                "last_updated": time.time(),
                "application_count": 0
            }
            
    def _evaluate_learning_effectiveness(self, learning_task: Dict[str, Any]) -> bool:
        """评估学习效果"""
        target = learning_task["target"]
        
        if learning_task["type"] == "concept":
            # 评估概念理解
            understanding_level = self._calculate_understanding_level(target)
            return understanding_level >= 0.7
        elif learning_task["type"] == "skill":
            # 评估技能掌握
            mastery_level = self._calculate_mastery_level(target)
            return mastery_level >= 0.7
            
        return False
        
    def _update_knowledge_base(self, learning_task: Dict[str, Any]):
        """更新知识库"""
        target = learning_task["target"]
        
        # 更新知识索引
        self._update_knowledge_index(target, learning_task["type"])
        
        # 更新向量数据库
        self._update_vector_db(target, learning_task)
        
        # 更新关系网络
        self._update_relationship_network(target)
        
    def _cleanup_learning_task(self, learning_task: Dict[str, Any]):
        """清理学习任务"""
        # 从活动任务中移除
        if learning_task["id"] in self.active_learning_tasks:
            del self.active_learning_tasks[learning_task["id"]]
            
        # 记录学习历史
        self.learning_history.append({
            "task_id": learning_task["id"],
            "type": learning_task["type"],
            "target": learning_task["target"],
            "status": learning_task["status"],
            "start_time": learning_task["start_time"],
            "end_time": time.time(),
            "progress": learning_task["progress"]
        })
        
        # 发布学习完成事件
        if self.event_system:
            self.event_system.publish("learning.task_completed", {
                "task_id": learning_task["id"],
                "status": learning_task["status"],
                "progress": learning_task["progress"]
            }) 
    
    def _fetch_wikipedia_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Wikipedia获取学习资源"""
        resources = []
        try:
            import wikipedia
            wikipedia.set_lang("zh")  # 设置中文
            
            # 搜索相关文章
            search_results = wikipedia.search(learning_need["target"], results=3)
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    resources.append({
                        "type": "concept",
                        "source": "wikipedia",
                        "title": title,
                        "content": {
                            "summary": page.summary,
                            "url": page.url,
                            "categories": page.categories
                        }
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # 处理消歧义页面
                    for option in e.options[:3]:
                        try:
                            page = wikipedia.page(option)
                            resources.append({
                                "type": "concept",
                                "source": "wikipedia",
                                "title": option,
                                "content": {
                                    "summary": page.summary,
                                    "url": page.url,
                                    "categories": page.categories
                                }
                            })
                        except:
                            continue
                except:
                    continue
                    
        except ImportError:
            self.logger.warning("Wikipedia API未安装")
            
        return resources
        
    def _fetch_arxiv_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从arXiv获取学习资源"""
        resources = []
        try:
            import arxiv
            
            # 构建搜索查询
            search = arxiv.Search(
                query=learning_need["target"],
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # 获取论文
            for result in search.results():
                resources.append({
                    "type": "concept",
                    "source": "arxiv",
                    "title": result.title,
                    "content": {
                        "summary": result.summary,
                        "url": result.entry_id,
                        "authors": [author.name for author in result.authors],
                        "published": result.published
                    }
                })
                
        except ImportError:
            self.logger.warning("arXiv API未安装")
            
        return resources
        
    def _fetch_github_resources(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从GitHub获取学习资源"""
        resources = []
        try:
            import requests
            
            # 构建GitHub API请求
            headers = {
                "Authorization": f"token {self.api_clients['github']}" if self.api_clients['github'] else None
            }
            
            # 搜索代码仓库
            search_url = f"https://api.github.com/search/repositories?q={learning_need['target']}&sort=stars"
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                repos = response.json()["items"][:3]  # 获取前3个最相关的仓库
                
                for repo in repos:
                    resources.append({
                        "type": "skill",
                        "source": "github",
                        "title": repo["full_name"],
                        "content": {
                            "description": repo["description"],
                            "url": repo["html_url"],
                            "stars": repo["stargazers_count"],
                            "language": repo["language"]
                        }
                    })
                    
        except ImportError:
            self.logger.warning("requests库未安装")
            
        return resources
        
    def _find_related_concepts(self, target: str) -> List[str]:
        """查找相关概念"""
        related = []
        
        # 从知识库中查找
        for concept, data in self.knowledge_base["concepts"].items():
            if self._calculate_similarity(target, concept) > 0.7:
                related.append(concept)
                
        # 从关系网络中查找
        if target in self.knowledge_base["relationships"]:
            related.extend(self.knowledge_base["relationships"][target])
            
        return list(set(related))
        
    def _find_related_skills(self, target: str) -> List[str]:
        """查找相关技能"""
        related = []
        
        # 从知识库中查找
        for skill, data in self.knowledge_base["skills"].items():
            if self._calculate_similarity(target, skill) > 0.7:
                related.append(skill)
                
        return list(set(related))
        
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1, str2).ratio()
        except:
            return 0.0
            
    def _generate_practice_questions(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成练习问题"""
        questions = []
        
        if learning_need["type"] == "concept":
            # 生成概念理解问题
            questions.extend(self._generate_concept_questions(learning_need["target"]))
        elif learning_need["type"] == "skill":
            # 生成技能练习问题
            questions.extend(self._generate_skill_questions(learning_need["target"]))
            
        return questions
        
    def _generate_concept_questions(self, concept: str) -> List[Dict[str, Any]]:
        """生成概念理解问题"""
        questions = []
        
        # 从知识库获取概念信息
        if concept in self.knowledge_base["concepts"]:
            concept_data = self.knowledge_base["concepts"][concept]
            
            # 生成定义问题
            questions.append({
                "type": "practice",
                "subtype": "definition",
                "question": f"请解释{concept}的定义",
                "answer": concept_data.get("definition", ""),
                "difficulty": "easy"
            })
            
            # 生成应用问题
            questions.append({
                "type": "practice",
                "subtype": "application",
                "question": f"请举例说明{concept}的应用场景",
                "answer": concept_data.get("examples", []),
                "difficulty": "medium"
            })
            
            # 生成分析问题
            questions.append({
                "type": "practice",
                "subtype": "analysis",
                "question": f"请分析{concept}与其他概念的关系",
                "answer": concept_data.get("relationships", {}),
                "difficulty": "hard"
            })
            
        return questions
        
    def _generate_skill_questions(self, skill: str) -> List[Dict[str, Any]]:
        """生成技能练习问题"""
        questions = []
        
        # 从知识库获取技能信息
        if skill in self.knowledge_base["skills"]:
            skill_data = self.knowledge_base["skills"][skill]
            
            # 生成基础练习
            questions.append({
                "type": "practice",
                "subtype": "basic",
                "question": f"请完成{skill}的基础练习",
                "answer": skill_data.get("basic_exercise", ""),
                "difficulty": "easy"
            })
            
            # 生成进阶练习
            questions.append({
                "type": "practice",
                "subtype": "advanced",
                "question": f"请完成{skill}的进阶练习",
                "answer": skill_data.get("advanced_exercise", ""),
                "difficulty": "medium"
            })
            
            # 生成综合练习
            questions.append({
                "type": "practice",
                "subtype": "comprehensive",
                "question": f"请完成{skill}的综合练习",
                "answer": skill_data.get("comprehensive_exercise", ""),
                "difficulty": "hard"
            })
            
        return questions
        
    def _generate_test_cases(self, learning_need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成测试用例"""
        test_cases = []
        
        if learning_need["type"] == "concept":
            # 生成概念测试用例
            test_cases.extend(self._generate_concept_tests(learning_need["target"]))
        elif learning_need["type"] == "skill":
            # 生成技能测试用例
            test_cases.extend(self._generate_skill_tests(learning_need["target"]))
            
        return test_cases
        
    def _generate_concept_tests(self, concept: str) -> List[Dict[str, Any]]:
        """生成概念测试用例"""
        test_cases = []
        
        # 从知识库获取概念信息
        if concept in self.knowledge_base["concepts"]:
            concept_data = self.knowledge_base["concepts"][concept]
            
            # 生成选择题
            test_cases.append({
                "type": "test",
                "subtype": "multiple_choice",
                "question": f"关于{concept}，以下哪个说法是正确的？",
                "options": concept_data.get("test_options", []),
                "correct_answer": concept_data.get("correct_option", ""),
                "explanation": concept_data.get("explanation", "")
            })
            
            # 生成判断题
            test_cases.append({
                "type": "test",
                "subtype": "true_false",
                "question": f"判断：{concept_data.get('test_statement', '')}",
                "correct_answer": concept_data.get("is_true", True),
                "explanation": concept_data.get("explanation", "")
            })
            
        return test_cases
        
    def _generate_skill_tests(self, skill: str) -> List[Dict[str, Any]]:
        """生成技能测试用例"""
        test_cases = []
        
        # 从知识库获取技能信息
        if skill in self.knowledge_base["skills"]:
            skill_data = self.knowledge_base["skills"][skill]
            
            # 生成实践测试
            test_cases.append({
                "type": "test",
                "subtype": "practical",
                "question": f"请完成以下{skill}实践任务：{skill_data.get('test_task', '')}",
                "expected_output": skill_data.get("expected_output", ""),
                "evaluation_criteria": skill_data.get("evaluation_criteria", [])
            })
            
            # 生成问题解决测试
            test_cases.append({
                "type": "test",
                "subtype": "problem_solving",
                "question": f"请解决以下{skill}相关问题：{skill_data.get('test_problem', '')}",
                "solution": skill_data.get("solution", ""),
                "evaluation_criteria": skill_data.get("evaluation_criteria", [])
            })
            
        return test_cases
        
    def _create_feedback_mechanism(self, learning_need: Dict[str, Any]) -> Dict[str, Any]:
        """创建反馈机制"""
        return {
            "type": "feedback",
            "mechanism": {
                "type": "adaptive",
                "parameters": {
                    "learning_rate": 0.1,
                    "feedback_threshold": 0.7,
                    "adjustment_interval": 60
                },
                "metrics": [
                    "accuracy",
                    "completion_time",
                    "error_rate",
                    "confidence"
                ],
                "adjustment_strategies": {
                    "high_error_rate": "increase_practice",
                    "low_confidence": "review_fundamentals",
                    "slow_completion": "optimize_process",
                    "high_accuracy": "increase_difficulty"
                }
            }
        }
        
    def _learn_concept(self, content: Dict[str, Any]):
        """学习概念"""
        # 提取关键信息
        concept = content.get("title", "")
        definition = content.get("content", {}).get("summary", "")
        examples = content.get("content", {}).get("examples", [])
        relationships = content.get("content", {}).get("categories", [])
        
        # 更新知识库
        if concept not in self.knowledge_base["concepts"]:
            self.knowledge_base["concepts"][concept] = {
                "definition": definition,
                "examples": examples,
                "relationships": relationships,
                "understanding_level": 0.0,
                "last_updated": time.time(),
                "source": content.get("source", "unknown")
            }
        else:
            # 更新现有概念
            existing = self.knowledge_base["concepts"][concept]
            existing["definition"] = self._merge_definitions(existing["definition"], definition)
            existing["examples"].extend(examples)
            existing["relationships"].extend(relationships)
            existing["last_updated"] = time.time()
            
    def _learn_skill(self, content: Dict[str, Any]):
        """学习技能"""
        # 提取关键信息
        skill = content.get("title", "")
        description = content.get("content", {}).get("description", "")
        examples = content.get("content", {}).get("examples", [])
        requirements = content.get("content", {}).get("requirements", [])
        
        # 更新知识库
        if skill not in self.knowledge_base["skills"]:
            self.knowledge_base["skills"][skill] = {
                "description": description,
                "examples": examples,
                "requirements": requirements,
                "mastery_level": 0.0,
                "last_updated": time.time(),
                "source": content.get("source", "unknown")
            }
        else:
            # 更新现有技能
            existing = self.knowledge_base["skills"][skill]
            existing["description"] = self._merge_descriptions(existing["description"], description)
            existing["examples"].extend(examples)
            existing["requirements"].extend(requirements)
            existing["last_updated"] = time.time()
            
    def _practice_skill(self, content: Dict[str, Any]):
        """练习技能"""
        # 执行练习
        result = self._execute_practice(content)
        
        # 评估练习结果
        evaluation = self._evaluate_practice_result(result, content)
        
        # 更新技能掌握度
        if content["target"] in self.knowledge_base["skills"]:
            skill_data = self.knowledge_base["skills"][content["target"]]
            skill_data["mastery_level"] = min(1.0, skill_data["mastery_level"] + evaluation["improvement"])
            
    def _test_understanding(self, content: Dict[str, Any]):
        """测试理解程度"""
        # 执行测试
        result = self._execute_test(content)
        
        # 评估测试结果
        evaluation = self._evaluate_test_result(result, content)
        
        # 更新理解度
        if content["target"] in self.knowledge_base["concepts"]:
            concept_data = self.knowledge_base["concepts"][content["target"]]
            concept_data["understanding_level"] = min(1.0, concept_data["understanding_level"] + evaluation["improvement"])
            
    def _merge_definitions(self, existing: str, new: str) -> str:
        """合并概念定义"""
        if not existing:
            return new
        if not new:
            return existing
            
        # 使用更详细的定义
        if len(new) > len(existing):
            return new
        return existing
        
    def _merge_descriptions(self, existing: str, new: str) -> str:
        """合并技能描述"""
        if not existing:
            return new
        if not new:
            return existing
            
        # 使用更详细的描述
        if len(new) > len(existing):
            return new
        return existing
        
    def _execute_practice(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """执行练习"""
        # 根据练习类型执行不同的练习
        if content["subtype"] == "basic":
            return self._execute_basic_practice(content)
        elif content["subtype"] == "advanced":
            return self._execute_advanced_practice(content)
        elif content["subtype"] == "comprehensive":
            return self._execute_comprehensive_practice(content)
            
        return {"status": "error", "message": "未知的练习类型"}
        
    def _execute_test(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """执行测试"""
        # 根据测试类型执行不同的测试
        if content["subtype"] == "multiple_choice":
            return self._execute_multiple_choice_test(content)
        elif content["subtype"] == "true_false":
            return self._execute_true_false_test(content)
        elif content["subtype"] == "practical":
            return self._execute_practical_test(content)
        elif content["subtype"] == "problem_solving":
            return self._execute_problem_solving_test(content)
            
        return {"status": "error", "message": "未知的测试类型"}
        
    def _evaluate_practice_result(self, result: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]:
        """评估练习结果"""
        evaluation = {
            "status": "success",
            "score": 0.0,
            "improvement": 0.0,
            "feedback": []
        }
        
        # 根据练习类型评估结果
        if content["subtype"] == "basic":
            evaluation["score"] = self._evaluate_basic_practice(result)
        elif content["subtype"] == "advanced":
            evaluation["score"] = self._evaluate_advanced_practice(result)
        elif content["subtype"] == "comprehensive":
            evaluation["score"] = self._evaluate_comprehensive_practice(result)
            
        # 计算改进程度
        evaluation["improvement"] = evaluation["score"] * 0.1  # 每次练习最多提升10%
        
        # 生成反馈
        if evaluation["score"] < 0.6:
            evaluation["feedback"].append("需要更多练习")
        elif evaluation["score"] < 0.8:
            evaluation["feedback"].append("表现良好，但还有提升空间")
        else:
            evaluation["feedback"].append("优秀表现")
            
        return evaluation
        
    def _evaluate_test_result(self, result: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]:
        """评估测试结果"""
        evaluation = {
            "status": "success",
            "score": 0.0,
            "improvement": 0.0,
            "feedback": []
        }
        
        # 根据测试类型评估结果
        if content["subtype"] == "multiple_choice":
            evaluation["score"] = self._evaluate_multiple_choice_test(result)
        elif content["subtype"] == "true_false":
            evaluation["score"] = self._evaluate_true_false_test(result)
        elif content["subtype"] == "practical":
            evaluation["score"] = self._evaluate_practical_test(result)
        elif content["subtype"] == "problem_solving":
            evaluation["score"] = self._evaluate_problem_solving_test(result)
            
        # 计算改进程度
        evaluation["improvement"] = evaluation["score"] * 0.2  # 每次测试最多提升20%
        
        # 生成反馈
        if evaluation["score"] < 0.6:
            evaluation["feedback"].append("需要加强学习")
        elif evaluation["score"] < 0.8:
            evaluation["feedback"].append("理解基本正确，但需要深化")
        else:
            evaluation["feedback"].append("理解透彻")
            
        return evaluation