"""
GHOST-AGI 自适应世界模型

该模块实现系统对环境的建模和预测能力，使GHOST-AGI能够理解和适应复杂变化的环境。
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict

class AdaptiveWorldModel:
    """自适应世界模型，提供预测性环境建模和动态环境适应能力"""
    
    def __init__(self, 
                 prediction_horizons: List[int] = [1, 5, 20], 
                 model_update_frequency: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        初始化自适应世界模型
        
        Args:
            prediction_horizons: 预测时间尺度列表(步数)
            model_update_frequency: 模型更新频率
            logger: 日志记录器
        """
        self.prediction_horizons = prediction_horizons
        self.model_update_frequency = model_update_frequency
        
        # 初始化日志
        self.logger = logger or self._setup_logger()
        
        # 环境表示
        self.environment_state = {}  # 当前环境状态
        self.observation_history = []  # 历史观察
        self.action_history = []  # 历史行动
        
        # 预测模型
        self.prediction_models = {}  # 不同尺度的预测模型
        self.prediction_accuracy = defaultdict(list)  # 预测准确性记录
        
        # 环境动态特性
        self.environment_dynamics = {
            'stability': 0.5,  # 环境稳定性
            'complexity': 0.5,  # 环境复杂性
            'responsiveness': 0.5,  # 对行动的响应程度
            'noise_level': 0.3  # 噪声水平
        }
        
        # 因果关系
        self.causal_graph = {
            'nodes': {},  # 变量节点
            'edges': []   # 因果关系边
        }
        
        # 抽象层次结构
        self.abstraction_levels = {}
        
        # 初始化各时间尺度的预测模型
        for horizon in prediction_horizons:
            self.prediction_models[horizon] = self._initialize_prediction_model(horizon)
        
        self.update_counter = 0
        
        self.logger.info("自适应世界模型初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AdaptiveWorldModel")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("world_model.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_prediction_model(self, horizon: int) -> Dict[str, Any]:
        """初始化预测模型"""
        return {
            'horizon': horizon,
            'type': 'simple', 
            'model': None,
            'uncertainty': 0.5,
            'last_update': time.time()
        }
    
    # 预测性环境建模方法
    def update_environment_state(self, observation: Dict[str, Any], action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        更新环境状态
        
        Args:
            observation: 环境观察
            action: 执行的行动(可选)
            
        Returns:
            更新后的环境状态
        """
        self.logger.debug("更新环境状态")
        
        # 添加时间戳
        observation['timestamp'] = time.time()
        
        # 更新当前状态
        self.environment_state.update(observation)
        
        # 添加到历史记录
        self.observation_history.append(observation)
        if action:
            self.action_history.append(action)
        
        # 限制历史长度
        max_history = 1000
        if len(self.observation_history) > max_history:
            self.observation_history = self.observation_history[-max_history:]
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
        
        # 增加更新计数
        self.update_counter += 1
        
        # 检查是否应该更新模型
        if self.update_counter >= self.model_update_frequency:
            self._update_prediction_models()
            self.update_counter = 0
        
        return self.environment_state
    
    def predict_future_states(self, 
                            steps: int, 
                            current_state: Optional[Dict[str, Any]] = None,
                            actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        预测未来状态
        
        Args:
            steps: 预测步数
            current_state: 起始状态(如果为None则使用当前环境状态)
            actions: 预测期间的行动序列
            
        Returns:
            预测结果
        """
        self.logger.debug(f"预测未来 {steps} 步状态")
        
        # 使用当前环境状态作为默认起始状态
        if current_state is None:
            current_state = self.environment_state.copy()
        
        # 找到最接近的预测模型
        closest_horizon = min(self.prediction_horizons, key=lambda x: abs(x - steps))
        model = self.prediction_models[closest_horizon]
        
        # 预测未来状态
        future_states = []
        uncertainty = []
        
        # 简化的预测实现
        # 实际应用中会使用更复杂的预测模型
        state = current_state.copy()
        for i in range(steps):
            action = actions[i] if actions and i < len(actions) else None
            
            # 简单演化模型 (实际应用中会有真实的预测模型)
            next_state = self._evolve_state(state, action, model)
            
            # 模拟不确定性增长
            step_uncertainty = model['uncertainty'] * (1 + 0.1 * i)
            
            future_states.append(next_state)
            uncertainty.append(step_uncertainty)
            
            # 更新状态
            state = next_state
        
        # 创建预测分支
        prediction_branches = [
            {
                'probability': 0.7,
                'states': future_states,
                'uncertainty': uncertainty
            }
        ]
        
        # 在实际应用中，可能会创建多个可能的未来分支
        if steps > 5:
            # 创建一个替代分支
            alternative_branch = {
                'probability': 0.3,
                'states': [self._generate_alternative_state(s) for s in future_states],
                'uncertainty': [u * 1.2 for u in uncertainty]
            }
            prediction_branches.append(alternative_branch)
        
        return {
            'prediction_steps': steps,
            'start_state': current_state,
            'branches': prediction_branches,
            'model_horizon': closest_horizon,
            'model_confidence': 1.0 - model['uncertainty']
        }
    
    def _evolve_state(self, state: Dict[str, Any], action: Optional[Dict[str, Any]], model: Dict[str, Any]) -> Dict[str, Any]:
        """简单状态演化模型"""
        # 这是一个简化的状态演化示例
        # 实际应用中会有更复杂的预测逻辑
        
        evolved = state.copy()
        evolved['timestamp'] = time.time()
        
        # 简单随机变化
        for key, value in state.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                noise = np.random.normal(0, self.environment_dynamics['noise_level'])
                evolved[key] = value + noise
                
                # 如果有动作，考虑动作影响
                if action and key in action:
                    evolved[key] += action[key] * self.environment_dynamics['responsiveness']
        
        return evolved
    
    def _generate_alternative_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成替代状态"""
        alternative = state.copy()
        
        # 在一些关键维度上做出不同的预测
        for key, value in state.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                if np.random.random() > 0.7:  # 30%的维度有不同预测
                    direction = 1 if np.random.random() > 0.5 else -1
                    alternative[key] = value + direction * np.random.random() * 0.2 * value
        
        return alternative
    
    def _update_prediction_models(self) -> None:
        """更新预测模型"""
        self.logger.debug("更新预测模型")
        
        if len(self.observation_history) < max(self.prediction_horizons) + 1:
            self.logger.info("历史数据不足，跳过模型更新")
            return
        
        # 评估当前模型性能
        for horizon in self.prediction_horizons:
            # 历史预测和实际结果对比
            predictions = []
            actuals = []
            
            for i in range(len(self.observation_history) - horizon):
                prediction = self._past_prediction(self.observation_history[i], horizon)
                actual = self.observation_history[i + horizon]
                
                if prediction and actual:
                    predictions.append(prediction)
                    actuals.append(actual)
            
            if predictions and actuals:
                accuracy = self._calculate_prediction_accuracy(predictions, actuals)
                self.prediction_accuracy[horizon].append(accuracy)
                
                # 更新模型不确定性
                self.prediction_models[horizon]['uncertainty'] = 1.0 - accuracy
                
                self.logger.debug(f"horizon {horizon} 模型评估，准确度: {accuracy:.4f}")
        
        # 更新环境动态特性
        self._update_environment_dynamics()
    
    def _past_prediction(self, state: Dict[str, Any], horizon: int) -> Optional[Dict[str, Any]]:
        """获取过去对特定状态的预测"""
        # 简化实现，实际应用中会有预测历史
        return None
    
    def _calculate_prediction_accuracy(self, predictions: List[Dict[str, Any]], actuals: List[Dict[str, Any]]) -> float:
        """计算预测准确度"""
        # 简化实现
        return 0.8  # 示例固定准确度
    
    def _update_environment_dynamics(self) -> None:
        """更新环境动态特性"""
        # 实际应用中会有更复杂的环境分析
        # 这里简单根据近期观察调整
        
        # 示例：评估环境稳定性
        if len(self.observation_history) > 10:
            recent = self.observation_history[-10:]
            variances = []
            
            for key in recent[0]:
                if key != 'timestamp' and isinstance(recent[0][key], (int, float)):
                    values = [obs.get(key, 0) for obs in recent if key in obs]
                    if values:
                        variances.append(np.var(values))
            
            if variances:
                avg_variance = np.mean(variances)
                stability = 1.0 / (1.0 + avg_variance)  # 高方差表示低稳定性
                self.environment_dynamics['stability'] = stability
                
                # 更新噪声水平估计
                self.environment_dynamics['noise_level'] = min(0.9, max(0.1, 1.0 - stability))
    
    # 动态环境适应方法
    def adapt_to_environment_change(self, change_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        适应环境变化
        
        Args:
            change_detection: 环境变化检测信息
            
        Returns:
            适应结果
        """
        self.logger.info(f"适应环境变化: {change_detection}")
        
        change_type = change_detection.get('type', 'unknown')
        severity = change_detection.get('severity', 0.5)
        affected_variables = change_detection.get('affected_variables', [])
        
        adaptation_actions = []
        
        if change_type == 'sudden_shift':
            # 突然变化：快速适应，降低旧数据权重
            adaptation_actions.append({
                'action': 'increase_learning_rate',
                'target': 'all_models',
                'factor': min(3.0, 1.0 + severity)
            })
            
            adaptation_actions.append({
                'action': 'reduce_history_weight',
                'target': 'all_models',
                'factor': max(0.2, 1.0 - severity)
            })
            
        elif change_type == 'drift':
            # 渐变：逐步适应
            adaptation_actions.append({
                'action': 'adjust_learning_rate',
                'target': 'all_models',
                'factor': 1.0 + 0.5 * severity
            })
            
        elif change_type == 'cyclic':
            # 周期性变化：寻找周期性模式
            adaptation_actions.append({
                'action': 'enable_seasonal_detection',
                'target': 'affected_variables',
                'variables': affected_variables
            })
        
        # 执行适应行动
        self._execute_adaptation_actions(adaptation_actions)
        
        return {
            'detected_change': change_detection,
            'adaptation_actions': adaptation_actions,
            'adapted_variables': affected_variables,
            'success': True
        }
    
    def _execute_adaptation_actions(self, actions: List[Dict[str, Any]]) -> None:
        """执行适应行动"""
        for action in actions:
            self.logger.debug(f"执行适应行动: {action}")
            # 实际实现中会根据action执行相应调整
    
    def detect_environment_change(self) -> Dict[str, Any]:
        """
        检测环境变化
        
        Returns:
            变化检测结果
        """
        self.logger.debug("检测环境变化")
        
        if len(self.observation_history) < 20:
            return {'detected': False, 'reason': 'insufficient_data'}
        
        # 分析近期观察数据
        recent = self.observation_history[-10:]
        older = self.observation_history[-20:-10]
        
        # 检查统计分布变化
        changes = {}
        significant_changes = []
        
        for key in recent[0]:
            if key != 'timestamp' and isinstance(recent[0][key], (int, float)):
                recent_values = [obs.get(key, 0) for obs in recent if key in obs]
                older_values = [obs.get(key, 0) for obs in older if key in obs]
                
                if len(recent_values) > 5 and len(older_values) > 5:
                    recent_mean = np.mean(recent_values)
                    older_mean = np.mean(older_values)
                    
                    if older_mean != 0:
                        change_ratio = abs(recent_mean - older_mean) / abs(older_mean)
                        changes[key] = change_ratio
                        
                        if change_ratio > 0.2:  # 20%变化阈值
                            significant_changes.append(key)
        
        if significant_changes:
            change_type = self._classify_change_type(significant_changes)
            severity = np.mean([changes[key] for key in significant_changes])
            
            return {
                'detected': True,
                'type': change_type,
                'affected_variables': significant_changes,
                'severity': min(1.0, severity),
                'timestamp': time.time()
            }
        
        return {'detected': False, 'reason': 'no_significant_changes'}
    
    def _classify_change_type(self, changed_variables: List[str]) -> str:
        """分类变化类型"""
        # 简化实现，实际应用中会有更复杂的分类逻辑
        if np.random.random() > 0.7:
            return 'sudden_shift'
        elif np.random.random() > 0.5:
            return 'drift'
        else:
            return 'cyclic'
    
    # 因果关系发现方法
    def discover_causal_relationships(self, 
                                    variables: List[str], 
                                    max_history: int = 100) -> Dict[str, Any]:
        """
        发现变量间的因果关系
        
        Args:
            variables: 要分析的变量列表
            max_history: 分析的最大历史长度
            
        Returns:
            发现的因果关系
        """
        self.logger.info(f"分析变量之间的因果关系: {variables}")
        
        if len(self.observation_history) < 10 or len(self.action_history) < 5:
            return {'status': 'insufficient_data'}
        
        # 限制分析历史长度
        history = self.observation_history[-min(len(self.observation_history), max_history):]
        actions = self.action_history[-min(len(self.action_history), max_history):]
        
        discovered_relations = []
        
        # 简化的因果关系分析
        # 实际应用中会使用更复杂的统计和干预方法
        
        # 1. 检查变量间的时间滞后相关性
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    correlation = self._calculate_lagged_correlation(var1, var2, history)
                    if abs(correlation) > 0.7:  # 强相关阈值
                        discovered_relations.append({
                            'source': var1,
                            'target': var2,
                            'type': 'correlation',
                            'strength': abs(correlation),
                            'direction': np.sign(correlation)
                        })
        
        # 2. 检查动作对变量的影响
        for action_key in actions[0] if actions else {}:
            for var in variables:
                impact = self._calculate_action_impact(action_key, var, history, actions)
                if abs(impact) > 0.3:  # 影响阈值
                    discovered_relations.append({
                        'source': action_key,
                        'target': var,
                        'type': 'action_impact',
                        'strength': abs(impact),
                        'direction': np.sign(impact)
                    })
        
        # 更新因果图
        for relation in discovered_relations:
            source = relation['source']
            target = relation['target']
            
            # 添加节点
            if source not in self.causal_graph['nodes']:
                self.causal_graph['nodes'][source] = {'id': source, 'type': 'variable'}
            if target not in self.causal_graph['nodes']:
                self.causal_graph['nodes'][target] = {'id': target, 'type': 'variable'}
            
            # 添加边
            edge_id = f"{source}_{target}"
            existing_edge = False
            
            for edge in self.causal_graph['edges']:
                if edge.get('id') == edge_id:
                    # 更新现有边
                    edge.update({
                        'strength': relation['strength'],
                        'last_updated': time.time()
                    })
                    existing_edge = True
                    break
            
            if not existing_edge:
                # 添加新边
                self.causal_graph['edges'].append({
                    'id': edge_id,
                    'source': source,
                    'target': target,
                    'strength': relation['strength'],
                    'type': relation['type'],
                    'discovered_at': time.time(),
                    'last_updated': time.time()
                })
        
        return {
            'status': 'success',
            'discovered_relations': discovered_relations,
            'causal_graph_size': {
                'nodes': len(self.causal_graph['nodes']),
                'edges': len(self.causal_graph['edges'])
            }
        }
    
    def _calculate_lagged_correlation(self, var1: str, var2: str, history: List[Dict[str, Any]]) -> float:
        """计算时间滞后相关性"""
        # 简化实现
        return np.random.random() * 2 - 1  # -1到1的随机值
    
    def _calculate_action_impact(self, action_key: str, var: str, 
                               history: List[Dict[str, Any]], 
                               actions: List[Dict[str, Any]]) -> float:
        """计算动作对变量的影响"""
        # 简化实现
        return np.random.random() * 0.8  # 0到0.8的随机值
    
    def predict_intervention_effect(self, 
                                  intervention: Dict[str, Any], 
                                  target_variables: List[str]) -> Dict[str, Any]:
        """
        预测干预行为的效果
        
        Args:
            intervention: 干预行为
            target_variables: 目标变量
            
        Returns:
            干预效果预测
        """
        self.logger.info(f"预测干预效果: {intervention} -> {target_variables}")
        
        effects = {}
        
        # 简化的干预效果预测
        # 实际应用中会使用更复杂的因果模型
        
        for var in target_variables:
            # 检查是否存在直接因果关系
            direct_effect = 0
            for edge in self.causal_graph['edges']:
                if edge['source'] in intervention and edge['target'] == var:
                    # 简化的效果计算
                    intervention_value = intervention[edge['source']]
                    direct_effect += intervention_value * edge['strength']
            
            # 模拟一些不确定性
            uncertainty = 0.2 + 0.1 * self.environment_dynamics['noise_level']
            noise = np.random.normal(0, uncertainty)
            
            effects[var] = {
                'expected_change': direct_effect,
                'confidence': 1.0 - uncertainty,
                'possible_range': [direct_effect - 2*uncertainty, direct_effect + 2*uncertainty]
            }
        
        return {
            'intervention': intervention,
            'predicted_effects': effects,
            'model_confidence': 1.0 - self.environment_dynamics['noise_level']
        }
    
    # 抽象层次方法
    def generate_abstract_representation(self, 
                                       abstraction_level: str = 'medium',
                                       focus_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        生成环境的抽象表示
        
        Args:
            abstraction_level: 抽象级别 ('low', 'medium', 'high')
            focus_variables: 关注的变量
            
        Returns:
            抽象表示
        """
        self.logger.debug(f"生成抽象表示，级别: {abstraction_level}")
        
        if not self.environment_state:
            return {'error': 'No environment state available'}
        
        # 确定要包含的变量
        variables = focus_variables or list(self.environment_state.keys())
        
        # 根据抽象级别确定详细程度
        if abstraction_level == 'low':
            # 低抽象级别：包含大多数细节
            detail_threshold = 0.2  # 只省略最不重要的20%
        elif abstraction_level == 'medium':
            # 中等抽象级别：平衡细节和概括
            detail_threshold = 0.5  # 省略不太重要的50%
        else:
            # 高抽象级别：只保留最重要信息
            detail_threshold = 0.8  # 只保留最重要的20%
        
        # 评估变量重要性
        var_importance = self._evaluate_variable_importance(variables)
        
        # 筛选变量
        selected_vars = [v for v, imp in var_importance.items() if imp >= detail_threshold]
        
        # 构建抽象表示
        abstraction = {}
        for var in selected_vars:
            if var in self.environment_state:
                abstraction[var] = self.environment_state[var]
        
        # 添加抽象特性
        abstraction['_abstraction_level'] = abstraction_level
        abstraction['_variable_count'] = len(selected_vars)
        abstraction['_timestamp'] = time.time()
        
        # 保存到抽象层次结构
        self.abstraction_levels[abstraction_level] = {
            'representation': abstraction,
            'variables': selected_vars,
            'created_at': time.time()
        }
        
        return abstraction
    
    def _evaluate_variable_importance(self, variables: List[str]) -> Dict[str, float]:
        """评估变量重要性"""
        importance = {}
        
        # 简化的重要性评估
        # 实际应用中会考虑因果中心性、信息价值等因素
        
        for var in variables:
            # 默认重要性
            base_importance = 0.5
            
            # 考虑因果关系中的中心性
            centrality_boost = 0
            for edge in self.causal_graph['edges']:
                if edge['source'] == var or edge['target'] == var:
                    centrality_boost += 0.1 * edge['strength']
            
            importance[var] = min(1.0, base_importance + centrality_boost)
        
        return importance
    
    # 系统API方法
    def get_model_status(self) -> Dict[str, Any]:
        """
        获取世界模型状态
        
        Returns:
            模型状态
        """
        accuracies = {}
        for horizon, acc_list in self.prediction_accuracy.items():
            if acc_list:
                accuracies[horizon] = round(np.mean(acc_list[-10:]), 4)
        
        return {
            'environment_dynamics': self.environment_dynamics,
            'observation_count': len(self.observation_history),
            'model_accuracies': accuracies,
            'prediction_horizons': self.prediction_horizons,
            'causal_graph_size': {
                'nodes': len(self.causal_graph['nodes']),
                'edges': len(self.causal_graph['edges'])
            },
            'abstraction_levels': list(self.abstraction_levels.keys())
        }
    
    def save_state(self, file_path: str) -> Dict[str, Any]:
        """
        保存世界模型状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存结果
        """
        self.logger.info(f"保存世界模型状态到: {file_path}")
        
        try:
            # 筛选可序列化的状态数据
            state = {
                'environment_state': self.environment_state,
                'environment_dynamics': self.environment_dynamics,
                'prediction_horizons': self.prediction_horizons,
                'model_update_frequency': self.model_update_frequency,
                'causal_graph': self.causal_graph,
                'saved_at': time.time()
            }
            
            # 保存最近的观察和行动历史
            recent_obs = self.observation_history[-100:] if self.observation_history else []
            recent_actions = self.action_history[-100:] if self.action_history else []
            state['recent_observations'] = recent_obs
            state['recent_actions'] = recent_actions
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return {'success': True, 'file_path': file_path}
        
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load_state(self, file_path: str) -> Dict[str, Any]:
        """
        加载世界模型状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载结果
        """
        self.logger.info(f"从 {file_path} 加载世界模型状态")
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.environment_state = state.get('environment_state', {})
            self.environment_dynamics = state.get('environment_dynamics', self.environment_dynamics)
            self.prediction_horizons = state.get('prediction_horizons', self.prediction_horizons)
            self.model_update_frequency = state.get('model_update_frequency', self.model_update_frequency)
            self.causal_graph = state.get('causal_graph', {'nodes': {}, 'edges': []})
            
            # 恢复历史
            self.observation_history = state.get('recent_observations', [])
            self.action_history = state.get('recent_actions', [])
            
            # 重新初始化预测模型
            for horizon in self.prediction_horizons:
                self.prediction_models[horizon] = self._initialize_prediction_model(horizon)
            
            return {'success': True, 'loaded_at': time.time()}
        
        except Exception as e:
            self.logger.error(f"加载状态失败: {str(e)}")
            return {'success': False, 'error': str(e)} 