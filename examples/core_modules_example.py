"""
GHOST-AGI 核心增强模块示例

该示例演示了GHOST-AGI七个核心增强模块的基本使用方法。
"""

import os
import sys
import time
import logging
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入七个核心增强模块
from evolution.code_evolution_engine import CodeEvolutionEngine
from reasoning.neuro_symbolic_integration import NeuroSymbolicIntegration
from learning.continual_learning import ContinualLearning
from motivation.intrinsic_motivation import IntrinsicMotivationSystem
from evolution.evolutionary_search import EvolutionarySearch
from memory.hierarchical_memory import HierarchicalMemory
from world_model.adaptive_world_model import AdaptiveWorldModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghost_agi_core_modules.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CoreModulesExample")

def setup_modules():
    """初始化所有核心增强模块"""
    logger.info("开始初始化GHOST-AGI核心增强模块")
    
    # 1. 代码自演化引擎
    code_evolution = CodeEvolutionEngine(
        code_repository_path="./",  # 当前项目路径
        test_suite_path="./tests"
    )
    
    # 2. 神经符号集成系统
    neuro_symbolic = NeuroSymbolicIntegration()
    
    # 3. 连续学习与防遗忘机制
    continual_learning = ContinualLearning(
        memory_size=500,
        ewc_lambda=5000.0
    )
    
    # 4. 内在动机系统
    intrinsic_motivation = IntrinsicMotivationSystem(
        curiosity_weight=0.6,
        mastery_weight=0.3,
        autonomy_weight=0.1
    )
    
    # 5. 自主进化搜索框架
    evolutionary_search = EvolutionarySearch(
        population_size=30,
        mutation_rate=0.1,
        crossover_rate=0.7
    )
    
    # 6. 分层记忆与知识整合系统
    hierarchical_memory = HierarchicalMemory(
        working_memory_size=10,
        episodic_memory_size=500,
        semantic_memory_size=1000
    )
    
    # 7. 自适应世界模型
    adaptive_world_model = AdaptiveWorldModel(
        prediction_horizons=[1, 5, 20],
        model_update_frequency=10
    )
    
    logger.info("所有核心增强模块初始化完成")
    
    return {
        "code_evolution": code_evolution,
        "neuro_symbolic": neuro_symbolic,
        "continual_learning": continual_learning,
        "intrinsic_motivation": intrinsic_motivation,
        "evolutionary_search": evolutionary_search,
        "hierarchical_memory": hierarchical_memory,
        "adaptive_world_model": adaptive_world_model
    }

def code_evolution_demo(code_evolution):
    """代码自演化引擎示例"""
    logger.info("\n=== 代码自演化引擎示例 ===")
    
    # 扫描代码库
    logger.info("扫描代码库...")
    code_evolution.scan_codebase()
    
    # 分析进化潜力
    logger.info("分析代码进化潜力...")
    opportunities = code_evolution.analyze_evolution_potential()
    logger.info(f"发现 {len(opportunities.get('high_complexity', []))} 个高复杂度区域")
    logger.info(f"发现 {len(opportunities.get('duplicated_code', []))} 个代码重复区域")
    
    # 模拟代码进化过程
    logger.info("正在准备代码进化过程...")
    
    # 注: 实际进化过程需要提供适应度函数和完整环境
    # 这里我们只是展示API的使用方法
    # evolution_report = code_evolution.evolve_codebase(
    #     iterations=5,
    #     population_size=10,
    #     evolution_strategy='gradual'
    # )
    
    logger.info("代码自演化引擎示例完成")

def neuro_symbolic_demo(neuro_symbolic):
    """神经符号集成系统示例"""
    logger.info("\n=== 神经符号集成系统示例 ===")
    
    # 创建一些示例概念
    concept1 = "人工智能"
    concept2 = "机器学习"
    concept3 = "知识表示"
    
    # 存储一些语义知识
    knowledge1 = {
        "定义": "人工智能是研究如何使计算机系统能模仿人类认知功能的一门学科",
        "子领域": ["机器学习", "自然语言处理", "计算机视觉"],
        "应用": ["智能助手", "自动驾驶", "推荐系统"]
    }
    
    knowledge2 = {
        "定义": "机器学习是人工智能的一个子领域，专注于开发能从数据中学习的算法",
        "方法": ["监督学习", "无监督学习", "强化学习"],
        "模型": ["神经网络", "决策树", "支持向量机"]
    }
    
    # 在符号知识库中存储概念
    neuro_symbolic.symbolic_kb['concepts'][concept1] = knowledge1
    neuro_symbolic.symbolic_kb['concepts'][concept2] = knowledge2
    
    # 添加关系
    relation = {
        "source": concept1,
        "target": concept2,
        "type": "has_subfield",
        "strength": 0.9
    }
    neuro_symbolic.knowledge_graph['relations'].append(relation)
    
    # 执行混合推理
    query = "人工智能的应用领域有哪些？"
    logger.info(f"执行混合推理，查询: '{query}'")
    
    result = neuro_symbolic.hybrid_reasoning(query, reasoning_type='symbolic')
    logger.info(f"推理结果置信度: {result.get('confidence', 0)}")
    
    # 解释推理过程
    trace_id = result.get('trace_id')
    if trace_id:
        explanation = neuro_symbolic.explain_reasoning(trace_id)
        logger.info(f"推理步骤数: {explanation.get('steps_count', 0)}")
    
    logger.info("神经符号集成系统示例完成")

def continual_learning_demo(continual_learning):
    """连续学习与防遗忘机制示例"""
    logger.info("\n=== 连续学习与防遗忘机制示例 ===")
    
    # 模拟几个任务的学习
    tasks = [
        {"task_id": "task1", "name": "图像分类", "data_size": 1000},
        {"task_id": "task2", "name": "语言翻译", "data_size": 2000},
        {"task_id": "task3", "name": "声音识别", "data_size": 1500}
    ]
    
    for i, task in enumerate(tasks):
        logger.info(f"学习任务 {i+1}: {task['name']}")
        
        # 在实际应用中，这里会提供真实的任务数据和模型
        # 这里我们只是展示API的使用方法
        result = continual_learning.learn_new_task(
            task_id=task["task_id"],
            task_data=task,
            model=None,  # 实际应用中提供模型
            learning_method='ewc' if i > 0 else 'standard'  # 第一个任务使用标准学习，后续任务使用EWC
        )
        
        logger.info(f"任务学习完成，性能指标: {result.get('performance', {})}")
        
        # 检测概念漂移
        if i > 0:
            drift_result = continual_learning.detect_concept_drift(
                task_id=tasks[i-1]["task_id"],
                current_data=task
            )
            logger.info(f"概念漂移检测结果: {drift_result.get('detected', False)}")
    
    # 生成记忆回放批次
    batch = continual_learning.generate_rehearsal_batch(
        batch_size=10, 
        task_weights={"task1": 0.3, "task2": 0.4, "task3": 0.3}
    )
    logger.info(f"生成记忆回放批次，大小: {len(batch)}")
    
    logger.info("连续学习与防遗忘机制示例完成")

def intrinsic_motivation_demo(intrinsic_motivation):
    """内在动机系统示例"""
    logger.info("\n=== 内在动机系统示例 ===")
    
    # 评估新奇性
    observation = {
        "type": "environment_state",
        "features": [0.2, 0.5, 0.8, 0.1],
        "complexity": 0.7
    }
    
    logger.info("评估观察的新颖性...")
    novelty_result = intrinsic_motivation.evaluate_novelty(observation)
    logger.info(f"新颖性评分: {novelty_result.get('novelty_score', 0)}")
    logger.info(f"探索建议: {novelty_result.get('recommendation', '')}")
    
    # 跟踪技能进步
    logger.info("跟踪技能进步...")
    skill_progress = intrinsic_motivation.track_skill_progress(
        skill_id="natural_language_processing",
        performance_metric=0.75
    )
    logger.info(f"当前熟练度: {skill_progress.get('current_proficiency', 0)}")
    logger.info(f"进步奖励: {skill_progress.get('progress_reward', 0)}")
    
    # 生成目标
    logger.info("生成新目标...")
    goal = intrinsic_motivation.generate_goal(
        context={"current_focus": "提升学习能力"}
    )
    logger.info(f"生成目标: {goal.get('description', '')}")
    logger.info(f"目标优先级: {goal.get('priority', 0)}")
    
    # 分解目标
    if goal.get('id'):
        logger.info("将目标分解为子目标...")
        subgoals = intrinsic_motivation.decompose_goal(goal['id'])
        logger.info(f"生成 {len(subgoals.get('subgoals', []))} 个子目标")
    
    # 更新情绪状态
    logger.info("更新情绪状态...")
    events = [
        {"type": "success", "intensity": 0.8},
        {"type": "discovery", "intensity": 0.9}
    ]
    emotion_update = intrinsic_motivation.update_emotional_state(events)
    logger.info(f"主导情绪: {emotion_update.get('dominant_emotion', '')}")
    logger.info(f"情绪价: {emotion_update.get('emotional_valence', 0)}")
    
    logger.info("内在动机系统示例完成")

def evolutionary_search_demo(evolutionary_search):
    """自主进化搜索框架示例"""
    logger.info("\n=== 自主进化搜索框架示例 ===")
    
    # 定义一个简单的个体模板
    individual_template = {
        "genes": [0.5, 0.5, 0.5, 0.5, 0.5],
        "structure": "linear"
    }
    
    # 初始化种群
    logger.info("初始化进化搜索种群...")
    evolutionary_search.initialize_population(individual_template)
    
    # 定义一个简单的适应度函数
    def simple_fitness(individual):
        # 简单示例：适应度是基因值之和
        return sum(individual.get('genes', []))
    
    # 执行进化
    logger.info("执行进化过程...")
    evolution_result = evolutionary_search.evolve(
        generations=10,
        fitness_function=simple_fitness,
        selection_method='tournament'
    )
    
    logger.info(f"进化完成，代数: {evolution_result.get('generations', 0)}")
    logger.info(f"最佳适应度: {evolution_result.get('best_fitness', 0)}")
    
    # 分析适应度景观
    logger.info("分析适应度景观...")
    landscape_analysis = evolutionary_search.analyze_fitness_landscape()
    logger.info(f"峰值数量: {landscape_analysis.get('peaks_count', 0)}")
    logger.info(f"推荐探索策略: {landscape_analysis.get('exploration_recommendation', '')}")
    
    logger.info("自主进化搜索框架示例完成")

def hierarchical_memory_demo(hierarchical_memory):
    """分层记忆与知识整合系统示例"""
    logger.info("\n=== 分层记忆与知识整合系统示例 ===")
    
    # 添加项目到工作记忆
    logger.info("添加项目到工作记忆...")
    item1 = {
        "content": "GHOST-AGI系统需要增强自主进化能力",
        "source": "user_input",
        "context": "planning"
    }
    
    result = hierarchical_memory.add_to_working_memory(item1, priority=0.8)
    logger.info(f"工作记忆大小: {result.get('working_memory_size', 0)}")
    
    # 查询工作记忆
    logger.info("查询工作记忆...")
    query_result = hierarchical_memory.query_working_memory({"source": "user_input"})
    logger.info(f"匹配项数: {len(query_result)}")
    
    # 存储情节记忆
    logger.info("存储情节记忆...")
    episode = {
        "type": "learning_session",
        "task": "实现分层记忆系统",
        "duration": 120,
        "outcome": "success",
        "key_insights": ["记忆整合可以提高效率", "工作记忆容量有限需要及时转移"]
    }
    result = hierarchical_memory.store_episode(episode, importance=0.7)
    logger.info(f"情节记忆大小: {result.get('episodic_memory_size', 0)}")
    
    # 存储语义知识
    logger.info("存储语义知识...")
    concept = "分层记忆系统"
    knowledge = {
        "定义": "一种模拟人类记忆层次结构的系统，包括工作记忆、情节记忆和语义记忆",
        "组成部分": ["工作记忆", "情节记忆", "语义记忆", "程序性记忆"],
        "优势": ["高效信息检索", "长期知识存储", "防止干扰"]
    }
    result = hierarchical_memory.store_semantic_knowledge(concept, knowledge)
    logger.info(f"语义记忆大小: {result.get('semantic_memory_size', 0)}")
    
    # 添加知识图谱关系
    logger.info("添加知识图谱关系...")
    relation_result = hierarchical_memory.add_relation(
        source="分层记忆系统",
        relation_type="is_part_of",
        target="认知架构",
        strength=0.8
    )
    logger.info(f"关系ID: {relation_result.get('relation_id', '')}")
    
    # 执行记忆巩固
    logger.info("执行记忆巩固过程...")
    consolidation_result = hierarchical_memory.consolidate_memories()
    logger.info(f"巩固统计: {consolidation_result.get('consolidation_stats', {})}")
    
    # 获取记忆系统统计信息
    stats = hierarchical_memory.get_memory_stats()
    logger.info(f"工作记忆: {stats.get('working_memory_size', 0)} 项")
    logger.info(f"情节记忆: {stats.get('episodic_memory_size', 0)} 项")
    logger.info(f"语义记忆: {stats.get('semantic_memory_size', 0)} 概念")
    
    logger.info("分层记忆与知识整合系统示例完成")

def adaptive_world_model_demo(adaptive_world_model):
    """自适应世界模型示例"""
    logger.info("\n=== 自适应世界模型示例 ===")
    
    # 更新环境状态
    logger.info("更新环境状态...")
    observation = {
        "temperature": 22.5,
        "humidity": 0.65,
        "light_level": 0.8,
        "noise_level": 0.3
    }
    adaptive_world_model.update_environment_state(observation)
    
    # 再更新几次状态，模拟环境变化
    for i in range(5):
        observation = {
            "temperature": 22.5 + np.random.normal(0, 0.5),
            "humidity": 0.65 + np.random.normal(0, 0.05),
            "light_level": max(0, min(1, 0.8 + np.random.normal(0, 0.1))),
            "noise_level": max(0, min(1, 0.3 + np.random.normal(0, 0.1)))
        }
        adaptive_world_model.update_environment_state(observation)
    
    # 预测未来状态
    logger.info("预测未来状态...")
    prediction = adaptive_world_model.predict_future_states(steps=3)
    
    logger.info(f"预测分支数: {len(prediction.get('branches', []))}")
    logger.info(f"预测模型置信度: {prediction.get('model_confidence', 0)}")
    
    # 检测环境变化
    logger.info("检测环境变化...")
    change_detection = adaptive_world_model.detect_environment_change()
    
    if change_detection.get('detected', False):
        logger.info(f"检测到环境变化，类型: {change_detection.get('type', '')}")
        
        # 适应环境变化
        adaptation = adaptive_world_model.adapt_to_environment_change(change_detection)
        logger.info(f"适应结果: {adaptation.get('success', False)}")
    else:
        logger.info("未检测到显著环境变化")
    
    # 发现因果关系
    logger.info("发现因果关系...")
    variables = ["temperature", "humidity", "light_level", "noise_level"]
    causal_discovery = adaptive_world_model.discover_causal_relationships(variables)
    
    if causal_discovery.get('status') == 'success':
        relations = causal_discovery.get('discovered_relations', [])
        logger.info(f"发现 {len(relations)} 个因果关系")
    
    # 生成抽象表示
    logger.info("生成环境抽象表示...")
    abstraction = adaptive_world_model.generate_abstract_representation(
        abstraction_level='medium',
        focus_variables=["temperature", "humidity"]
    )
    logger.info(f"抽象表示变量数: {abstraction.get('_variable_count', 0)}")
    
    # 获取模型状态
    status = adaptive_world_model.get_model_status()
    logger.info(f"环境稳定性: {status.get('environment_dynamics', {}).get('stability', 0)}")
    logger.info(f"环境噪声水平: {status.get('environment_dynamics', {}).get('noise_level', 0)}")
    
    logger.info("自适应世界模型示例完成")

def main():
    """主函数"""
    logger.info("======= GHOST-AGI 核心增强模块示例 =======")
    
    # 初始化所有模块
    modules = setup_modules()
    
    # 运行各模块示例
    code_evolution_demo(modules["code_evolution"])
    neuro_symbolic_demo(modules["neuro_symbolic"])
    continual_learning_demo(modules["continual_learning"])
    intrinsic_motivation_demo(modules["intrinsic_motivation"])
    evolutionary_search_demo(modules["evolutionary_search"])
    hierarchical_memory_demo(modules["hierarchical_memory"])
    adaptive_world_model_demo(modules["adaptive_world_model"])
    
    logger.info("======= 所有示例完成 =======")

if __name__ == "__main__":
    main() 