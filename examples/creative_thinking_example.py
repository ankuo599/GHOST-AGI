"""
GHOST-AGI 创造性思维引擎示例

该示例展示了GHOST-AGI的创造性思维引擎功能，包括概念混合、类比推理、发散思维、约束放松和横向思维。
"""

import sys
import os
import json
import time
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.creative_thinking_engine import CreativeThinkingEngine

def print_separator(message):
    """打印分隔符"""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50)

def format_json(data, indent=2):
    """格式化JSON数据"""
    return json.dumps(data, ensure_ascii=False, indent=indent)

def main():
    print_separator("GHOST-AGI 创造性思维引擎示例")
    
    # 初始化创造性思维引擎
    creative_engine = CreativeThinkingEngine()
    
    # ---------- 测试概念混合 ----------
    print_separator("1. 概念混合示例")
    
    blend_problem = {
        "type": "design",
        "description": "设计一种结合智能手机和健康监测的新产品",
        "domains": ["科技", "医疗"]
    }
    
    blend_result = creative_engine.generate_creative_idea(
        problem=blend_problem,
        thinking_mode="conceptual_blending"
    )
    
    print("概念混合结果:")
    print(f"问题: {blend_problem['description']}")
    
    if "idea" in blend_result and "concept" in blend_result["idea"]:
        concept = blend_result["idea"]["concept"]
        print(f"\n混合概念: {concept.get('name')}")
        print(f"详细描述: {concept.get('detailed_description')}")
        print("\n属性:")
        for prop in concept.get("properties", [])[:5]:  # 只显示前5个属性
            print(f" - {prop}")
            
        print("\n应用场景:")
        for app in concept.get("applications", []):
            print(f" - {app.get('title')}: {app.get('description')}")
            
    print(f"\n评估结果:")
    if "evaluation" in blend_result:
        eval_data = blend_result["evaluation"]
        print(f"新颖性: {eval_data.get('novelty', 0):.2f}")
        print(f"实用性: {eval_data.get('usefulness', 0):.2f}")
        print(f"可行性: {eval_data.get('feasibility', 0):.2f}")
        print(f"总体评分: {eval_data.get('overall_score', 0):.2f}")
        print(f"评价: {eval_data.get('evaluation_text', '')}")
    
    # ---------- 测试类比推理 ----------
    print_separator("2. 类比推理示例")
    
    analogy_problem = {
        "type": "innovation",
        "description": "如何将自然界的生态系统原理应用到城市规划中",
        "source_domain": "生物学",
        "target_domain": "城市规划"
    }
    
    analogy_result = creative_engine.generate_creative_idea(
        problem=analogy_problem,
        thinking_mode="analogical_reasoning"
    )
    
    print("类比推理结果:")
    print(f"问题: {analogy_problem['description']}")
    
    if "idea" in analogy_result and "analogy" in analogy_result["idea"]:
        analogy = analogy_result["idea"]["analogy"]
        print(f"\n类比关系: {analogy.get('explanation', '')}")
        
        print("\n创新见解:")
        for insight in analogy_result["idea"].get("insights", []):
            print(f" - {insight.get('description')}")
            print(f"   应用: {insight.get('application')}")
    
    # ---------- 测试发散思维 ----------
    print_separator("3. 发散思维示例")
    
    divergent_problem = {
        "type": "ideation",
        "description": "为远程教育开发创新解决方案",
        "central_concept": "远程教育"
    }
    
    divergent_result = creative_engine.generate_creative_idea(
        problem=divergent_problem,
        thinking_mode="divergent_thinking"
    )
    
    print("发散思维结果:")
    print(f"问题: {divergent_problem['description']}")
    print(f"中心概念: {divergent_result['idea'].get('central_concept', '')}")
    
    if "idea" in divergent_result and "branches" in divergent_result["idea"]:
        branches = divergent_result["idea"]["branches"]
        print(f"\n生成了 {len(branches)} 个思维分支:")
        
        for i, branch in enumerate(branches):
            print(f"\n分支 {i+1}: {branch.get('perspective')}")
            print("子想法:")
            for sub_idea in branch.get("sub_ideas", [])[:3]:  # 只显示前3个子想法
                print(f" - {sub_idea}")
        
        print(f"\n多样性评分: {divergent_result['idea'].get('diversity', 0):.2f}")
    
    # ---------- 测试约束放松 ----------
    print_separator("4. 约束放松示例")
    
    constraint_problem = {
        "type": "optimization",
        "description": "在有限预算和时间内提高软件开发效率",
        "constraints": [
            {"name": "预算约束", "description": "项目预算有限", "importance": "high"},
            {"name": "时间约束", "description": "项目截止日期紧迫", "importance": "high"},
            {"name": "人力约束", "description": "团队规模固定", "importance": "medium"}
        ]
    }
    
    constraint_result = creative_engine.generate_creative_idea(
        problem=constraint_problem,
        thinking_mode="constraint_relaxation"
    )
    
    print("约束放松结果:")
    print(f"问题: {constraint_problem['description']}")
    
    if "idea" in constraint_result and "relaxations" in constraint_result["idea"]:
        relaxations = constraint_result["idea"]["relaxations"]
        print(f"\n放松了 {len(relaxations)} 个约束:")
        
        for relaxation in relaxations:
            constraint = relaxation.get("constraint", {})
            print(f"\n约束: {constraint.get('name')} ({constraint.get('importance')})")
            print(f"描述: {constraint.get('description')}")
            print("放松方案:")
            for idea in relaxation.get("relaxation_ideas", []):
                print(f" - {idea.get('title')}: {idea.get('description')}")
        
        print("\n综合解决方案:")
        for solution in constraint_result["idea"].get("solutions", []):
            print(f" - {solution.get('title')}: {solution.get('description')}")
    
    # ---------- 测试横向思维 ----------
    print_separator("5. 横向思维示例")
    
    lateral_problem = {
        "type": "innovation",
        "description": "重新思考传统办公空间的设计和使用方式"
    }
    
    lateral_result = creative_engine.generate_creative_idea(
        problem=lateral_problem,
        thinking_mode="lateral_thinking"
    )
    
    print("横向思维结果:")
    print(f"问题: {lateral_problem['description']}")
    
    if "idea" in lateral_result:
        idea_data = lateral_result["idea"]
        
        if "perspective_shifts" in idea_data:
            shifts = idea_data["perspective_shifts"]
            print(f"\n视角转换 ({len(shifts)}):")
            for shift in shifts[:2]:  # 只显示前2个转换
                print(f" - {shift.get('description')}")
        
        if "provocations" in idea_data:
            provocations = idea_data["provocations"]
            print(f"\n挑战性假设 ({len(provocations)}):")
            for prov in provocations[:2]:  # 只显示前2个假设
                print(f" - {prov.get('statement')}")
        
        if "insights" in idea_data:
            insights = idea_data["insights"]
            print(f"\n关键洞见 ({len(insights)}):")
            for insight in insights:
                print(f" - {insight.get('content')}")
                print(f"   潜在价值: {insight.get('potential_value', 0):.2f}")
    
    # ---------- 自动选择思维模式 ----------
    print_separator("6. 自动选择思维模式示例")
    
    auto_problem = {
        "type": "general",
        "description": "如何在保持社交距离的情况下创造有意义的社交体验"
    }
    
    auto_result = creative_engine.generate_creative_idea(problem=auto_problem)
    
    print("自动选择思维模式结果:")
    print(f"问题: {auto_problem['description']}")
    print(f"自动选择的思维模式: {auto_result.get('thinking_mode')}")
    
    if "evaluation" in auto_result:
        eval_data = auto_result["evaluation"]
        print(f"\n总体评分: {eval_data.get('overall_score', 0):.2f}")
        print(f"评价: {eval_data.get('evaluation_text', '')}")
        
        print("\n优势:")
        for strength in eval_data.get("strengths", []):
            print(f" - {strength}")
            
        print("\n劣势:")
        for weakness in eval_data.get("weaknesses", []):
            print(f" - {weakness}")
    
    print("\n完整结果已保存到文件。")
    
    # 将完整结果保存到文件
    with open("creative_thinking_results.json", "w", encoding="utf-8") as f:
        results = {
            "conceptual_blending": blend_result,
            "analogical_reasoning": analogy_result,
            "divergent_thinking": divergent_result,
            "constraint_relaxation": constraint_result,
            "lateral_thinking": lateral_result,
            "auto_mode": auto_result
        }
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 