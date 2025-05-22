"""
自主进化系统演示
"""

from metacognition.self_evolution import SelfEvolutionEngine
import time

def main():
    # 创建进化引擎实例
    engine = SelfEvolutionEngine()
    
    print("=== 开始自主进化演示 ===")
    
    # 1. 备份当前代码
    print("\n1. 备份当前代码...")
    backup_result = engine.backup_code("metacognition.cognitive_monitor")
    print(f"备份结果: {backup_result}")
    
    # 2. 执行代码进化
    print("\n2. 执行代码进化...")
    def evolution_strategy(code):
        # 示例：优化代码注释
        code = code.replace("# TODO", "# 已完成")
        code = code.replace("# FIXME", "# 已修复")
        return code
    
    evolve_result = engine.evolve_code("metacognition.cognitive_monitor", evolution_strategy)
    print(f"进化结果: {evolve_result}")
    
    # 3. 评估进化效果
    print("\n3. 评估进化效果...")
    eval_result = engine.evaluate_evolution()
    print(f"评估结果: {eval_result}")
    
    # 4. 如果需要，恢复备份
    if eval_result["status"] == "fail":
        print("\n4. 恢复备份...")
        restore_result = engine.restore_backup("metacognition.cognitive_monitor")
        print(f"恢复结果: {restore_result}")
    
    print("\n=== 自主进化演示完成 ===")

if __name__ == "__main__":
    main() 