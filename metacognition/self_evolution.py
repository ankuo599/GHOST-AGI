"""
自主进化模块 (Self-Evolution)

该模块负责系统的自我修改、自我优化和自我扩展。
支持代码自演化、结构自优化、功能自扩展等能力。
"""

import os
import importlib
import shutil
import time
import logging

class SelfEvolutionEngine:
    def __init__(self, target_modules=None, backup_dir="evolution_backups"):
        self.target_modules = target_modules or ["metacognition.cognitive_monitor"]
        self.backup_dir = backup_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def backup_code(self, module_name):
        """备份目标模块代码"""
        try:
            module = importlib.import_module(module_name)
            file_path = module.__file__
            backup_path = os.path.join(self.backup_dir, f"{os.path.basename(file_path)}.{int(time.time())}.bak")
            shutil.copy(file_path, backup_path)
            self.logger.info(f"已备份 {file_path} 到 {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"备份失败: {e}")
            return None

    def evolve_code(self, module_name, evolution_fn):
        """对目标模块进行自演化(自动修改)"""
        backup = self.backup_code(module_name)
        if not backup:
            return {"status": "error", "message": "备份失败，终止演化"}
        try:
            module = importlib.import_module(module_name)
            file_path = module.__file__
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            evolved_code = evolution_fn(code)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(evolved_code)
            self.logger.info(f"{module_name} 已完成自演化")
            return {"status": "success", "message": "自演化完成"}
        except Exception as e:
            self.logger.error(f"自演化失败: {e}")
            return {"status": "error", "message": str(e)}

    def restore_backup(self, module_name):
        """恢复最近的备份"""
        try:
            module = importlib.import_module(module_name)
            file_path = module.__file__
            backups = [f for f in os.listdir(self.backup_dir) if f.startswith(os.path.basename(file_path))]
            if not backups:
                return {"status": "error", "message": "无可用备份"}
            latest_backup = sorted(backups)[-1]
            backup_path = os.path.join(self.backup_dir, latest_backup)
            shutil.copy(backup_path, file_path)
            self.logger.info(f"已恢复 {file_path} 到 {backup_path}")
            return {"status": "success", "message": "已恢复最近备份"}
        except Exception as e:
            self.logger.error(f"恢复失败: {e}")
            return {"status": "error", "message": str(e)}

    def auto_evolve_all(self, evolution_fn):
        """对所有目标模块自动自演化"""
        results = {}
        for module in self.target_modules:
            results[module] = self.evolve_code(module, evolution_fn)
        return results

    def evaluate_evolution(self, test_cmd="pytest", module_name=None):
        """评估自演化效果，自动运行测试"""
        import subprocess
        try:
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("自演化后测试全部通过")
                return {"status": "success", "output": result.stdout}
            else:
                self.logger.warning("自演化后测试未全部通过")
                return {"status": "fail", "output": result.stdout + result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# 示例：简单的进化函数

def simple_evolution_fn(code):
    """示例：将所有注释中的'TODO'替换为'已处理'"""
    return code.replace("TODO", "已处理")

# 用法示例
if __name__ == "__main__":
    engine = SelfEvolutionEngine()
    result = engine.auto_evolve_all(simple_evolution_fn)
    print(result)
    eval_result = engine.evaluate_evolution()
    print(eval_result) 