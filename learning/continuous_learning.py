"""
持续学习系统
实现多种学习策略和元认知监控
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
import time
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class LearningTask:
    """学习任务"""
    id: str
    name: str
    type: str  # 任务类型：classification, regression, reinforcement
    data: Dict[str, Any]  # 训练数据
    model: Optional[nn.Module] = None  # 模型
    optimizer: Optional[torch.optim.Optimizer] = None  # 优化器
    metrics: Dict[str, float] = None  # 性能指标
    status: str = "pending"  # 任务状态
    created_at: float = time.time()
    updated_at: float = time.time()

@dataclass
class LearningStrategy:
    """学习策略"""
    name: str
    type: str  # 策略类型：exploration, exploitation, curriculum, active, transfer
    parameters: Dict[str, Any]  # 策略参数
    performance: Dict[str, float] = None  # 策略性能
    created_at: float = time.time()
    updated_at: float = time.time()

class ContinuousLearner:
    """持续学习器"""
    def __init__(self, model_dir: str = "models"):
        self.tasks: Dict[str, LearningTask] = {}
        self.strategies: Dict[str, LearningStrategy] = {}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ContinuousLearner")
        self.performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._init_strategies()
        
    def _init_strategies(self):
        """初始化学习策略"""
        # 探索策略
        self.strategies["exploration"] = LearningStrategy(
            name="exploration",
            type="exploration",
            parameters={
                "epsilon": 0.1,  # 探索率
                "decay_rate": 0.995,  # 衰减率
                "min_epsilon": 0.01  # 最小探索率
            }
        )
        
        # 利用策略
        self.strategies["exploitation"] = LearningStrategy(
            name="exploitation",
            type="exploitation",
            parameters={
                "confidence_threshold": 0.8,  # 置信度阈值
                "max_samples": 1000  # 最大样本数
            }
        )
        
        # 课程学习策略
        self.strategies["curriculum"] = LearningStrategy(
            name="curriculum",
            type="curriculum",
            parameters={
                "difficulty_levels": 5,  # 难度等级
                "samples_per_level": 100  # 每级样本数
            }
        )
        
        # 主动学习策略
        self.strategies["active"] = LearningStrategy(
            name="active",
            type="active",
            parameters={
                "query_strategy": "uncertainty",  # 查询策略
                "batch_size": 10,  # 批次大小
                "max_queries": 100  # 最大查询次数
            }
        )
        
        # 迁移学习策略
        self.strategies["transfer"] = LearningStrategy(
            name="transfer",
            type="transfer",
            parameters={
                "source_domains": [],  # 源领域
                "target_domain": None,  # 目标领域
                "transfer_type": "fine_tuning"  # 迁移类型
            }
        )
        
    def create_task(self, task: LearningTask) -> bool:
        """创建学习任务"""
        try:
            self.tasks[task.id] = task
            return True
        except Exception as e:
            self.logger.error(f"创建任务失败: {str(e)}")
            return False
            
    def train_task(self, task_id: str, strategy_name: str) -> bool:
        """训练任务"""
        if task_id not in self.tasks or strategy_name not in self.strategies:
            return False
            
        task = self.tasks[task_id]
        strategy = self.strategies[strategy_name]
        
        try:
            # 准备数据
            train_data = self._prepare_data(task.data, strategy)
            
            # 训练模型
            task.model.train()
            for epoch in range(strategy.parameters.get("epochs", 10)):
                for batch in train_data:
                    # 前向传播
                    outputs = task.model(batch["inputs"])
                    loss = self._calculate_loss(outputs, batch["targets"])
                    
                    # 反向传播
                    task.optimizer.zero_grad()
                    loss.backward()
                    task.optimizer.step()
                    
                # 更新策略参数
                self._update_strategy(strategy, epoch)
                
                # 评估性能
                metrics = self._evaluate_task(task)
                self.performance_history[task_id].append({
                    "epoch": epoch,
                    "metrics": metrics,
                    "strategy": strategy_name
                })
                
            # 更新任务状态
            task.status = "completed"
            task.updated_at = time.time()
            task.metrics = metrics
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练任务失败: {str(e)}")
            task.status = "failed"
            return False
            
    def _prepare_data(self, data: Dict[str, Any], 
                     strategy: LearningStrategy) -> DataLoader:
        """准备训练数据"""
        if strategy.type == "curriculum":
            return self._prepare_curriculum_data(data, strategy)
        elif strategy.type == "active":
            return self._prepare_active_data(data, strategy)
        else:
            return DataLoader(
                CustomDataset(data),
                batch_size=strategy.parameters.get("batch_size", 32),
                shuffle=True
            )
            
    def _prepare_curriculum_data(self, data: Dict[str, Any], 
                               strategy: LearningStrategy) -> DataLoader:
        """准备课程学习数据"""
        # 按难度排序数据
        sorted_data = sorted(
            data["samples"],
            key=lambda x: x["difficulty"]
        )
        
        # 分批返回数据
        current_level = 0
        samples_per_level = strategy.parameters["samples_per_level"]
        
        while current_level < strategy.parameters["difficulty_levels"]:
            level_data = sorted_data[
                current_level * samples_per_level:
                (current_level + 1) * samples_per_level
            ]
            
            yield DataLoader(
                CustomDataset({"samples": level_data}),
                batch_size=strategy.parameters.get("batch_size", 32),
                shuffle=True
            )
            
            current_level += 1
            
    def _prepare_active_data(self, data: Dict[str, Any], 
                           strategy: LearningStrategy) -> DataLoader:
        """准备主动学习数据"""
        # 选择最有价值的样本
        if strategy.parameters["query_strategy"] == "uncertainty":
            selected_samples = self._select_uncertain_samples(
                data["samples"],
                strategy.parameters["batch_size"]
            )
        else:
            selected_samples = data["samples"]
            
        return DataLoader(
            CustomDataset({"samples": selected_samples}),
            batch_size=strategy.parameters["batch_size"],
            shuffle=True
        )
        
    def _select_uncertain_samples(self, samples: List[Dict[str, Any]], 
                                batch_size: int) -> List[Dict[str, Any]]:
        """选择不确定性高的样本"""
        # 计算每个样本的不确定性
        uncertainties = []
        for sample in samples:
            with torch.no_grad():
                outputs = self.tasks[sample["task_id"]].model(sample["inputs"])
                uncertainty = self._calculate_uncertainty(outputs)
                uncertainties.append((sample, uncertainty))
                
        # 选择不确定性最高的样本
        selected = sorted(
            uncertainties,
            key=lambda x: x[1],
            reverse=True
        )[:batch_size]
        
        return [s[0] for s in selected]
        
    def _calculate_uncertainty(self, outputs: torch.Tensor) -> float:
        """计算预测的不确定性"""
        probs = torch.softmax(outputs, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return float(entropy)
        
    def _calculate_loss(self, outputs: torch.Tensor, 
                       targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        return nn.CrossEntropyLoss()(outputs, targets)
        
    def _update_strategy(self, strategy: LearningStrategy, epoch: int):
        """更新策略参数"""
        if strategy.type == "exploration":
            # 更新探索率
            strategy.parameters["epsilon"] = max(
                strategy.parameters["min_epsilon"],
                strategy.parameters["epsilon"] * strategy.parameters["decay_rate"]
            )
            
    def _evaluate_task(self, task: LearningTask) -> Dict[str, float]:
        """评估任务性能"""
        task.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in DataLoader(
                CustomDataset(task.data),
                batch_size=32
            ):
                outputs = task.model(batch["inputs"])
                preds = torch.argmax(outputs, dim=-1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch["targets"].cpu().numpy())
                
        return {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, average="weighted"),
            "recall": recall_score(targets, predictions, average="weighted"),
            "f1": f1_score(targets, predictions, average="weighted")
        }
        
    def save_model(self, task_id: str) -> bool:
        """保存模型"""
        if task_id not in self.tasks:
            return False
            
        try:
            task = self.tasks[task_id]
            model_path = self.model_dir / f"{task_id}.pt"
            
            torch.save({
                "model_state": task.model.state_dict(),
                "optimizer_state": task.optimizer.state_dict(),
                "metrics": task.metrics,
                "updated_at": task.updated_at
            }, model_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False
            
    def load_model(self, task_id: str) -> bool:
        """加载模型"""
        if task_id not in self.tasks:
            return False
            
        try:
            task = self.tasks[task_id]
            model_path = self.model_dir / f"{task_id}.pt"
            
            if not model_path.exists():
                return False
                
            checkpoint = torch.load(model_path)
            task.model.load_state_dict(checkpoint["model_state"])
            task.optimizer.load_state_dict(checkpoint["optimizer_state"])
            task.metrics = checkpoint["metrics"]
            task.updated_at = checkpoint["updated_at"]
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
            
    def get_performance_history(self, task_id: str) -> List[Dict[str, Any]]:
        """获取性能历史"""
        return self.performance_history.get(task_id, [])
        
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """获取策略性能"""
        if strategy_name not in self.strategies:
            return {}
            
        strategy = self.strategies[strategy_name]
        if not strategy.performance:
            return {}
            
        return strategy.performance
        
    def update_strategy(self, strategy_name: str, 
                       parameters: Dict[str, Any]) -> bool:
        """更新策略参数"""
        if strategy_name not in self.strategies:
            return False
            
        try:
            strategy = self.strategies[strategy_name]
            strategy.parameters.update(parameters)
            strategy.updated_at = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"更新策略失败: {str(e)}")
            return False

class CustomDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data: Dict[str, Any]):
        self.samples = data["samples"]
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "inputs": torch.tensor(sample["inputs"], dtype=torch.float32),
            "targets": torch.tensor(sample["targets"], dtype=torch.long)
        } 