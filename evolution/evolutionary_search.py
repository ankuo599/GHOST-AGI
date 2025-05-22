"""
GHOST-AGI 自主进化搜索框架

该模块实现高效解空间搜索能力，使GHOST-AGI能够高效探索复杂问题空间并发现创新解决方案。
"""

import numpy as np
import random
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from collections import defaultdict

class EvolutionarySearch:
    """自主进化搜索框架，提供高效解空间搜索能力"""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 selection_pressure: float = 0.8,
                 logger: Optional[logging.Logger] = None):
        """
        初始化进化搜索框架
        
        Args:
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            selection_pressure: 选择压力
            logger: 日志记录器
        """
        # 进化参数
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 进化算法组件
        self.selection_methods = {
            "tournament": self._tournament_selection,
            "roulette": self._roulette_selection,
            "rank": self._rank_selection,
            "elitism": self._elitism_selection
        }
        
        self.crossover_methods = {
            "one_point": self._one_point_crossover,
            "two_point": self._two_point_crossover,
            "uniform": self._uniform_crossover,
            "blend": self._blend_crossover
        }
        
        self.mutation_methods = {
            "random": self._random_mutation,
            "gaussian": self._gaussian_mutation,
            "swap": self._swap_mutation,
            "inversion": self._inversion_mutation
        }
        
        # 进化状态
        self.current_population = []
        self.fitness_history = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # 搜索空间适应
        self.problem_dimension = 0
        self.problem_bounds = []
        self.constraints = []
        self.fitness_function = None
        
        # 自适应参数
        self.adaptive_params = {
            "enabled": True,
            "adaptation_rate": 0.1,
            "history_window": 10
        }
        
        # 多目标优化支持
        self.objectives = []
        self.pareto_front = []
        
        # 搜索统计
        self.search_stats = {
            "iterations": 0,
            "evaluations": 0,
            "improvement_trend": [],
            "diversity_history": [],
            "last_improvement": 0,
            "stagnation_counter": 0
        }
        
        self.logger.info("自主进化搜索框架初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("EvolutionarySearch")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("evolutionary_search.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def setup_problem(self, 
                     dimension: int, 
                     bounds: List[Tuple[float, float]], 
                     fitness_function: Callable, 
                     constraints: Optional[List[Callable]] = None,
                     objectives: Optional[List[Callable]] = None) -> None:
        """
        配置问题空间
        
        Args:
            dimension: 问题维度
            bounds: 每个维度的边界 [(min1, max1), (min2, max2), ...]
            fitness_function: 适应度函数
            constraints: 约束函数列表
            objectives: 多目标优化的目标函数列表
        """
        self.logger.info(f"配置问题空间: 维度={dimension}")
        
        # 设置问题空间
        self.problem_dimension = dimension
        self.problem_bounds = bounds
        self.fitness_function = fitness_function
        self.constraints = constraints or []
        self.objectives = objectives or []
        
        # 重置搜索状态
        self.current_population = []
        self.fitness_history = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.pareto_front = []
        
        # 重置统计
        self.search_stats = {
            "iterations": 0,
            "evaluations": 0,
            "improvement_trend": [],
            "diversity_history": [],
            "last_improvement": 0,
            "stagnation_counter": 0
        }
    
    def initialize_population(self) -> None:
        """初始化种群"""
        self.logger.info(f"初始化大小为 {self.population_size} 的种群")
        
        if not self.problem_bounds or self.problem_dimension == 0:
            self.logger.error("未配置问题空间，无法初始化种群")
            return
        
        self.current_population = []
        
        # 生成初始个体
        for i in range(self.population_size):
            individual = []
            
            # 在每个维度的边界内随机生成值
            for dim in range(self.problem_dimension):
                if dim < len(self.problem_bounds):
                    lower, upper = self.problem_bounds[dim]
                    value = lower + random.random() * (upper - lower)
                else:
                    # 默认范围 [0, 1]
                    value = random.random()
                
                individual.append(value)
            
            # 确保个体遵守约束条件
            individual = self._repair_individual(individual)
            
            # 添加到种群
            self.current_population.append({
                "genotype": individual,
                "fitness": None,
                "age": 0,
                "origin": "initialization",
                "objectives": []
            })
        
        self.logger.info("种群初始化完成")
    
    def _repair_individual(self, individual: List[float]) -> List[float]:
        """修复违反约束的个体"""
        # 应用边界约束
        for i in range(len(individual)):
            if i < len(self.problem_bounds):
                lower, upper = self.problem_bounds[i]
                individual[i] = max(lower, min(upper, individual[i]))
        
        # 应用自定义约束 (简化版本)
        for constraint in self.constraints:
            while not self._check_constraint(individual, constraint):
                # 随机修改一个维度的值
                dim = random.randint(0, len(individual) - 1)
                if dim < len(self.problem_bounds):
                    lower, upper = self.problem_bounds[dim]
                    individual[dim] = lower + random.random() * (upper - lower)
                else:
                    individual[dim] = random.random()
        
        return individual
    
    def _check_constraint(self, individual: List[float], constraint: Callable) -> bool:
        """检查个体是否满足约束"""
        try:
            return constraint(individual)
        except Exception as e:
            self.logger.warning(f"约束检查异常: {str(e)}")
            return False
    
    def evaluate_population(self) -> Dict[str, Any]:
        """
        评估种群适应度
        
        Returns:
            评估结果
        """
        self.logger.info(f"评估第 {self.generation} 代种群")
        
        if not self.fitness_function:
            self.logger.error("未设置适应度函数，无法评估种群")
            return {"error": "未设置适应度函数"}
        
        fitness_scores = []
        
        for i, individual in enumerate(self.current_population):
            if individual["fitness"] is None:  # 只评估新个体
                try:
                    # 评估单目标适应度
                    fitness = self.fitness_function(individual["genotype"])
                    individual["fitness"] = fitness
                    
                    # 评估多目标适应度
                    if self.objectives:
                        objectives = []
                        for obj_func in self.objectives:
                            obj_value = obj_func(individual["genotype"])
                            objectives.append(obj_value)
                        individual["objectives"] = objectives
                    
                    fitness_scores.append(fitness)
                    
                    # 更新最佳个体
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_individual = individual.copy()
                        self.search_stats["last_improvement"] = self.generation
                        self.search_stats["stagnation_counter"] = 0
                    
                    # 增加评估计数
                    self.search_stats["evaluations"] += 1
                    
                except Exception as e:
                    self.logger.error(f"评估个体 {i} 时出错: {str(e)}")
                    individual["fitness"] = float('-inf')
                    fitness_scores.append(float('-inf'))
            else:
                fitness_scores.append(individual["fitness"])
        
        # 更新种群年龄
        for individual in self.current_population:
            individual["age"] += 1
        
        # 计算统计信息
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        min_fitness = min(fitness_scores) if fitness_scores else 0
        max_fitness = max(fitness_scores) if fitness_scores else 0
        
        # 记录历史
        generation_record = {
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "min_fitness": min_fitness,
            "max_fitness": max_fitness,
            "diversity": self._calculate_diversity(),
            "timestamp": time.time()
        }
        self.fitness_history.append(generation_record)
        
        # 更新搜索统计
        self.search_stats["improvement_trend"].append(max_fitness)
        self.search_stats["diversity_history"].append(self._calculate_diversity())
        
        # 检查是否停滞
        if self.generation > self.search_stats["last_improvement"]:
            self.search_stats["stagnation_counter"] += 1
        
        # 更新多目标优化的Pareto前沿
        if self.objectives:
            self._update_pareto_front()
        
        return {
            "avg_fitness": avg_fitness,
            "min_fitness": min_fitness,
            "max_fitness": max_fitness,
            "best_fitness_ever": self.best_fitness,
            "diversity": self._calculate_diversity(),
            "evaluated_individuals": len(fitness_scores)
        }
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if not self.current_population:
            return 0.0
        
        # 计算种群中心
        center = np.zeros(self.problem_dimension)
        for individual in self.current_population:
            center += np.array(individual["genotype"])
        center /= len(self.current_population)
        
        # 计算到中心的平均距离
        total_distance = 0.0
        for individual in self.current_population:
            distance = np.linalg.norm(np.array(individual["genotype"]) - center)
            total_distance += distance
        
        avg_distance = total_distance / len(self.current_population)
        
        # 正规化多样性度量，以问题空间对角线长度为参考
        if self.problem_bounds:
            space_diagonal = 0.0
            for lower, upper in self.problem_bounds:
                space_diagonal += (upper - lower) ** 2
            space_diagonal = np.sqrt(space_diagonal)
            
            if space_diagonal > 0:
                normalized_diversity = avg_distance / space_diagonal
                return normalized_diversity
        
        return avg_distance
    
    def _update_pareto_front(self) -> None:
        """更新Pareto前沿"""
        if not self.objectives or not self.current_population:
            return
        
        # 重置Pareto前沿
        self.pareto_front = []
        
        # 检查每个个体是否在Pareto前沿
        for i, ind_i in enumerate(self.current_population):
            dominated = False
            
            for j, ind_j in enumerate(self.current_population):
                if i == j:
                    continue
                
                # 检查ind_j是否支配ind_i
                if all(ind_j["objectives"][k] <= ind_i["objectives"][k] for k in range(len(self.objectives))) and \
                   any(ind_j["objectives"][k] < ind_i["objectives"][k] for k in range(len(self.objectives))):
                    dominated = True
                    break
            
            if not dominated:
                self.pareto_front.append(ind_i.copy())
    
    def evolve_population(self, 
                        selection_method: str = "tournament", 
                        crossover_method: str = "two_point",
                        mutation_method: str = "gaussian") -> Dict[str, Any]:
        """
        进化种群
        
        Args:
            selection_method: 选择方法
            crossover_method: 交叉方法
            mutation_method: 变异方法
            
        Returns:
            进化结果
        """
        self.logger.info(f"进化第 {self.generation} 代种群")
        
        if not self.current_population:
            self.logger.error("种群为空，无法进行进化")
            return {"error": "种群为空"}
        
        # 选择父代
        selection_func = self.selection_methods.get(selection_method, self._tournament_selection)
        parents = selection_func(self.current_population)
        
        # 生成子代
        offspring = []
        
        # 交叉
        crossover_func = self.crossover_methods.get(crossover_method, self._two_point_crossover)
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover_func(parent1["genotype"], parent2["genotype"])
                    
                    offspring.append({
                        "genotype": child1,
                        "fitness": None,
                        "age": 0,
                        "origin": "crossover",
                        "objectives": []
                    })
                    
                    offspring.append({
                        "genotype": child2,
                        "fitness": None,
                        "age": 0,
                        "origin": "crossover",
                        "objectives": []
                    })
        
        # 变异
        mutation_func = self.mutation_methods.get(mutation_method, self._gaussian_mutation)
        for individual in offspring:
            if random.random() < self.mutation_rate:
                individual["genotype"] = mutation_func(individual["genotype"])
                individual["origin"] = individual["origin"] + "+mutation"
        
        # 确保子代满足约束条件
        for individual in offspring:
            individual["genotype"] = self._repair_individual(individual["genotype"])
        
        # 合并精英个体和子代
        elite_count = max(1, int(self.population_size * 0.1))  # 保留10%的精英
        elite_individuals = self._select_elite(self.current_population, elite_count)
        
        # 从子代中选择剩余个体
        remaining_count = self.population_size - len(elite_individuals)
        if len(offspring) > remaining_count:
            offspring = offspring[:remaining_count]
        
        # 如果子代不足，随机生成新个体补充
        while len(offspring) < remaining_count:
            new_individual = []
            for dim in range(self.problem_dimension):
                if dim < len(self.problem_bounds):
                    lower, upper = self.problem_bounds[dim]
                    value = lower + random.random() * (upper - lower)
                else:
                    value = random.random()
                new_individual.append(value)
            
            offspring.append({
                "genotype": self._repair_individual(new_individual),
                "fitness": None,
                "age": 0,
                "origin": "random",
                "objectives": []
            })
        
        # 更新种群
        self.current_population = elite_individuals + offspring
        
        # 自适应参数调整
        if self.adaptive_params["enabled"]:
            self._adapt_parameters()
        
        # 增加代数
        self.generation += 1
        self.search_stats["iterations"] += 1
        
        return {
            "generation": self.generation,
            "elite_count": len(elite_individuals),
            "offspring_count": len(offspring),
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate
        }
    
    def _adapt_parameters(self) -> None:
        """自适应调整进化参数"""
        if len(self.fitness_history) < self.adaptive_params["history_window"]:
            return
        
        recent_history = self.fitness_history[-self.adaptive_params["history_window"]:]
        
        # 计算改进率
        improvements = [h["max_fitness"] for h in recent_history]
        if len(improvements) >= 2:
            improvement_rate = (improvements[-1] - improvements[0]) / max(1e-10, improvements[0])
            
            # 种群多样性
            diversity = self._calculate_diversity()
            
            # 根据改进率和多样性调整参数
            adaptation_rate = self.adaptive_params["adaptation_rate"]
            
            if improvement_rate < 0.001:  # 停滞
                if diversity < 0.1:  # 多样性低，增加变异率
                    self.mutation_rate += adaptation_rate
                    self.crossover_rate -= adaptation_rate * 0.5
                else:  # 多样性尚可，增加选择压力
                    self.selection_pressure += adaptation_rate
            else:  # 有改进
                if diversity < 0.1:  # 多样性低，略微增加变异
                    self.mutation_rate += adaptation_rate * 0.3
                else:  # 一切正常，微调
                    self.mutation_rate *= (1.0 - adaptation_rate * 0.1)
                    self.crossover_rate *= (1.0 + adaptation_rate * 0.1)
            
            # 确保参数在合理范围内
            self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))
            self.crossover_rate = max(0.5, min(0.95, self.crossover_rate))
            self.selection_pressure = max(0.5, min(0.9, self.selection_pressure))
    
    def _select_elite(self, population: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """选择精英个体"""
        sorted_population = sorted(population, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else float('-inf'), reverse=True)
        return sorted_population[:count]
    
    # 选择方法
    def _tournament_selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """锦标赛选择"""
        selected = []
        tournament_size = max(2, int(len(population) * 0.2))  # 约20%的种群参与每次锦标赛
        
        while len(selected) < len(population):
            # 随机选择候选者
            candidates = random.sample(population, tournament_size)
            
            # 选择最优个体
            winner = max(candidates, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else float('-inf'))
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """轮盘赌选择"""
        selected = []
        
        # 计算总适应度和调整后的适应度
        total_fitness = 0
        min_fitness = float('inf')
        
        for ind in population:
            fitness = ind["fitness"] if ind["fitness"] is not None else float('-inf')
            min_fitness = min(min_fitness, fitness)
        
        # 调整为非负值
        adjusted_fitness = []
        adjusted_total = 0
        
        for ind in population:
            fitness = ind["fitness"] if ind["fitness"] is not None else float('-inf')
            adj_fitness = fitness - min_fitness + 1e-10  # 确保非负且非零
            adjusted_fitness.append(adj_fitness)
            adjusted_total += adj_fitness
        
        # 轮盘赌选择
        while len(selected) < len(population):
            pick = random.uniform(0, adjusted_total)
            current = 0
            
            for i, adj_fitness in enumerate(adjusted_fitness):
                current += adj_fitness
                if current > pick:
                    selected.append(population[i])
                    break
        
        return selected
    
    def _rank_selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """排名选择"""
        selected = []
        
        # 按适应度排序
        sorted_population = sorted(population, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else float('-inf'))
        
        # 分配选择概率按排名
        ranks = list(range(1, len(sorted_population) + 1))
        rank_sum = sum(ranks)
        selection_probs = [r / rank_sum for r in ranks]
        
        # 基于排名概率选择
        while len(selected) < len(population):
            picked_idx = random.choices(range(len(sorted_population)), weights=selection_probs)[0]
            selected.append(sorted_population[picked_idx])
        
        return selected
    
    def _elitism_selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """精英选择"""
        sorted_population = sorted(population, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else float('-inf'), reverse=True)
        return sorted_population[:len(population)]
    
    # 交叉方法
    def _one_point_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """单点交叉"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """两点交叉"""
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1, parent2
        
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """均匀交叉"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        child1 = []
        child2 = []
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def _blend_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """混合交叉"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        alpha = 0.5  # 混合系数
        
        child1 = []
        child2 = []
        
        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            
            # 扩展范围
            min_bound = min_val - range_val * alpha
            max_bound = max_val + range_val * alpha
            
            # 在扩展范围内生成子代
            child1.append(random.uniform(min_bound, max_bound))
            child2.append(random.uniform(min_bound, max_bound))
        
        return child1, child2
    
    # 变异方法
    def _random_mutation(self, individual: List[float]) -> List[float]:
        """随机变异"""
        mutated = individual.copy()
        
        # 随机选择维度进行变异
        mutation_points = max(1, int(len(individual) * self.mutation_rate))
        mutation_indices = random.sample(range(len(individual)), mutation_points)
        
        for idx in mutation_indices:
            if idx < len(self.problem_bounds):
                lower, upper = self.problem_bounds[idx]
                mutated[idx] = lower + random.random() * (upper - lower)
            else:
                mutated[idx] = random.random()
        
        return mutated
    
    def _gaussian_mutation(self, individual: List[float]) -> List[float]:
        """高斯变异"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # 计算变异幅度
                if i < len(self.problem_bounds):
                    lower, upper = self.problem_bounds[i]
                    range_val = upper - lower
                    sigma = range_val * 0.1  # 标准差为10%的范围
                else:
                    sigma = 0.1
                
                # 应用高斯变异
                mutation = random.gauss(0, sigma)
                mutated[i] += mutation
        
        return mutated
    
    def _swap_mutation(self, individual: List[float]) -> List[float]:
        """交换变异"""
        if len(individual) < 2:
            return individual
        
        mutated = individual.copy()
        
        # 随机选择两个位置交换
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def _inversion_mutation(self, individual: List[float]) -> List[float]:
        """反转变异"""
        if len(individual) < 2:
            return individual
        
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            # 选择子序列
            idx1 = random.randint(0, len(individual) - 2)
            idx2 = random.randint(idx1 + 1, len(individual) - 1)
            
            # 反转子序列
            mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])
        
        return mutated
    
    def run_evolutionary_search(self, 
                              max_generations: int = 100,
                              target_fitness: Optional[float] = None,
                              selection_method: str = "tournament",
                              crossover_method: str = "two_point",
                              mutation_method: str = "gaussian",
                              early_stopping: bool = True,
                              early_stopping_generations: int = 20) -> Dict[str, Any]:
        """
        运行进化搜索
        
        Args:
            max_generations: 最大代数
            target_fitness: 目标适应度
            selection_method: 选择方法
            crossover_method: 交叉方法
            mutation_method: 变异方法
            early_stopping: 是否启用早停
            early_stopping_generations: 早停代数
            
        Returns:
            搜索结果
        """
        self.logger.info(f"开始进化搜索: 最大代数={max_generations}, 选择={selection_method}, 交叉={crossover_method}, 变异={mutation_method}")
        
        start_time = time.time()
        
        # 初始化种群（如果尚未初始化）
        if not self.current_population:
            self.initialize_population()
        
        best_fitness_history = []
        
        for gen in range(max_generations):
            self.generation = gen
            
            # 评估种群
            eval_results = self.evaluate_population()
            
            # 记录最佳适应度
            current_best = eval_results.get("max_fitness", float('-inf'))
            best_fitness_history.append(current_best)
            
            # 检查是否达到目标适应度
            if target_fitness is not None and current_best >= target_fitness:
                self.logger.info(f"达到目标适应度 {target_fitness}，在第 {gen} 代停止")
                break
            
            # 检查早停条件
            if early_stopping and gen > early_stopping_generations:
                recent_best = best_fitness_history[-early_stopping_generations:]
                if all(abs(recent_best[i] - recent_best[0]) < 1e-6 for i in range(1, len(recent_best))):
                    self.logger.info(f"连续 {early_stopping_generations} 代没有改进，在第 {gen} 代停止")
                    break
            
            # 进化种群
            if gen < max_generations - 1:  # 最后一代不需要进化
                self.evolve_population(
                    selection_method=selection_method,
                    crossover_method=crossover_method,
                    mutation_method=mutation_method
                )
        
        elapsed_time = time.time() - start_time
        
        # 最终评估
        final_eval = self.evaluate_population()
        
        # 准备搜索结果
        search_result = {
            "best_individual": self.best_individual["genotype"] if self.best_individual else None,
            "best_fitness": self.best_fitness,
            "generations": self.generation + 1,
            "evaluations": self.search_stats["evaluations"],
            "elapsed_time": elapsed_time,
            "fitness_history": [h["max_fitness"] for h in self.fitness_history],
            "diversity_history": [h["diversity"] for h in self.fitness_history]
        }
        
        if self.objectives:
            search_result["pareto_front_size"] = len(self.pareto_front)
            search_result["pareto_front"] = [
                {
                    "genotype": ind["genotype"],
                    "objectives": ind["objectives"]
                }
                for ind in self.pareto_front
            ]
        
        self.logger.info(f"进化搜索完成: 最佳适应度={self.best_fitness}, 代数={self.generation+1}, 耗时={elapsed_time:.2f}秒")
        
        return search_result
    
    def get_search_status(self) -> Dict[str, Any]:
        """
        获取搜索状态
        
        Returns:
            搜索状态
        """
        return {
            "current_generation": self.generation,
            "population_size": len(self.current_population),
            "best_fitness_ever": self.best_fitness,
            "evaluations": self.search_stats["evaluations"],
            "stagnation_generations": self.search_stats["stagnation_counter"],
            "current_diversity": self._calculate_diversity() if self.current_population else 0.0,
            "adaptive_parameters": {
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "selection_pressure": self.selection_pressure
            }
        }
    
    def save_state(self, file_path: str) -> Dict[str, Any]:
        """
        保存搜索状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存结果
        """
        try:
            state = {
                "search_parameters": {
                    "population_size": self.population_size,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "selection_pressure": self.selection_pressure
                },
                "problem_definition": {
                    "dimension": self.problem_dimension,
                    "bounds": self.problem_bounds
                },
                "current_state": {
                    "generation": self.generation,
                    "best_fitness": self.best_fitness,
                    "best_individual": self.best_individual["genotype"] if self.best_individual else None
                },
                "search_stats": self.search_stats,
                "saved_at": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"搜索状态已保存到: {file_path}")
            
            return {"success": True, "file_path": file_path}
        
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            return {"success": False, "error": str(e)} 