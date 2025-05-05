import time
import numpy as np
import pandas as pd
from ..core.momentum_model import MomentumMLModel
from ..config.model_config import ModelConfig
import psutil
import os
import logging
from typing import Dict, List, Tuple

class Benchmark:
    """性能基准测试类"""
    
    def __init__(self):
        """初始化基准测试"""
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process(os.getpid())
        self.model = MomentumMLModel()
        
        # 生成测试数据
        self.test_data = self._generate_test_data()
        self.test_labels = np.random.randint(0, 2, 1000)
    
    def _generate_test_data(self, size: int = 1000) -> pd.DataFrame:
        """生成测试数据"""
        data = {}
        for col in self.model.config.feature_columns:
            data[col] = np.random.rand(size)
        return pd.DataFrame(data)
    
    def _measure_memory(self) -> float:
        """测量当前内存使用"""
        return self.process.memory_info().rss / 1024 / 1024  # MB
    
    def _measure_cpu(self) -> float:
        """测量当前CPU使用"""
        return self.process.cpu_percent()
    
    def run_prediction_benchmark(self, n_runs: int = 100) -> Dict[str, float]:
        """
        运行预测性能基准测试
        
        Args:
            n_runs: 运行次数
            
        Returns:
            性能指标字典
        """
        start_memory = self._measure_memory()
        start_cpu = self._measure_cpu()
        start_time = time.time()
        
        for _ in range(n_runs):
            self.model.predict(self.test_data)
        
        end_time = time.time()
        end_cpu = self._measure_cpu()
        end_memory = self._measure_memory()
        
        return {
            'total_time': end_time - start_time,
            'avg_time': (end_time - start_time) / n_runs,
            'memory_used': end_memory - start_memory,
            'cpu_used': end_cpu - start_cpu
        }
    
    def run_training_benchmark(self, n_runs: int = 10) -> Dict[str, float]:
        """
        运行训练性能基准测试
        
        Args:
            n_runs: 运行次数
            
        Returns:
            性能指标字典
        """
        start_memory = self._measure_memory()
        start_cpu = self._measure_cpu()
        start_time = time.time()
        
        for _ in range(n_runs):
            self.model.train(self.test_data, self.test_labels)
        
        end_time = time.time()
        end_cpu = self._measure_cpu()
        end_memory = self._measure_memory()
        
        return {
            'total_time': end_time - start_time,
            'avg_time': (end_time - start_time) / n_runs,
            'memory_used': end_memory - start_memory,
            'cpu_used': end_cpu - start_cpu
        }
    
    def run_scalability_test(self, sizes: List[int] = [100, 1000, 10000]) -> List[Dict[str, float]]:
        """
        运行可扩展性测试
        
        Args:
            sizes: 数据大小列表
            
        Returns:
            各规模下的性能指标列表
        """
        results = []
        for size in sizes:
            test_data = self._generate_test_data(size)
            start_time = time.time()
            self.model.predict(test_data)
            end_time = time.time()
            
            results.append({
                'size': size,
                'time': end_time - start_time,
                'memory': self._measure_memory()
            })
        
        return results
    
    def print_results(self, results: Dict[str, float], test_name: str):
        """打印测试结果"""
        self.logger.info(f"\n{test_name} 测试结果:")
        self.logger.info(f"总时间: {results['total_time']:.4f}秒")
        self.logger.info(f"平均时间: {results['avg_time']:.4f}秒")
        self.logger.info(f"内存使用: {results['memory_used']:.2f}MB")
        self.logger.info(f"CPU使用: {results['cpu_used']:.2f}%")

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行基准测试
    benchmark = Benchmark()
    
    # 预测性能测试
    pred_results = benchmark.run_prediction_benchmark()
    benchmark.print_results(pred_results, "预测")
    
    # 训练性能测试
    train_results = benchmark.run_training_benchmark()
    benchmark.print_results(train_results, "训练")
    
    # 可扩展性测试
    scalability_results = benchmark.run_scalability_test()
    for result in scalability_results:
        logging.info(f"\n数据规模: {result['size']}")
        logging.info(f"预测时间: {result['time']:.4f}秒")
        logging.info(f"内存使用: {result['memory']:.2f}MB") 