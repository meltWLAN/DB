#!/usr/bin/env python3
"""
策略参数自动优化模块
提供自动化的策略参数优化功能，寻找最佳参数组合
"""
import os
import sys
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入策略模块
from ma_cross_strategy import MACrossStrategy
from momentum_analysis import MomentumAnalyzer
from gui_controller import GuiController

# 配置日志
logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"strategy_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """策略参数优化器类"""
    
    def __init__(self, use_tushare=False):
        """初始化优化器"""
        self.controller = GuiController(use_tushare=use_tushare)
        self.ma_strategy = MACrossStrategy(use_tushare=use_tushare)
        self.momentum_analyzer = MomentumAnalyzer(use_tushare=use_tushare)
        
        # 创建结果目录
        self.results_dir = os.path.join("results", "optimization")
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("策略参数优化器初始化完成")
    
    def optimize_ma_parameters(self, stock_list=None, 
                             short_ma_range=(3, 20), 
                             long_ma_range=(20, 120),
                             stop_loss_range=(0.03, 0.1),
                             initial_capital=100000,
                             scoring_metric="sharpe_ratio",
                             max_combinations=100,
                             parallel=True):
        """
        优化均线交叉策略参数
        
        参数:
            stock_list: DataFrame, 股票列表 
            short_ma_range: tuple, 短期均线参数范围(start, end)
            long_ma_range: tuple, 长期均线参数范围(start, end)
            stop_loss_range: tuple, 止损参数范围(start, end)
            initial_capital: float, 初始资金
            scoring_metric: str, 评分指标 ('total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown')
            max_combinations: int, 最大参数组合数量
            parallel: bool, 是否使用并行处理
        
        返回:
            dict, 最佳参数和回测结果
        """
        # 准备股票列表
        if stock_list is None:
            logger.info("获取股票列表...")
            stock_list = self.controller.get_stock_list()
            # 如果样本太大，限制数量
            if len(stock_list) > 20:
                logger.info(f"限制股票样本数量为20（原数量: {len(stock_list)}）")
                stock_list = stock_list.sample(20, random_state=42)
        
        # 生成参数组合
        short_ma_values = list(range(short_ma_range[0], short_ma_range[1] + 1))
        long_ma_values = list(range(long_ma_range[0], long_ma_range[1] + 1))
        stop_loss_values = np.linspace(stop_loss_range[0], stop_loss_range[1], 5).tolist()
        
        # 筛选有效的参数组合(确保短期均线小于长期均线)
        parameter_combinations = []
        for short_ma in short_ma_values:
            for long_ma in long_ma_values:
                if short_ma < long_ma:
                    for stop_loss in stop_loss_values:
                        parameter_combinations.append({
                            'short_ma': short_ma,
                            'long_ma': long_ma,
                            'stop_loss_pct': stop_loss,
                            'initial_capital': initial_capital
                        })
        
        # 如果组合太多，随机抽样
        if len(parameter_combinations) > max_combinations:
            logger.info(f"参数组合数量({len(parameter_combinations)})超过最大限制({max_combinations})，随机抽样")
            np.random.seed(42)
            parameter_combinations = np.random.choice(parameter_combinations, max_combinations, replace=False).tolist()
        
        logger.info(f"开始优化 {len(parameter_combinations)} 种参数组合...")
        
        # 选择并行或串行处理
        if parallel and len(parameter_combinations) > 10:
            results = self._parallel_optimize(stock_list, parameter_combinations, scoring_metric)
        else:
            results = self._sequential_optimize(stock_list, parameter_combinations, scoring_metric)
        
        # 查找最佳参数
        best_params, best_result = self._find_best_parameters(results, scoring_metric)
        
        # 输出结果
        self._output_optimization_results(results, best_params, best_result, scoring_metric)
        
        logger.info(f"参数优化完成，最佳参数: {best_params}")
        
        return {
            'best_parameters': best_params,
            'best_result': best_result,
            'all_results': results
        }
    
    def _parallel_optimize(self, stock_list, parameter_combinations, scoring_metric):
        """并行优化参数"""
        results = []
        # 获取可用CPU核心数
        max_workers = min(os.cpu_count() or 4, 8)  # 最多使用8个核心
        logger.info(f"使用并行处理，工作进程数: {max_workers}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = []
            for params in parameter_combinations:
                futures.append(executor.submit(
                    self._evaluate_parameters, stock_list, params, scoring_metric
                ))
            
            # 获取结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="优化进度"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"参数评估出错: {str(e)}")
        
        return results
    
    def _sequential_optimize(self, stock_list, parameter_combinations, scoring_metric):
        """串行优化参数"""
        results = []
        for params in tqdm(parameter_combinations, desc="优化进度"):
            try:
                result = self._evaluate_parameters(stock_list, params, scoring_metric)
                results.append(result)
            except Exception as e:
                logger.error(f"参数评估出错: {str(e)}")
        
        return results
    
    def _evaluate_parameters(self, stock_list, params, scoring_metric):
        """评估单组参数"""
        try:
            # 运行策略
            strategy_results = self.ma_strategy.run_strategy(
                stock_list=stock_list,
                short_ma=params['short_ma'],
                long_ma=params['long_ma'],
                initial_capital=params['initial_capital'],
                stop_loss_pct=params['stop_loss_pct']
            )
            
            # 计算综合绩效
            performance_metrics = self._calculate_portfolio_performance(strategy_results)
            
            # 返回参数和评估结果
            return {
                'parameters': params,
                'performance': performance_metrics,
                'stock_results': strategy_results
            }
        except Exception as e:
            logger.error(f"评估参数时出错: {params}, 错误: {str(e)}")
            raise
    
    def _calculate_portfolio_performance(self, strategy_results):
        """计算投资组合整体表现"""
        if not strategy_results or len(strategy_results) == 0:
            return {
                'total_return': -999,
                'annual_return': -999,
                'max_drawdown': 1,
                'win_rate': 0,
                'sharpe_ratio': -999,
                'sortino_ratio': -999,
                'profit_factor': 0,
                'recovery_factor': 0,
                'trades_count': 0
            }
        
        # 提取有效结果
        valid_results = [r for r in strategy_results if r.get('total_return', None) is not None]
        if not valid_results:
            return {
                'total_return': -999,
                'annual_return': -999,
                'max_drawdown': 1,
                'win_rate': 0,
                'sharpe_ratio': -999,
                'sortino_ratio': -999,
                'profit_factor': 0,
                'recovery_factor': 0,
                'trades_count': 0
            }
        
        # 计算平均值
        total_returns = [float(r.get('total_return', 0)) for r in valid_results]
        annual_returns = [float(r.get('annual_return', 0)) for r in valid_results]
        max_drawdowns = [float(r.get('max_drawdown', 1)) for r in valid_results]
        win_rates = [float(r.get('win_rate', 0)) for r in valid_results]
        trades_counts = [int(r.get('trades_count', 0)) for r in valid_results]
        
        # 计算夏普比率和索提诺比率
        returns_array = np.array(total_returns)
        valid_returns = returns_array[~np.isnan(returns_array) & ~np.isinf(returns_array)]
        
        if len(valid_returns) > 0:
            returns_mean = np.mean(valid_returns)
            returns_std = np.std(valid_returns) if len(valid_returns) > 1 else 0.001
            
            # 夏普比率 (假设无风险收益率为0.03)
            sharpe_ratio = (returns_mean - 0.03) / returns_std if returns_std > 0 else 0
            
            # 索提诺比率 (只考虑负收益的标准差)
            negative_returns = valid_returns[valid_returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 1 else 0.001
            sortino_ratio = (returns_mean - 0.03) / downside_std if downside_std > 0 else 0
            
            # 计算盈亏比
            profit_factor = 1.0
            if sum(r < 0 for r in valid_returns) > 0:
                profit_factor = sum(max(0, r) for r in valid_returns) / abs(sum(min(0, r) for r in valid_returns))
            
            # 计算恢复因子
            avg_max_drawdown = np.mean(max_drawdowns)
            recovery_factor = returns_mean / avg_max_drawdown if avg_max_drawdown > 0 else 0
        else:
            sharpe_ratio = -999
            sortino_ratio = -999
            profit_factor = 0
            recovery_factor = 0
        
        return {
            'total_return': np.mean(total_returns),
            'annual_return': np.mean(annual_returns),
            'max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean(win_rates),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'trades_count': np.mean(trades_counts)
        }
    
    def _find_best_parameters(self, results, scoring_metric):
        """查找最佳参数"""
        if not results:
            logger.warning("没有有效的优化结果")
            return None, None
        
        # 按评分指标排序
        reverse = True  # 大部分指标是越大越好
        if scoring_metric == 'max_drawdown':  # 最大回撤是越小越好
            reverse = False
        
        sorted_results = sorted(
            results, 
            key=lambda x: x['performance'].get(scoring_metric, -999 if reverse else 999), 
            reverse=reverse
        )
        
        best_result = sorted_results[0]
        best_params = best_result['parameters']
        
        return best_params, best_result
    
    def _output_optimization_results(self, results, best_params, best_result, scoring_metric):
        """输出优化结果"""
        # 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(self.results_dir, f"ma_optimization_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 1. 保存所有结果到CSV
        results_data = []
        for result in results:
            row = {}
            # 添加参数
            for key, value in result['parameters'].items():
                row[key] = value
            # 添加性能指标
            for key, value in result['performance'].items():
                row[key] = value
            results_data.append(row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            results_csv = os.path.join(result_dir, "all_results.csv")
            results_df.to_csv(results_csv, index=False)
            logger.info(f"已保存所有优化结果到: {results_csv}")
        
        # 2. 生成参数敏感性分析图表
        self._generate_sensitivity_charts(results, result_dir, scoring_metric)
        
        # 3. 保存最佳参数的回测结果
        if best_result and 'stock_results' in best_result:
            best_results_df = pd.DataFrame(best_result['stock_results'])
            best_csv = os.path.join(result_dir, "best_results.csv")
            best_results_df.to_csv(best_csv, index=False)
            logger.info(f"已保存最佳参数回测结果到: {best_csv}")
        
        # 4. 输出结果摘要
        summary_path = os.path.join(result_dir, "optimization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"均线交叉策略参数优化结果摘要\n")
            f.write(f"优化时间: {timestamp}\n")
            f.write(f"评分指标: {scoring_metric}\n\n")
            
            f.write("最佳参数:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n最佳性能指标:\n")
            for key, value in best_result['performance'].items():
                f.write(f"  {key}: {value:.4f}\n")
        
        logger.info(f"已输出优化摘要到: {summary_path}")
    
    def _generate_sensitivity_charts(self, results, result_dir, scoring_metric):
        """生成参数敏感性分析图表"""
        if not results:
            return
        
        # 准备数据
        data = []
        for result in results:
            params = result['parameters']
            performance = result['performance']
            data.append({
                'short_ma': params['short_ma'],
                'long_ma': params['long_ma'],
                'stop_loss_pct': params['stop_loss_pct'],
                scoring_metric: performance.get(scoring_metric, 0)
            })
        
        df = pd.DataFrame(data)
        
        # 创建图表目录
        charts_dir = os.path.join(result_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. 短期均线敏感性
        plt.figure(figsize=(10, 6))
        grouped = df.groupby('short_ma')[scoring_metric].mean().reset_index()
        plt.plot(grouped['short_ma'], grouped[scoring_metric], marker='o')
        plt.xlabel('短期均线')
        plt.ylabel(scoring_metric)
        plt.title(f'短期均线对{scoring_metric}的影响')
        plt.grid(True)
        plt.savefig(os.path.join(charts_dir, f"short_ma_sensitivity.png"))
        plt.close()
        
        # 2. 长期均线敏感性
        plt.figure(figsize=(10, 6))
        grouped = df.groupby('long_ma')[scoring_metric].mean().reset_index()
        plt.plot(grouped['long_ma'], grouped[scoring_metric], marker='o')
        plt.xlabel('长期均线')
        plt.ylabel(scoring_metric)
        plt.title(f'长期均线对{scoring_metric}的影响')
        plt.grid(True)
        plt.savefig(os.path.join(charts_dir, f"long_ma_sensitivity.png"))
        plt.close()
        
        # 3. 短期/长期均线比率敏感性
        plt.figure(figsize=(10, 6))
        df['ma_ratio'] = df['short_ma'] / df['long_ma']
        # 分箱
        df['ma_ratio_bin'] = pd.cut(df['ma_ratio'], bins=10)
        grouped = df.groupby('ma_ratio_bin')[scoring_metric].mean().reset_index()
        plt.plot(range(len(grouped)), grouped[scoring_metric], marker='o')
        plt.xticks(range(len(grouped)), [str(x) for x in grouped['ma_ratio_bin']], rotation=45)
        plt.xlabel('短期/长期均线比率')
        plt.ylabel(scoring_metric)
        plt.title(f'均线比率对{scoring_metric}的影响')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"ma_ratio_sensitivity.png"))
        plt.close()
        
        # 4. 止损比例敏感性
        plt.figure(figsize=(10, 6))
        grouped = df.groupby('stop_loss_pct')[scoring_metric].mean().reset_index()
        plt.plot(grouped['stop_loss_pct'], grouped[scoring_metric], marker='o')
        plt.xlabel('止损比例')
        plt.ylabel(scoring_metric)
        plt.title(f'止损比例对{scoring_metric}的影响')
        plt.grid(True)
        plt.savefig(os.path.join(charts_dir, f"stop_loss_sensitivity.png"))
        plt.close()
        
        # 5. 热力图：短期均线vs长期均线
        plt.figure(figsize=(12, 8))
        pivot = pd.pivot_table(df, values=scoring_metric, index='short_ma', columns='long_ma', aggfunc='mean')
        plt.imshow(pivot, aspect='auto', cmap='viridis')
        plt.colorbar(label=scoring_metric)
        plt.xlabel('长期均线')
        plt.ylabel('短期均线')
        plt.title(f'短期均线vs长期均线对{scoring_metric}的影响')
        
        # 添加刻度
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"ma_heatmap.png"))
        plt.close()
        
        logger.info(f"已生成参数敏感性分析图表到: {charts_dir}")

    def optimize_momentum_parameters(self, stock_list=None, 
                                  rsi_range=(9, 25), 
                                  macd_fast_range=(8, 20),
                                  macd_slow_range=(20, 40),
                                  volume_ma_range=(5, 30),
                                  scoring_metric="total_score",
                                  max_combinations=100,
                                  parallel=True):
        """
        优化动量策略参数
        
        参数:
            stock_list: DataFrame, 股票列表 
            rsi_range: tuple, RSI参数范围(start, end)
            macd_fast_range: tuple, MACD快线参数范围(start, end)
            macd_slow_range: tuple, MACD慢线参数范围(start, end)
            volume_ma_range: tuple, 成交量均线参数范围(start, end)
            scoring_metric: str, 评分指标 ('total_score', 'momentum_score', 'trend_score')
            max_combinations: int, 最大参数组合数量
            parallel: bool, 是否使用并行处理
        
        返回:
            dict, 最佳参数和分析结果
        """
        logger.info("动量策略参数优化功能尚未实现")
        # 这里实现动量策略参数优化逻辑，类似于均线交叉策略优化
        
        return {
            'best_parameters': {},
            'best_result': {},
            'all_results': []
        }

def main():
    """主函数"""
    try:
        print("====== 策略参数自动优化工具 ======")
        print("1. 均线交叉策略参数优化")
        print("2. 动量策略参数优化")
        print("3. 退出")
        
        choice = input("请选择要执行的功能 (1-3): ")
        
        if choice == '1':
            optimizer = StrategyOptimizer(use_tushare=False)
            
            # 获取参数
            try:
                short_ma_min = int(input("请输入短期均线最小值 (默认3): ") or "3")
                short_ma_max = int(input("请输入短期均线最大值 (默认20): ") or "20")
                long_ma_min = int(input("请输入长期均线最小值 (默认20): ") or "20")
                long_ma_max = int(input("请输入长期均线最大值 (默认60): ") or "60")
                stop_loss_min = float(input("请输入止损比例最小值 (默认0.03): ") or "0.03")
                stop_loss_max = float(input("请输入止损比例最大值 (默认0.1): ") or "0.1")
                initial_capital = float(input("请输入初始资金 (默认100000): ") or "100000")
                max_combinations = int(input("请输入最大参数组合数 (默认50): ") or "50")
                
                scoring_metrics = ["total_return", "annual_return", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
                print("评分指标选项:")
                for i, metric in enumerate(scoring_metrics, 1):
                    print(f"{i}. {metric}")
                metric_choice = int(input("请选择评分指标 (1-5，默认3): ") or "3")
                scoring_metric = scoring_metrics[metric_choice - 1]
                
                parallel = input("是否使用并行处理? (y/n，默认y): ").lower() != 'n'
                
                print("\n开始优化参数...")
                
                result = optimizer.optimize_ma_parameters(
                    short_ma_range=(short_ma_min, short_ma_max),
                    long_ma_range=(long_ma_min, long_ma_max),
                    stop_loss_range=(stop_loss_min, stop_loss_max),
                    initial_capital=initial_capital,
                    scoring_metric=scoring_metric,
                    max_combinations=max_combinations,
                    parallel=parallel
                )
                
                best_params = result['best_parameters']
                performance = result['best_result']['performance']
                
                print("\n====== 优化结果 ======")
                print(f"最佳参数:")
                for key, value in best_params.items():
                    print(f"  {key}: {value}")
                    
                print(f"\n性能指标:")
                for key, value in performance.items():
                    print(f"  {key}: {value:.4f}")
                
                print(f"\n结果详情已保存到 'results/optimization' 目录")
                
            except Exception as e:
                logger.error(f"均线交叉策略参数优化失败: {str(e)}")
                print(f"优化过程出错: {str(e)}")
        
        elif choice == '2':
            optimizer = StrategyOptimizer(use_tushare=False)
            print("动量策略参数优化功能尚未实现")
            # 这里实现动量策略参数优化用户交互
        
        elif choice == '3':
            print("退出程序")
        
        else:
            print("无效的选择")
    
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
        print(f"程序出错: {str(e)}")

if __name__ == "__main__":
    main() 