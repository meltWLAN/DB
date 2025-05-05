"""
股票分析系统 - 命令行版本
绕过GUI和依赖问题
"""
import os
import sys
import time
import logging
import platform
from pathlib import Path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cli_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CLI")

# 确保目录存在
for dirname in ["data", "logs", "results", "cache"]:
    os.makedirs(os.path.join(os.getcwd(), dirname), exist_ok=True)

class StockAnalyzer:
    """股票分析器命令行版本"""
    
    def __init__(self):
        """初始化分析器"""
        self.data_dir = "data"
        self.results_dir = "results"
        logger.info("初始化股票分析器")
    
    def load_modules(self):
        """加载必要的模块"""
        try:
            print("加载模块...")
            
            # 尝试导入常用模块
            for module_name in ["numpy", "pandas", "matplotlib"]:
                try:
                    __import__(module_name)
                    print(f"✓ {module_name}已加载")
                except ImportError:
                    print(f"✗ {module_name}未安装")
                    return False
            
            return True
        except Exception as e:
            print(f"模块加载错误: {e}")
            logger.error(f"模块加载错误: {e}")
            return False
    
    def list_stocks(self):
        """列出可用的股票数据"""
        print("\n可用的股票数据:")
        
        try:
            data_files = list(Path(self.data_dir).glob("*.csv"))
            
            if not data_files:
                print("未找到股票数据文件")
                return
            
            for i, file in enumerate(data_files[:10]):
                size = file.stat().st_size // 1024
                print(f"  {i+1}. {file.name} ({size}KB)")
            
            if len(data_files) > 10:
                print(f"  ... 还有 {len(data_files) - 10} 个文件")
        except Exception as e:
            print(f"列出股票数据出错: {e}")
            logger.error(f"列出股票数据出错: {e}")
    
    def list_results(self):
        """列出分析结果"""
        print("\n分析结果文件:")
        
        try:
            result_files = list(Path(self.results_dir).glob("*.*"))
            
            if not result_files:
                print("未找到结果文件")
                return
            
            for i, file in enumerate(result_files[:10]):
                size = file.stat().st_size // 1024
                print(f"  {i+1}. {file.name} ({size}KB)")
            
            if len(result_files) > 10:
                print(f"  ... 还有 {len(result_files) - 10} 个文件")
        except Exception as e:
            print(f"列出结果文件出错: {e}")
            logger.error(f"列出结果文件出错: {e}")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 50)
        print(" 股票分析系统 - 命令行版本")
        print("=" * 50)
        print("1. 系统诊断")
        print("2. 列出股票数据")
        print("3. 列出分析结果")
        print("4. 生成示例数据")
        print("5. 退出")
        print("-" * 50)
        
        choice = input("请选择操作 [1-5]: ")
        return choice
    
    def run(self):
        """运行分析器"""
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.run_diagnostics()
            elif choice == '2':
                self.list_stocks()
            elif choice == '3':
                self.list_results()
            elif choice == '4':
                self.generate_sample_data()
            elif choice == '5':
                print("退出程序")
                break
            else:
                print("无效的选择，请重试")
    
    def run_diagnostics(self):
        """运行系统诊断"""
        print("\n系统诊断:")
        print(f"操作系统: {platform.system()} {platform.release()}")
        print(f"Python版本: {sys.version.split()[0]}")
        print(f"工作目录: {os.getcwd()}")
        
        # 检查目录
        for dirname in ["data", "logs", "results", "cache"]:
            path = os.path.join(os.getcwd(), dirname)
            if os.path.exists(path):
                print(f"✓ 目录 {dirname} 存在")
            else:
                print(f"✗ 目录 {dirname} 不存在")
        
        # 检查Python包
        print("\n已安装的关键包:")
        packages = [
            "numpy", "pandas", "matplotlib", "scipy", 
            "sklearn", "statsmodels", "yfinance", "requests"
        ]
        
        for package in packages:
            try:
                __import__(package)
                module = sys.modules[package]
                version = getattr(module, "__version__", "未知")
                print(f"  ✓ {package} - 版本: {version}")
            except ImportError:
                print(f"  ✗ {package} - 未安装")
            except Exception as e:
                print(f"  ? {package} - 出错: {str(e)}")
    
    def generate_sample_data(self):
        """生成示例数据"""
        try:
            import_success = True
            try:
                import numpy as np
            except ImportError:
                print("错误: 需要安装numpy库")
                import_success = False
            
            try:
                import pandas as pd
            except ImportError:
                print("错误: 需要安装pandas库")
                import_success = False
            
            if not import_success:
                print("无法生成示例数据，请先安装所需的Python库")
                return
            
            print("生成示例股票数据...")
            
            # 创建日期范围
            dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='B')
            
            # 生成3只股票的数据
            stocks = ['SAMPLE001', 'SAMPLE002', 'SAMPLE003']
            
            for stock in stocks:
                print(f"生成 {stock} 的数据...")
                
                # 生成随机价格数据
                np.random.seed(hash(stock) % 100)
                
                # 基础价格和趋势
                base_price = np.random.uniform(10, 100)
                trend = np.random.uniform(-0.0001, 0.0002)
                
                # 生成价格序列
                prices = []
                price = base_price
                
                for i in range(len(dates)):
                    # 添加随机波动
                    daily_return = np.random.normal(trend, 0.015)
                    price = price * (1 + daily_return)
                    prices.append(price)
                
                # 确保价格为正
                prices = np.maximum(prices, 0.1)
                
                # 生成成交量
                volume = np.random.randint(1000, 100000, size=len(dates))
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'Date': dates,
                    'Open': prices * (1 - np.random.uniform(0, 0.01, len(prices))),
                    'High': prices * (1 + np.random.uniform(0, 0.02, len(prices))),
                    'Low': prices * (1 - np.random.uniform(0, 0.02, len(prices))),
                    'Close': prices,
                    'Volume': volume
                })
                
                # 保存到CSV
                output_file = os.path.join(self.data_dir, f"{stock}.csv")
                df.to_csv(output_file, index=False)
                print(f"✓ 数据已保存到 {output_file}")
            
            print("示例数据生成完成")
            
        except Exception as e:
            print(f"生成示例数据出错: {e}")
            logger.error(f"生成示例数据出错: {e}", exc_info=True)

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("股票分析系统 - 命令行版本")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print("-" * 60)
    
    # 环境变量
    print("\n环境变量:")
    for key in ['PYTHONPATH', 'PYTHONHOME', 'PYENV_ROOT', 'PYENV_VERSION']:
        if key in os.environ:
            print(f"  {key}: {os.environ[key]}")
    
    print("-" * 60)

def main():
    """主函数"""
    print_system_info()
    
    try:
        analyzer = StockAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}", exc_info=True)
        print(f"程序运行错误: {str(e)}")
    
    print("\n程序已退出")

if __name__ == "__main__":
    main() 