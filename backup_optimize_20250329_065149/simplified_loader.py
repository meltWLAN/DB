#!/usr/bin/env python3
"""
简化版股票分析系统加载器 - 诊断版本
"""
import os
import sys
import platform
import subprocess
import time
import logging
from pathlib import Path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/loader_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimplifiedLoader")

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("股票分析系统 - 诊断工具")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print("-" * 60)
    
    # 环境变量
    print("\n环境变量:")
    for key in ['PYTHONPATH', 'PYTHONHOME', 'PYENV_ROOT', 'PYENV_VERSION', 'PATH']:
        if key in os.environ:
            print(f"  {key}: {os.environ[key]}")
    
    # Python路径
    print("\nPython路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print("-" * 60)

def check_python_packages():
    """检查Python包的安装情况"""
    print("\n已安装的关键包:")
    packages = [
        "numpy", "pandas", "matplotlib", "tkinter", "scipy", 
        "sklearn", "statsmodels", "yfinance", "requests"
    ]
    
    for package in packages:
        try:
            if package == "tkinter":
                import tkinter
                print(f"  ✓ {package} - 版本: {tkinter.TkVersion}")
            else:
                __import__(package)
                module = sys.modules[package]
                version = getattr(module, "__version__", "未知")
                print(f"  ✓ {package} - 版本: {version}")
        except ImportError:
            print(f"  ✗ {package} - 未安装")
        except Exception as e:
            print(f"  ? {package} - 出错: {str(e)}")
    
    print("-" * 60)

def list_source_files():
    """列出源代码文件"""
    print("\n源代码文件:")
    
    py_files = list(Path(".").glob("*.py"))
    py_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, file in enumerate(py_files[:10]):
        try:
            size = file.stat().st_size // 1024
            modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(file.stat().st_mtime))
            print(f"  {i+1}. {file.name} ({size}KB, {modified})")
        except:
            print(f"  {i+1}. {file.name}")
    
    if len(py_files) > 10:
        print(f"  ... 还有 {len(py_files) - 10} 个文件")
    
    print("-" * 60)

def check_gui_support():
    """检查GUI支持情况"""
    print("\nGUI支持检查:")
    
    try:
        import tkinter
        print(f"  tkinter可用 - 版本: {tkinter.TkVersion}")
        
        # 尝试创建一个隐藏的窗口
        try:
            root = tkinter.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            print(f"  屏幕分辨率: {screen_width}x{screen_height}")
            root.destroy()
            print("  ✓ 可以创建Tk窗口")
        except Exception as e:
            print(f"  ✗ 创建Tk窗口失败: {str(e)}")
            
    except ImportError:
        print("  ✗ tkinter不可用")
    except Exception as e:
        print(f"  ✗ tkinter导入出错: {str(e)}")
    
    print("-" * 60)

def install_required_packages():
    """安装必要的包"""
    print("\n正在安装必要的Python包...")
    
    packages = [
        "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", 
        "pytz", "python-dateutil", "statsmodels", "yfinance", 
        "requests", "beautifulsoup4", "lxml", "sqlalchemy", "pillow"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            print(f"  ✓ {package} 安装成功")
        except Exception as e:
            print(f"  ✗ {package} 安装失败: {str(e)}")
    
    print("包安装完成")
    print("-" * 60)

def create_minimal_gui():
    """创建最小化的GUI文件来测试"""
    minimal_gui = "minimal_gui.py"
    print(f"\n创建最小化GUI测试文件: {minimal_gui}")
    
    with open(minimal_gui, "w") as f:
        f.write("""#!/usr/bin/env python3
import os
import sys
import logging
import time

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/minimal_gui_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MinimalGUI")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    logger.info("tkinter导入成功")
    
    class MinimalGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("最小化股票分析系统")
            self.root.geometry("600x400")
            
            # 创建界面
            self.create_widgets()
            
            logger.info("GUI初始化完成")
        
        def create_widgets(self):
            # 顶部标签
            tk.Label(self.root, text="股票分析系统 - 最小化测试版", 
                    font=("Arial", 16)).pack(pady=20)
            
            # 系统信息
            info_frame = ttk.LabelFrame(self.root, text="系统信息")
            info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Python信息
            tk.Label(info_frame, text=f"Python版本: {sys.version}", 
                    anchor="w").pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(info_frame, text=f"Tkinter版本: {tk.TkVersion}", 
                    anchor="w").pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(info_frame, text=f"工作目录: {os.getcwd()}", 
                    anchor="w").pack(fill=tk.X, padx=5, pady=2)
            
            # 测试按钮
            ttk.Button(self.root, text="测试消息框", 
                      command=self.test_messagebox).pack(pady=20)
            
            ttk.Button(self.root, text="退出", 
                      command=self.root.quit).pack(pady=5)
        
        def test_messagebox(self):
            messagebox.showinfo("测试", "GUI系统正常工作!")
            logger.info("消息框测试成功")
    
    def main():
        logger.info("启动最小化GUI")
        root = tk.Tk()
        app = MinimalGUI(root)
        root.mainloop()
        logger.info("GUI已关闭")
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    logger.error(f"运行出错: {str(e)}", exc_info=True)
    print(f"错误: {str(e)}")
""")
    
    # 添加执行权限
    os.chmod(minimal_gui, 0o755)
    print(f"已创建最小化GUI测试文件: {minimal_gui}")
    print("-" * 60)

def create_cli_version():
    """创建命令行版本"""
    cli_file = "cli_analyzer.py"
    print(f"\n创建命令行版本: {cli_file}")
    
    with open(cli_file, "w") as f:
        f.write("""#!/usr/bin/env python3
'''
股票分析系统 - 命令行版本
'''
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
            import numpy as np
            print("✓ NumPy已加载")
            
            import pandas as pd
            print("✓ Pandas已加载")
            
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            print("✓ Matplotlib已加载")
            
            return True
        except ImportError as e:
            print(f"模块导入错误: {e}")
            logger.error(f"模块导入错误: {e}")
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
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 50)
        print(" 股票分析系统 - 命令行版本")
        print("=" * 50)
        print("1. 系统诊断")
        print("2. 列出股票数据")
        print("3. 生成示例数据")
        print("4. 简单分析")
        print("5. 退出")
        print("-" * 50)
        
        choice = input("请选择操作 [1-5]: ")
        return choice
    
    def run(self):
        """运行分析器"""
        if not self.load_modules():
            print("无法加载必要的模块，退出程序")
            return
        
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.run_diagnostics()
            elif choice == '2':
                self.list_stocks()
            elif choice == '3':
                self.generate_sample_data()
            elif choice == '4':
                self.run_simple_analysis()
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
    
    def generate_sample_data(self):
        """生成示例数据"""
        try:
            import numpy as np
            import pandas as pd
            
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
    
    def run_simple_analysis(self):
        """运行简单分析"""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            # 查找数据文件
            data_files = list(Path(self.data_dir).glob("*.csv"))
            
            if not data_files:
                print("未找到股票数据文件，请先生成示例数据")
                return
            
            # 列出可用的文件
            print("\n可用的股票数据文件:")
            for i, file in enumerate(data_files):
                print(f"{i+1}. {file.name}")
            
            # 请求用户选择
            choice = input("\n请选择要分析的股票编号 [1-{}]: ".format(len(data_files)))
            
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(data_files):
                    print("无效的选择")
                    return
                
                selected_file = data_files[idx]
                print(f"分析 {selected_file.name}...")
                
                # 读取数据
                df = pd.read_csv(selected_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # 计算简单指标
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                
                # 计算每日回报率
                df['Daily_Return'] = df['Close'].pct_change() * 100
                
                # 计算波动率 (20日标准差)
                df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
                
                # 打印基本统计信息
                print("\n基本统计信息:")
                print(f"数据时间范围: {df.index.min().date()} 到 {df.index.max().date()}")
                print(f"交易日数量: {len(df)}")
                print(f"起始价格: {df['Close'].iloc[0]:.2f}")
                print(f"结束价格: {df['Close'].iloc[-1]:.2f}")
                print(f"价格变化: {(df['Close'].iloc[-1] - df['Close'].iloc[0]):.2f} " 
                      f"({(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.2f}%)")
                print(f"最高价格: {df['High'].max():.2f}")
                print(f"最低价格: {df['Low'].min():.2f}")
                print(f"平均成交量: {df['Volume'].mean():.0f}")
                print(f"20日波动率: {df['Volatility'].iloc[-1]:.2f}%")
                
                # 生成图表
                plt.figure(figsize=(12, 8))
                
                # 绘制价格和均线
                plt.subplot(2, 1, 1)
                plt.plot(df.index, df['Close'], label='收盘价')
                plt.plot(df.index, df['MA5'], label='5日均线')
                plt.plot(df.index, df['MA20'], label='20日均线')
                plt.plot(df.index, df['MA60'], label='60日均线')
                plt.title(f'{selected_file.stem} 价格走势')
                plt.ylabel('价格')
                plt.legend()
                plt.grid(True)
                
                # 绘制成交量
                plt.subplot(2, 1, 2)
                plt.bar(df.index, df['Volume'], color='gray', alpha=0.7)
                plt.title('成交量')
                plt.ylabel('成交量')
                plt.grid(True)
                
                plt.tight_layout()
                
                # 保存图表
                output_file = os.path.join(self.results_dir, f"{selected_file.stem}_analysis.png")
                plt.savefig(output_file)
                print(f"\n分析图表已保存到 {output_file}")
                
                # 保存结果数据
                output_csv = os.path.join(self.results_dir, f"{selected_file.stem}_results.csv")
                df.to_csv(output_csv)
                print(f"分析数据已保存到 {output_csv}")
                
            except ValueError:
                print("请输入有效的数字")
            
        except Exception as e:
            print(f"分析过程出错: {e}")
            logger.error(f"分析过程出错: {e}", exc_info=True)

def main():
    """主函数"""
    analyzer = StockAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
""")
    
    # 添加执行权限
    os.chmod(cli_file, 0o755)
    print(f"已创建命令行版本: {cli_file}")
    print("-" * 60)

def main():
    """主函数"""
    print_system_info()
    check_python_packages()
    list_source_files()
    check_gui_support()
    
    # 询问是否继续
    print("\n请选择操作:")
    print("1. 安装必要的Python包")
    print("2. 创建最小化GUI测试文件")
    print("3. 创建命令行版本")
    print("4. 退出")
    
    try:
        choice = input("请输入选择 [1-4]: ").strip()
        
        if choice == "1":
            install_required_packages()
        elif choice == "2":
            create_minimal_gui()
        elif choice == "3":
            create_cli_version()
        else:
            print("已退出")
            return
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        print(f"出现错误: {str(e)}")
    
    print("\n诊断完成。如有必要，请查看日志文件以获取更多详细信息。")

if __name__ == "__main__":
    main() 