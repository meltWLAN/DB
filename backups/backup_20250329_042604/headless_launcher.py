#!/usr/bin/env python3
"""
股票分析系统 - 无界面启动器
绕过pyenv环境问题，无需使用tkinter界面
"""
import os
import sys
import logging
import subprocess
import platform
import time
import traceback
from pathlib import Path

# 配置日志
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'headless_launcher_{time.strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HeadlessLauncher")

def find_system_python():
    """查找系统Python路径"""
    system_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    if platform.system() == "Windows":
        system_paths = [
            r"C:\Python39\python.exe",
            r"C:\Python310\python.exe",
            r"C:\Program Files\Python39\python.exe",
            r"C:\Program Files\Python310\python.exe"
        ]
    
    for path in system_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            logger.info(f"找到系统Python: {path}")
            return path
    
    logger.warning("未找到系统Python")
    return None

def check_pyenv_status():
    """检查pyenv状态"""
    try:
        if "PYENV_ROOT" in os.environ:
            return f"已检测到pyenv环境变量: {os.environ.get('PYENV_ROOT')}"
        
        # 检查配置文件
        home = os.path.expanduser("~")
        config_files = [
            os.path.join(home, ".bashrc"), 
            os.path.join(home, ".zshrc"),
            os.path.join(home, ".bash_profile"),
            os.path.join(home, ".profile")
        ]
        
        for file in config_files:
            if os.path.exists(file):
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                        if "pyenv init" in content:
                            return f"在{os.path.basename(file)}中发现pyenv配置"
                except:
                    pass
        
        return "未检测到pyenv配置"
    except Exception as e:
        logger.error(f"检查pyenv状态出错: {str(e)}")
        return f"检查出错: {str(e)}"

def find_main_script():
    """查找主程序脚本"""
    main_scripts = [
        "stock_analysis_gui.py",
        "simple_gui.py",
        "integrated_system.py",
        "start_gui.py",
        "direct_start.py"
    ]
    
    for script in main_scripts:
        if os.path.exists(script):
            logger.info(f"找到主程序脚本: {script}")
            return script
    
    logger.warning("未找到主程序脚本")
    return None

def create_temp_script(main_script):
    """创建临时运行脚本"""
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, f"temp_runner_{time.strftime('%Y%m%d_%H%M%S')}.py")
    logger.info(f"创建临时脚本: {temp_file_path}")
    
    with open(temp_file_path, 'w') as f:
        f.write(f"""
#!/usr/bin/env python3
# 临时隔离环境运行脚本
import os
import sys
import logging
import importlib.util
import traceback
import time

# 配置日志
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'isolated_run_{{time.strftime("%Y%m%d_%H%M%S")}}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IsolatedRunner')

logger.info("隔离环境启动中...")
logger.info(f"Python路径: {{sys.executable}}")
logger.info(f"Python版本: {{sys.version}}")
logger.info(f"当前目录: {{os.getcwd()}}")

# 设置环境变量
os.environ["PYTHONPATH"] = os.getcwd()
if 'PYTHONHOME' in os.environ:
    logger.info(f"移除PYTHONHOME环境变量: {{os.environ.pop('PYTHONHOME')}}")

# 确保必要的目录存在
for dirname in ["data", "logs", "results", "cache"]:
    os.makedirs(os.path.join(os.getcwd(), dirname), exist_ok=True)

try:
    logger.info(f"正在加载: {main_script}")
    
    # 添加当前目录到Python路径
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    # 使用spec导入模块
    spec = importlib.util.spec_from_file_location("main_module", "{main_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_module"] = module
    spec.loader.exec_module(module)
    
    # 运行main函数
    if hasattr(module, 'main'):
        logger.info("执行main函数")
        module.main()
    else:
        logger.warning("找不到main函数，尝试直接运行模块")
    
    logger.info("程序运行完成")
except Exception as e:
    logger.error(f"运行失败: {{str(e)}}")
    logger.error(traceback.format_exc())
    print(f"运行失败: {{str(e)}}")
""")
    
    return temp_file_path

def run_with_system_python(system_python, script_path, mode="direct"):
    """使用系统Python运行脚本"""
    logger.info(f"使用 {system_python} {mode} 模式运行 {script_path}")
    
    # 准备环境变量
    env = os.environ.copy()
    env["PYENV_VERSION"] = "system"
    env["PYENV_DISABLE_PROMPT"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    
    if "PYTHONHOME" in env:
        logger.info(f"移除PYTHONHOME环境变量: {env.pop('PYTHONHOME')}")
    
    try:
        # 执行Python脚本
        process = subprocess.Popen(
            [system_python, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"启动进程 PID: {process.pid}")
        print(f"已启动 {os.path.basename(script_path)} (PID: {process.pid})")
        
        # 读取一部分输出（不阻塞等待所有输出）
        try:
            stdout, stderr = process.communicate(timeout=2)
            if stdout:
                logger.info(f"输出: {stdout[:500]}")
            if stderr:
                logger.warning(f"错误: {stderr[:500]}")
        except subprocess.TimeoutExpired:
            logger.info("进程正在后台运行")
        
        print(f"程序已在后台启动，日志文件: {log_file}")
        return True
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        print(f"启动失败: {str(e)}")
        return False

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("股票分析系统 - 无界面启动器")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"当前目录: {os.getcwd()}")
    print("-" * 60)

def main():
    """主函数"""
    try:
        print_system_info()
        
        # 检查是否已经使用系统Python运行
        current_python = sys.executable
        logger.info(f"当前Python路径: {current_python}")
        
        if current_python.startswith(("/usr/bin/", "/usr/local/bin/", "/Library/Developer/CommandLineTools/usr/bin/")):
            logger.info("已使用系统Python运行")
            
            # 查找主程序脚本
            main_script = find_main_script()
            if not main_script:
                print("错误: 未找到主程序脚本，无法启动")
                return 1
            
            print(f"找到主程序: {main_script}")
            
            # 检查pyenv状态
            pyenv_status = check_pyenv_status()
            print(f"pyenv状态: {pyenv_status}")
            
            print("\n请选择启动模式:")
            print("1. 直接启动")
            print("2. 隔离环境启动")
            print("3. 退出")
            
            choice = input("请输入选择 [1-3]: ").strip()
            
            if choice == "1":
                print("\n正在直接启动...")
                success = run_with_system_python(current_python, main_script, "direct")
                if not success:
                    print("直接启动失败，尝试隔离环境启动...")
                    temp_script = create_temp_script(main_script)
                    run_with_system_python(current_python, temp_script, "isolated")
            elif choice == "2":
                print("\n正在准备隔离环境...")
                temp_script = create_temp_script(main_script)
                run_with_system_python(current_python, temp_script, "isolated")
            else:
                print("已取消启动")
                return 0
            
        else:
            logger.info("需要切换到系统Python")
            
            # 查找系统Python
            system_python = find_system_python()
            if not system_python:
                print("错误: 未找到系统Python，无法启动")
                return 1
            
            # 使用系统Python重新运行此脚本
            print(f"需要使用系统Python运行，正在切换...")
            logger.info(f"使用系统Python重新运行: {system_python}")
            
            # 执行新进程而不是替换当前进程
            subprocess.call([system_python, __file__])
            
    except Exception as e:
        logger.error(f"启动器错误: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"启动器错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 