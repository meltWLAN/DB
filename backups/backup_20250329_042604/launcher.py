#!/usr/bin/env python3
"""
股票分析系统启动器 - 自动绕过pyenv问题
"""
import os
import sys
import subprocess
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Launcher')

def find_system_python():
    """查找系统Python路径"""
    system_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3"
    ]
    
    for path in system_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            logger.info(f"找到系统Python: {path}")
            return path
    
    logger.error("无法找到系统Python")
    return None

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
    
    logger.error("无法找到主程序脚本")
    return None

def create_temp_script(main_script):
    """创建临时运行脚本"""
    temp_script = "temp_launcher.py"
    logger.info(f"创建临时启动脚本: {temp_script}")
    
    with open(temp_script, "w") as f:
        f.write(f"""
import os
import sys
import importlib.util
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TempLauncher')

# 设置环境变量
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

try:
    logger.info(f"正在加载程序: {main_script}")
    
    # 使用spec导入模块
    spec = importlib.util.spec_from_file_location("main_module", "{main_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_module"] = module
    spec.loader.exec_module(module)
    
    # 执行main函数
    if hasattr(module, 'main'):
        logger.info("执行main函数")
        module.main()
    else:
        logger.warning("找不到main函数，尝试执行__main__逻辑")
        
    logger.info("程序运行完成")
except Exception as e:
    logger.error(f"运行失败: {{str(e)}}", exc_info=True)
""")
    
    return temp_script

def run_with_system_python(system_python, script_path):
    """使用系统Python运行脚本"""
    logger.info(f"使用 {system_python} 运行 {script_path}")
    
    # 准备环境变量
    env = os.environ.copy()
    env["PYENV_VERSION"] = "system"
    env["PYENV_DISABLE_PROMPT"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    
    try:
        process = subprocess.Popen(
            [system_python, script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"启动进程 PID: {process.pid}")
        
        # 读取并记录输出
        for line in process.stdout:
            print(line.strip())
        
        # 等待进程完成
        process.wait()
        return process.returncode
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        return 1

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统启动器 - 绕过pyenv问题")
    print("=" * 60)
    
    # 查找系统Python
    system_python = find_system_python()
    if not system_python:
        print("错误: 无法找到系统Python，无法启动")
        return 1
    
    # 查找主程序脚本
    main_script = find_main_script()
    if not main_script:
        print("错误: 无法找到主程序脚本，无法启动")
        return 1
    
    # 创建临时脚本
    temp_script = create_temp_script(main_script)
    
    # 使用系统Python运行
    try:
        print(f"正在启动系统，请稍候...")
        return_code = run_with_system_python(system_python, temp_script)
        
        # 清理临时文件
        if os.path.exists(temp_script):
            os.remove(temp_script)
            
        if return_code != 0:
            print(f"程序异常退出，退出码: {return_code}")
            return return_code
            
        return 0
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        # 清理临时文件
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return 0
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        print(f"启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 