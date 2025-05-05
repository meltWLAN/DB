#!/usr/bin/env python3
"""
股票分析系统 - 直接启动器
绕过pyenv环境问题，确保系统能稳定启动
"""
import os
import sys
import logging
import subprocess
import platform
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame
from pathlib import Path
import traceback
import tempfile
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'launcher_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DirectLauncher")

class LauncherGUI:
    """启动器GUI类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("股票分析系统 - 启动器")
        self.root.geometry("600x400")
        self.system_python = self.find_system_python()
        
        # 创建UI
        self.create_ui()
        
        # 系统信息
        self.detect_system_info()
    
    def create_ui(self):
        """创建UI界面"""
        # 标题
        title_frame = Frame(self.root, pady=10)
        title_frame.pack(fill=tk.X)
        Label(title_frame, text="股票分析系统启动器", font=("Arial", 16, "bold")).pack()
        
        # 系统信息区域
        info_frame = Frame(self.root, padx=20, pady=10)
        info_frame.pack(fill=tk.X)
        
        Label(info_frame, text="系统信息:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.system_info_label = Label(info_frame, text="检测中...", justify=tk.LEFT)
        self.system_info_label.pack(anchor=tk.W, pady=5)
        
        # Python信息
        Label(info_frame, text="Python路径:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.python_info_label = Label(info_frame, text=self.system_python or "未找到", justify=tk.LEFT)
        self.python_info_label.pack(anchor=tk.W, pady=5)
        
        # 发现的主程序
        Label(info_frame, text="主程序:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.main_file_label = Label(info_frame, text="正在检测...", justify=tk.LEFT)
        self.main_file_label.pack(anchor=tk.W, pady=5)
        
        # pyenv状态
        Label(info_frame, text="pyenv状态:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.pyenv_status_label = Label(info_frame, text="正在检测...", justify=tk.LEFT)
        self.pyenv_status_label.pack(anchor=tk.W, pady=5)
        
        # 启动按钮
        button_frame = Frame(self.root, pady=20)
        button_frame.pack()
        
        Button(button_frame, text="直接启动", 
               command=self.launch_direct, 
               width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        Button(button_frame, text="隔离环境启动", 
               command=self.launch_isolated, 
               width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = Label(self.root, textvariable=self.status_var, 
                               bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def detect_system_info(self):
        """检测系统信息"""
        # 系统信息
        system_info = f"操作系统: {platform.system()} {platform.release()}\n"
        system_info += f"Python版本: {sys.version.split()[0]}\n"
        system_info += f"当前目录: {os.getcwd()}"
        self.system_info_label.config(text=system_info)
        
        # 检测pyenv状态
        pyenv_status = self.check_pyenv_status()
        self.pyenv_status_label.config(text=pyenv_status)
        
        # 检测主程序文件
        main_script = self.find_main_script()
        if main_script:
            self.main_file_label.config(text=main_script)
        else:
            self.main_file_label.config(text="未找到主程序文件")
    
    def find_system_python(self):
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
    
    def check_pyenv_status(self):
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
                    with open(file, 'r') as f:
                        content = f.read()
                        if "pyenv init" in content:
                            return f"在{os.path.basename(file)}中发现pyenv配置"
            
            return "未检测到pyenv配置"
        except Exception as e:
            logger.error(f"检查pyenv状态出错: {str(e)}")
            return f"检查出错: {str(e)}"
    
    def find_main_script(self):
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
    
    def launch_direct(self):
        """直接启动"""
        if not self.system_python:
            messagebox.showerror("错误", "未找到系统Python，无法启动")
            return
        
        main_script = self.find_main_script()
        if not main_script:
            messagebox.showerror("错误", "未找到主程序脚本，无法启动")
            return
        
        self.status_var.set(f"正在启动 {main_script}...")
        
        # 创建环境变量
        env = os.environ.copy()
        env["PYENV_VERSION"] = "system"
        env["PYENV_DISABLE_PROMPT"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        
        # 启动进程
        try:
            logger.info(f"使用 {self.system_python} 直接启动 {main_script}")
            subprocess.Popen(
                [self.system_python, main_script],
                env=env,
                close_fds=True
            )
            self.status_var.set(f"{main_script} 已启动")
        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            messagebox.showerror("启动失败", f"启动失败: {str(e)}")
            self.status_var.set("启动失败")
    
    def launch_isolated(self):
        """使用隔离环境启动"""
        if not self.system_python:
            messagebox.showerror("错误", "未找到系统Python，无法启动")
            return
        
        main_script = self.find_main_script()
        if not main_script:
            messagebox.showerror("错误", "未找到主程序脚本，无法启动")
            return
        
        self.status_var.set("正在准备隔离环境...")
        
        # 创建临时启动脚本
        try:
            temp_script = self._create_temp_script(main_script)
            
            # 创建环境变量
            env = os.environ.copy()
            env["PYENV_VERSION"] = "system"
            env["PYENV_DISABLE_PROMPT"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            
            # 启动进程
            logger.info(f"使用隔离环境启动 {main_script}")
            subprocess.Popen(
                [self.system_python, temp_script],
                env=env,
                close_fds=True
            )
            self.status_var.set(f"{main_script} 已在隔离环境中启动")
        except Exception as e:
            logger.error(f"隔离环境启动失败: {str(e)}")
            messagebox.showerror("启动失败", f"隔离环境启动失败: {str(e)}")
            self.status_var.set("启动失败")
    
    def _create_temp_script(self, main_script):
        """创建临时运行脚本"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
        temp_file_path = temp_file.name
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('isolated_run.log'),
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
os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

try:
    logger.info(f"正在加载: {main_script}")
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
finally:
    # 程序结束后自动删除临时文件
    try:
        if os.path.exists("{temp_file_path}"):
            logger.info("清理临时文件")
            os.unlink("{temp_file_path}")
    except:
        pass
""")
        
        return temp_file_path

def main():
    """主函数"""
    try:
        # 检查是否已经使用系统Python运行
        current_python = sys.executable
        logger.info(f"当前Python路径: {current_python}")
        
        if current_python.startswith(("/usr/bin/", "/usr/local/bin/", "/Library/Developer/CommandLineTools/usr/bin/")):
            logger.info("已使用系统Python运行")
            
            # 创建GUI
            root = tk.Tk()
            app = LauncherGUI(root)
            root.mainloop()
        else:
            logger.info("需要切换到系统Python")
            
            # 查找系统Python
            system_python = None
            for path in ["/usr/bin/python3", "/usr/local/bin/python3", 
                        "/Library/Developer/CommandLineTools/usr/bin/python3",
                        "/opt/homebrew/bin/python3"]:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    system_python = path
                    break
            
            if not system_python:
                logger.error("未找到系统Python")
                print("错误: 未找到系统Python")
                return 1
            
            # 使用系统Python重新运行此脚本
            logger.info(f"使用系统Python重新运行: {system_python}")
            os.execl(system_python, system_python, __file__)
            
    except Exception as e:
        logger.error(f"启动器错误: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"启动器错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 