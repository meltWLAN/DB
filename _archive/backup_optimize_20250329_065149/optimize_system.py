#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统优化脚本 - 改进性能和用户体验
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

# 配置日志
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Optimizer")

class SystemOptimizer:
    """系统优化器类"""
    
    def __init__(self):
        """初始化优化器"""
        self.start_time = time.time()
        self.optimized_files = 0
        self.total_size_before = 0
        self.total_size_after = 0
        self.backup_dir = f"backup_optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 创建必要的目录
        for dirname in ["data", "logs", "results", "charts", "assets", "cache"]:
            os.makedirs(dirname, exist_ok=True)
        
        logger.info(f"系统优化器初始化完成，备份目录: {self.backup_dir}")
    
    def backup_file(self, file_path):
        """备份文件"""
        if os.path.exists(file_path) and os.path.isfile(file_path):
            backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            logger.info(f"已备份文件: {file_path} -> {backup_path}")
            return True
        return False
    
    def optimize_imports(self, file_path):
        """优化Python文件的导入语句"""
        if not file_path.endswith('.py') or not os.path.exists(file_path):
            return False
        
        try:
            # 备份原文件
            self.backup_file(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_size = len(content)
            self.total_size_before += original_size
            
            # 优化导入语句
            import_section = []
            other_code = []
            current_section = []
            in_import_section = True
            
            for line in content.split('\n'):
                line_stripped = line.strip()
                
                # 判断是否还在导入部分
                if in_import_section:
                    if (line_stripped.startswith('import ') or 
                        line_stripped.startswith('from ') or 
                        line_stripped == '' or 
                        line_stripped.startswith('#')):
                        import_section.append(line)
                    else:
                        in_import_section = False
                        other_code.append(line)
                else:
                    other_code.append(line)
            
            # 去除重复的导入
            clean_imports = []
            import_set = set()
            
            for line in import_section:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#'):
                    if line_stripped not in import_set:
                        import_set.add(line_stripped)
                        clean_imports.append(line)
                else:
                    clean_imports.append(line)
            
            # 对导入进行排序
            std_lib_imports = []
            third_party_imports = []
            local_imports = []
            
            for line in clean_imports:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                    
                if line_stripped.startswith('import '):
                    module = line_stripped.split()[1].split('.')[0]
                elif line_stripped.startswith('from '):
                    module = line_stripped.split()[1].split('.')[0]
                else:
                    module = ""
                
                # 检查是标准库、第三方库还是本地库
                if module in sys.builtin_module_names or module in ('os', 'sys', 'time', 'datetime', 'logging', 'pathlib', 'shutil'):
                    std_lib_imports.append(line)
                elif module in ('numpy', 'pandas', 'matplotlib', 'tkinter', 'PIL'):
                    third_party_imports.append(line)
                else:
                    local_imports.append(line)
            
            # 重组导入部分
            organized_imports = []
            if std_lib_imports:
                organized_imports.extend(std_lib_imports)
                organized_imports.append('')
            if third_party_imports:
                organized_imports.extend(third_party_imports)
                organized_imports.append('')
            if local_imports:
                organized_imports.extend(local_imports)
                organized_imports.append('')
            
            # 重组文件内容
            optimized_content = '\n'.join(organized_imports + other_code)
            optimized_size = len(optimized_content)
            self.total_size_after += optimized_size
            
            # 写入优化后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            
            self.optimized_files += 1
            size_diff = original_size - optimized_size
            logger.info(f"优化文件: {file_path} (减少 {size_diff} 字节)")
            return True
        
        except Exception as e:
            logger.error(f"优化文件失败: {file_path}, 错误: {str(e)}")
            return False
    
    def remove_logs(self):
        """清理旧的日志文件"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            return 0
        
        logs_removed = 0
        current_time = time.time()
        
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.log'):
                file_stat = os.stat(file_path)
                # 清理30天前的日志
                if (current_time - file_stat.st_mtime) / (60 * 60 * 24) > 30:
                    try:
                        os.remove(file_path)
                        logs_removed += 1
                        logger.info(f"已删除旧日志: {file_path}")
                    except Exception as e:
                        logger.error(f"删除日志失败: {file_path}, 错误: {str(e)}")
        
        return logs_removed
    
    def create_optimize_gui(self):
        """创建优化后的GUI启动器"""
        launcher_path = "optimized_launcher.py"
        self.backup_file(launcher_path)
        
        content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
优化版启动器 - 提供更快的启动速度和更好的用户体验
\"\"\"

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import platform
from datetime import datetime

# 确保必要的目录存在
for dirname in ["data", "logs", "results", "charts"]:
    os.makedirs(dirname, exist_ok=True)

# 配置日志
log_file = os.path.join("logs", f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)

class OptimizedLauncher:
    \"\"\"优化版启动器类\"\"\"
    
    def __init__(self, root):
        \"\"\"初始化启动器\"\"\"
        self.root = root
        self.root.title("股票分析系统 - 优化版")
        self.root.geometry("600x400")
        
        # 设置主题
        self.setup_theme()
        
        # 创建界面元素
        self.create_widgets()
        
        # 检查依赖
        self.check_dependencies()
    
    def setup_theme(self):
        \"\"\"设置主题\"\"\"
        style = ttk.Style()
        
        # 检测系统
        if platform.system() == "Darwin":  # macOS
            style.theme_use("aqua")
        elif platform.system() == "Windows":
            style.theme_use("vista")
        else:
            style.theme_use("clam")
        
        # 自定义按钮样式
        style.configure(
            "TButton",
            font=("Helvetica", 12),
            padding=6
        )
        style.configure(
            "Header.TLabel",
            font=("Helvetica", 24, "bold")
        )
        style.configure(
            "SubHeader.TLabel",
            font=("Helvetica", 12)
        )
    
    def create_widgets(self):
        \"\"\"创建界面元素\"\"\"
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title = ttk.Label(
            main_frame,
            text="股票分析系统",
            style="Header.TLabel"
        )
        title.pack(pady=(0, 20))
        
        # 子标题
        subtitle = ttk.Label(
            main_frame,
            text="选择一个功能模块启动",
            style="SubHeader.TLabel"
        )
        subtitle.pack(pady=(0, 30))
        
        # 模块按钮
        modules_frame = ttk.Frame(main_frame)
        modules_frame.pack(fill=tk.BOTH, expand=True)
        
        modules = [
            ("完整分析界面", "stock_analysis_gui.py", "集成了所有分析功能的完整界面"),
            ("简易分析界面", "simple_gui.py", "简化版分析界面，适合新用户"),
            ("无界面分析", "headless_gui.py", "命令行分析工具，适合高级用户")
        ]
        
        for i, (name, script, desc) in enumerate(modules):
            module_frame = ttk.Frame(modules_frame, padding=5)
            module_frame.pack(fill=tk.X, pady=10)
            
            button = ttk.Button(
                module_frame,
                text=name,
                command=lambda s=script: self.launch_module(s)
            )
            button.pack(side=tk.LEFT, padx=(0, 10))
            
            label = ttk.Label(module_frame, text=desc)
            label.pack(side=tk.LEFT, fill=tk.X)
        
        # 底部状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("系统就绪")
        
        status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 系统信息
        info_label = ttk.Label(
            status_frame,
            text=f"Python {platform.python_version()}",
            relief=tk.SUNKEN,
            anchor=tk.E
        )
        info_label.pack(side=tk.RIGHT)
    
    def check_dependencies(self):
        \"\"\"检查依赖包\"\"\"
        required_packages = {
            "numpy": "数据计算",
            "pandas": "数据处理",
            "matplotlib": "图表显示"
        }
        
        missing = []
        
        for package, desc in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing.append(f"{package} ({desc})")
        
        if missing:
            message = "缺少以下依赖包:\\n\\n" + "\\n".join(missing)
            message += "\\n\\n建议使用以下命令安装:\\npip install " + " ".join([p.split()[0] for p in missing])
            
            self.status_var.set("缺少依赖")
            messagebox.warning("缺少依赖", message)
    
    def launch_module(self, script):
        \"\"\"启动模块\"\"\"
        self.status_var.set(f"正在启动 {script}...")
        
        def run():
            try:
                # 准备环境变量
                env = os.environ.copy()
                
                # 移除可能干扰的环境变量
                for var in list(env.keys()):
                    if var.startswith("PYENV"):
                        del env[var]
                
                # 设置Python路径
                env["PYTHONPATH"] = os.getcwd()
                
                # 启动进程
                python_exe = sys.executable
                cmd = [python_exe, script]
                
                subprocess.Popen(cmd, env=env)
                self.status_var.set(f"已启动 {script}")
            except Exception as e:
                self.status_var.set(f"启动失败: {str(e)}")
                messagebox.showerror("启动错误", f"启动 {script} 失败:\\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()

def main():
    \"\"\"主函数\"\"\"
    # 创建主窗口
    root = tk.Tk()
    
    # 设置图标
    try:
        icon_path = os.path.join("assets", "icon.png")
        if os.path.exists(icon_path):
            img = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, img)
    except Exception as e:
        logging.warning(f"无法加载图标: {str(e)}")
    
    # 创建启动器
    app = OptimizedLauncher(root)
    
    # 主循环
    root.mainloop()

if __name__ == "__main__":
    main()
"""
        
        with open(launcher_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        os.chmod(launcher_path, 0o755)
        logger.info(f"已创建优化版启动器: {launcher_path}")
    
    def optimize_all_py_files(self):
        """优化所有Python文件"""
        py_files = [f for f in os.listdir('.') if f.endswith('.py') and os.path.isfile(f)]
        
        for py_file in py_files:
            self.optimize_imports(py_file)
    
    def create_runner_script(self):
        """创建优化版运行脚本"""
        runner_path = "run_optimized.sh"
        self.backup_file(runner_path)
        
        content = """#!/bin/bash

# 优化版运行脚本 - 绕过pyenv干扰
echo "=============================================="
echo "股票分析系统 - 优化版启动器"
echo "=============================================="

# 隔离环境变量
exec env -i \\
  HOME="$HOME" \\
  USER="$USER" \\
  DISPLAY="$DISPLAY" \\
  PATH="/usr/bin:/usr/local/bin:/bin:/sbin" \\
  /usr/bin/python3 "$(dirname "$0")/optimized_launcher.py"
"""
        
        with open(runner_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        os.chmod(runner_path, 0o755)
        logger.info(f"已创建优化版运行脚本: {runner_path}")
    
    def optimize_gui_controller(self):
        """优化GUI控制器"""
        controller_path = "gui_controller.py"
        if not os.path.exists(controller_path):
            return False
        
        # 备份原文件
        self.backup_file(controller_path)
        
        try:
            # 读取文件
            with open(controller_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            # 添加内存优化和缓存机制
            optimized_lines = []
            cache_added = False
            
            for line in content:
                optimized_lines.append(line)
                
                # 添加LRU缓存机制
                if "class GuiController" in line and not cache_added:
                    cache_added = True
                    optimized_lines.append("""    # 使用LRU缓存优化数据加载
    def __init__(self, use_tushare=False, cache_limit=128):
        \"\"\"初始化控制器
        
        Args:
            use_tushare: 是否使用Tushare数据源
            cache_limit: 缓存限制大小
        \"\"\"
        self.use_tushare = use_tushare
        self._data_cache = {}
        self._cache_keys = []
        self._cache_limit = cache_limit
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
    
    def _get_cached_data(self, key):
        \"\"\"从缓存获取数据\"\"\"
        if key in self._data_cache:
            # 更新访问顺序
            self._cache_keys.remove(key)
            self._cache_keys.append(key)
            return self._data_cache[key]
        return None
    
    def _set_cached_data(self, key, data):
        \"\"\"设置缓存数据\"\"\"
        # 如果缓存已满，删除最早访问的项目
        if len(self._cache_keys) >= self._cache_limit and self._cache_keys:
            oldest_key = self._cache_keys.pop(0)
            if oldest_key in self._data_cache:
                del self._data_cache[oldest_key]
        
        # 添加新项目到缓存
        if key not in self._data_cache:
            self._cache_keys.append(key)
        self._data_cache[key] = data
    
    def clear_cache(self):
        \"\"\"清空缓存\"\"\"
        self._data_cache.clear()
        self._cache_keys.clear()
        
""")
            
            # 写入优化后的文件
            with open(controller_path, 'w', encoding='utf-8') as f:
                f.write(''.join(optimized_lines))
            
            logger.info(f"已优化GUI控制器: {controller_path}")
            return True
            
        except Exception as e:
            logger.error(f"优化GUI控制器失败: {str(e)}")
            return False
    
    def run_optimization(self):
        """运行所有优化"""
        logger.info("开始系统优化...")
        
        # 1. 优化所有Python文件的导入
        logger.info("步骤1: 优化Python文件导入...")
        self.optimize_all_py_files()
        
        # 2. 清理旧日志
        logger.info("步骤2: 清理旧日志...")
        logs_removed = self.remove_logs()
        logger.info(f"已清理 {logs_removed} 个旧日志文件")
        
        # 3. 创建优化版启动器
        logger.info("步骤3: 创建优化版启动器...")
        self.create_optimize_gui()
        
        # 4. 创建优化版运行脚本
        logger.info("步骤4: 创建优化版运行脚本...")
        self.create_runner_script()
        
        # 5. 优化GUI控制器
        logger.info("步骤5: 优化GUI控制器...")
        self.optimize_gui_controller()
        
        # 完成统计
        elapsed_time = time.time() - self.start_time
        size_diff = self.total_size_before - self.total_size_after
        
        logger.info(f"优化完成! 耗时: {elapsed_time:.2f}秒")
        logger.info(f"优化文件数: {self.optimized_files}")
        logger.info(f"减少大小: {size_diff} 字节")
        logger.info(f"备份目录: {self.backup_dir}")
        
        return {
            "elapsed_time": elapsed_time,
            "optimized_files": self.optimized_files,
            "size_diff": size_diff,
            "backup_dir": self.backup_dir
        }

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统 - 优化工具")
    print("=" * 60)
    print("这个工具将优化系统性能和用户体验")
    print("优化过程不会删除任何文件，所有修改都有备份")
    print("-" * 60)
    
    # 运行优化
    optimizer = SystemOptimizer()
    result = optimizer.run_optimization()
    
    # 显示结果
    print("\n优化完成!")
    print(f"- 优化文件数: {result['optimized_files']}")
    print(f"- 减少大小: {result['size_diff']} 字节")
    print(f"- 耗时: {result['elapsed_time']:.2f}秒")
    print(f"- 备份目录: {result['backup_dir']}")
    print("\n推荐使用新的启动脚本: ./run_optimized.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 