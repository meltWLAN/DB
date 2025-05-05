"""
检查系统依赖包是否安装
"""

import importlib
import subprocess
import sys

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# 系统必需的包
required_packages = [
    'pandas',
    'numpy',
    'tushare',
    'akshare',
    'matplotlib',
    'pyarrow',  # 用于parquet文件
    'requests',
    'jqdatasdk',  # JoinQuant SDK
]

# 检查每个包是否已安装
missing_packages = []
for package in required_packages:
    if not check_package(package):
        missing_packages.append(package)

# 显示结果
if missing_packages:
    print("以下包需要安装:")
    for package in missing_packages:
        print(f"  - {package}")
    
    # 询问是否安装缺失的包
    install = input("是否要安装这些包? [y/N]: ")
    if install.lower() == 'y':
        for package in missing_packages:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("所有缺失的包都已安装完成。")
    else:
        print("请手动安装缺失的包后再运行系统。")
        sys.exit(1)
else:
    print("所有必需的包都已安装。")
    print("系统环境检查通过！") 