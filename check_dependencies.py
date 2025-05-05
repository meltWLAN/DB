#!/usr/bin/env python3
"""
Check if necessary dependencies are available for the market overview modules
"""
import sys
import importlib.util

def check_module(module_name):
    """Check if a module is installed"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ Module {module_name} is NOT installed")
        return False
    else:
        print(f"✅ Module {module_name} is installed")
        return True

def main():
    """Check all required modules"""
    required_modules = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tushare",
        "logging",
        "datetime",
        "requests",
        "json",
    ]
    
    missing = []
    for module in required_modules:
        if not check_module(module):
            missing.append(module)
    
    if missing:
        print("\nMissing modules:")
        for module in missing:
            print(f"  - {module}")
        print("\nInstall them using:")
        print(f"pip install {' '.join(missing)}")
        return 1
    else:
        print("\nAll required modules are installed!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 