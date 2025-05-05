# 自动加载动量分析模块修复
try:
    from . import momentum_fix_integration
except ImportError:
    pass
