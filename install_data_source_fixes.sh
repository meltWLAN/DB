#!/bin/bash

# 数据源修复安装脚本
# 用于自动应用所有数据源修复，解决市场概览刷新慢和接口参数问题

# 显示彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}      数据源修复安装程序${NC}"
echo -e "${BLUE}=======================================${NC}"

# 检查Python环境
echo -e "\n${YELLOW}[1/5] 检查Python环境...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo -e "${GREEN}✓ 找到Python 3${NC}"
elif command -v python &>/dev/null; then
    PYTHON="python"
    echo -e "${GREEN}✓ 找到Python${NC}"
else
    echo -e "${RED}✗ 未找到Python解释器${NC}"
    exit 1
fi

# 检查必要的Python包
echo -e "\n${YELLOW}[2/5] 检查必要的Python包...${NC}"
$PYTHON -c "import pandas, numpy, tushare" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 所有必要的Python包已安装${NC}"
else
    echo -e "${YELLOW}! 缺少一些必要的Python包，尝试安装...${NC}"
    $PYTHON -m pip install pandas numpy tushare akshare --quiet
    
    # 再次检查
    $PYTHON -c "import pandas, numpy, tushare" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 已成功安装所有必要的Python包${NC}"
    else
        echo -e "${RED}✗ 无法安装所有必要的Python包${NC}"
        exit 1
    fi
fi

# 备份原始文件
echo -e "\n${YELLOW}[3/5] 备份原始文件...${NC}"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR/src/enhanced/data/fetchers

# 备份核心文件
if [ -f "src/enhanced/data/fetchers/data_source_manager.py" ]; then
    cp src/enhanced/data/fetchers/data_source_manager.py $BACKUP_DIR/src/enhanced/data/fetchers/
    echo -e "${GREEN}✓ 已备份 data_source_manager.py${NC}"
fi

if [ -f "src/enhanced/data/fetchers/tushare_fetcher.py" ]; then
    cp src/enhanced/data/fetchers/tushare_fetcher.py $BACKUP_DIR/src/enhanced/data/fetchers/
    echo -e "${GREEN}✓ 已备份 tushare_fetcher.py${NC}"
fi

if [ -f "src/enhanced/data/fetchers/akshare_fetcher.py" ]; then
    cp src/enhanced/data/fetchers/akshare_fetcher.py $BACKUP_DIR/src/enhanced/data/fetchers/
    echo -e "${GREEN}✓ 已备份 akshare_fetcher.py${NC}"
fi

# 应用修复
echo -e "\n${YELLOW}[4/5] 应用数据源修复...${NC}"

# 运行修复脚本
$PYTHON unified_data_source_fix.py

# 如果修复脚本成功执行，将修复脚本安装到系统中
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 数据源修复脚本已成功执行${NC}"
    
    # 创建自动加载修复的初始化文件
    echo -e "\n${YELLOW}[5/5] 创建自动加载修复...${NC}"
    
    # 创建自动修复初始化脚本
    cat > src/enhanced/data/fixes.py << EOF
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源修复模块
在系统启动时自动应用所有数据源修复
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def apply_data_source_fixes():
    """应用数据源修复"""
    logger.info("应用数据源修复...")
    
    try:
        # 1. 修复TuShare数据接口
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 保存原始实现
        original_get_stock_index_data = EnhancedTushareFetcher.get_stock_index_data
        
        # 定义修复后的方法
        def fixed_get_stock_index_data(self, index_code, start_date=None, end_date=None):
            """修复的获取指数数据方法"""
            try:
                # 标准化指数代码
                if '.' not in index_code:
                    if index_code.startswith('000'):
                        index_code = f"{index_code}.SH"
                    elif index_code.startswith('399'):
                        index_code = f"{index_code}.SZ"
                
                # 标准化日期格式
                if start_date and '-' in start_date:
                    start_date = start_date.replace('-', '')
                if end_date and '-' in end_date:
                    end_date = end_date.replace('-', '')
                
                # 调用原始方法
                return original_get_stock_index_data(self, index_code, start_date, end_date)
            except Exception as e:
                logger.error(f"获取指数 {index_code} 数据失败: {str(e)}")
                return None
        
        # 应用修复
        EnhancedTushareFetcher.get_stock_index_data = fixed_get_stock_index_data
        logger.info("已修复TuShare指数数据获取方法")
        
        # 2. 修复DataSourceManager
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        
        # 修复get_market_overview方法
        original_get_market_overview = DataSourceManager.get_market_overview
        
        def fixed_get_market_overview(self, trade_date=None):
            """修复的市场概览获取方法"""
            try:
                # 调用原始方法
                result = original_get_market_overview(self, trade_date)
                
                # 处理返回类型
                import pandas as pd
                if isinstance(result, pd.DataFrame):
                    if not result.empty:
                        # 转换为字典
                        dict_result = {}
                        for col in result.columns:
                            dict_result[col] = result.iloc[0][col]
                        
                        # 确保日期存在
                        if 'date' not in dict_result and trade_date:
                            dict_result['date'] = trade_date
                        
                        return dict_result
                    else:
                        return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
                elif result is None:
                    return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
                elif isinstance(result, dict):
                    return result
                
                else:
                    return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
            except Exception as e:
                logger.error(f"获取市场概览失败: {str(e)}")
                return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
        
        # 应用修复
        DataSourceManager.get_market_overview = fixed_get_market_overview
        logger.info("已修复市场概览获取方法")
        
        # 3. 添加/修复get_stock_data方法
        if not hasattr(DataSourceManager, 'get_stock_data'):
            def get_stock_data(self, stock_code, start_date=None, end_date=None, limit=None):
                """获取股票日线数据"""
                try:
                    # 调用get_daily_data方法
                    data = self.get_daily_data(stock_code, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # 处理日期列名兼容性
                        date_col = 'date' if 'date' in data.columns else 'trade_date'
                        
                        # 根据限制条数截取数据
                        if limit is not None and len(data) > limit:
                            data = data.sort_values(date_col, ascending=False).head(limit).sort_values(date_col)
                        
                        return data
                    else:
                        return None
                    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 数据失败: {str(e)}")
                    return None
            
            # 添加方法
            DataSourceManager.get_stock_data = get_stock_data
            logger.info("已添加get_stock_data方法")
        
        return True
        
    except Exception as e:
        logger.error(f"应用数据源修复失败: {str(e)}")
        return False

# 在模块导入时自动应用修复
apply_data_source_fixes()
EOF
    
    # 创建自动加载修复的__init__.py文件
    if [ -f "src/enhanced/data/__init__.py" ]; then
        # 如果文件已存在，添加导入语句
        grep -q "from . import fixes" src/enhanced/data/__init__.py || echo -e "\n# 自动加载数据源修复\nfrom . import fixes" >> src/enhanced/data/__init__.py
    else
        # 如果文件不存在，创建新文件
        mkdir -p src/enhanced/data
        cat > src/enhanced/data/__init__.py << EOF
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据模块
"""

# 自动加载数据源修复
from . import fixes
EOF
    fi
    
    echo -e "${GREEN}✓ 自动加载修复已安装${NC}"
    echo -e "${GREEN}✓ 数据源修复已成功安装${NC}"
    
    echo -e "\n${BLUE}=======================================${NC}"
    echo -e "${GREEN}修复已成功安装!${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo -e "原始文件已备份到: ${YELLOW}${BACKUP_DIR}${NC}"
    echo -e "现在可以启动系统，市场概览应该能正常工作了。"
    
else
    echo -e "${RED}✗ 数据源修复脚本执行失败${NC}"
    exit 1
fi 