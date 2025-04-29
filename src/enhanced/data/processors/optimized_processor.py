#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的数据处理器
提供高效的数据处理和转换功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

# 设置日志
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """
    增强版数据处理器
    提供数据清洗、转换和处理功能
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典，包含处理参数
        """
        self.config = config or {}
        self.data_quality_threshold = self.config.get('data_quality_threshold', 0.8)
        self.max_missing_days = self.config.get('max_missing_days', 5)
        self.min_trading_days = self.config.get('min_trading_days', 20)
        
        logger.info("增强版数据处理器初始化完成")
    
    def process_daily_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理日线数据，包括填充缺失值、计算涨跌幅等
        
        Args:
            data: 日线数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if data is None or data.empty:
            logger.warning("处理的数据为空")
            return pd.DataFrame()
        
        try:
            # 复制数据以避免修改原始数据
            processed_data = data.copy()
            
            # 确保日期是索引
            if 'date' in processed_data.columns:
                processed_data['date'] = pd.to_datetime(processed_data['date'])
                processed_data.set_index('date', inplace=True)
            
            # 按日期排序
            processed_data.sort_index(inplace=True)
            
            # 填充缺失值
            processed_data.fillna(method='ffill', inplace=True)
            
            # 计算涨跌幅
            if 'close' in processed_data.columns:
                processed_data['pct_change'] = processed_data['close'].pct_change() * 100
                
            # 计算均线
            if 'close' in processed_data.columns:
                for period in [5, 10, 20, 30, 60]:
                    processed_data[f'ma{period}'] = processed_data['close'].rolling(period).mean()
            
            # 计算成交量变化率
            if 'volume' in processed_data.columns:
                processed_data['volume_ratio'] = processed_data['volume'] / processed_data['volume'].rolling(5).mean()
                
            return processed_data
            
        except Exception as e:
            logger.error(f"处理日线数据时出错: {str(e)}")
            return data 