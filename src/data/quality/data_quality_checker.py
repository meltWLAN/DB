"""
数据质量检查模块
负责数据质量检查和验证
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ...utils.logger import get_logger

class DataQualityChecker:
    """数据质量检查器类"""
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.logger = get_logger("DataQualityChecker")
        
    def check_stock_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查股票数据质量
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            Dict[str, Any]: 质量检查结果，包含：
                - is_valid: 是否有效
                - completeness: 完整性得分
                - consistency: 一致性得分
                - timeliness: 时效性得分
                - issues: 问题列表
        """
        try:
            if df is None or df.empty:
                return {
                    "is_valid": False,
                    "completeness": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0,
                    "issues": ["数据为空"]
                }
                
            # 检查完整性
            completeness = self._check_completeness(df)
            
            # 检查一致性
            consistency = self._check_consistency(df)
            
            # 检查时效性
            timeliness = self._check_timeliness(df)
            
            # 收集问题
            issues = []
            if completeness < 0.9:
                issues.append("数据完整性不足")
            if consistency < 0.9:
                issues.append("数据一致性不足")
            if timeliness < 0.9:
                issues.append("数据时效性不足")
                
            # 判断是否有效
            is_valid = all(score >= 0.9 for score in [completeness, consistency, timeliness])
            
            return {
                "is_valid": is_valid,
                "completeness": completeness,
                "consistency": consistency,
                "timeliness": timeliness,
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error("检查数据质量失败", exc_info=e)
            return {
                "is_valid": False,
                "completeness": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "issues": [f"检查过程出错: {str(e)}"]
            }
            
    def _check_completeness(self, df: pd.DataFrame) -> float:
        """
        检查数据完整性
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            float: 完整性得分
        """
        try:
            # 检查缺失值
            missing_ratio = df.isnull().mean()
            
            # 检查异常值
            z_scores = np.abs((df - df.mean()) / df.std())
            outlier_ratio = (z_scores > 3).mean()
            
            # 计算完整性得分
            completeness = 1 - (missing_ratio.mean() + outlier_ratio.mean()) / 2
            
            return max(0, min(1, completeness))
            
        except Exception as e:
            self.logger.error("检查数据完整性失败", exc_info=e)
            return 0.0
            
    def _check_consistency(self, df: pd.DataFrame) -> float:
        """
        检查数据一致性
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            float: 一致性得分
        """
        try:
            # 检查价格一致性
            price_consistency = (
                (df["high"] >= df["open"]) &
                (df["high"] >= df["close"]) &
                (df["low"] <= df["open"]) &
                (df["low"] <= df["close"])
            ).mean()
            
            # 检查成交量一致性
            volume_consistency = (df["volume"] >= 0).mean()
            
            # 计算一致性得分
            consistency = (price_consistency + volume_consistency) / 2
            
            return max(0, min(1, consistency))
            
        except Exception as e:
            self.logger.error("检查数据一致性失败", exc_info=e)
            return 0.0
            
    def _check_timeliness(self, df: pd.DataFrame) -> float:
        """
        检查数据时效性
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            float: 时效性得分
        """
        try:
            # 检查日期连续性
            dates = pd.to_datetime(df.index)
            date_diff = dates.diff().dt.days
            continuity = (date_diff == 1).mean()
            
            # 检查最新数据时间
            latest_date = dates.max()
            current_date = datetime.now()
            days_diff = (current_date - latest_date).days
            
            # 计算时效性得分
            if days_diff <= 1:
                timeliness = 1.0
            elif days_diff <= 3:
                timeliness = 0.8
            elif days_diff <= 5:
                timeliness = 0.6
            else:
                timeliness = 0.4
                
            # 综合得分
            timeliness = (continuity + timeliness) / 2
            
            return max(0, min(1, timeliness))
            
        except Exception as e:
            self.logger.error("检查数据时效性失败", exc_info=e)
            return 0.0
            
    def check_stock_list(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查股票列表数据质量
        
        Args:
            df: 股票列表DataFrame
            
        Returns:
            Dict[str, Any]: 质量检查结果
        """
        try:
            if df is None or df.empty:
                return {
                    "is_valid": False,
                    "completeness": 0.0,
                    "consistency": 0.0,
                    "issues": ["数据为空"]
                }
                
            # 检查完整性
            completeness = self._check_list_completeness(df)
            
            # 检查一致性
            consistency = self._check_list_consistency(df)
            
            # 收集问题
            issues = []
            if completeness < 0.9:
                issues.append("股票列表完整性不足")
            if consistency < 0.9:
                issues.append("股票列表一致性不足")
                
            # 判断是否有效
            is_valid = all(score >= 0.9 for score in [completeness, consistency])
            
            return {
                "is_valid": is_valid,
                "completeness": completeness,
                "consistency": consistency,
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error("检查股票列表质量失败", exc_info=e)
            return {
                "is_valid": False,
                "completeness": 0.0,
                "consistency": 0.0,
                "issues": [f"检查过程出错: {str(e)}"]
            }
            
    def _check_list_completeness(self, df: pd.DataFrame) -> float:
        """检查股票列表完整性"""
        try:
            # 检查必要字段
            required_fields = ["code", "name", "industry"]
            field_completeness = df[required_fields].notnull().mean()
            
            # 检查数据量
            min_stocks = 3000  # 最小股票数量
            quantity_completeness = min(1, len(df) / min_stocks)
            
            # 计算完整性得分
            completeness = (field_completeness.mean() + quantity_completeness) / 2
            
            return max(0, min(1, completeness))
            
        except Exception as e:
            self.logger.error("检查股票列表完整性失败", exc_info=e)
            return 0.0
            
    def _check_list_consistency(self, df: pd.DataFrame) -> float:
        """检查股票列表一致性"""
        try:
            # 检查代码格式
            code_format = df["code"].str.match(r"^\d{6}$").mean()
            
            # 检查重复值
            duplicates = df.duplicated(subset=["code"]).mean()
            uniqueness = 1 - duplicates
            
            # 计算一致性得分
            consistency = (code_format + uniqueness) / 2
            
            return max(0, min(1, consistency))
            
        except Exception as e:
            self.logger.error("检查股票列表一致性失败", exc_info=e)
            return 0.0
            
    def generate_quality_report(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Args:
            stock_data: 股票数据字典，key为股票代码，value为DataFrame
            
        Returns:
            Dict[str, Any]: 质量报告
        """
        try:
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_stocks": len(stock_data),
                "quality_summary": {
                    "excellent": 0,
                    "good": 0,
                    "fair": 0,
                    "poor": 0
                },
                "issues_summary": {},
                "stock_details": {}
            }
            
            # 检查每只股票的数据质量
            for stock_code, df in stock_data.items():
                quality_result = self.check_stock_data(df)
                
                # 更新质量统计
                if quality_result["is_valid"]:
                    report["quality_summary"]["excellent"] += 1
                elif quality_result["completeness"] >= 0.8:
                    report["quality_summary"]["good"] += 1
                elif quality_result["completeness"] >= 0.6:
                    report["quality_summary"]["fair"] += 1
                else:
                    report["quality_summary"]["poor"] += 1
                    
                # 更新问题统计
                for issue in quality_result["issues"]:
                    report["issues_summary"][issue] = report["issues_summary"].get(issue, 0) + 1
                    
                # 记录详细信息
                report["stock_details"][stock_code] = quality_result
                
            return report
            
        except Exception as e:
            self.logger.error("生成质量报告失败", exc_info=e)
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            } 