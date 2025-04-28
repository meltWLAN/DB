"""
价格形态识别模块
提供对常见技术分析形态的识别功能，如头肩顶/底、双顶/底、旗形等
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PricePatternRecognizer:
    """价格形态识别器类"""
    
    def __init__(self, window_size=20, extrema_window=5, similarity_threshold=0.75):
        """
        初始化价格形态识别器
        
        Args:
            window_size: 识别窗口大小
            extrema_window: 极值点检测窗口大小
            similarity_threshold: 形态相似度阈值
        """
        self.window_size = window_size
        self.extrema_window = extrema_window
        self.similarity_threshold = similarity_threshold
    
    def find_extrema(self, data, column='close', order=5):
        """
        查找价格序列中的极大值和极小值点
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            order: 极值点的窗口大小
            
        Returns:
            tuple: (极大值索引, 极小值索引)
        """
        # 确保数据包含目标列
        if column not in data.columns:
            logger.error(f"数据中不包含列 {column}")
            return [], []
        
        # 查找极大值和极小值点
        prices = data[column].values
        max_idx = argrelextrema(prices, np.greater, order=order)[0]
        min_idx = argrelextrema(prices, np.less, order=order)[0]
        
        return max_idx, min_idx
    
    def detect_head_and_shoulders(self, data, column='close', price_tolerance=0.03):
        """
        识别头肩顶/底形态
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            price_tolerance: 左右肩高度相似性容差
            
        Returns:
            dict: 包含形态识别结果的字典
        """
        # 查找极值点
        max_idx, min_idx = self.find_extrema(data, column)
        
        if len(max_idx) < 3 or len(min_idx) < 2:
            return {"pattern": None, "confidence": 0, "details": {}}
        
        patterns = []
        
        # 识别头肩顶形态 (头肩顶: 三个峰，中间峰最高)
        for i in range(len(max_idx) - 2):
            # 获取三个连续峰值
            left_shoulder_idx = max_idx[i]
            head_idx = max_idx[i+1]
            right_shoulder_idx = max_idx[i+2]
            
            left_shoulder = data[column].iloc[left_shoulder_idx]
            head = data[column].iloc[head_idx]
            right_shoulder = data[column].iloc[right_shoulder_idx]
            
            # 检查中间峰是否最高，以及左右肩是否近似等高
            if (head > left_shoulder and 
                head > right_shoulder and
                abs(left_shoulder - right_shoulder) < price_tolerance * left_shoulder):
                
                # 查找颈线(两肩之间的低点连线)
                neck_line_idx = [idx for idx in min_idx if left_shoulder_idx < idx < right_shoulder_idx]
                
                if len(neck_line_idx) < 2:
                    continue
                
                # 计算颈线
                neck_line_start = data[column].iloc[neck_line_idx[0]]
                neck_line_end = data[column].iloc[neck_line_idx[-1]]
                
                # 量化形态的完美度/置信度
                height = head - (neck_line_start + neck_line_end) / 2
                shoulder_ratio = abs(left_shoulder - right_shoulder) / left_shoulder
                
                confidence = (1 - shoulder_ratio) * (height / head) * 100
                
                pattern_dict = {
                    "pattern": "head_and_shoulders_top",
                    "confidence": min(100, max(0, confidence)),
                    "details": {
                        "left_shoulder_idx": int(left_shoulder_idx),
                        "head_idx": int(head_idx),
                        "right_shoulder_idx": int(right_shoulder_idx),
                        "neck_line": [int(neck_line_idx[0]), int(neck_line_idx[-1])],
                        "target_price": neck_line_end - height
                    }
                }
                patterns.append(pattern_dict)
        
        # 识别头肩底形态 (头肩底: 三个谷，中间谷最低)
        for i in range(len(min_idx) - 2):
            # 获取三个连续谷值
            left_shoulder_idx = min_idx[i]
            head_idx = min_idx[i+1]
            right_shoulder_idx = min_idx[i+2]
            
            left_shoulder = data[column].iloc[left_shoulder_idx]
            head = data[column].iloc[head_idx]
            right_shoulder = data[column].iloc[right_shoulder_idx]
            
            # 检查中间谷是否最低，以及左右肩是否近似等低
            if (head < left_shoulder and 
                head < right_shoulder and
                abs(left_shoulder - right_shoulder) < price_tolerance * left_shoulder):
                
                # 查找颈线(两肩之间的高点连线)
                neck_line_idx = [idx for idx in max_idx if left_shoulder_idx < idx < right_shoulder_idx]
                
                if len(neck_line_idx) < 2:
                    continue
                
                # 计算颈线
                neck_line_start = data[column].iloc[neck_line_idx[0]]
                neck_line_end = data[column].iloc[neck_line_idx[-1]]
                
                # 量化形态的完美度/置信度
                depth = (neck_line_start + neck_line_end) / 2 - head
                shoulder_ratio = abs(left_shoulder - right_shoulder) / left_shoulder
                
                confidence = (1 - shoulder_ratio) * (depth / (neck_line_start + neck_line_end) * 2) * 100
                
                pattern_dict = {
                    "pattern": "head_and_shoulders_bottom",
                    "confidence": min(100, max(0, confidence)),
                    "details": {
                        "left_shoulder_idx": int(left_shoulder_idx),
                        "head_idx": int(head_idx),
                        "right_shoulder_idx": int(right_shoulder_idx),
                        "neck_line": [int(neck_line_idx[0]), int(neck_line_idx[-1])],
                        "target_price": neck_line_end + depth
                    }
                }
                patterns.append(pattern_dict)
        
        # 如果找到多个形态，返回置信度最高的
        if patterns:
            return max(patterns, key=lambda x: x["confidence"])
        else:
            return {"pattern": None, "confidence": 0, "details": {}}
    
    def detect_double_tops_bottoms(self, data, column='close', price_tolerance=0.03):
        """
        识别双顶/双底形态
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            price_tolerance: 两个顶/底的高度相似性容差
            
        Returns:
            dict: 包含形态识别结果的字典
        """
        # 查找极值点
        max_idx, min_idx = self.find_extrema(data, column)
        
        if len(max_idx) < 2 or len(min_idx) < 1:
            return {"pattern": None, "confidence": 0, "details": {}}
        
        patterns = []
        
        # 识别双顶形态 (两个近似等高的峰)
        for i in range(len(max_idx) - 1):
            # 获取两个连续峰值
            top1_idx = max_idx[i]
            top2_idx = max_idx[i+1]
            
            # 确保两个峰之间至少有一个明显的谷
            between_idx = [idx for idx in min_idx if top1_idx < idx < top2_idx]
            if not between_idx:
                continue
                
            mid_idx = between_idx[0]
            
            top1 = data[column].iloc[top1_idx]
            top2 = data[column].iloc[top2_idx]
            mid_val = data[column].iloc[mid_idx]
            
            # 检查两个峰是否近似等高，以及中间谷是否足够低
            if (abs(top1 - top2) < price_tolerance * top1 and
                top1 > mid_val and
                top2 > mid_val and
                (top1 + top2) / 2 - mid_val > 0.03 * top1):
                
                # 计算颈线
                neck_line = mid_val
                
                # 量化形态的完美度/置信度
                height = (top1 + top2) / 2 - neck_line
                top_ratio = abs(top1 - top2) / top1
                
                confidence = (1 - top_ratio) * (height / ((top1 + top2) / 2)) * 100
                
                pattern_dict = {
                    "pattern": "double_top",
                    "confidence": min(100, max(0, confidence)),
                    "details": {
                        "top1_idx": int(top1_idx),
                        "top2_idx": int(top2_idx),
                        "mid_idx": int(mid_idx),
                        "neck_line": float(neck_line),
                        "target_price": neck_line - height
                    }
                }
                patterns.append(pattern_dict)
        
        # 识别双底形态 (两个近似等低的谷)
        for i in range(len(min_idx) - 1):
            # 获取两个连续谷值
            bottom1_idx = min_idx[i]
            bottom2_idx = min_idx[i+1]
            
            # 确保两个谷之间至少有一个明显的峰
            between_idx = [idx for idx in max_idx if bottom1_idx < idx < bottom2_idx]
            if not between_idx:
                continue
                
            mid_idx = between_idx[0]
            
            bottom1 = data[column].iloc[bottom1_idx]
            bottom2 = data[column].iloc[bottom2_idx]
            mid_val = data[column].iloc[mid_idx]
            
            # 检查两个谷是否近似等低，以及中间峰是否足够高
            if (abs(bottom1 - bottom2) < price_tolerance * bottom1 and
                bottom1 < mid_val and
                bottom2 < mid_val and
                mid_val - (bottom1 + bottom2) / 2 > 0.03 * bottom1):
                
                # 计算颈线
                neck_line = mid_val
                
                # 量化形态的完美度/置信度
                depth = neck_line - (bottom1 + bottom2) / 2
                bottom_ratio = abs(bottom1 - bottom2) / bottom1
                
                confidence = (1 - bottom_ratio) * (depth / ((bottom1 + bottom2) / 2)) * 100
                
                pattern_dict = {
                    "pattern": "double_bottom",
                    "confidence": min(100, max(0, confidence)),
                    "details": {
                        "bottom1_idx": int(bottom1_idx),
                        "bottom2_idx": int(bottom2_idx),
                        "mid_idx": int(mid_idx),
                        "neck_line": float(neck_line),
                        "target_price": neck_line + depth
                    }
                }
                patterns.append(pattern_dict)
        
        # 如果找到多个形态，返回置信度最高的
        if patterns:
            return max(patterns, key=lambda x: x["confidence"])
        else:
            return {"pattern": None, "confidence": 0, "details": {}}
    
    def detect_triangle_patterns(self, data, column='close', min_points=5):
        """
        识别三角形形态(上升三角形、下降三角形、对称三角形)
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            min_points: 形成三角形的最小点数
            
        Returns:
            dict: 包含形态识别结果的字典
        """
        if len(data) < min_points:
            return {"pattern": None, "confidence": 0, "details": {}}
        
        # 查找极值点
        max_idx, min_idx = self.find_extrema(data, column)
        
        if len(max_idx) < 2 or len(min_idx) < 2:
            return {"pattern": None, "confidence": 0, "details": {}}
            
        # 保留最近的N个高点和低点
        recent_max = max_idx[-min_points:] if len(max_idx) >= min_points else max_idx
        recent_min = min_idx[-min_points:] if len(min_idx) >= min_points else min_idx
        
        # 计算高点和低点的线性回归
        if len(recent_max) >= 2:
            x_max = recent_max
            y_max = data[column].iloc[recent_max].values
            max_slope, max_intercept, _, _, _ = stats.linregress(x_max, y_max)
        else:
            max_slope = 0
            
        if len(recent_min) >= 2:
            x_min = recent_min
            y_min = data[column].iloc[recent_min].values
            min_slope, min_intercept, _, _, _ = stats.linregress(x_min, y_min)
        else:
            min_slope = 0
            
        # 计算R平方以评估拟合质量
        if len(recent_max) >= 2:
            max_r2 = stats.pearsonr(x_max, y_max)[0] ** 2
        else:
            max_r2 = 0
            
        if len(recent_min) >= 2:
            min_r2 = stats.pearsonr(x_min, y_min)[0] ** 2
        else:
            min_r2 = 0
        
        # 识别三角形类型
        pattern = None
        confidence = 0
        details = {}
        
        # 对称三角形: 高点下降，低点上升
        if max_slope < -0.01 and min_slope > 0.01 and max_r2 > 0.7 and min_r2 > 0.7:
            pattern = "symmetric_triangle"
            confidence = (max_r2 + min_r2) * 50  # 平均R平方*100
            
            # 计算收敛点 (两条趋势线相交的点)
            if max_slope != min_slope:  # 避免除以零
                intersect_x = (min_intercept - max_intercept) / (max_slope - min_slope)
                intersect_y = max_slope * intersect_x + max_intercept
                
                # 计算突破方向 (基于形态前的趋势)
                pre_pattern_avg = data[column].iloc[max(0, recent_max[0]-10):recent_max[0]].mean()
                current_avg = data[column].iloc[-5:].mean()
                
                breakout_direction = "up" if current_avg > pre_pattern_avg else "down"
                target_move = abs(data[column].iloc[recent_max[0]] - data[column].iloc[recent_min[0]])
                target_price = intersect_y + target_move if breakout_direction == "up" else intersect_y - target_move
                
                details = {
                    "max_points": [int(x) for x in recent_max],
                    "min_points": [int(x) for x in recent_min],
                    "convergence_point": (int(intersect_x), float(intersect_y)),
                    "breakout_direction": breakout_direction,
                    "target_price": float(target_price)
                }
        
        # 上升三角形: 高点平，低点上升
        elif abs(max_slope) < 0.01 and min_slope > 0.01 and min_r2 > 0.7:
            pattern = "ascending_triangle"
            confidence = min_r2 * 100
            
            # 水平阻力位
            resistance = data[column].iloc[recent_max].mean()
            
            # 计算目标价格 (突破后的预期移动)
            height = resistance - data[column].iloc[recent_min[0]]
            target_price = resistance + height
            
            details = {
                "max_points": [int(x) for x in recent_max],
                "min_points": [int(x) for x in recent_min],
                "resistance": float(resistance),
                "target_price": float(target_price)
            }
        
        # 下降三角形: 高点下降，低点平
        elif max_slope < -0.01 and abs(min_slope) < 0.01 and max_r2 > 0.7:
            pattern = "descending_triangle"
            confidence = max_r2 * 100
            
            # 水平支撑位
            support = data[column].iloc[recent_min].mean()
            
            # 计算目标价格 (突破后的预期移动)
            height = data[column].iloc[recent_max[0]] - support
            target_price = support - height
            
            details = {
                "max_points": [int(x) for x in recent_max],
                "min_points": [int(x) for x in recent_min],
                "support": float(support),
                "target_price": float(target_price)
            }
        
        return {
            "pattern": pattern,
            "confidence": min(100, max(0, confidence)),
            "details": details
        }
    
    def detect_flag_pennant(self, data, column='close', min_pole_return=0.1):
        """
        识别旗形与三角旗形态
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            min_pole_return: 形成旗杆的最小价格变动比例
            
        Returns:
            dict: 包含形态识别结果的字典
        """
        if len(data) < 20:  # 需要足够的数据来形成旗杆和旗帜
            return {"pattern": None, "confidence": 0, "details": {}}
        
        # 寻找潜在的旗杆(快速且单向的价格变动)
        returns = data[column].pct_change()
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # 查找显著的价格趋势作为潜在旗杆
        for i in range(5, len(cumulative_returns) - 10):
            # 计算前i天的累积收益
            pole_return = cumulative_returns.iloc[i]
            
            # 检查是否有足够强的价格趋势
            if abs(pole_return) >= min_pole_return:
                # 确定趋势方向
                trend_up = pole_return > 0
                
                # 分析之后的价格波动，寻找整合(旗帜)
                flag_start = i
                flag_end = min(i + 15, len(data) - 1)  # 最多15天的整合期
                
                flag_data = data.iloc[flag_start:flag_end+1]
                
                if len(flag_data) < 5:  # 需要足够的数据点来形成旗帜
                    continue
                
                # 计算旗帜期间的价格范围
                flag_high = flag_data[column].max()
                flag_low = flag_data[column].min()
                flag_range_pct = (flag_high - flag_low) / flag_data[column].iloc[0]
                
                # 判断是旗形还是三角旗
                flag_prices = flag_data[column].values
                x = np.arange(len(flag_prices))
                
                # 计算趋势线拟合
                try:
                    # 高点趋势线
                    max_idx, _ = self.find_extrema(flag_data, column, order=2)
                    if len(max_idx) >= 2:
                        high_x = max_idx
                        high_y = flag_prices[max_idx]
                        high_slope, _, _, _, _ = stats.linregress(high_x, high_y)
                    else:
                        high_slope = 0
                    
                    # 低点趋势线
                    _, min_idx = self.find_extrema(flag_data, column, order=2)
                    if len(min_idx) >= 2:
                        low_x = min_idx
                        low_y = flag_prices[min_idx]
                        low_slope, _, _, _, _ = stats.linregress(low_x, low_y)
                    else:
                        low_slope = 0
                    
                    # 三角旗：趋势线汇聚
                    is_pennant = (trend_up and high_slope < -0.001 and low_slope > 0.001) or \
                                 (not trend_up and high_slope < -0.001 and low_slope > 0.001)
                    
                    # 旗形：平行通道
                    is_flag = (abs(high_slope - low_slope) < 0.005) and (high_slope * (1 if trend_up else -1) < 0)
                    
                    # 验证交易量衰减
                    volume_decline = False
                    if 'vol' in data.columns:
                        avg_pole_vol = data['vol'].iloc[max(0, flag_start-5):flag_start].mean()
                        avg_flag_vol = flag_data['vol'].mean()
                        volume_decline = avg_flag_vol < avg_pole_vol
                    
                    # 计算目标价格(继续旗杆方向的同等幅度移动)
                    pole_height = abs(data[column].iloc[flag_start] - data[column].iloc[max(0, flag_start-5)])
                    target_price = flag_data[column].iloc[-1] + (pole_height if trend_up else -pole_height)
                    
                    # 识别形态类型和置信度
                    if is_pennant:
                        pattern = "bull_pennant" if trend_up else "bear_pennant"
                        confidence = min(100, (abs(pole_return) / min_pole_return) * 70 + (30 if volume_decline else 0))
                    elif is_flag:
                        pattern = "bull_flag" if trend_up else "bear_flag"
                        confidence = min(100, (abs(pole_return) / min_pole_return) * 70 + (30 if volume_decline else 0))
                    else:
                        continue  # 不是我们要找的形态
                    
                    return {
                        "pattern": pattern,
                        "confidence": confidence,
                        "details": {
                            "pole_start": int(max(0, flag_start-5)),
                            "pole_end": int(flag_start),
                            "flag_start": int(flag_start),
                            "flag_end": int(flag_end),
                            "trend_direction": "up" if trend_up else "down",
                            "target_price": float(target_price)
                        }
                    }
                
                except Exception as e:
                    logger.error(f"旗形/三角旗识别异常: {str(e)}")
                    continue
        
        return {"pattern": None, "confidence": 0, "details": {}}
    
    def detect_patterns(self, data, column='close'):
        """
        综合检测所有支持的价格形态
        
        Args:
            data: 股票数据DataFrame
            column: 用于分析的价格列名
            
        Returns:
            dict: 包含形态识别结果的字典
        """
        patterns = []
        
        # 检测头肩顶/底
        hs_pattern = self.detect_head_and_shoulders(data, column)
        if hs_pattern["pattern"]:
            patterns.append(hs_pattern)
        
        # 检测双顶/双底
        dt_pattern = self.detect_double_tops_bottoms(data, column)
        if dt_pattern["pattern"]:
            patterns.append(dt_pattern)
        
        # 检测三角形
        tri_pattern = self.detect_triangle_patterns(data, column)
        if tri_pattern["pattern"]:
            patterns.append(tri_pattern)
        
        # 检测旗形和三角旗
        flag_pattern = self.detect_flag_pennant(data, column)
        if flag_pattern["pattern"]:
            patterns.append(flag_pattern)
        
        # 返回置信度最高的形态
        if patterns:
            return max(patterns, key=lambda x: x["confidence"])
        else:
            return {"pattern": None, "confidence": 0, "details": {}} 