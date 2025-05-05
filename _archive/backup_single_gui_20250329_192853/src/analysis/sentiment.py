import pandas as pd
import numpy as np
import logging
import re
import jieba
import jieba.analyse
from collections import Counter
from datetime import datetime, timedelta
from snownlp import SnowNLP
from ..config import SENTIMENT_PARAMS

class SentimentAnalyzer:
    """市场情绪分析类"""
    
    def __init__(self, params=None):
        """初始化
        
        Args:
            params: 情绪分析参数，默认使用配置文件中的参数
        """
        self.params = params or SENTIMENT_PARAMS
        self.logger = logging.getLogger(__name__)
        
        # 加载情绪词典 (可以根据需要扩展)
        self.positive_words = set(['上涨', '涨停', '突破', '拉升', '走强', '爆发', '利好', '增长', '回升',
                                '强势', '机会', '看多', '热点', '活跃', '反弹', '向好', '牛市', '赚钱'])
        
        self.negative_words = set(['下跌', '跌停', '破位', '下挫', '走弱', '暴跌', '利空', '下降', '回落',
                                '弱势', '风险', '看空', '低迷', '萎靡', '回调', '看淡', '熊市', '亏损'])
        
        # 初始化结巴分词
        jieba.initialize()
    
    def analyze_text_sentiment(self, text):
        """分析文本情绪
        
        Args:
            text: 文本内容
            
        Returns:
            dict: 情绪分析结果，包含得分和标签
        """
        if not text or not isinstance(text, str):
            return {'score': 0.5, 'label': 'neutral'}
        
        try:
            # 使用SnowNLP进行情感分析
            sentiment_score = SnowNLP(text).sentiments
            
            # 结巴分词提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
            keywords_dict = dict(keywords)
            
            # 计算正面词和负面词的加权得分
            positive_score = sum(keywords_dict.get(word, 0) for word in self.positive_words if word in text)
            negative_score = sum(keywords_dict.get(word, 0) for word in self.negative_words if word in text)
            
            # 调整情感得分 (结合SnowNLP的得分和关键词分析的结果)
            adjusted_score = (sentiment_score + (positive_score - negative_score + 1) / 2) / 2
            adjusted_score = max(0, min(1, adjusted_score))  # 确保在0-1范围内
            
            # 根据得分确定情绪标签
            if adjusted_score >= 0.7:
                label = 'very_positive'
            elif adjusted_score >= 0.55:
                label = 'positive'
            elif adjusted_score >= 0.45:
                label = 'neutral'
            elif adjusted_score >= 0.3:
                label = 'negative'
            else:
                label = 'very_negative'
            
            return {
                'score': adjusted_score,
                'label': label,
                'snowlp_score': sentiment_score,
                'positive_keywords': positive_score,
                'negative_keywords': negative_score
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {'score': 0.5, 'label': 'neutral'}
    
    def analyze_news_sentiment(self, news_data):
        """分析新闻情绪
        
        Args:
            news_data: DataFrame，包含新闻数据
            
        Returns:
            DataFrame: 添加了情绪分析结果的新闻数据
        """
        if news_data is None or news_data.empty:
            self.logger.warning("Empty news data provided")
            return news_data
        
        df = news_data.copy()
        
        # 确保必要的列存在
        required_cols = ['title', 'content', 'publish_time']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Required column {col} not found in news data")
                df[col] = ''
        
        # 分析标题和内容的情绪
        df['title_sentiment'] = df['title'].apply(self.analyze_text_sentiment)
        df['content_sentiment'] = df['content'].apply(self.analyze_text_sentiment)
        
        # 提取情绪得分和标签
        df['title_sentiment_score'] = df['title_sentiment'].apply(lambda x: x['score'])
        df['title_sentiment_label'] = df['title_sentiment'].apply(lambda x: x['label'])
        df['content_sentiment_score'] = df['content_sentiment'].apply(lambda x: x['score'])
        df['content_sentiment_label'] = df['content_sentiment'].apply(lambda x: x['label'])
        
        # 计算综合情绪得分 (标题占60%，内容占40%)
        df['sentiment_score'] = df['title_sentiment_score'] * 0.6 + df['content_sentiment_score'] * 0.4
        
        # 综合情绪标签
        df['sentiment_label'] = pd.cut(
            df['sentiment_score'],
            bins=[0, 0.3, 0.45, 0.55, 0.7, 1],
            labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        )
        
        # 提取关键词 (可用于后续分析热点)
        df['keywords'] = df.apply(
            lambda row: jieba.analyse.extract_tags(f"{row['title']} {row['content']}", topK=10),
            axis=1
        )
        
        # 删除中间计算结果
        df = df.drop(['title_sentiment', 'content_sentiment'], axis=1)
        
        return df
    
    def analyze_stock_news_sentiment(self, stock_news_data, days=7):
        """分析特定股票的新闻情绪
        
        Args:
            stock_news_data: DataFrame，包含股票相关的新闻数据
            days: 分析的天数，默认为7天
            
        Returns:
            dict: 股票新闻情绪分析结果
        """
        if stock_news_data is None or stock_news_data.empty:
            self.logger.warning("Empty stock news data provided")
            return {}
        
        df = stock_news_data.copy()
        
        # 确保日期列存在
        if 'publish_time' not in df.columns:
            self.logger.warning("publish_time column not found in stock news data")
            return {}
            
        # 转换日期格式
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        
        # 过滤最近n天的新闻
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_news = df[df['publish_time'] >= cutoff_date]
        
        if recent_news.empty:
            self.logger.warning(f"No news found for the last {days} days")
            return {}
        
        # 分析情绪 (如果尚未分析)
        if 'sentiment_score' not in recent_news.columns:
            recent_news = self.analyze_news_sentiment(recent_news)
        
        # 计算各情绪类别的占比
        sentiment_counts = recent_news['sentiment_label'].value_counts(normalize=True).to_dict()
        
        # 计算情绪平均分
        avg_sentiment_score = recent_news['sentiment_score'].mean()
        
        # 提取高频关键词
        all_keywords = []
        for keywords in recent_news['keywords']:
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(10)
        
        # 计算情绪变化趋势
        recent_news = recent_news.sort_values('publish_time')
        sentiment_trend = recent_news.groupby(recent_news['publish_time'].dt.date)['sentiment_score'].mean()
        
        latest_sentiment = sentiment_trend.iloc[-1] if len(sentiment_trend) > 0 else None
        sentiment_change = None
        if len(sentiment_trend) > 1:
            sentiment_change = latest_sentiment - sentiment_trend.iloc[-2]
        
        # 情绪趋势描述
        trend_description = 'stable'
        if sentiment_change is not None:
            if sentiment_change > 0.1:
                trend_description = 'strong_improving'
            elif sentiment_change > 0.05:
                trend_description = 'improving'
            elif sentiment_change < -0.1:
                trend_description = 'strong_deteriorating'
            elif sentiment_change < -0.05:
                trend_description = 'deteriorating'
        
        # 返回汇总结果
        result = {
            'avg_sentiment_score': avg_sentiment_score,
            'sentiment_distribution': sentiment_counts,
            'top_keywords': top_keywords,
            'sentiment_trend': sentiment_trend.to_dict(),
            'latest_sentiment': latest_sentiment,
            'sentiment_change': sentiment_change,
            'trend_description': trend_description,
            'news_count': len(recent_news)
        }
        
        return result
    
    def analyze_industry_news_sentiment(self, industry_news_data, days=7):
        """分析行业新闻情绪
        
        Args:
            industry_news_data: DataFrame，包含行业相关的新闻数据
            days: 分析的天数，默认为7天
            
        Returns:
            dict: 行业新闻情绪分析结果
        """
        # 实现类似于股票新闻情绪分析的方法，但针对行业
        # 这里可以添加行业特定的分析逻辑
        return self.analyze_stock_news_sentiment(industry_news_data, days)
    
    def detect_sentiment_change(self, sentiment_data, threshold=0.1):
        """检测情绪变化
        
        Args:
            sentiment_data: Series，包含情绪得分数据
            threshold: 变化阈值，默认为0.1
            
        Returns:
            list: 情绪显著变化的日期及变化值
        """
        if sentiment_data is None or len(sentiment_data) < 2:
            return []
        
        # 计算日环比变化
        sentiment_change = sentiment_data.diff()
        
        # 找出显著变化的点
        significant_changes = []
        for date, change in sentiment_change.items():
            if abs(change) >= threshold:
                significant_changes.append({
                    'date': date,
                    'change': change,
                    'direction': 'up' if change > 0 else 'down'
                })
        
        return significant_changes
    
    def combine_technical_and_sentiment(self, technical_data, sentiment_data):
        """结合技术指标和情绪数据
        
        Args:
            technical_data: DataFrame，包含股票技术指标数据
            sentiment_data: DataFrame，包含股票情绪数据
            
        Returns:
            DataFrame: 结合了技术指标和情绪数据的结果
        """
        if technical_data is None or technical_data.empty:
            self.logger.warning("Empty technical data provided")
            return None
            
        if sentiment_data is None or sentiment_data.empty:
            self.logger.warning("Empty sentiment data provided")
            return technical_data
        
        # 确保两个数据集都有日期列
        tech_date_col = 'date' if 'date' in technical_data.columns else 'datetime'
        sent_date_col = 'date' if 'date' in sentiment_data.columns else 'publish_time'
        
        # 转换日期格式
        technical_data[tech_date_col] = pd.to_datetime(technical_data[tech_date_col])
        sentiment_data[sent_date_col] = pd.to_datetime(sentiment_data[sent_date_col])
        
        # 按日期聚合情绪数据
        daily_sentiment = sentiment_data.groupby(
            sentiment_data[sent_date_col].dt.date
        )['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'daily_sentiment_score']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        # 合并数据
        result = pd.merge(
            technical_data,
            daily_sentiment,
            left_on=pd.to_datetime(technical_data[tech_date_col]).dt.date,
            right_on=daily_sentiment['date'].dt.date,
            how='left'
        )
        
        # 填充缺失值 (前向填充情绪数据)
        result['daily_sentiment_score'] = result['daily_sentiment_score'].fillna(method='ffill')
        
        # 移除重复的日期列
        result = result.drop('key_0', axis=1) if 'key_0' in result.columns else result
        result = result.drop('date_y', axis=1) if 'date_y' in result.columns else result
        
        # 计算平均情绪得分 (5日移动平均)
        result['sentiment_ma5'] = result['daily_sentiment_score'].rolling(window=5).mean()
        
        # 计算情绪变化
        result['sentiment_change'] = result['daily_sentiment_score'].diff()
        
        # 计算强情绪信号
        result['strong_positive_sentiment'] = result['daily_sentiment_score'] > 0.7
        result['strong_negative_sentiment'] = result['daily_sentiment_score'] < 0.3
        
        # 计算技术与情绪一致性
        if 'close' in result.columns:
            # 价格上涨和情绪乐观的一致性
            result['price_up_sentiment_positive'] = (
                (result['close'] > result['close'].shift(1)) & 
                (result['daily_sentiment_score'] > 0.55)
            )
            
            # 价格下跌和情绪悲观的一致性
            result['price_down_sentiment_negative'] = (
                (result['close'] < result['close'].shift(1)) & 
                (result['daily_sentiment_score'] < 0.45)
            )
            
            # 情绪与价格背离
            result['sentiment_price_divergence'] = (
                ((result['close'] > result['close'].shift(1)) & (result['daily_sentiment_score'] < result['daily_sentiment_score'].shift(1))) |
                ((result['close'] < result['close'].shift(1)) & (result['daily_sentiment_score'] > result['daily_sentiment_score'].shift(1)))
            )
        
        return result
    
    def calculate_market_sentiment_index(self, market_data, news_data=None):
        """计算市场情绪指数
        
        Args:
            market_data: DataFrame，包含大盘指数数据
            news_data: DataFrame，包含市场新闻数据，默认为None
            
        Returns:
            DataFrame: 添加了市场情绪指数的数据
        """
        if market_data is None or market_data.empty:
            self.logger.warning("Empty market data provided")
            return None
        
        df = market_data.copy()
        
        # 技术指标贡献的情绪因子 (50%)
        
        # 1. 计算涨跌比例
        df['up_down_ratio'] = df['up_count'] / df['down_count'] if 'up_count' in df.columns and 'down_count' in df.columns else np.nan
        
        # 2. 计算市场动量 (20日收益率)
        df['market_momentum'] = df['close'].pct_change(20)
        
        # 3. 计算波动率 (20日收盘价标准差/均值)
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # 4. 计算成交量变化
        df['volume_change'] = df['volume'].pct_change()
        
        # 5. 计算MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 6. 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 新闻情绪贡献的因子 (50%)
        sentiment_score = 0.5  # 默认中性
        
        if news_data is not None and not news_data.empty:
            # 分析新闻情绪
            news_with_sentiment = self.analyze_news_sentiment(news_data)
            
            # 按日期聚合新闻情绪
            daily_news_sentiment = news_with_sentiment.groupby(
                pd.to_datetime(news_with_sentiment['publish_time']).dt.date
            )['sentiment_score'].mean()
            
            # 合并到市场数据
            df_dates = pd.to_datetime(df['date']).dt.date if 'date' in df.columns else pd.to_datetime(df.index).date
            for i, date in enumerate(df_dates):
                if date in daily_news_sentiment.index:
                    sentiment_score = daily_news_sentiment.loc[date]
                    df.loc[df.index[i], 'news_sentiment'] = sentiment_score
            
            # 填充缺失的新闻情绪数据
            df['news_sentiment'] = df['news_sentiment'].fillna(method='ffill').fillna(0.5)
        else:
            df['news_sentiment'] = 0.5
            
        # 组合技术因子计算技术情绪得分 (0-1范围)
        
        # 将涨跌比例标准化为0-1
        up_down_normalized = 1 / (1 + np.exp(-2 * (df['up_down_ratio'] - 1)))
        
        # 将市场动量标准化为0-1
        momentum_normalized = 1 / (1 + np.exp(-20 * df['market_momentum']))
        
        # 将波动率反向标准化为0-1 (波动率高表示恐慌)
        volatility_normalized = 1 - 1 / (1 + np.exp(-5 * (df['volatility'] - df['volatility'].mean()) / df['volatility'].std()))
        
        # 将成交量变化标准化为0-1
        volume_normalized = 1 / (1 + np.exp(-2 * df['volume_change']))
        
        # 将MACD标准化为0-1
        macd_normalized = 1 / (1 + np.exp(-5 * df['macd_hist'] / df['close']))
        
        # 将RSI标准化为0-1
        rsi_normalized = df['rsi'] / 100
        
        # 计算技术情绪得分 (加权平均)
        df['tech_sentiment'] = (
            up_down_normalized * 0.2 +
            momentum_normalized * 0.2 +
            volatility_normalized * 0.15 +
            volume_normalized * 0.15 +
            macd_normalized * 0.15 +
            rsi_normalized * 0.15
        )
        
        # 计算综合情绪指数 (技术和新闻各占50%)
        df['market_sentiment_index'] = df['tech_sentiment'] * 0.5 + df['news_sentiment'] * 0.5
        
        # 计算情绪区间
        df['sentiment_regime'] = pd.cut(
            df['market_sentiment_index'],
            bins=[0, 0.3, 0.45, 0.55, 0.7, 1],
            labels=['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        )
        
        # 计算情绪变化
        df['sentiment_change'] = df['market_sentiment_index'].diff()
        
        # 计算10日情绪均线
        df['sentiment_ma10'] = df['market_sentiment_index'].rolling(10).mean()
        
        # 清理中间计算列
        df = df.drop(['ema12', 'ema26'], axis=1)
        
        return df 