"""
报告模板模块
用于生成股票分析和回测报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
from pathlib import Path
import jinja2
from ..config.settings import RESULTS_DIR

class ReportTemplate:
    """报告模板基类"""
    
    def __init__(self, title: str = "股票分析报告", output_dir: Optional[Path] = None):
        """
        初始化报告模板
        
        Args:
            title: 报告标题
            output_dir: 输出目录
        """
        self.title = title
        self.creation_time = datetime.now()
        
        if output_dir is None:
            self.output_dir = RESULTS_DIR / "reports"
        else:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化Jinja2环境
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def _generate_filename(self, prefix: str, extension: str) -> str:
        """生成文件名"""
        timestamp = self.creation_time.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"
        
    def generate_report(self, data: Dict[str, Any], format: str = "html") -> str:
        """
        生成报告
        
        Args:
            data: 报告数据
            format: 报告格式，支持html, pdf, json
            
        Returns:
            str: 报告文件路径
        """
        if format == "html":
            return self._generate_html(data)
        elif format == "pdf":
            return self._generate_pdf(data)
        elif format == "json":
            return self._generate_json(data)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
            
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        raise NotImplementedError("子类必须实现此方法")
        
    def _generate_pdf(self, data: Dict[str, Any]) -> str:
        """生成PDF报告"""
        raise NotImplementedError("子类必须实现此方法")
        
    def _generate_json(self, data: Dict[str, Any]) -> str:
        """生成JSON报告"""
        filename = self._generate_filename("report", "json")
        filepath = self.output_dir / filename
        
        # 处理无法序列化的对象
        serializable_data = self._make_serializable(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
        return str(filepath)
        
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的形式"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, pd.DataFrame):
            return {
                "type": "DataFrame",
                "data": obj.to_dict(orient="records"),
                "index": list(obj.index),
                "columns": list(obj.columns)
            }
        elif isinstance(obj, pd.Series):
            return {
                "type": "Series",
                "data": obj.to_dict(),
                "name": obj.name
            }
        elif isinstance(obj, np.ndarray):
            return {
                "type": "ndarray",
                "data": obj.tolist()
            }
        elif isinstance(obj, datetime):
            return {
                "type": "datetime",
                "data": obj.isoformat()
            }
        else:
            return str(obj)
            
class StockAnalysisReport(ReportTemplate):
    """股票分析报告"""
    
    def __init__(self, title: str = "股票分析报告", output_dir: Optional[Path] = None):
        """初始化股票分析报告"""
        super().__init__(title, output_dir)
        
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        # 检查必要数据
        required_fields = ["stock_info", "price_data", "indicators", "recommendations"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少必要的数据字段: {field}")
                
        # 加载模板
        try:
            template = self.jinja_env.get_template("stock_analysis.html")
        except jinja2.exceptions.TemplateNotFound:
            # 如果模板不存在，使用内置模板
            template_str = self._get_default_stock_template()
            template = self.jinja_env.from_string(template_str)
            
        # 准备数据
        context = {
            "title": self.title,
            "creation_time": self.creation_time.strftime("%Y-%m-%d %H:%M:%S"),
            "stock_info": data["stock_info"],
            "price_data": self._prepare_price_data(data["price_data"]),
            "indicators": data["indicators"],
            "recommendations": data["recommendations"],
            "summary": data.get("summary", "")
        }
        
        # 生成报告
        html_content = template.render(**context)
        
        # 保存报告
        filename = self._generate_filename("stock_analysis", "html")
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(filepath)
        
    def _generate_pdf(self, data: Dict[str, Any]) -> str:
        """生成PDF报告"""
        # 首先生成HTML报告
        html_path = self._generate_html(data)
        
        # 然后转换为PDF
        from weasyprint import HTML
        
        pdf_filename = self._generate_filename("stock_analysis", "pdf")
        pdf_filepath = self.output_dir / pdf_filename
        
        HTML(html_path).write_pdf(pdf_filepath)
        
        return str(pdf_filepath)
        
    def _prepare_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """准备价格数据用于渲染"""
        # 生成价格图表
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # 绘制收盘价
        ax.plot(df.index, df['close'], color='black', linewidth=1.5)
        
        # 添加移动平均线
        if 'ma5' in df.columns:
            ax.plot(df.index, df['ma5'], color='red', linewidth=1, label='MA5')
        if 'ma10' in df.columns:
            ax.plot(df.index, df['ma10'], color='blue', linewidth=1, label='MA10')
        if 'ma20' in df.columns:
            ax.plot(df.index, df['ma20'], color='green', linewidth=1, label='MA20')
            
        ax.set_title("价格走势")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 格式化日期
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        # 保存图表
        chart_filename = self._generate_filename("price_chart", "png")
        chart_filepath = self.output_dir / chart_filename
        fig.savefig(chart_filepath)
        
        # 计算摘要统计信息
        stats = {
            "start_date": df.index[0].strftime("%Y-%m-%d") if isinstance(df.index[0], datetime) else str(df.index[0]),
            "end_date": df.index[-1].strftime("%Y-%m-%d") if isinstance(df.index[-1], datetime) else str(df.index[-1]),
            "days": len(df),
            "latest_price": df['close'].iloc[-1],
            "change": df['close'].iloc[-1] - df['close'].iloc[0],
            "change_pct": (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
            "max_price": df['close'].max(),
            "min_price": df['close'].min(),
            "volatility": df['close'].pct_change().std() * 100,
            "chart_path": chart_filename
        }
        
        # 返回摘要数据和最近的价格数据
        return {
            "stats": stats,
            "recent_data": df.tail(10).reset_index().to_dict(orient="records")
        }
        
    def _get_default_stock_template(self) -> str:
        """获取默认股票分析报告模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .summary { margin-bottom: 30px; }
                .chart { margin: 20px 0; text-align: center; }
                .chart img { max-width: 100%; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .footer { margin-top: 50px; text-align: center; color: #777; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>生成时间: {{ creation_time }}</p>
                </div>
                
                <div class="summary">
                    <h2>股票信息</h2>
                    <table>
                        <tr>
                            <th>代码</th>
                            <td>{{ stock_info.code }}</td>
                            <th>名称</th>
                            <td>{{ stock_info.name }}</td>
                        </tr>
                        <tr>
                            <th>行业</th>
                            <td>{{ stock_info.industry }}</td>
                            <th>市场</th>
                            <td>{{ stock_info.market }}</td>
                        </tr>
                    </table>
                    
                    <h2>价格分析</h2>
                    <table>
                        <tr>
                            <th>分析周期</th>
                            <td>{{ price_data.stats.start_date }} 至 {{ price_data.stats.end_date }} ({{ price_data.stats.days }}天)</td>
                            <th>最新价格</th>
                            <td>{{ "%.2f"|format(price_data.stats.latest_price) }}</td>
                        </tr>
                        <tr>
                            <th>价格变化</th>
                            <td>{{ "%.2f"|format(price_data.stats.change) }} ({{ "%.2f"|format(price_data.stats.change_pct) }}%)</td>
                            <th>波动率</th>
                            <td>{{ "%.2f"|format(price_data.stats.volatility) }}%</td>
                        </tr>
                        <tr>
                            <th>最高价</th>
                            <td>{{ "%.2f"|format(price_data.stats.max_price) }}</td>
                            <th>最低价</th>
                            <td>{{ "%.2f"|format(price_data.stats.min_price) }}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="chart">
                    <h2>价格走势图</h2>
                    <img src="{{ price_data.stats.chart_path }}" alt="价格走势图">
                </div>
                
                <div class="indicators">
                    <h2>技术指标</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                            <th>信号</th>
                        </tr>
                        {% for indicator in indicators %}
                        <tr>
                            <td>{{ indicator.name }}</td>
                            <td>{{ indicator.value }}</td>
                            <td>{{ indicator.signal }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="recommendations">
                    <h2>投资建议</h2>
                    <table>
                        <tr>
                            <th>建议</th>
                            <th>置信度</th>
                            <th>描述</th>
                        </tr>
                        {% for rec in recommendations %}
                        <tr>
                            <td>{{ rec.action }}</td>
                            <td>{{ "%.2f"|format(rec.confidence) }}</td>
                            <td>{{ rec.reason }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if summary %}
                <div class="summary">
                    <h2>总结</h2>
                    <p>{{ summary }}</p>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>本报告由股票推荐系统自动生成，仅供参考，不构成投资建议。</p>
                </div>
            </div>
        </body>
        </html>
        """
        
class BacktestReport(ReportTemplate):
    """回测报告"""
    
    def __init__(self, title: str = "策略回测报告", output_dir: Optional[Path] = None):
        """初始化回测报告"""
        super().__init__(title, output_dir)
        
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        # 检查必要数据
        required_fields = ["strategy_info", "performance", "trades"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少必要的数据字段: {field}")
                
        # 加载模板
        try:
            template = self.jinja_env.get_template("backtest_report.html")
        except jinja2.exceptions.TemplateNotFound:
            # 如果模板不存在，使用内置模板
            template_str = self._get_default_backtest_template()
            template = self.jinja_env.from_string(template_str)
            
        # 生成性能图表
        chart_filename = self._generate_performance_chart(data["performance"])
        
        # 准备数据
        context = {
            "title": self.title,
            "creation_time": self.creation_time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy_info": data["strategy_info"],
            "performance": data["performance"],
            "trades": self._prepare_trades_data(data["trades"]),
            "chart_path": chart_filename
        }
        
        # 生成报告
        html_content = template.render(**context)
        
        # 保存报告
        filename = self._generate_filename("backtest_report", "html")
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(filepath)
        
    def _generate_pdf(self, data: Dict[str, Any]) -> str:
        """生成PDF报告"""
        # 首先生成HTML报告
        html_path = self._generate_html(data)
        
        # 然后转换为PDF
        from weasyprint import HTML
        
        pdf_filename = self._generate_filename("backtest_report", "pdf")
        pdf_filepath = self.output_dir / pdf_filename
        
        HTML(html_path).write_pdf(pdf_filepath)
        
        return str(pdf_filepath)
        
    def _generate_performance_chart(self, performance: Dict[str, Any]) -> str:
        """生成性能图表"""
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # 提取数据
        equity_curve = performance.get("equity_curve")
        benchmark = performance.get("benchmark")
        drawdowns = performance.get("drawdowns")
        
        # 创建子图
        if drawdowns is not None:
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        else:
            ax1 = fig.add_subplot(1, 1, 1)
            ax2 = None
            
        # 绘制权益曲线
        if equity_curve is not None:
            if isinstance(equity_curve, pd.Series):
                ax1.plot(equity_curve.index, equity_curve, linewidth=2, label='策略')
            else:
                dates = range(len(equity_curve))
                ax1.plot(dates, equity_curve, linewidth=2, label='策略')
                
        # 绘制基准（如果有）
        if benchmark is not None:
            if isinstance(benchmark, pd.Series):
                ax1.plot(benchmark.index, benchmark, linewidth=2, label='基准', alpha=0.7)
            else:
                dates = range(len(benchmark))
                ax1.plot(dates, benchmark, linewidth=2, label='基准', alpha=0.7)
                
        ax1.set_title("策略表现")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylabel("权益")
        
        # 绘制回撤（如果有）
        if drawdowns is not None and ax2 is not None:
            if isinstance(drawdowns, pd.Series):
                ax2.fill_between(drawdowns.index, 0, -drawdowns, color='red', alpha=0.3)
            else:
                dates = range(len(drawdowns))
                ax2.fill_between(dates, 0, -np.array(drawdowns), color='red', alpha=0.3)
                
            ax2.set_title("回撤")
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_ylabel("回撤率")
            ax2.set_xlabel("日期")
            
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        chart_filename = self._generate_filename("performance_chart", "png")
        chart_filepath = self.output_dir / chart_filename
        fig.savefig(chart_filepath)
        
        return chart_filename
        
    def _prepare_trades_data(self, trades_data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """准备交易数据用于渲染"""
        if isinstance(trades_data, pd.DataFrame):
            # 转换为字典列表
            trades = trades_data.to_dict(orient="records")
        else:
            trades = trades_data
            
        # 处理日期格式
        for trade in trades:
            if "entry_date" in trade and isinstance(trade["entry_date"], datetime):
                trade["entry_date"] = trade["entry_date"].strftime("%Y-%m-%d")
            if "exit_date" in trade and isinstance(trade["exit_date"], datetime):
                trade["exit_date"] = trade["exit_date"].strftime("%Y-%m-%d")
                
        return trades
        
    def _get_default_backtest_template(self) -> str:
        """获取默认回测报告模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .summary { margin-bottom: 30px; }
                .chart { margin: 20px 0; text-align: center; }
                .chart img { max-width: 100%; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .footer { margin-top: 50px; text-align: center; color: #777; font-size: 0.8em; }
                .metrics { display: flex; flex-wrap: wrap; }
                .metric { width: 25%; padding: 10px; box-sizing: border-box; }
                .metric-value { font-size: 1.5em; font-weight: bold; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>生成时间: {{ creation_time }}</p>
                </div>
                
                <div class="summary">
                    <h2>策略信息</h2>
                    <table>
                        <tr>
                            <th>策略名称</th>
                            <td>{{ strategy_info.name }}</td>
                            <th>回测周期</th>
                            <td>{{ strategy_info.period }}</td>
                        </tr>
                        <tr>
                            <th>初始资金</th>
                            <td>{{ strategy_info.initial_capital }}</td>
                            <th>资金曲线</th>
                            <td>{{ strategy_info.final_capital }}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="chart">
                    <h2>策略表现</h2>
                    <img src="{{ chart_path }}" alt="策略表现">
                </div>
                
                <div class="metrics">
                    <h2>性能指标</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div>总收益率</div>
                            <div class="metric-value {% if performance.total_return > 0 %}positive{% else %}negative{% endif %}">
                                {{ "%.2f"|format(performance.total_return * 100) }}%
                            </div>
                        </div>
                        <div class="metric">
                            <div>年化收益率</div>
                            <div class="metric-value {% if performance.annual_return > 0 %}positive{% else %}negative{% endif %}">
                                {{ "%.2f"|format(performance.annual_return * 100) }}%
                            </div>
                        </div>
                        <div class="metric">
                            <div>夏普比率</div>
                            <div class="metric-value {% if performance.sharpe_ratio > 1 %}positive{% else %}negative{% endif %}">
                                {{ "%.2f"|format(performance.sharpe_ratio) }}
                            </div>
                        </div>
                        <div class="metric">
                            <div>最大回撤</div>
                            <div class="metric-value negative">
                                {{ "%.2f"|format(performance.max_drawdown * 100) }}%
                            </div>
                        </div>
                        <div class="metric">
                            <div>胜率</div>
                            <div class="metric-value">
                                {{ "%.2f"|format(performance.win_rate * 100) }}%
                            </div>
                        </div>
                        <div class="metric">
                            <div>盈亏比</div>
                            <div class="metric-value">
                                {{ "%.2f"|format(performance.profit_factor) }}
                            </div>
                        </div>
                        <div class="metric">
                            <div>交易次数</div>
                            <div class="metric-value">
                                {{ performance.trade_count }}
                            </div>
                        </div>
                        <div class="metric">
                            <div>Beta</div>
                            <div class="metric-value">
                                {{ "%.2f"|format(performance.beta) }}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="trades">
                    <h2>交易记录</h2>
                    <table>
                        <tr>
                            <th>股票</th>
                            <th>方向</th>
                            <th>开仓日期</th>
                            <th>开仓价格</th>
                            <th>平仓日期</th>
                            <th>平仓价格</th>
                            <th>收益率</th>
                        </tr>
                        {% for trade in trades %}
                        <tr>
                            <td>{{ trade.symbol }}</td>
                            <td>{{ trade.direction }}</td>
                            <td>{{ trade.entry_date }}</td>
                            <td>{{ "%.2f"|format(trade.entry_price) }}</td>
                            <td>{{ trade.exit_date }}</td>
                            <td>{{ "%.2f"|format(trade.exit_price) }}</td>
                            <td class="{% if trade.return > 0 %}positive{% else %}negative{% endif %}">
                                {{ "%.2f"|format(trade.return * 100) }}%
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="footer">
                    <p>本报告由股票推荐系统回测引擎自动生成，仅供参考，不构成投资建议。</p>
                </div>
            </div>
        </body>
        </html>
        """ 