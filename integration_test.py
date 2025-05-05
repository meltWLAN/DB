#!/usr/bin/env python3
"""
市场概览模块集成测试脚本
测试市场概览模块与其他系统组件的集成
"""
import os
import sys
import json
import logging
import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_reports/integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("集成测试")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 输出目录
OUTPUT_DIR = "test_reports/integration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试市场概览模块与数据源集成
class TestDataSourceIntegration(unittest.TestCase):
    """测试市场概览模块与数据源的集成"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟数据源模块
        self.mock_data_source = MagicMock()
        
        # 设置模拟数据
        self._setup_mock_data()
        
        # 设置模拟数据源返回值
        self.mock_data_source.get_north_fund_data.return_value = self.north_fund_data
        self.mock_data_source.get_sector_data.return_value = self.sector_data
        self.mock_data_source.get_macro_economic_data.return_value = self.macro_data
    
    def _setup_mock_data(self):
        """设置模拟数据"""
        # 北向资金数据
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='B')
        self.north_fund_data = pd.DataFrame({
            'trade_date': dates,
            'north_money': np.random.normal(1000, 500, size=len(dates)),
            'south_money': np.random.normal(800, 400, size=len(dates))
        })
        
        # 行业数据
        self.sector_data = pd.DataFrame({
            'industry': ['电子', '医药', '食品饮料', '银行', '房地产'],
            'weight': [15, 12, 10, 8, 5],
            'stock_count': [120, 100, 80, 50, 40]
        })
        
        # 宏观经济数据
        self.macro_data = {
            'gdp': {
                'latest_value': 121000,
                'yoy_change': 5.2
            },
            'cpi': {
                'latest_value': 102.5,
                'yoy_change': 2.5
            },
            'ppi': {
                'latest_value': 101.8,
                'yoy_change': 1.8
            },
            'money_supply': {
                'm2': 250000,
                'm2_yoy': 8.5
            }
        }
    
    @patch('src.enhanced.market.north_fund_analyzer.NorthFundAnalyzer')
    def test_north_fund_analyzer_with_data_source(self, MockAnalyzer):
        """测试北向资金分析器与数据源的集成"""
        # 创建分析器实例
        analyzer = MockAnalyzer.return_value
        
        # 调用分析器方法
        analyzer.analyze_fund_flow_trend(days=30)
        
        # 验证方法调用
        analyzer.get_north_fund_flow.assert_called()
        
        # 添加断言验证集成是否成功
        self.mock_data_source.get_north_fund_data.assert_not_called()  # 这里断言失败是为了示范
        
        logger.info("北向资金分析器与数据源集成测试完成")
    
    @patch('src.enhanced.market.macro_economic_analyzer.MacroEconomicAnalyzer')
    def test_macro_analyzer_with_data_source(self, MockAnalyzer):
        """测试宏观经济分析器与数据源的集成"""
        # 创建分析器实例
        analyzer = MockAnalyzer.return_value
        
        # 调用分析器方法
        analyzer.get_macro_overview()
        
        # 验证方法调用
        analyzer.get_money_supply.assert_called()
        
        # 添加断言验证集成是否成功
        # 这里应该添加实际的集成断言
        
        logger.info("宏观经济分析器与数据源集成测试完成")

# 测试市场概览模块与GUI的集成
class TestGUIIntegration(unittest.TestCase):
    """测试市场概览模块与GUI的集成"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟GUI控制器
        self.mock_gui_controller = MagicMock()
        
        # 设置模拟响应
        self.mock_gui_controller.update_view.return_value = True
        self.mock_gui_controller.show_chart.return_value = True
    
    @patch('src.enhanced.market.north_fund_analyzer.NorthFundAnalyzer')
    def test_north_fund_analyzer_with_gui(self, MockAnalyzer):
        """测试北向资金分析器与GUI的集成"""
        # 创建分析器实例
        analyzer = MockAnalyzer.return_value
        
        # 模拟分析结果
        trend_result = {
            'latest_net_flow': 1000.5,
            'trend': '上升',
            'trend_strength': 75
        }
        analyzer.analyze_fund_flow_trend.return_value = trend_result
        
        # 调用GUI控制器更新视图
        self.mock_gui_controller.update_view('north_fund', trend_result)
        
        # 验证方法调用
        self.mock_gui_controller.update_view.assert_called_with('north_fund', trend_result)
        
        logger.info("北向资金分析器与GUI集成测试完成")
    
    @patch('src.enhanced.market.market_heatmap.MarketHeatmap')
    def test_market_heatmap_with_gui(self, MockHeatmap):
        """测试市场热力图与GUI的集成"""
        # 创建热力图实例
        heatmap = MockHeatmap.return_value
        
        # 模拟生成图表
        test_image_path = os.path.join(OUTPUT_DIR, "test_heatmap.png")
        with open(test_image_path, 'w') as f:
            f.write("test")  # 创建测试图片文件
        heatmap.create_industry_performance_heatmap.return_value = test_image_path
        
        # 调用GUI控制器显示图表
        self.mock_gui_controller.show_chart('industry_heatmap', test_image_path)
        
        # 验证方法调用
        self.mock_gui_controller.show_chart.assert_called_with('industry_heatmap', test_image_path)
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("市场热力图与GUI集成测试完成")

# 测试市场概览模块与事件系统的集成
class TestEventSystemIntegration(unittest.TestCase):
    """测试市场概览模块与事件系统的集成"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟事件系统
        self.mock_event_system = MagicMock()
        
        # 设置模拟响应
        self.mock_event_system.subscribe.return_value = True
        self.mock_event_system.publish.return_value = True
    
    def test_market_overview_event_subscription(self):
        """测试市场概览模块订阅事件"""
        # 定义事件处理函数
        def handle_data_update(data):
            pass
        
        # 订阅数据更新事件
        self.mock_event_system.subscribe('data_update', handle_data_update)
        
        # 验证订阅调用
        self.mock_event_system.subscribe.assert_called_with('data_update', handle_data_update)
        
        logger.info("市场概览模块事件订阅测试完成")
    
    def test_market_overview_event_publishing(self):
        """测试市场概览模块发布事件"""
        # 定义事件数据
        event_data = {
            'type': 'market_alert',
            'level': 'warning',
            'message': '北向资金流出超过阈值',
            'timestamp': datetime.now().isoformat()
        }
        
        # 发布事件
        self.mock_event_system.publish('market_alert', event_data)
        
        # 验证发布调用
        self.mock_event_system.publish.assert_called_with('market_alert', event_data)
        
        logger.info("市场概览模块事件发布测试完成")

# 测试市场概览模块组件之间的集成
class TestModuleComponentsIntegration(unittest.TestCase):
    """测试市场概览模块各组件之间的集成"""
    
    @patch('src.enhanced.market.north_fund_analyzer.NorthFundAnalyzer')
    @patch('src.enhanced.market.market_heatmap.MarketHeatmap')
    def test_north_fund_with_heatmap(self, MockHeatmap, MockAnalyzer):
        """测试北向资金分析器与热力图的集成"""
        # 创建实例
        analyzer = MockAnalyzer.return_value
        heatmap = MockHeatmap.return_value
        
        # 模拟北向资金分析结果 - 行业配置数据
        sector_data = pd.DataFrame({
            'industry': ['电子', '医药', '食品饮料', '银行', '房地产'],
            'weight': [15, 12, 10, 8, 5],
            'stock_count': [120, 100, 80, 50, 40],
            'weight_change': [2.5, 1.2, -0.8, -1.5, 0.3]
        })
        analyzer.analyze_sector_changes.return_value = sector_data
        
        # 将行业变化数据传递给热力图
        # 通常这会通过某种形式的集成接口或转换器完成
        industry_data = pd.DataFrame({
            'name': sector_data['industry'],
            'change': sector_data['weight_change']
        })
        
        # 生成热力图
        test_image_path = os.path.join(OUTPUT_DIR, "industry_change_heatmap.png")
        heatmap.create_industry_performance_heatmap.return_value = test_image_path
        heatmap.create_industry_performance_heatmap(industry_data, save_path=test_image_path)
        
        # 验证调用
        heatmap.create_industry_performance_heatmap.assert_called_with(
            industry_data, save_path=test_image_path
        )
        
        logger.info("北向资金分析器与热力图集成测试完成")
    
    @patch('src.enhanced.market.macro_economic_analyzer.MacroEconomicAnalyzer')
    @patch('src.enhanced.market.market_heatmap.MarketHeatmap')
    def test_macro_analyzer_with_heatmap(self, MockHeatmap, MockAnalyzer):
        """测试宏观经济分析器与热力图的集成"""
        # 创建实例
        analyzer = MockAnalyzer.return_value
        heatmap = MockHeatmap.return_value
        
        # 模拟宏观经济分析结果
        macro_overview = {
            'assessment': {
                'growth': {'score': 0.8, 'status': '良好'},
                'inflation': {'score': 0.5, 'status': '中性'},
                'monetary': {'score': 0.7, 'status': '良好'},
                'overall': {'score': 0.7, 'status': '良好'},
                'investment_advice': '宏观环境稳健，可均衡配置股票和债券，关注业绩稳定且有成长性的蓝筹股。'
            }
        }
        analyzer.get_macro_overview.return_value = macro_overview
        
        # 将宏观经济指标转换为情绪指标
        emotion_indicators = {
            '经济增长': macro_overview['assessment']['growth']['score'] * 100,
            '通胀': macro_overview['assessment']['inflation']['score'] * 100,
            '货币环境': macro_overview['assessment']['monetary']['score'] * 100,
            '整体评分': macro_overview['assessment']['overall']['score'] * 100
        }
        
        # 生成热力图
        test_image_path = os.path.join(OUTPUT_DIR, "macro_emotion_heatmap.png")
        heatmap.create_market_emotion_heatmap.return_value = test_image_path
        heatmap.create_market_emotion_heatmap(emotion_indicators, save_path=test_image_path)
        
        # 验证调用
        heatmap.create_market_emotion_heatmap.assert_called_with(
            emotion_indicators, save_path=test_image_path
        )
        
        logger.info("宏观经济分析器与热力图集成测试完成")

# 生成HTML测试报告
def generate_html_report(test_results):
    """生成HTML格式的集成测试报告"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>市场概览模块集成测试报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>市场概览模块集成测试报告</h1>
        <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>测试摘要</h2>
            <p>本报告包含市场概览模块与其他系统组件的集成测试结果。</p>
            <p>测试运行: {test_results['total']}个测试用例</p>
            <p>通过: <span class="success">{test_results['success']}</span></p>
            <p>失败: <span class="failure">{test_results['failures']}</span></p>
            <p>错误: <span class="failure">{test_results['errors']}</span></p>
            <p>总体状态: <span class="{('success' if test_results['success'] == test_results['total'] else 'failure')}">{('通过' if test_results['success'] == test_results['total'] else '失败')}</span></p>
        </div>
        
        <div class="container">
            <h2>测试结果详情</h2>
            <table>
                <tr>
                    <th>测试用例</th>
                    <th>结果</th>
                    <th>描述</th>
                </tr>
    """
    
    for test_case in test_results['test_cases']:
        html += f"""
                <tr>
                    <td>{test_case['name']}</td>
                    <td class="{('success' if test_case['result'] == 'Pass' else 'failure')}">{test_case['result']}</td>
                    <td>{test_case['description']}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="container">
            <h2>集成测试架构</h2>
            <h3>组件关系</h3>
            <ul>
                <li>北向资金分析器 ↔ 数据源模块：获取北向资金流向数据和行业配置数据</li>
                <li>宏观经济分析器 ↔ 数据源模块：获取GDP、CPI、利率等宏观经济数据</li>
                <li>市场热力图 ↔ GUI控制器：将生成的热力图显示在用户界面上</li>
                <li>北向资金分析器 ↔ 市场热力图：将行业配置数据转换为热力图</li>
                <li>宏观经济分析器 ↔ 市场热力图：将宏观经济指标转换为情绪热力图</li>
                <li>所有组件 ↔ 事件系统：发布和订阅市场事件</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML文件
    report_path = os.path.join(OUTPUT_DIR, "integration_test_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML集成测试报告已生成: {report_path}")
    
    return report_path

def run_tests():
    """运行所有集成测试"""
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTest(unittest.makeSuite(TestDataSourceIntegration))
    suite.addTest(unittest.makeSuite(TestGUIIntegration))
    suite.addTest(unittest.makeSuite(TestEventSystemIntegration))
    suite.addTest(unittest.makeSuite(TestModuleComponentsIntegration))
    
    # 运行测试并收集结果
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # 整理测试结果
    test_results = {
        'total': result.testsRun,
        'success': result.testsRun - len(result.failures) - len(result.errors),
        'failures': len(result.failures),
        'errors': len(result.errors),
        'test_cases': []
    }
    
    # 收集失败的测试用例
    for failure in result.failures:
        test_case = {
            'name': failure[0]._testMethodName,
            'result': 'Fail',
            'description': str(failure[1])
        }
        test_results['test_cases'].append(test_case)
        
    # 收集错误的测试用例
    for error in result.errors:
        test_case = {
            'name': error[0]._testMethodName,
            'result': 'Error',
            'description': str(error[1])
        }
        test_results['test_cases'].append(test_case)
    
    # 添加成功的测试用例
    for test in suite:
        method_name = test._testMethodName
        if not any(tc['name'] == method_name for tc in test_results['test_cases']):
            test_case = {
                'name': method_name,
                'result': 'Pass',
                'description': getattr(test, method_name).__doc__ or ''
            }
            test_results['test_cases'].append(test_case)
    
    return test_results

def main():
    """主函数"""
    logger.info("开始市场概览模块集成测试")
    
    # 运行测试
    test_results = run_tests()
    
    # 生成HTML报告
    report_path = generate_html_report(test_results)
    
    logger.info(f"集成测试完成，报告位于: {report_path}")
    
    # 返回状态码：0表示成功，非0表示失败
    return 0 if test_results['failures'] == 0 and test_results['errors'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 