#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据修复模块
包含各种数据源修复
"""

from .market_overview_fix import apply_market_overview_fix

# 自动应用市场概览修复
apply_market_overview_fix()
