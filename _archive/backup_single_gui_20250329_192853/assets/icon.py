#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成系统图标文件
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

def create_icon(size=64, output_file="icon.png"):
    """创建一个简单的图标"""
    # 创建画布
    img = Image.new('RGBA', (size, size), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制背景圆
    circle_color = (65, 105, 225)  # 蓝色
    draw.ellipse([(0, 0), (size, size)], fill=circle_color)
    
    # 绘制图表线条
    line_color = (255, 255, 255)  # 白色
    
    # 绘制上升线
    points = [
        (size * 0.2, size * 0.7),
        (size * 0.4, size * 0.5),
        (size * 0.6, size * 0.6),
        (size * 0.8, size * 0.3)
    ]
    draw.line(points, fill=line_color, width=3)
    
    # 保存图标
    img.save(output_file)
    print(f"图标已保存至: {output_file}")
    return output_file

if __name__ == "__main__":
    # 确保assets目录存在
    os.makedirs("assets", exist_ok=True)
    
    # 生成图标
    icon_path = os.path.join("assets", "icon.png")
    create_icon(output_file=icon_path) 