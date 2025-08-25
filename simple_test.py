#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的立体视觉处理器测试脚本
跳过YAML文件加载，直接测试图像处理功能
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def test_basic_functionality():
    """测试基本功能"""
    
    # 检查数据目录是否存在
    data_dir = Path("/home/xtf/Downloads/MODD2_video_data/video_data/kope67-00-00004500-00005050")
    if not data_dir.exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        return False
    
    frames_dir = data_dir / "frames"
    if not frames_dir.exists():
        print(f"错误: frames目录不存在: {frames_dir}")
        return False
    
    try:
        # 获取图像文件列表
        left_images = sorted(list(frames_dir.glob("*L.jpg")))
        right_images = sorted(list(frames_dir.glob("*R.jpg")))
        
        if not left_images or not right_images:
            print("错误: 未找到图像文件")
            return False
        
        print(f"找到 {len(left_images)} 个左图像")
        print(f"找到 {len(right_images)} 个右图像")
        
        # 测试加载第一对图像
        if left_images and right_images:
            left_path = left_images[0]
            right_path = right_images[0]
            
            print(f"测试加载图像对: {left_path.name} 和 {right_path.name}")
            
            # 加载图像
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            
            if left_img is None or right_img is None:
                print("错误: 无法加载图像")
                return False
            
            print(f"左图像尺寸: {left_img.shape}")
            print(f"右图像尺寸: {right_img.shape}")
            
            # 测试转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            print(f"灰度图尺寸: {left_gray.shape}")
            
            # 测试立体匹配（使用默认参数）
            print("测试立体匹配...")
            stereo = cv2.StereoBM_create()
            disparity = stereo.compute(left_gray, right_gray)
            
            print(f"视差图尺寸: {disparity.shape}")
            print(f"视差图数据类型: {disparity.dtype}")
            print(f"视差图值范围: {disparity.min()} 到 {disparity.max()}")
            
            print("所有基本功能测试通过！")
            return True
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("立体视觉处理器基本功能测试")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n测试成功！基本功能正常。")
        print("\n现在可以运行完整的处理器:")
        print("python stereo_vision_processor.py /home/xtf/Downloads/MODD2_video_data/video_data/kope67-00-00004500-00005050 --max-frames 3")
    else:
        print("\n测试失败！请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
