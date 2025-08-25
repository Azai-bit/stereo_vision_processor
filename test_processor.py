#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试立体视觉处理器的脚本
"""

import sys
from pathlib import Path
from stereo_vision_processor import StereoVisionProcessor

def test_processor():
    """测试处理器基本功能"""
    
    # 检查数据目录是否存在
    data_dir = "/home/xtf/Downloads/MODD2_video_data/video_data/kope67-00-00004500-00005050"
    if not Path(data_dir).exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保在正确的目录中运行此脚本")
        return False
    
    try:
        # 创建处理器
        print("正在初始化处理器...")
        processor = StereoVisionProcessor(data_dir)
        
        # 测试获取图像对
        print("正在获取图像对...")
        image_pairs = processor.get_image_pairs()
        
        if not image_pairs:
            print("错误: 未找到图像对")
            return False
        
        print(f"找到 {len(image_pairs)} 对图像")
        
        # 测试加载第一对图像
        if image_pairs:
            print("正在测试图像加载...")
            left_path, right_path = image_pairs[0]
            left_img, right_img = processor.load_image_pair(left_path, right_path)
            
            print(f"左图像尺寸: {left_img.shape}")
            print(f"右图像尺寸: {right_img.shape}")
            
            # 测试视差计算
            print("正在测试视差计算...")
            disparity = processor.compute_disparity(left_img, right_img)
            print(f"视差图尺寸: {disparity.shape}")
            
            print("所有测试通过！")
            return True
            
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main():
    """主函数"""
    print("立体视觉处理器测试")
    print("=" * 50)
    
    success = test_processor()
    
    if success:
        print("\n测试成功！处理器可以正常使用。")
        print("\n使用示例:")
        opened_path = input("请输入要打开的文件夹路径:")#kope71-01-00011520-00011800/frames/00011520L.jpg
        data_dir = opened_path.split("/")[0]
        start_frame = opened_path.split("/")[2].split("L")[0]
        print(f"python3 stereo_vision_processor.py {data_dir} --start-frame {start_frame} --max-frames 1 --save")
    else:
        print("\n测试失败！请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
