#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
立体视觉图像处理和可视化工具
用于处理MODD2视频数据集中的立体图像对
"""

import cv2
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, List, Optional
import glob

class StereoVisionProcessor:
    """立体视觉图像处理器"""
    
    def __init__(self, data_dir: str):
        """
        初始化处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.calibration_data = None
        self.stereo_params = None
        self.load_calibration()
        self.load_stereo_params()
        
    def load_calibration(self):
        """加载相机标定参数"""
        calib_file = self.data_dir / "calibration.yaml"
        if calib_file.exists():
            try:
                with open(calib_file, 'r') as f:
                    self.calibration_data = yaml.safe_load(f)
                print(f"已加载标定参数: {calib_file}")
            except yaml.YAMLError as e:
                print(f"警告: 标定文件解析失败，使用默认参数: {e}")
                self.calibration_data = None
        else:
            print(f"警告: 未找到标定文件 {calib_file}")
            self.calibration_data = None
            
    def load_stereo_params(self):
        """加载立体匹配参数"""
        stereo_file = self.data_dir / "stereo-method.yaml"
        if stereo_file.exists():
            try:
                with open(stereo_file, 'r') as f:
                    self.stereo_params = yaml.safe_load(f)
                print(f"已加载立体匹配参数: {stereo_file}")
            except yaml.YAMLError as e:
                print(f"警告: 立体匹配参数文件解析失败，使用默认参数: {e}")
                self.stereo_params = None
        else:
            print(f"警告: 未找到立体匹配参数文件 {stereo_file}")
            self.stereo_params = None
    
    def get_image_pairs(self, start_frame: int = None, end_frame: int = None) -> List[Tuple[str, str]]:
        """
        获取图像对列表
        
        Args:
            start_frame: 起始帧号
            end_frame: 结束帧号
            
        Returns:
            图像对路径列表
        """
        frames_dir = self.data_dir / "frames"
        if not frames_dir.exists():
            print(f"错误: 未找到frames目录 {frames_dir}")
            return []
            
        # 获取所有左图像
        left_images = sorted(glob.glob(str(frames_dir / "*L.jpg")))
        right_images = sorted(glob.glob(str(frames_dir / "*R.jpg")))
        
        if len(left_images) != len(right_images):
            print(f"警告: 左右图像数量不匹配 L:{len(left_images)} R:{len(right_images)}")
            
        # 过滤帧范围
        if start_frame is not None or end_frame is not None:
            filtered_pairs = []
            for left_img, right_img in zip(left_images, right_images):
                frame_num = int(Path(left_img).stem.replace('L', ''))
                if start_frame is not None and frame_num < start_frame:
                    continue
                if end_frame is not None and frame_num > end_frame:
                    continue
                filtered_pairs.append((left_img, right_img))
            return filtered_pairs
            
        return list(zip(left_images, right_images))
    
    def load_image_pair(self, left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载图像对
        
        Args:
            left_path: 左图像路径
            right_path: 右图像路径
            
        Returns:
            左右图像数组
        """
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            raise ValueError(f"无法加载图像: {left_path} 或 {right_path}")
            
        return left_img, right_img
    
    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        计算视差图
        
        Args:
            left_img: 左图像
            right_img: 右图像
            
        Returns:
            视差图
        """
        if self.stereo_params is None:
            print("警告: 使用默认立体匹配参数")
            # 使用默认参数
            stereo = cv2.StereoBM_create()
        else:
            # 使用配置文件中的参数
            stereo = cv2.StereoBM_create(
                numDisparities=self.stereo_params['NumDisparities'],
                blockSize=self.stereo_params['SADWindowSize']
            )
            stereo.setPreFilterType(self.stereo_params['PreFilterType'])
            stereo.setPreFilterSize(self.stereo_params['PreFilterSize'])
            stereo.setPreFilterCap(self.stereo_params['PreFilterCap'])
            stereo.setMinDisparity(self.stereo_params['MinDisparity'])
            stereo.setTextureThreshold(self.stereo_params['TextureThreshold'])
            stereo.setUniquenessRatio(self.stereo_params['UniquenessRatio'])
            stereo.setSpeckleWindowSize(self.stereo_params['SpeckleWindowSize'])
            stereo.setSpeckleRange(self.stereo_params['SpeckleRange'])
            stereo.setDisp12MaxDiff(self.stereo_params['Disp12MaxDiff'])
        
        # 转换为灰度图
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
            
        # 计算视差
        disparity = stereo.compute(left_gray, right_gray)
        return disparity
    
    def visualize_stereo_pair(self, left_img: np.ndarray, right_img: np.ndarray, 
                            disparity: np.ndarray = None, title: str = "立体图像对"):
        """
        可视化立体图像对
        
        Args:
            left_img: 左图像
            right_img: 右图像
            disparity: 视差图
            title: 图像标题
        """
        if disparity is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 左图像
            axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Left")
            axes[0].axis('off')
            
            # 右图像
            axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title("right")
            axes[1].axis('off')
            
            # 视差图
            axes[2].imshow(disparity, cmap='plasma')
            axes[2].set_title("Parallax")
            axes[2].axis('off')
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左图像
            axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Left")
            axes[0].axis('off')
            
            # 右图像
            axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title("right")
            axes[1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def process_sequence(self, start_frame: int = None, end_frame: int = None, 
                        max_frames: int = 10, show_disparity: bool = True):
        """
        处理图像序列
        
        Args:
            start_frame: 起始帧号
            end_frame: 结束帧号
            max_frames: 最大处理帧数
            show_disparity: 是否显示视差图
        """
        image_pairs = self.get_image_pairs(start_frame, end_frame)
        
        if not image_pairs:
            print("未找到图像对")
            return
            
        print(f"找到 {len(image_pairs)} 对图像")
        
        # 限制处理帧数
        if len(image_pairs) > max_frames:
            image_pairs = image_pairs[:max_frames]
            print(f"限制处理帧数为 {max_frames}")
        
        for i, (left_path, right_path) in enumerate(image_pairs):
            print(f"处理第 {i+1}/{len(image_pairs)} 对图像...")
            
            try:
                left_img, right_img = self.load_image_pair(left_path, right_path)
                
                if show_disparity:
                    disparity = self.compute_disparity(left_img, right_img)
                    self.visualize_stereo_pair(left_img, right_img, disparity, 
                                             f"帧 {i+1}: {Path(left_path).stem}")
                else:
                    self.visualize_stereo_pair(left_img, right_img, 
                                             title=f"帧 {i+1}: {Path(left_path).stem}")
                
                # 等待用户按键继续
                input("按回车键继续下一帧...")
                
            except Exception as e:
                print(f"处理图像对时出错: {e}")
                continue
    
    def save_processed_results(self, output_dir: str, start_frame: int = None, 
                              end_frame: int = None, max_frames: int = 10):
        """
        保存处理结果
        
        Args:
            output_dir: 输出目录
            start_frame: 起始帧号
            end_frame: 结束帧号
            max_frames: 最大处理帧数
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_pairs = self.get_image_pairs(start_frame, end_frame)
        
        if not image_pairs:
            print("未找到图像对")
            return
            
        if len(image_pairs) > max_frames:
            image_pairs = image_pairs[:max_frames]
        
        for i, (left_path, right_path) in enumerate(image_pairs):
            print(f"保存第 {i+1}/{len(image_pairs)} 对图像结果...")
            
            try:
                left_img, right_img = self.load_image_pair(left_path, right_path)
                disparity = self.compute_disparity(left_img, right_img)
                
                # 保存视差图
                frame_name = Path(left_path).stem.replace('L', '')
                disparity_path = output_path / f"disparity_{frame_name}.png"
                cv2.imwrite(str(disparity_path), disparity)
                
                # 保存左右图像
                left_output = output_path / f"left_{frame_name}.jpg"
                right_output = output_path / f"right_{frame_name}.jpg"
                cv2.imwrite(str(left_output), left_img)
                cv2.imwrite(str(right_output), right_img)
                
            except Exception as e:
                print(f"保存结果时出错: {e}")
                continue
        
        print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="立体视觉图像处理工具")
    parser.add_argument("data_dir", help="数据目录路径")
    parser.add_argument("--start-frame", type=int, help="起始帧号")
    parser.add_argument("--end-frame", type=int, help="结束帧号")
    parser.add_argument("--max-frames", type=int, default=5, help="最大处理帧数")
    parser.add_argument("--no-disparity", action="store_true", help="不显示视差图")
    parser.add_argument("--save", action="store_true", help="保存处理结果")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = StereoVisionProcessor(args.data_dir)
    
    if args.save:
        # 保存结果
        processor.save_processed_results(
            args.output_dir, 
            args.start_frame, 
            args.end_frame, 
            args.max_frames
        )
    else:
        # 交互式处理
        processor.process_sequence(
            args.start_frame, 
            args.end_frame, 
            args.max_frames, 
            not args.no_disparity
        )


if __name__ == "__main__":
    main()
