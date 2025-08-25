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
        self.rectify_maps_left = None  # (map1, map2)
        self.rectify_maps_right = None  # (map1, map2)
        self.rectify_Q = None
        self.load_calibration()
        self.load_stereo_params()
        # 延迟到首次用到图像尺寸时再生成重映射表
        
    def load_calibration(self):
        """加载相机标定参数（支持OpenCV YAML via FileStorage）"""
        calib_file = self.data_dir/"calibration.yaml"
        if not calib_file.exists():
            print(f"警告: 未找到标定文件 {calib_file}")
            self.calibration_data = None
            return

        # 优先用 OpenCV FileStorage 读取，兼容 %YAML 与 !!opencv-matrix
        fs = cv2.FileStorage(str(calib_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"警告: 标定文件无法打开: {calib_file}")
            self.calibration_data = None
            return

        def _read_mat(name):
            node = fs.getNode(name)
            return None if node.empty() else node.mat()

        def _read_seq(name):
            node = fs.getNode(name)
            if node.empty():
                return None
            if node.isSeq():
                return [int(node.at(i).real()) for i in range(node.size())]
            mat = node.mat()
            if mat is not None and mat.size >= 2:
                return [int(mat[0, 0]), int(mat[0, 1])]
            return None

        M1 = _read_mat('M1')
        M2 = _read_mat('M2')
        D1 = _read_mat('D1')
        D2 = _read_mat('D2')
        R = _read_mat('R')
        T = _read_mat('T')
        image_size = _read_seq('imageSize')
        fs.release()

        if any(x is None for x in [M1, M2, D1, D2, R, T]) or image_size is None:
            print(f"警告: 标定文件缺少必要字段，已忽略: {calib_file}")
            self.calibration_data = None
            return

        # 规范化类型
        self.calibration_data = {
            'M1': np.asarray(M1, dtype=np.float64),
            'M2': np.asarray(M2, dtype=np.float64),
            'D1': np.asarray(D1, dtype=np.float64).reshape(-1, 1),
            'D2': np.asarray(D2, dtype=np.float64).reshape(-1, 1),
            'R': np.asarray(R, dtype=np.float64),
            'T': np.asarray(T, dtype=np.float64).reshape(3, 1),
            'imageSize': (int(image_size[0]), int(image_size[1])),  # (w, h)
        }
        print(f"已加载标定参数: {calib_file}")
            
    def load_stereo_params(self):
        """加载立体匹配参数"""
        stereo_file = self.data_dir/"stereo-method.yaml"
        if not stereo_file.exists():
            print(f"警告: 未找到立体匹配参数文件 {stereo_file}")
            self.stereo_params = None
            return

        # 用 FileStorage 读取为标量
        fs = cv2.FileStorage(str(stereo_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"警告: 立体匹配参数文件无法打开: {stereo_file}")
            self.stereo_params = None
            return

        def _read_int(name, default=None):
            node = fs.getNode(name)
            if node.empty():
                return default
            try:
                return int(round(node.real()))
            except Exception:
                return default

        params = {
            'MethodName': 'BM',
            'PreFilterType': _read_int('PreFilterType', 1),
            'PreFilterSize': _read_int('PreFilterSize', 9),
            'PreFilterCap': _read_int('PreFilterCap', 31),
            'SADWindowSize': _read_int('SADWindowSize', 9),
            'MinDisparity': _read_int('MinDisparity', 0),
            'NumDisparities': _read_int('NumDisparities', 128),
            'TextureThreshold': _read_int('TextureThreshold', 10),
            'UniquenessRatio': _read_int('UniquenessRatio', 15),
            'SpeckleWindowSize': _read_int('SpeckleWindowSize', 100),
            'SpeckleRange': _read_int('SpeckleRange', 32),
            'Disp12MaxDiff': _read_int('Disp12MaxDiff', 1),
        }
        fs.release()
        self.stereo_params = params
        print(f"已加载立体匹配参数: {stereo_file}")

    def _ensure_rectify_maps(self, frame_size: Tuple[int, int]):
        """根据标定数据与当前帧尺寸，准备重映射表。"""
        if self.calibration_data is None:
            return
        # 如果已经生成且尺寸匹配，直接返回
        if self.rectify_maps_left is not None:
            return

        M1 = self.calibration_data['M1']
        M2 = self.calibration_data['M2']
        D1 = self.calibration_data['D1']
        D2 = self.calibration_data['D2']
        R = self.calibration_data['R']
        T = self.calibration_data['T']

        # 选用标定文件的尺寸，若与实际帧不同也能工作，OpenCV会做调整
        w_h = self.calibration_data.get('imageSize', (frame_size[0], frame_size[1]))
        image_size_cv = (int(w_h[0]), int(w_h[1]))  # (w, h)

        flags = cv2.CALIB_ZERO_DISPARITY
        alpha = 0  # 去黑边
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            M1, D1, M2, D2, image_size_cv, R, T, flags=flags, alpha=alpha
        )

        self.rectify_Q = Q
        # 生成重映射表（按当前帧尺寸创建）
        self.rectify_maps_left = cv2.initUndistortRectifyMap(
            M1, D1, R1, P1, image_size_cv, cv2.CV_16SC2
        )
        self.rectify_maps_right = cv2.initUndistortRectifyMap(
            M2, D2, R2, P2, image_size_cv, cv2.CV_16SC2
        )

    def rectify_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对左右图进行去畸变+校正重映射。若无标定信息则原样返回。"""
        if self.calibration_data is None:
            return left_img, right_img
        h, w = left_img.shape[:2]
        self._ensure_rectify_maps((w, h))
        if self.rectify_maps_left is None or self.rectify_maps_right is None:
            return left_img, right_img
        left_rect = cv2.remap(left_img, self.rectify_maps_left[0], self.rectify_maps_left[1], interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.rectify_maps_right[0], self.rectify_maps_right[1], interpolation=cv2.INTER_LINEAR)
        return left_rect, right_rect
    
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
        
        # 使用标定信息进行重映射与校正
        left_img, right_img = self.rectify_pair(left_img, right_img)
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
    
    def process_image(self, left_img: np.ndarray, right_img: np.ndarray, disparity: np.ndarray):
        """
        处理图像
        基于视差图估计无障碍方向，并在左图上进行可视化叠加。
        """
        # 复制左图作为绘制底图
        if left_img is None or disparity is None:
            return left_img
        overlay = left_img.copy()

        # 视差来自 StereoBM，通常放大16倍；将无效和负值过滤
        disp = disparity.astype(np.float32)
        disp[disp <= 0] = np.nan
        disp = disp / 16.0

        h, w = disp.shape[:2]
        if w < 10 or h < 10:
            return overlay

        # 在中间水平带评估每列的“远距离评分”（取较小分位数代表最近障碍，分位数越小越远则分数越低）
        top = int(0.45 * h)
        bot = int(0.65 * h)
        roi = disp[top:bot, :]

        # 按列计算分位数（忽略NaN）
        # 小视差 => 远 => 我们要找分位数最小的列
        col_scores = np.full((w,), np.nan, dtype=np.float32)
        for x in range(w):
            col = roi[:, x]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                continue
            # 使用20分位作为列的代表（抑制少量异常）
            col_scores[x] = np.percentile(valid, 20)

        # 平滑列分数以避免毛刺
        if np.all(np.isnan(col_scores)):
            return overlay
        kernel = 31
        pad = kernel // 2
        # 以NaN感知的方式做简单滑窗：用有值窗口的均值近似
        smoothed = np.copy(col_scores)
        for x in range(w):
            l = max(0, x - pad)
            r = min(w, x + pad + 1)
            win = col_scores[l:r]
            v = win[~np.isnan(win)]
            if v.size > 0:
                smoothed[x] = np.mean(v)

        # 选择分数最小的列（距离最远，障碍最少）
        valid_idx = np.where(~np.isnan(smoothed))[0]
        if valid_idx.size == 0:
            return overlay
        best_col = int(valid_idx[np.argmin(smoothed[valid_idx])])

        # 在图像上绘制指示（竖线与箭头）
        color = (255, 0, 0)
        thickness = max(2, w // 400)
        cv2.line(overlay, (best_col, 0), (best_col, h - 1), color, thickness)
        # 画一个自下而上的小箭头
        base = (best_col, h - 10)
        tip = (best_col, h - 60)
        cv2.arrowedLine(overlay, base, tip, color, thickness, tipLength=0.3)

        # 叠加文本
        text = f"x={best_col}"
        cv2.putText(overlay, text, (max(5, best_col - 120), max(20, tip[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 4, cv2.LINE_AA)

        return overlay

    def load_imu_for_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """读取与帧号对应的IMU三轴数据。

        期望文件: data_dir/imu/XXXXXXXX.txt，每行一个浮点数，前三行为三轴。
        
        Args:
            frame_number: 帧号（整数）
        Returns:
            np.ndarray shape (3,) 或 None（找不到/解析失败）
        """
        imu_dir = self.data_dir / "imu"
        if not imu_dir.exists():
            return None
        fname = f"{frame_number:08d}.txt"
        imu_path = imu_dir / fname
        if not imu_path.exists():
            return None
        try:
            with open(imu_path, 'r') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
            values: List[float] = []
            for ln in lines[:3]:
                try:
                    values.append(float(ln))
                except ValueError:
                    pass
            if len(values) < 3:
                return None
            return np.array(values[:3], dtype=float)
        except Exception:
            return None

    def visualize_stereo_pair(self, left_img: np.ndarray, right_img: np.ndarray, 
                            disparity: np.ndarray = None, processed_img: Optional[np.ndarray] = None, 
                            imu: Optional[np.ndarray] = None, title: str = "Visualization",save: bool = False):
        """
        可视化立体图像对
        
        Args:
            left_img: 左图像
            right_img: 右图像
            disparity: 视差图
            title: 图像标题
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 5))
        
        # 左图像
        axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Left")
        axes[0, 0].axis('off')
        
        # 右图像
        axes[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("right")
        axes[0, 1].axis('off')
        
        # 视差图
        axes[1, 0].imshow(disparity, cmap='plasma')
        axes[1, 0].set_title("Parallax")
        axes[1, 0].axis('off')

        # 右下角：优先显示处理后图像，并叠加IMU；若无处理图，则显示IMU柱状图
        if processed_img is not None:
            axes[1, 1].imshow(processed_img, cmap='plasma')
            axes[1, 1].set_title("Processed")
            axes[1, 1].axis('off')
            if imu is not None:
                txt = f"IMU\nax: {imu[0]:.3f}\nay: {imu[1]:.3f}\naz: {imu[2]:.3f}"
                axes[1, 1].text(0.02, 0.98, txt, transform=axes[1, 1].transAxes,
                                va='top', ha='left', color='w', fontsize=9,
                                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=4))
        else:
            axes[1, 1].axis('on')
            axes[1, 1].clear()
            if imu is not None:
                axes[1, 1].bar(['ax', 'ay', 'az'], imu, color=['#4e79a7', '#59a14f', '#e15759'])
                axes[1, 1].set_title('IMU (ax, ay, az)')
                axes[1, 1].grid(True, linestyle='--', alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No IMU', ha='center', va='center')
                axes[1, 1].set_title('IMU')
        
        plt.suptitle(title)
        plt.tight_layout()
        if save:
            plt.savefig("save_pic/out.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        


    
    ##Main function
    def process_sequence(self, start_frame: int = None, end_frame: int = None, 
                        max_frames: int = 10, save: bool = False):
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
            print("Error: No image pairs found")
            return
            
        print(f"Found {len(image_pairs)} image pairs")
        
        # 限制处理帧数
        if len(image_pairs) > max_frames:
            image_pairs = image_pairs[:max_frames]
            print(f"Limit the number of frames to {max_frames}")
        
        for i, (left_path, right_path) in enumerate(image_pairs):
            print(f"Processing frame {i+1}/{len(image_pairs)}...")
            
            try:
                left_img, right_img = self.load_image_pair(left_path, right_path)
                #计算视差图
                disparity = self.compute_disparity(left_img, right_img)
                #计算处理后的图像
                processed_img = self.process_image(left_img, right_img, disparity)
                # 读取IMU
                frame_str = Path(left_path).stem.replace('L', '')
                try:
                    frame_num = int(frame_str)
                except ValueError:
                    frame_num = None
                imu_vec = self.load_imu_for_frame(frame_num) if frame_num is not None else None

                self.visualize_stereo_pair(left_img, right_img, disparity, processed_img,
                                            imu=imu_vec, title=f"Frame {i+1}: {Path(left_path).stem}",save=save)

                
                # 等待用户按键继续
                input("按回车键继续下一帧...")
            except Exception as e:
                print(f"处理图像对时出错: {e}")
                continue
    
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="立体视觉图像处理工具")
    parser.add_argument("data_dir", default="kope67-00-00004500-00005050",help="数据目录路径")
    parser.add_argument("--start-frame", type=int, help="起始帧号")
    parser.add_argument("--end-frame", type=int, help="结束帧号")
    parser.add_argument("--max-frames", type=int, default=5, help="最大处理帧数")
    parser.add_argument("--no-disparity", action="store_true", help="不显示视差图")
    parser.add_argument("--save", action="store_true", help="保存处理结果")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = StereoVisionProcessor(args.data_dir)
    
    # 交互式处理
    processor.process_sequence(
        args.start_frame, 
        args.end_frame, 
        args.max_frames, 
        args.save
    )


if __name__ == "__main__":
    main()
