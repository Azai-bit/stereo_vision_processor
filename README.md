# 立体视觉图像处理工具

这是一个用于处理[MODD2](https://box.vicos.si/borja/viamaro/index.html)视频数据集中立体图像对的Python工具，支持图像可视化、视差图计算和OpenCV处理。
![example](/save_pic/out.png "vision_processor")

## 使用方法

### 克隆项目
```bash
git clone https://github.com/Azai-bit/stereo_vision_processor.git
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本用法
```bash
python3 stereo_vision_processor.py kope67-00-00004500-00005050 \
  --start-frame 4500 \
  --end-frame 4510 \
  --max-frames 5 \
  --save
```
或运行test_processor.py，输入要查看的图片路径，自动生成运行脚本

### 命令行参数

- `data_dir`: 数据目录路径（可选，默认内置一个示例目录）
- `--start-frame`: 起始帧号
- `--end-frame`: 结束帧号
- `--max-frames`: 最大处理帧数（默认：5）
- `--save`: 保存当前可视化（保存到 `save_pic/out.png`）

## 数据目录结构

工具期望的数据目录结构如下：

```
data_directory/
├── calibration.yaml      # 相机标定参数（OpenCV FileStorage 格式）
├── stereo-method.yaml    # 立体匹配参数（OpenCV FileStorage 格式）
├── frames/               # 图像帧目录
│   ├── 00004500L.jpg    # 左图像
│   ├── 00004500R.jpg    # 右图像
│   ├── 00004501L.jpg
│   ├── 00004501R.jpg
│   └── ...
└── imu/                  # IMU数据（可选）
    ├── 00004500.txt     # 前三行分别为 ax, ay, az（浮点数）
    ├── 00004501.txt
    └── ...
```

## 输出说明

当使用 `--save` 选项时，会将当前可视化窗口保存为：

- `save_pic/out.png`: 包含 Left/Right/Parallax/Processed（含 IMU 叠加）四宫格视图


## 注意事项

1. 确保数据目录包含完整的标定文件和图像帧；标定文件必须为 OpenCV YAML 格式（`%YAML:1.0`、`!!opencv-matrix`）
2. 左右图像必须成对存在且命名规范（以 `L`/`R` 结尾）
3. 处理大量帧时建议使用 `--max-frames` 限制
4. 视差图由 StereoBM 计算，默认像素视差扩大 16 倍（已在内部缩放处理）
5. IMU 文件若缺失或格式不符，会自动忽略
6. 若标定文件无法解析，将回退到默认参数（仍可运行但无精准校正）

## 故障排除
- 如果出现“无法加载图像”错误，检查图像文件路径和权限
- 确保安装了所有必需的 Python 包

