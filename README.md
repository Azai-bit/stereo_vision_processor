# 立体视觉图像处理工具

这是一个用于处理[MODD2](https://box.vicos.si/borja/viamaro/index.html)视频数据集中立体图像对的Python工具，支持图像可视化、视差图计算和OpenCV处理。

## 功能特性

- 自动加载相机标定参数和立体匹配参数
- 支持批量处理图像序列
- 实时计算和显示视差图
- 交互式图像浏览
- 批量保存处理结果
- 支持帧范围过滤

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
# 处理指定数据目录的图像
python stereo_vision_processor.py kope67-00-00004500-00005050

# 限制处理帧数
python stereo_vision_processor.py kope67-00-00004500-00005050 --max-frames 3

# 指定帧范围
python stereo_vision_processor.py kope67-00-00004500-00005050 --start-frame 4500 --end-frame 4510
```

### 高级选项

```bash
# 不显示视差图，只显示左右图像
python stereo_vision_processor.py kope67-00-00004500-00005050 --no-disparity

# 保存处理结果到指定目录
python stereo_vision_processor.py kope67-00-00004500-00005050 --save --output-dir results

# 组合使用
python stereo_vision_processor.py kope67-00-00004500-00005050 \
    --start-frame 4500 \
    --end-frame 4510 \
    --max-frames 5 \
    --save \
    --output-dir output_frames
```

### 命令行参数

- `data_dir`: 数据目录路径（必需）
- `--start-frame`: 起始帧号
- `--end-frame`: 结束帧号
- `--max-frames`: 最大处理帧数（默认：5）
- `--no-disparity`: 不显示视差图
- `--save`: 保存处理结果
- `--output-dir`: 输出目录（默认：output）

## 数据目录结构

工具期望的数据目录结构如下：

```
data_directory/
├── calibration.yaml      # 相机标定参数
├── stereo-method.yaml    # 立体匹配参数
├── frames/               # 图像帧目录
│   ├── 00004500L.jpg    # 左图像
│   ├── 00004500R.jpg    # 右图像
│   ├── 00004501L.jpg
│   ├── 00004501R.jpg
│   └── ...
└── imu/                  # IMU数据（可选）
    ├── 00004500.txt
    ├── 00004501.txt
    └── ...
```

## 输出说明

当使用 `--save` 选项时，工具会生成以下文件：

- `disparity_XXXXX.png`: 视差图
- `left_XXXXX.jpg`: 左图像副本
- `right_XXXXX.jpg`: 右图像副本

其中 `XXXXX` 是帧号。

## 示例

### 交互式浏览

```bash
# 浏览前5帧图像，显示视差图
python stereo_vision_processor.py kope67-00-00004500-00005050 --max-frames 5
```

### 批量处理

```bash
# 处理4500-4510帧，保存结果
python stereo_vision_processor.py kope67-00-00004500-00005050 \
    --start-frame 4500 \
    --end-frame 4510 \
    --save \
    --output-dir processed_frames
```

## 注意事项

1. 确保数据目录包含完整的标定文件和图像帧
2. 左右图像必须成对存在且命名规范
3. 处理大量帧时建议使用 `--max-frames` 限制
4. 视差图计算需要足够的计算资源

## 故障排除

- 如果出现"无法加载图像"错误，检查图像文件路径和权限
- 如果标定文件加载失败，工具会使用默认参数
- 确保安装了所有必需的Python包
