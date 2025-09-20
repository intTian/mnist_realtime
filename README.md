# CNN手写数字识别

## 问题描述

利用卷积[神经网络](https://so.csdn.net/so/search?q=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&spm=1001.2101.3001.7020)将MNIST数据集的28×28像素的灰度手写数字图片识别为相应的数字。

## 数据描述

MNIST数据集是28×28像素的灰度手写数字图片，其中数字的范围从0到9

具体如下所示（参考自Tensorflow官方文档）：

train-images-idx3-ubyte.gz:训练集图片，55000张训练图片, 5000张验证图片

train-labels-idx1-ubyte.gz:训练集图片对应的数字标签

t10k-images-idx3-ubyte.gz:测试集图片，10000张图片

t10k-labels-idx1-ubyte.gz:测试集图片对应的数字标签

## 网络结构

### 输入层

将数据输入到神经网络中

输入层的结构是多维的 

MNIST数据集中是28×28像素的灰度图片 因此输入为28×28的二维矩阵

### 卷积层

使用卷积核提取特征

概念：

局部感受野： 类似滑动窗口 以窗口的范围去提取对应范围的神经元携带的特征

共享权值：对于同一个卷积核 权重是相同的 滑动过程中 权重不变

一个N×N的图像经过M×M的卷积核卷积后将得到（N-M+1）×（N-M+1）的输出。

卷积后输出的矩阵数据成为特征映射图，一个卷积核输出一个特征映射图，卷积操作是一种线性计算，因此通常在卷积后进行一次非线性映射。

### 激励层

激活函数 ReLU Sigmoid

映射到0-1之间 便于收敛 

### 池化层

池化层是将卷积得到的特征映射图进行稀疏处理，减少数据量，操作与卷积基本相似，不同的是卷积操作是一种线性计算，而池化的计算方法更多样化，一般有如下计算方式：

最大池化：取样池中的最大值作为池化结果

均值池化：取样池中的平均值作为池化结果

还有重叠池化、均方池化、归一化池化等方法。

### 全连接层

在网络的末端对提取后的特征进行恢复，重新拟合，减少因为特征提取而造成的特征丢失。

全连接层的神经元数需要根据经验和实验结果进行反复调参。

### 输出层

输出层用于将最终的结果输出，针对不同的问题，输出层的结构也不相同

例如MNIST数据集识别问题中，输出层为有10个神经元的向量 识别0-9

## 示例网络结构

### 模型结构

模型包括输入层

两个卷积层

两个池化层

全连接层

输出层

### 特征图输出大小计算公式

其中卷积和池化操作的特征图输出大小计算公式为：

![a3155bbbd0fb09abba3c8d2af25e76fe](https://i-blog.csdnimg.cn/blog_migrate/a3155bbbd0fb09abba3c8d2af25e76fe.png)

ImageWidth：图片宽度

Padding：边缘补齐像素数

KernelSize：卷积核宽度

Stride：移动步长

### 图示

具体模型结构如下图：

<img title="" src="https://i-blog.csdnimg.cn/blog_migrate/5a667e7cd8e715b6bfd18946f13f8b87.png" alt="5a667e7cd8e715b6bfd18946f13f8b87" data-align="center" style="zoom:67%;">

实时手写数字识别
=================

项目简介
----

本项目基于 **CNN（卷积神经网络）** 和 **OpenCV** 实现实时手写数字识别：使用 MNIST 数据集训练 CNN 模型，通过电脑摄像头捕获实时画面，对画面中的手写数字进行预处理后输入模型，最终在画面上标注识别结果（数字 + 置信度），支持视频录制功能。

## 核心功能

1. **实时摄像头捕获**：调用电脑默认摄像头，获取实时视频帧。
2. **智能图像预处理**：针对实时场景的噪声、光线变化，优化数字区域检测与图像适配（转为 MNIST 标准的 28×28 黑底白字灰度图）。
3. **CNN 模型预测**：加载预训练模型，输出数字预测结果及置信度（0-9 数字识别）。
4. **预测稳定性优化**：连续多帧相同预测才更新结果，减少识别跳变。
5. **视频录制**：支持按 `r` 键开启 / 停止录制，自动生成带时间戳的 MP4 视频文件。

环境依赖
----

需提前安装以下 Python 库，建议使用 `pip` 安装：

bash
    # 核心依赖
    pip install opencv-python  # 摄像头捕获与图像处理
    pip install torch torchvision  # PyTorch 框架与图像工具
    pip install numpy  # 数值计算

* **Python 版本**：3.7+
* **设备支持**：支持 CPU 运行；若有 NVIDIA 显卡，可安装 CUDA 版本 PyTorch 加速预测（需匹配显卡驱动版本）。

快速开始
----

### 1. 准备预训练模型

项目运行需 `mnist_cnn_model.pth` 预训练模型文件，获取方式有两种：

* **方式 1：自行训练**：使用 MNIST 数据集训练 CNN 模型（模型结构见主程序 `CNNModel` 类），训练后保存为 `mnist_cnn_model.pth`。
  
  * 训练参考代码（简化版）：
    python
    
        import torch
        from torchvision.datasets import MNIST
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from real_time_digit_recognition import CNNModel  # 导入主程序中的模型类
        
        # 数据加载与预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 模型初始化与训练（省略优化器、损失函数、训练循环等代码）
        model = CNNModel()
        # ...（训练逻辑）...
        
        # 保存模型
        torch.save(model.state_dict(), 'mnist_cnn_model.pth')

* **方式 2：使用现有模型**：从可靠渠道获取与 `CNNModel` 结构匹配的预训练模型文件，命名为 `mnist_cnn_model.pth` 并放在主程序同级目录。

### 2. 运行实时识别

将 `mnist_cnn_model.pth` 与主程序文件放在同一目录，执行以下命令：

bash
    python real_time_digit_recognition.py

### 3. 操作说明

运行后会弹出两个窗口：

* **Real-time Digit Recognition**：原始摄像头画面，显示识别结果（绿色字体）和录制状态（红色 "RECORDING" 标识）。
* **Preprocessed Image**：预处理后的图像（二值化结果），辅助查看数字区域检测效果。

| 按键    | 功能描述            |
| ----- | --------------- |
| `Esc` | 退出程序            |
| `r`   | 切换录制状态（开启 / 停止） |

关键模块说明
------

### 1. CNN 模型结构（`CNNModel` 类）

模型针对 MNIST 数据集设计，共 5 层（2 个卷积块 + 2 个全连接层）：

* **卷积块 1**：32 个 3×3 卷积核 → ReLU 激活 → 2×2 最大池化（输出特征图尺寸：14×14）。
* **卷积块 2**：64 个 3×3 卷积核 → ReLU 激活 → 2×2 最大池化（输出特征图尺寸：7×7）。
* **全连接层**：7×7×64 特征展平 → 128 维隐藏层（ReLU 激活 + Dropout 防过拟合）→ 10 维输出（对应 0-9 数字）。

### 2. 图像预处理（`preprocess_image` 函数）

针对实时场景优化，核心步骤：

1. **灰度转换**：去除颜色信息，转为单通道灰度图。
2. **高斯去噪**：5×5 高斯核平滑图像，减少噪声干扰。
3. **自适应二值化**：根据局部光线调整阈值，实现 “白底黑字→黑底白字”（匹配 MNIST 格式）。
4. **轮廓检测**：优先选择图像中心区域、面积符合条件的轮廓（定位手写数字）。
5. **尺寸适配**：将数字区域缩放到 24×24（留 2 像素边距），居中放置到 28×28 画布。
6. **归一化**：转为张量并按 MNIST 均值 / 标准差归一化，适配模型输入。


