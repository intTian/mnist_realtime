"""
实时手写数字识别
CNN利用MNIST数据集训练 结合OpenCV摄像头实现实时识别
"""

"""
实现步骤
摄像头图像捕获：用 OpenCV 的VideoCapture调用摄像头，获取实时帧。
图像预处理：MNIST 模型输入为(1, 1, 28, 28)的灰度图（黑底白字），因此需要对摄像头图像做以下处理：
转为灰度图
二值化（分离前景和背景）
调整尺寸为 28x28
反转颜色（MNIST 是黑底白字，摄像头可能是白底黑字）
归一化（像素值缩放到 0-1）
模型预测：将预处理后的图像输入 PyTorch 模型，得到预测结果。
实时显示：在摄像头画面上标注预测结果（数字 + 置信度）。
"""

"""
导入库
"""
import cv2  # 用于摄像头捕获和图像处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络层
import numpy as np  # 数值计算
from torchvision import transforms  # 图像转换工具

"""
定义 CNN 模型（与训练时一致）
"""
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一个卷积块：32个3x3卷积核 + ReLU + 2x2最大池化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块：64个3x3卷积核 + ReLU + 2x2最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层：展平特征后连接128维隐藏层 + Dropout + 10维输出（0-9）
        self.fc1 = nn.Linear(7*7*64, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
    
    # 前向传播：定义数据流动路径
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 7*7*64)  # 展平特征图为一维向量
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
"""
图像预处理函数（适配模型输入）
要解决实时识别中噪声多、数字区域不聚焦的问题，
通过去噪、区域定位和针对性处理，让输入模型的图像更接近 MNIST 数据集的特征（黑底白字、数字居中、无多余干扰）。
"""
def preprocess_image(frame):
    # 1. 转灰度图 数字识别不需要颜色信息
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 增强去噪（更大的高斯核，适应更多噪声）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 5x5核增强去噪
    
    # 3. 自适应二值化（调整参数适应不同光线）
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 黑字白底 -> 白字黑底（符合MNIST）
        blockSize=15,  # 更大的局部区域，减少光线干扰
        C=3  # 调整常数项，增强对比度
    )
    
    # 4. 形态学操作（更柔和的去噪，避免数字变形）
    kernel = np.ones((2, 2), np.uint8)  # 更小的核，减少数字腐蚀
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # 5. 轮廓检测优化：优先选择中心位置的轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_roi = None
    
    if contours:  # 确保有轮廓
        # 计算图像中心坐标
        img_center_x, img_center_y = frame.shape[1] // 2, frame.shape[0] // 2
        
        # 筛选符合条件的轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 调整面积阈值（适应不同大小的数字）
            if 200 < area < 10000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                # 放宽宽高比限制，允许更扁或更长的数字
                if 0.3 < aspect_ratio < 3.0:
                    # 计算轮廓中心与图像中心的距离
                    cnt_center_x = x + w // 2
                    cnt_center_y = y + h // 2
                    distance = ((cnt_center_x - img_center_x) **2 + 
                               (cnt_center_y - img_center_y)** 2) **0.5  # 欧氏距离
                    valid_contours.append((distance, x, y, w, h, area))
        
        if valid_contours:
            # 优先按距离排序（最近的优先），距离相同则按面积排序（更大的优先）
            valid_contours.sort(key=lambda x: (x[0], -x[5]))
            # 选择最优轮廓
            _, x, y, w, h, _ = valid_contours[0]
            valid_roi = (x, y, w, h)
    
    # 6. 提取ROI并扩展边界（动态扩展，避免裁剪）
    if valid_roi is not None:
        x, y, w, h = valid_roi
        # 动态扩展边界（按比例扩展）
        expand = int(max(w, h) * 0.2)  # 扩展20%的边界
        x = max(0, x - expand)
        y = max(0, y - expand)
        w = min(thresh.shape[1] - x, w + 2*expand)
        h = min(thresh.shape[0] - y, h + 2*expand)
        roi = thresh[y:y+h, x:x+w]
        # 可视化ROI（调试用，可选）
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 红色框标记ROI
    else:
        roi = thresh  # 未找到轮廓时用全图
    
    # 7. 缩放优化：避免过度压缩（最长边设为24，留2像素边距）
    h, w = roi.shape
    if h == 0 or w == 0:  # 防止空图像报错
        return torch.zeros(1, 1, 28, 28), thresh
    
    # 调整缩放比例
    if h > w:
        scale = 24 / h  # 最长边为24，留2像素边距
        new_h = 24
        new_w = int(w * scale)
    else:
        scale = 24 / w
        new_w = 24
        new_h = int(h * scale)
    
    # 确保缩放后尺寸不为0
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)  # 保留细节的插值方式
    
    # 8. 居中放置（28x28画布）
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_roi
    
    # 9. 转换为张量（与训练一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    processed = transform(canvas)

    # 确保是PyTorch张量
    if not isinstance(processed, torch.Tensor):
        processed = torch.from_numpy(processed)

    # 增加批次维度
    processed = processed.unsqueeze(0)
    # 返回处理后的张量、中间过程图像（用于调试显示）
    return processed, thresh


"""
模型加载函数
"""
def load_model(model_path):
    # 自动选择设备（GPU优先）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型并加载训练好的参数
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 设置为评估模式（关闭Dropout）
    return model, device

"""
实时识别主函数（带录制功能）
按 'r' 键开始/停止录制
按 'esc' 键退出程序
"""
def real_time_recognition(model_path):
    model, device = load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 初始化视频编写器
    out = None
    is_recording = False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # pyright: ignore[reportAttributeAccessIssue]


    # 预测稳定性优化：连续多帧相同预测才更新结果（减少跳变）
    stable_pred = None
    pred_count = 0
    same_pred_threshold = 60  # 连续多少帧相同才确认
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，退出程序...")
            break
        
        # 预处理
        processed_img, thresh = preprocess_image(frame)
        
        # 模型预测
        with torch.no_grad():
            processed_img = processed_img.to(device)#转到设备上
            outputs = model(processed_img)#预测
            probs = torch.nn.functional.softmax(outputs, dim=1)#转换为概率
            predicted_prob, predicted_idx = torch.max(probs, 1)#获取概率和索引
            current_pred = predicted_idx.item()#当前预测结果
            confidence = predicted_prob.item() * 100#置信度百分比
        
        # 稳定性处理：连续相同预测才更新
        if current_pred == stable_pred:
            pred_count += 1
        else:
            stable_pred = current_pred
            pred_count = 1
        
        # 达到阈值才显示该预测（否则保持上一个稳定结果）
        display_pred = stable_pred if pred_count >= same_pred_threshold else stable_pred
        
        # 标注结果和录制状态
        color = (0, 255, 0) # 绿色字体
    
        cv2.putText(frame, 
                f'predict: {display_pred }  confidence: {confidence:.1f}%', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,color, 2)
        
        # 显示录制状态
        if is_recording:
            cv2.putText(frame, "RECORDING", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 如果正在录制，写入帧
            if out is not None:
                out.write(frame)

       
        # 显示原始画面和预处理后的图像
        cv2.imshow('Real-time Digit Recognition', frame)
        cv2.imshow('Preprocessed Image', thresh)

        # 按键处理
        key = cv2.waitKey(1)
        if key == 27:# ESC键
            break
        elif key == ord('r'):
            # 切换录制状态
            is_recording = not is_recording
            if is_recording:
                # 开始录制，创建VideoWriter
                import time
                timestamp = int(time.time())
                filename = f"recording_{timestamp}.mp4"
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
                print(f"开始录制: {filename}")
            else:
                # 停止录制，释放VideoWriter
                if out is not None:
                    out.release()
                    out = None
                    print("停止录制")
    
    # 释放资源
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

"""
程序入口
"""
if __name__ == "__main__":
     #创建窗口
    cv2.namedWindow('Real-time Digit Recognition', cv2.WINDOW_NORMAL)#WINDOW_NORMAL可调大小
    cv2.namedWindow('Preprocessed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preprocessed Image', 640, 480)
    cv2.resizeWindow('Real-time Digit Recognition', 640, 480)
    # 运行实时识别（需指定训练好的模型路径）
    real_time_recognition('mnist_cnn_model.pth')