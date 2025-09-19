"""
用tensorflow实现
"""
import tensorflow as tf
import numpy as np

# 准备数据：面积（x）和房价（y）
x = np.array([100, 150, 200, 250], dtype=float)
y = np.array([50, 75, 100, 125], dtype=float)  # 假设1平米=0.5万
#数据归一化
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# 构建模型：一个简单的线性层（y = wx + b）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])  # 1个输出，输入是1个数
])

# 编译模型：定义优化器和损失函数
model.compile(optimizer='sgd', loss='mean_squared_error')

# 训练模型：让电脑学习规律
model.fit(x, y, epochs=1000)  # 训练1000次

# 预测：输入300平米，应该输出150万
input_data = np.array(300)
input_data = np.expand_dims(input_data, axis=0)  # 变为(1,)
input_data = np.expand_dims(input_data, axis=1)  # 变为(1, 1)
print(model.predict(input_data))  # 输出接近[[150.]]
# """
# 用pytorch实现
# """
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import torch.nn.functional as F

# # 定义神经网络模型
# #一层线性层 一个神经元
# """
# 深度学习模型通常不会一次只处理一个样本，而是批量处理多个样本（比如一次处理 32 个、64 个样本），这样能提高计算效率（利用 GPU 并行计算）并稳定训练过程。
# 二维张量的形状 [N, D] 正好对应这种批量处理模式：
# N：批量大小（batch size），表示一次处理的样本数量
# D：每个样本的特征数量
# """
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 输入1个特征(房屋面积)，输出1个特征(房价)
#         self.fc = nn.Linear(1, 1)  # 线性层: y = wx + b

#     def forward(self, x):
#         # 前向传播: 直接通过线性层计算
#         x = self.fc(x)
#         return x
    
# # 创建模型实例
# model = Net()

# # 定义损失函数：均方误差损失(MSE)
# # 适合回归问题，计算预测值与真实值的平方差的平均值
# criterion = nn.MSELoss()  

# # 定义优化器：随机梯度下降(SGD)
# # lr=0.0001是学习率，需要根据数据范围调整
# optimizer = optim.SGD(model.parameters(), lr=0.0001)  


# # 1. 准备原始数据
# x = np.array([100, 150, 200, 250], dtype=np.float32)  # 房屋面积
# y = np.array([50, 75, 100, 125], dtype=np.float32)    # 房价

# # 2. 数据归一化（关键！将数据缩放到0~1范围）
# x_mean, x_std = x.mean(), x.std()  # 计算均值和标准差
# y_mean, y_std = y.mean(), y.std()

# x_norm = (x - x_mean) / x_std  # 标准化：(x - 均值) / 标准差
# y_norm = (y - y_mean) / y_std

# # 3. 转换为张量并调整形状
# x_tensor = torch.from_numpy(x_norm).view(-1, 1)
# y_tensor = torch.from_numpy(y_norm).view(-1, 1)

# # 4. 训练模型
# for epoch in range(10000):
#     output = model(x_tensor)
#     loss = criterion(output, y_tensor)
    
#     optimizer.zero_grad()  # 清空梯度
#     loss.backward()        # 反向传播
#     optimizer.step()       # 更新参数
    
#     if (epoch + 1) % 1000 == 0:
#         print(f'Epoch [{epoch+1}/10000], Loss: {loss.item():.6f}')

# # 5. 预测300平米房价（需要先归一化输入，再还原输出）
# # 归一化输入
# x_test = np.array([300], dtype=np.float32)
# x_test_norm = (x_test - x_mean) / x_std
# x_test_tensor = torch.from_numpy(x_test_norm).view(-1, 1)

# # 模型预测（得到归一化的结果）
# y_pred_norm = model(x_test_tensor)

# # 还原为原始房价范围
# y_pred = y_pred_norm.item() * y_std + y_mean

# print(f'\n最终损失: {loss.item():.6f}')
# print(f'300平米房屋的预测价格: {y_pred:.2f}万元')

# # 打印模型参数
# for name, param in model.named_parameters():
#     print(f'{name}: {param.item():.6f}')
