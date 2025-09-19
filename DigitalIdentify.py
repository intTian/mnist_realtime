"""
TODO:加载包
"""
# 导入必要的库
import torch  # PyTorch主库，用于构建和训练神经网络
import torch.nn as nn  # 神经网络模块，包含各种层和激活函数
import torch.optim as optim  # 优化器模块，包含各种优化算法
from torch.utils.data import DataLoader  # 数据加载工具，用于批量加载数据
from torchvision import datasets, transforms  # 计算机视觉工具，包含数据集和数据转换

"""
TODO:数据预处理
1.归一化
2.one-hot编码
3.打乱数据

什么叫作数据预处理 方法 意义？
transforms.Compose使用
"""
# 数据预处理转换
# Compose用于组合多个转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor格式(0-1范围) 将二维数组转换为张量 并归一化
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化：MNIST数据集的均值和标准差
])
"""
TODO:加载数据
MNIST数据集是28×28像素的灰度手写数字图片，共有10个类别（数字0-9）
该数据集是机器学习领域中非常经典的一个数据集，主要用于训练和测试手写数字识别的算法。
MNIST数据集由0〜9手写数字图片和数字标签组成，包含60000个训练样本和10000个测试样本，每个样本都是一张28 * 28像素的灰度手写数字图片。

数据集内容
训练集: 60000张图片
测试集: 10000张图片
图片格式: 28 * 28像素的灰度图片

怎么加载数据？
datasets常用数据集及加载方法
数据加载器 使用及意义
"""
# 加载MNIST训练集
# root：数据保存路径，train=True表示加载训练集，download=True表示如果没有数据则下载
train_dataset = datasets.MNIST(
    root='./data',#数据保存路径
    train=True,#加载训练集
    download=True,#如果没有数据则下载
    transform=transform #数据预处理
)
# 加载MNIST测试集
# train=False表示加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
# batch_size：每次训练的样本数，shuffle=True表示打乱数据顺序
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False  # 测试集不需要打乱
)

"""
TODO:CNN模型定义
输入层、两个卷积层、两个池化层、全连接层和输出层

如何设计结构 每个层的意义是什么？
维度的变化
损失函数 优化器的选择
"""
# 定义CNN模型
class CNNModel(nn.Module):  # 继承nn.Module，自定义模型的基类
    def __init__(self):
        super(CNNModel, self).__init__()  # 调用父类构造函数
        """MNIST 图片是单通道灰度图，输入到模型的数据形状为：(batch_size, 1, 28, 28)
        batch_size：批次大小（一次训练的样本数，如 32）
        1：通道数（灰度图为 1，彩色图为 3）
        28, 28：图片的高和宽
        """
        # 第一个卷积块：卷积层 + 激活函数 + 池化层
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入通道数：灰度图为1
            out_channels=32,  # 输出通道数(卷积核数量)
            kernel_size=3,  # 卷积核大小3x3
            stride=1,  # 步长
            padding=1  # 填充，保持输出尺寸
        )
        """维度计算：输出形状 = (batch_size, 32, 28, 28)
        通道数变为 32（与卷积核数量一致）
        高和宽不变：(28 + 2×1 - 3) / 1 + 1 = 28
        """

        self.relu1 = nn.ReLU()  # ReLU激活函数
        """维度不变：(batch_size, 32, 28, 28)
        激活函数只改变数值，不改变形状）
        """

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        """维度计算：输出形状 = (batch_size, 32, 14, 14)
        高和宽减半：28 / 2 = 14（池化窗口和步长都是 2） 28+2*0-2/2+1=14
        """

        # 第二个卷积块
        self.conv2 = nn.Conv2d(
            in_channels=32,  # 输入通道数与上一层输出一致
            out_channels=64,  # 输出通道数
            kernel_size=3,
            stride=1,
            padding=1
        )
        """维度计算：输出形状 = (batch_size, 64, 14, 14)
        通道数变为 64
        高和宽不变：(14 + 2×1 - 3) / 1 + 1 = 14
        """

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        """维度计算：输出形状 = (batch_size, 64, 7, 7)
        高和宽再减半：14 / 2 = 7 (14+2*0-2)/2+1=7
        """
    
        """维度计算：展平后形状 = (batch_size, 7×7×64) = (batch_size, 3136)
        全连接层之前的展平操作
        在forward函数中，卷积特征需要展平为一维向量才能输入全连接层：
        x = x.view(-1, 7*7*64)

        

        7×7：池化后的高和宽
        64：通道数
        总特征数：7×7×64 = 3136
        """

        # 全连接层 对于每个样本 有多少个特征数 就有都是个神经元连接
        self.fc1 = nn.Linear(
            in_features=7*7*64,  # 输入特征数：池化后7x7，64个通道
            out_features=128  # 输出特征数
        )
        self.relu3 = nn.ReLU()

        """Dropout层：随机丢弃部分神经元，防止过拟合
        #在训练时，每次前向传播会随机让 20% 的神经元暂时失效（输出置为 0），且每次随机选择的神经元不同
        # 在测试 / 预测时，Dropout 层不工作，所有神经元正常参与计算，但会自动将输出结果乘以 0.8（补偿训练时被丢弃的神经元贡献）
        为什么能防止过拟合？
            减少神经元依赖：避免模型过度依赖某些 "强势神经元"（类似避免学生抄答案），迫使模型学习更通用的规律
            模拟集成学习：每次训练相当于在不同的子网络上学习，最终模型是多个子网络的 "平均效果"
            增加随机性：给训练过程加入噪声，使模型对输入的微小变化更稳健
        """
        self.dropout = nn.Dropout(0.2)  # Dropout层，防止过拟合
        """维度计算：输出形状 = (batch_size, 128)
        （通过矩阵乘法将 3136 维特征压缩为 128 维）
        relu3和dropout不改变维度，输出仍为(batch_size, 128)
        """

        # 输出层：10个类别(0-9)
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )
        """维度计算：最终输出形状 = (batch_size, 10)
        （10 个数值分别对应数字 0~9 的预测概率）
        """
    
    # 前向传播函数：定义数据在模型中的流动路径
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平特征图：从二维转为一维
        #view函数用于调整张量的形状 相对于reshape函数 结果为（批次数，特征数）
        x = x.view(-1, 7*7*64)  # -1表示自动计算批量大小
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        return x

# 初始化模型时添加设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)  # 将模型移到GPU

# # 初始化模型
# model = CNNModel()

# 定义损失函数：交叉熵损失，适用于多分类问题
criterion = nn.CrossEntropyLoss()

# 定义优化器：Adam优化器，学习率0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
TODO:模型训练
参数说明：
model：要训练的神经网络模型
train_loader：训练数据集的DataLoader对象
criterion：损失函数，用于计算模型输出和真实标签之间的差异
optimizer：优化器，用于更新模型参数
epochs：训练的轮数，表示整个训练数据集将被迭代多少次

代码要注意几点
开始训练模式 两层循环 使用优化器 数据加载器
"""
def train(model, train_loader, criterion, optimizer, epochs=10):
    # 设置为训练模式：启用Dropout等训练特有的层
    model.train()
    
    for epoch in range(epochs):  # 迭代训练轮数
        running_loss = 0.0  # 记录总损失
        # train_loader的结构：(data, target) data:输入数据 target:标签
        for batch_idx, (data, target) in enumerate(train_loader):
             # 将数据移到GPU
            data, target = data.to(device), target.to(device)

            # 梯度清零：防止上一轮的梯度影响当前轮
            optimizer.zero_grad()
            
            # 前向传播：计算模型输出
            outputs = model(data)
            
            # 计算损失
            loss = criterion(outputs, target)
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 更新参数：根据梯度调整权重
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
            
            # 每100个批次打印一次信息
            if batch_idx % 300 == 99:
                print(f'轮次 [{epoch+1}/{epochs}], 批次 [{batch_idx+1}/{len(train_loader)}], 损失: {running_loss/100:.4f}')
                running_loss = 0.0  # 重置损失

# 执行训练
train(model, train_loader, criterion, optimizer, epochs=10)

"""
TODO:模型保存与加载
保存模型参数：torch.save(model.state_dict(), 'model.pth')
加载模型参数：model.load_state_dict(torch.load('model.pth'))
"""
# # 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存为 mnist_cnn_model.pth")

# 加载模型函数
def load_model(model_path):
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式：禁用Dropout等层
    return model

# 保存模型后，可通过以下方式加载并评估
model = load_model('mnist_cnn_model.pth')

"""
TODO:模型评估
评估使用哪些数据集
评估模式
"""
# 模型评估
def evaluate(model, test_loader):
    # 设置为评估模式：关闭Dropout等层
    model.eval()
    correct = 0  # 正确预测的数量
    total = 0    # 总样本数
    
    # 关闭梯度计算：节省内存，加速计算
    with torch.no_grad():
        for data, target in test_loader:
             # 将数据移到GPU
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs = model(data)
            
            # 取概率最大的类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            # 累计总样本数和正确样本数
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 计算准确率
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

# 执行评估
evaluate(model, test_loader)

"""
TODO:模型预测
预测使用哪些数据集
"""
# 模型预测示例
def predict_sample(model, test_loader, sample_index=0):
    model.eval()  # 评估模式
    
    # 获取测试集中的一个样本
    data, target = test_loader.dataset[sample_index]

    data = data.unsqueeze(0).to(device)  #增加批次维度 迁移数据到GPU
    
    with torch.no_grad():#禁用梯度计算
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        
    """
    test_loader.dataset 是 MNIST 数据集对象，直接通过索引（[sample_index]）获取样本时，返回的 target 是原生整数（如 5、7 等），而不是张量。
    只有当通过 DataLoader 批量加载数据时（如训练循环中的 for data, target in train_loader）
    target 才会被封装成张量（因为 DataLoader 会自动将批次数据转换为张量）。
    """
    print(f'预测结果: {predicted.item()}, 真实标签: {target}')#target不需要加item()

# 预测第一个样本
predict_sample(model, test_loader, sample_index=0)









