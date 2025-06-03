# 导入必要的库
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块

class DouZeroNet(nn.Module):  # 定义DouZero网络类
    def __init__(self, state_dim, action_dim):  # 初始化网络
        super().__init__()  # 调用父类初始化
        assert len(state_dim) == 3, "state_dim should be (channels, height, width)"
        assert state_dim[0] > 0, "number of input channels should be positive"
        assert action_dim > 0, "action_dim should be positive"
        self.conv = nn.Sequential(  # 定义卷积层序列
            nn.Conv2d(state_dim[0], 32, kernel_size=3, padding=1),  # 第一个卷积层：输入通道到32通道
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第二个卷积层：32通道到64通道
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 第三个卷积层：64通道到128通道
            nn.ReLU()  # ReLU激活函数
        )
        
        self.gru = nn.GRU(128, 128, batch_first=True)  # 定义GRU层：处理序列特征
        
        self.fc = nn.Sequential(  # 定义全连接层序列
            nn.Linear(128, 256),  # 第一个全连接层：GRU输出到256维
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(256, action_dim)  # 第二个全连接层：256维到动作空间维度
        )
    
    def forward(self, x):  # 定义前向传播
        batch_size = x.size(0)  # 获取批次大小
        conv_out = self.conv(x)  # (batch_size, 128, height, width)
        conv_out = conv_out.view(batch_size, 128, -1).transpose(1, 2)  # (batch_size, height*width, 128)
        
        # GRU处理序列特征
        gru_out, _ = self.gru(conv_out)  # 通过GRU层处理序列
        gru_out = gru_out[:, -1, :]  # 取最后一个时间步的输出 (batch_size, 128)
        
        return self.fc(gru_out)  # 通过全连接层得到最终输出