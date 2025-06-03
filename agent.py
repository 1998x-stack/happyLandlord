# 导入必要的库
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入numpy用于数值计算
import random  # 导入random用于随机数生成
import torch.optim as optim  # 导入优化器
import torch.nn as nn  # 导入神经网络模块
from network import DouZeroNet  # 导入DouZero网络模型
from config import Config  # 导入配置类
from typing import List  # 导入类型提示

class DQNAgent:  # 定义DQN智能体类
    def __init__(self, state_dim, action_dim, device="cpu"):  # 初始化智能体
        self.config = Config()  # 加载配置
        self.state_dim = state_dim  # 设置状态维度
        self.action_dim = action_dim  # 设置动作维度
        self.device = device  # 设置设备（CPU/GPU）
        
        self.q_net = DouZeroNet(state_dim, action_dim).to(device)  # 创建Q网络
        self.target_net = DouZeroNet(state_dim, action_dim).to(device)  # 创建目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 复制Q网络参数到目标网络
        self.target_net.eval()  # 将目标网络设置为评估模式
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.LR)  # 创建Adam优化器
        self.loss_fn = nn.MSELoss()  # 创建均方误差损失函数
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], epsilon: float) -> int:  # 选择动作
        if random.random() < epsilon:  # 如果随机数小于探索率
            return random.choice(legal_actions)  # 随机选择一个合法动作
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 将状态转换为张量
        with torch.no_grad():  # 不计算梯度
            q_values = self.q_net(state_tensor)  # 获取Q值
        
        # 只考虑合法动作
        legal_q_values = q_values[0, legal_actions]  # 获取合法动作的Q值
        best_action_idx = torch.argmax(legal_q_values).item()  # 获取最大Q值对应的动作索引
        return legal_actions[best_action_idx]  # 返回最佳动作
    
    def update(self, batch: list) -> float:  # 更新网络
        states, actions, rewards, next_states, dones = zip(*batch)  # 解压批次数据
        
        states = torch.FloatTensor(np.array(states)).to(self.device)  # 转换状态为张量
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # 转换动作为张量
        rewards = torch.FloatTensor(rewards).to(self.device)  # 转换奖励为张量
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # 转换下一状态为张量
        dones = torch.FloatTensor(dones).to(self.device)  # 转换完成标志为张量
        
        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions).squeeze(1)  # 获取当前状态的Q值
        
        # 计算目标Q值
        with torch.no_grad():  # 不计算梯度
            next_q = self.target_net(next_states).max(1)[0]  # 获取下一状态的最大Q值
            target_q = rewards + self.config.GAMMA * next_q * (1 - dones)  # 计算目标Q值
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)  # 计算均方误差损失
        
        # 优化
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        
        return loss.item()  # 返回损失值
    
    def update_target_net(self):  # 更新目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将Q网络参数复制到目标网络
    
    def save(self, path):  # 保存模型
        torch.save(self.q_net.state_dict(), path)  # 保存Q网络参数
    
    def load(self, path):  # 加载模型
        self.q_net.load_state_dict(torch.load(path))  # 加载Q网络参数
        self.target_net.load_state_dict(self.q_net.state_dict())  # 同步目标网络参数