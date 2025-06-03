# 导入必要的库
import random  # 导入random用于随机采样
from collections import deque  # 导入deque用于实现双端队列
from config import Config  # 导入配置类

class ReplayMemory:  # 定义经验回放记忆类
    def __init__(self, capacity):  # 初始化记忆类
        self.capacity = capacity  # 设置记忆容量
        self.buffer = deque(maxlen=capacity)  # 创建固定长度的双端队列
    
    def push(self, state, action, reward, next_state, done):  # 添加经验到记忆
        self.buffer.append((state, action, reward, next_state, done))  # 将经验元组添加到队列
    
    def sample(self, batch_size):  # 采样经验批次
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))  # 随机采样指定大小的批次
        states, actions, rewards, next_states, dones = zip(*batch)  # 解压批次数据
        return states, actions, rewards, next_states, dones  # 返回解压后的数据
    
    def __len__(self):  # 获取记忆长度
        return len(self.buffer)  # 返回当前记忆中的经验数量

class TeamMemory:  # 定义队伍记忆类
    def __init__(self, capacity):  # 初始化队伍记忆类
        self.capacity = capacity  # 设置记忆容量
        self.buffer = deque(maxlen=capacity)  # 创建固定长度的双端队列
        self.team_strategies = {}  # 初始化队伍策略字典
    
    def add_experience(self, state, action, reward, next_state, done, team_id):  # 添加队伍经验
        self.buffer.append((state, action, reward, next_state, done, team_id))  # 将带队伍ID的经验元组添加到队列
    
    def sample(self, batch_size):  # 采样队伍经验批次
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))  # 随机采样指定大小的批次
        states, actions, rewards, next_states, dones, team_ids = zip(*batch)  # 解压批次数据
        return states, actions, rewards, next_states, dones, team_ids  # 返回解压后的数据
    
    def update_team_strategy(self, team_id, strategy_vector):  # 更新队伍策略
        self.team_strategies[team_id] = strategy_vector  # 更新指定队伍的策略向量
    
    def get_team_strategy(self, team_id):  # 获取队伍策略
        return self.team_strategies.get(team_id, None)  # 返回指定队伍的策略向量，如果不存在则返回None