# 导入必要的库
import os  # 导入操作系统模块
from datetime import datetime  # 导入日期时间模块

class Config:  # 定义配置类
    # 环境设置
    NUM_PLAYERS = 4  # 玩家数量
    TEAM_A = [0, 2]  # A队玩家编号
    TEAM_B = [1, 3]  # B队玩家编号
    STATE_SHAPE = (6, 5, 15)  # 状态空间形状（通道数，高度，宽度）
    
    # 训练参数
    GAMMA = 0.99  # 折扣因子
    LR = 1e-3  # 学习率
    BATCH_SIZE = 32  # 批次大小
    TARGET_UPDATE = 100  # 目标网络更新频率
    MEMORY_CAPACITY = 10000  # 经验回放缓冲区容量
    NUM_EPISODES = 10000  # 训练回合数
    EPSILON_START = 0.9  # 初始探索率
    EPSILON_END = 0.05  # 最终探索率
    EPSILON_DECAY = 1000  # 探索率衰减步数
    
    # 日志和模型保存
    LOG_DIR = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 日志目录（使用当前时间戳）
    MODEL_DIR = "models"  # 模型保存目录
    SAVE_INTERVAL = 100  # 模型保存间隔
    
    # TensorBoard
    TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")  # TensorBoard日志目录
    
    @classmethod
    def create_dirs(cls):  # 创建必要的目录
        os.makedirs(cls.LOG_DIR, exist_ok=True)  # 创建日志目录
        os.makedirs(cls.MODEL_DIR, exist_ok=True)  # 创建模型保存目录
        os.makedirs(cls.TENSORBOARD_DIR, exist_ok=True)  # 创建TensorBoard日志目录

config = Config()  # 创建配置实例
config.create_dirs()  # 创建必要的目录