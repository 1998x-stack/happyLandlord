import os
from datetime import datetime

class Config:
    # 环境设置
    NUM_PLAYERS = 4
    TEAM_A = [0, 2]
    TEAM_B = [1, 3]
    STATE_SHAPE = (6, 5, 15)
    
    # 训练参数
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE = 100
    MEMORY_CAPACITY = 10000
    NUM_EPISODES = 10000
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 1000
    
    # 日志和模型保存
    LOG_DIR = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_DIR = "models"
    SAVE_INTERVAL = 100
    
    # TensorBoard
    TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.TENSORBOARD_DIR, exist_ok=True)

config = Config()
config.create_dirs()