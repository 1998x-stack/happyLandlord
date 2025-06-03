from trainer import Trainer
from utils import init_logging, set_seed
from loguru import logger

def main():
    # 初始化配置和日志
    from config import Config
    init_logging()
    set_seed(42)
    
    logger.info("启动腾讯欢乐斗地主2v2强化学习训练")
    logger.info(f"日志目录: {Config.LOG_DIR}")
    logger.info(f"模型目录: {Config.MODEL_DIR}")
    logger.info(f"TensorBoard目录: {Config.TENSORBOARD_DIR}")
    
    # 创建并运行训练器
    trainer = Trainer()
    trainer.run_training()

if __name__ == "__main__":
    main()