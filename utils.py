import numpy as np
import torch
import random
import os
from loguru import logger

def set_seed(seed):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")

def init_logging():
    """初始化日志系统"""
    from .config import Config
    logger.add(f"{Config.LOG_DIR}/system.log", rotation="100 MB", level="INFO")
    logger.info("日志系统初始化完成")

def log_memory_usage():
    """记录内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    logger.info(f"内存使用: {mem:.2f} GB")
    return mem

def log_gpu_usage():
    """记录GPU使用情况"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"GPU {i} 使用: {mem:.2f}/{total_mem:.2f} GB")
        return mem
    return 0