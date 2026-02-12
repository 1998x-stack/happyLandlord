# 腾讯欢乐斗地主2V2强化学习项目 - 用户指南

## 目录
1. [项目概述](#项目概述)
2. [系统要求](#系统要求)
3. [安装指南](#安装指南)
4. [快速入门](#快速入门)
5. [项目结构](#项目结构)
6. [配置说明](#配置说明)
7. [运行训练](#运行训练)
8. [监控与分析](#监控与分析)
9. [故障排除](#故障排除)
10. [进阶使用](#进阶使用)

## 项目概述

本项目是一个完整的腾讯欢乐斗地主2V2强化学习系统，实现了：
- 完整的游戏规则模拟
- 多智能体协同训练
- CNN+GRU神经网络架构
- 团队经验回放机制
- 实时监控与日志记录

## 系统要求

### 硬件要求
- CPU: Intel/AMD 4核或更高
- 内存: 8GB RAM 或更高 (推荐 16GB+)
- 存储: 500MB 可用空间
- GPU: (可选) CUDA兼容显卡用于加速训练

### 软件要求
- Python 3.7或更高版本
- pip 包管理器
- (可选) CUDA 10.2+ (如使用GPU)

## 安装指南

### 1. 环境准备
```bash
# 检查Python版本 (需要3.7+)
python --version

# 克隆项目或下载源码
git clone <repository-url>
# 或手动下载zip文件并解压
```

### 2. 创建虚拟环境 (推荐)
```bash
# 创建虚拟环境
python -m venv happylandlord_env

# 激活虚拟环境
# Windows:
happylandlord_env\Scripts\activate
# macOS/Linux:
source happylandlord_env/bin/activate

# 升级pip
pip install --upgrade pip
```

### 3. 安装依赖
```bash
cd happyLandlord
pip install -r requirements.txt
```

### 4. 验证安装
```bash
# 测试基本导入
python -c "from environment import LandlordEnv2v2; print('安装成功')"
```

## 快速入门

### 1. 运行基本训练
```bash
# 开始训练
python main.py
```

### 2. 查看实时监控
```bash
# 在新的终端窗口中
tensorboard --logdir=logs
```

### 3. 查看训练日志
```bash
# 实时查看训练进度
tail -f logs/*/training.log
```

### 4. 运行测试
```bash
# 运行单元测试
python -m unittest test_happy_landlord
```

## 项目结构

```
happyLandlord/
├── main.py              # 主程序入口
├── config.py            # 项目配置
├── agent.py             # 智能体实现
├── network.py           # 神经网络定义
├── environment.py       # 游戏环境
├── trainer.py           # 训练器
├── memory.py            # 经验回放
├── utils.py             # 工具函数
├── test_happy_landlord.py # 单元测试
├── logs/                # 训练日志
├── models/              # 模型保存目录
├── requirements.txt     # 依赖包
├── README.md           # 项目说明
├── USER_GUIDE.md       # 本用户指南
├── ENV.md              # 环境文档
└── PROJECT_SUMMARY.md   # 项目总结
```

## 配置说明

### 主要配置参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_PLAYERS` | 4 | 游戏玩家数量 |
| `TEAM_A` | [0, 2] | A队玩家ID |
| `TEAM_B` | [1, 3] | B队玩家ID |
| `STATE_SHAPE` | (6, 5, 15) | 状态张量维度 |
| `GAMMA` | 0.99 | 折扣因子 |
| `LR` | 1e-3 | 学习率 |
| `BATCH_SIZE` | 32 | 批次大小 |
| `TARGET_UPDATE` | 100 | 目标网络更新频率 |
| `MEMORY_CAPACITY` | 10000 | 经验回放缓冲区容量 |
| `NUM_EPISODES` | 10000 | 训练回合数 |
| `EPSILON_START` | 0.9 | 初始探索率 |
| `EPSILON_END` | 0.05 | 最终探索率 |
| `EPSILON_DECAY` | 1000 | 探索率衰减步数 |

### 修改配置
可以直接编辑 `config.py` 文件来修改参数，例如：
```python
NUM_EPISODES = 5000  # 减少训练回合数
BATCH_SIZE = 64      # 增大批次大小
LR = 5e-4           # 降低学习率
```

## 运行训练

### 1. 基础训练
```bash
python main.py
```

### 2. 自定义训练参数
创建一个自定义脚本来修改特定参数：
```python
# custom_train.py
from trainer import Trainer
from config import Config

# 修改配置
Config.NUM_EPISODES = 5000
Config.BATCH_SIZE = 64

# 创建训练器
trainer = Trainer()
trainer.run_training()
```

### 3. 训练中断与恢复
- 训练过程中按 `Ctrl+C` 可以安全停止
- 模型会自动保存，可重新启动继续训练
- 检查点保存在 `models/` 目录下

### 4. 训练状态说明
- **Team A**: 玩家0和玩家2组成的队伍
- **Team B**: 玩家1和玩家3组成的队伍
- **奖励机制**: 击败对手获得正奖励，被击败获得负奖励
- **炸弹倍数**: 影响最终得分

## 监控与分析

### 1. TensorBoard 监控
```bash
# 启动TensorBoard
tensorboard --logdir=logs

# 在浏览器中打开 http://localhost:6006
```

监控指标包括：
- Loss: 网络损失值
- Reward: 团队奖励
- Win Rate: 胜率
- Legal Actions Count: 合法动作数量

### 2. 日志分析
日志文件位置：`logs/<timestamp>/`
- `training.log`: 训练过程详细日志
- `system.log`: 系统信息日志
- TensorBoard日志: 图形化监控数据

### 3. 模型评估
训练完成后，可以在 `models/` 目录找到保存的模型文件。

### 4. 性能指标
- **收敛性**: 观察损失值是否下降稳定
- **胜率**: 团队A vs 团队B的胜率变化
- **奖励**: 平均奖励值的变化趋势

## 故障排除

### 常见问题

#### 1. 内存不足
**症状**: 训练过程中出现内存错误
**解决方案**:
- 减少 `BATCH_SIZE`
- 减少 `MEMORY_CAPACITY`
- 关闭其他占用内存的应用

#### 2. 训练不收敛
**症状**: 损失值波动很大或不下降
**解决方案**:
- 降低学习率 (`LR`)
- 增加批次大小 (`BATCH_SIZE`)
- 调整探索率参数

#### 3. 依赖安装失败
**症状**: pip安装报错
**解决方案**:
```bash
# 清理缓存
pip cache purge
# 升级pip
pip install --upgrade pip
# 重新安装
pip install -r requirements.txt
```

#### 4. 环境导入错误
**症状**: `ImportError` 错误
**解决方案**:
- 确保在正确的虚拟环境中
- 重新安装依赖
- 检查Python版本

### 调试技巧
```bash
# 运行单个episode测试
python -c "
from environment import LandlordEnv2v2
env = LandlordEnv2v2(seed=42)
state = env.reset()
print('环境正常初始化')
"

# 检查网络是否正常工作
python -c "
import torch
from network import DouZeroNet
from config import Config
net = DouZeroNet(Config.STATE_SHAPE, 10)
dummy_input = torch.randn(1, *Config.STATE_SHAPE)
output = net(dummy_input)
print(f'网络前向传播成功，输出形状: {output.shape}')
"
```

## 进阶使用

### 1. 自定义智能体
```python
# 创建自定义智能体
from agent import DQNAgent

class CustomAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, device="cpu"):
        super().__init__(state_dim, action_dim, device)
        # 自定义初始化逻辑
    
    def custom_action_selection(self, state, legal_actions, epsilon):
        # 实现自定义动作选择策略
        pass
```

### 2. 调整奖励函数
修改 `environment.py` 中的奖励计算逻辑来改变训练目标。

### 3. 更改网络架构
在 `network.py` 中修改CNN+GRU架构以适应特定需求。

### 4. 数据分析脚本
创建分析脚本查看训练效果：
```python
# analyze_results.py
import os
import pandas as pd

def analyze_logs():
    log_dirs = [d for d in os.listdir('logs') if os.path.isdir(os.path.join('logs', d))]
    for log_dir in log_dirs:
        print(f"Log directory: {log_dir}")
        # 分析具体日志文件
```

### 5. 批量训练实验
```python
# experiment_runner.py
from config import Config
from trainer import Trainer

experiments = [
    {'LR': 1e-3, 'BATCH_SIZE': 32},
    {'LR': 5e-4, 'BATCH_SIZE': 64},
    {'LR': 1e-4, 'BATCH_SIZE': 128},
]

for exp_config in experiments:
    # 设置实验参数
    Config.LR = exp_config['LR']
    Config.BATCH_SIZE = exp_config['BATCH_SIZE']
    
    # 运行实验
    trainer = Trainer()
    trainer.run_training()
```

### 6. 性能优化建议
- 在支持CUDA的机器上使用GPU训练
- 适当调大批次大小以提高效率
- 使用多个环境并行训练（需额外开发）
- 定期清理旧的日志文件

---

## 技术支持

如遇问题，请参考：
- 项目文档 (`README.md`, `ENV.md`)
- 单元测试文件 (`test_happy_landlord.py`)
- 检查GitHub Issues
- 运行 `python -m unittest test_happy_landlord` 验证功能

## 版本更新

项目会持续优化改进，建议定期更新：
```bash
git pull origin main  # 如通过git克隆
```

---
*祝您使用愉快！如有疑问，请参阅相关文档或联系技术支持。*