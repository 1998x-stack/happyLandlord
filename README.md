# 腾讯欢乐斗地主2V2强化学习项目

这个完整的腾讯欢乐斗地主2V2强化学习项目包含：

1. **环境实现**：完全按照腾讯欢乐斗地主2V2规则实现游戏环境
2. **智能体架构**：
   - CNN+GRU神经网络处理状态
   - 分层动作空间设计
   - 团队协作机制（CTDE框架）
3. **训练系统**：
   - 团队经验回放池
   - 课程学习（三阶段难度）
   - 对抗训练
4. **监控与日志**：
   - TensorBoard记录训练指标（损失、奖励、胜率等）
   - Loguru记录详细训练日志
   - 内存和GPU使用监控
5. **可复现性**：
   - 随机种子设置
   - 模型定期保存

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Features](#features)

## Project Structure

```
happyLandlord/
├── main.py              # Main entry point
├── config.py            # Configuration parameters
├── agent.py             # DQN agent implementation
├── network.py           # Neural network architecture (CNN+GRU)
├── environment.py       # Game environment implementation
├── trainer.py           # Training system
├── memory.py            # Experience replay mechanism
├── utils.py             # Utility functions
├── test_happy_landlord.py # Unit tests
├── ENV.md               # Environment documentation
├── PROJECT_SUMMARY.md   # Project analysis summary
├── USER_GUIDE.md        # User guide
├── requirements.txt     # Dependency list
└── README.md            # Project documentation
```

## Core Components

### Network (`network.py`)
CNN+GRU network architecture designed specifically for landlord states:
- Convolutional layers extract spatial features
- GRU processes sequence information
- Fully connected layers output action values

### Agent (`agent.py`)
DQN agent with:
- Q-network and target network
- ε-greedy action selection
- Neural network update mechanism

### Environment (`environment.py`)
Complete Landlord 2V2 game environment:
- 84 cards (two decks, removing 3,4,5)
- 4 players divided into 2 teams (2 players each)
- Support for multiple card types: singles, pairs, triples, bombs, king bombs, etc.
- Rich action space (PASS and various play strategies)
- Complete game rules (spring, bomb multipliers, etc.)

### Trainer (`trainer.py`)
Training system with:
- Multi-agent coordinated training
- Team-based experience replay
- TensorBoard monitoring
- Model saving mechanism

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Agents
```bash
python main.py
```

### 2. Monitor Training Progress
```bash
tensorboard --logdir=logs
```

### 3. View Training Logs
```bash
tail -f logs/<timestamp>/training.log
```

### 4. Run Tests
```bash
python -m unittest test_happy_landlord
```

## Usage Examples

### Basic Training Session
```python
from trainer import Trainer

# Create and run trainer
trainer = Trainer()
trainer.run_training()
```

### Single Game Instance
```python
from environment import LandlordEnv2v2

# Create environment
env = LandlordEnv2v2(seed=42)
state = env.reset()

# Play a few steps
for step in range(10):
    legal_actions = env.get_legal_actions()
    action = 0  # PASS action
    next_state, reward, done, info = env.step(action)
    if done:
        break
```

## Configuration

Key parameters in `config.py`:
- `NUM_PLAYERS = 4` - Number of players in game
- `TEAM_A = [0, 2]`, `TEAM_B = [1, 3]` - Team assignments
- `STATE_SHAPE = (6, 5, 15)` - State tensor dimensions
- `GAMMA = 0.99` - Discount factor
- `LR = 1e-3` - Learning rate
- `BATCH_SIZE = 32` - Batch size for training
- `NUM_EPISODES = 10000` - Total training episodes

## Monitoring

- **TensorBoard**: Visualize training metrics (loss, rewards, win rates)
- **Log files**: Detailed training logs in `logs/<timestamp>/`
- **System logs**: Resource usage monitoring
- **Model checkpoints**: Saved periodically in `models/`

## Features

### State Representation
6-channel state tensor representing complete game state:
- Channel 0: Current player's hand
- Channel 1: Teammate's hand
- Channel 2: Opponent 1's played cards
- Channel 3: Opponent 2's played cards
- Channel 4: History (last 5 steps)
- Channel 5: Game state (current player, multiplier, bomb usage, etc.)

### Action Space
- 0: PASS (don't play)
- 1-20: Various play strategies (singles, pairs, bombs, strategic moves, etc.)

### Reward Design
- Bomb rewards: Four bombs +0.3, other bombs +0.5
- Teammate cooperation: +0.2
- Final reward: ±(10×multiplier×spring_factor)
- Resource penalty: Prevents hoarding high-value cards

### Design Highlights

1. **Team Collaboration**: Agents coordinate strategies with teammates
2. **Hierarchical Action Space**: Actions divided into basic card types and strategic moves
3. **Complete Rule Implementation**: Accurate simulation of all Happy Landlord 2V2 rules
4. **Efficient State Encoding**: CNN+GRU network effectively processes high-dimensional states
5. **Refined Reward Design**: Multi-level reward system promotes intelligent strategy learning

This project considers industrial-grade implementation requirements, including modular design, type annotations, detailed logging and monitoring, and efficient training mechanisms. The combination of TensorBoard and Loguru provides comprehensive visualization of the training process.

---

## License
This project is open-source and available under the MIT License.