# 🎯 腾讯欢乐斗地主2V2强化学习项目
*Mastering Multi-Agent Coordination with Deep Reinforcement Learning*

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-deep--learning-orange.svg)](https://pytorch.org/)
[![Reinforcement Learning](https://img.shields.io/badge/reinforcement--learning-green.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Multi-Agent](https://img.shields.io/badge/multi--agent-coordination-red.svg)](https://en.wikipedia.org/wiki/Multi-agent_system)

**🎯 Advanced Reinforcement Learning for Chinese Landlord Game**  
**👥 Multi-Agent Team Coordination Framework**  
**🧠 CNN+GRU Deep Neural Networks**

</div>

---

## 🌟 Project Overview

This is an **advanced reinforcement learning implementation** for Tencent's Happy Landlord 2V2, featuring cutting-edge multi-agent coordination and deep learning techniques. Perfect for researchers and developers interested in:

- 🤖 **Deep Q-Network (DQN)** applications
- 👥 **Multi-Agent Reinforcement Learning** (MARL)
- 🧠 **Team Coordination** algorithms
- 🎮 **Complex Game AI** development
- 📊 **Real-time Performance Monitoring**

### 🎯 Key Features

| Feature | Description | Technical Details |
|---------|-------------|-------------------|
| **CNN+GRU Network** | State-of-the-art architecture | Spatial + Sequential feature extraction |
| **Team Coordination** | CTDE framework | Centralized Training, Decentralized Execution |
| **Rich Action Space** | Hierarchical action design | 20+ strategic play options |
| **Complete Ruleset** | Accurate game simulation | Spring detection, bomb mechanics |
| **Advanced Training** | Curriculum learning | Three-phase difficulty progression |
| **Real-time Monitoring** | TensorBoard integration | Loss, reward, win-rate tracking |

---

## 🚀 Quick Demo

```python
from trainer import Trainer

# Initialize advanced multi-agent trainer
trainer = Trainer()

# Start sophisticated training process
trainer.run_training()

# Monitor in real-time with TensorBoard
# tensorboard --logdir=logs
```

---

## 📊 Training Visualization

Monitor your AI's progress with comprehensive metrics:
- 📈 **Loss Functions**: Network convergence tracking
- 💰 **Rewards**: Team performance indicators  
- 🏆 **Win Rates**: Agent effectiveness metrics
- 🧠 **Action Distribution**: Strategy learning patterns
- 💾 **Resource Usage**: GPU/Memory utilization

---

## 🏆 Multi-Agent Architecture

### Team-Based Coordination
- **Team A**: Agents 0 & 2 (Coordinated strategy)
- **Team B**: Agents 1 & 3 (Competitive play)
- **Communication**: Implicit through shared training signals
- **Strategy**: Collaborative bomb usage, coordinated attacks

### Network Architecture
```python
# CNN+GRU for Complex State Processing
Conv Layers → GRU Sequence Processing → FC Action Values
    ↓              ↓                    ↓
Spatial     Sequential          Action
Features    Patterns           Probabilities
```

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version
```

### Quick Setup
```bash
# Clone and install dependencies
git clone <repo-url>
cd happyLandlord
pip install -r requirements.txt

# Start training immediately
python main.py
```

### Advanced Configuration
```bash
# Monitor training in real-time
tensorboard --logdir=logs

# View detailed logs
tail -f logs/*/training.log
```

---

## 🧪 Experiments & Research

### Ready-to-Use Experiments
```python
# Multi-team competition study
from trainer import Trainer

# Configure different learning rates
trainer = Trainer()
trainer.config.LR = 1e-4  # Lower learning rate

# Adjust team coordination weights
trainer.config.TEAM_COOP_WEIGHT = 0.8

# Run comparative studies
trainer.run_training()
```

### Research Applications
- **Team Coordination**: Study agent collaboration strategies
- **Transfer Learning**: Apply learned strategies to similar games
- **Curriculum Learning**: Progressive difficulty adaptation
- **Multi-Agent Competition**: Team vs team dynamics

---

## 🎓 Learning Path for RL Researchers

### Beginner → Intermediate → Expert
1. **Start** with basic DQN implementation
2. **Progress** to multi-agent coordination
3. **Master** team-based strategy learning
4. **Advance** to complex game AI research

### Key Concepts Practiced
- Deep Q-Networks (DQN)
- Experience Replay
- Target Networks
- Multi-Agent RL
- Team Coordination (CTDE)
- Curriculum Learning
- Reward Engineering
- State Representation Learning

---

## 📚 Educational Value

### Perfect for:
- **University Courses**: Reinforcement Learning, Game AI
- **Research Projects**: Multi-agent systems, team coordination
- **Industry Applications**: Game AI, strategic decision making
- **Personal Learning**: Deep RL concepts, practical implementation

### What You'll Learn:
- 🧠 Building complex neural networks (CNN+GRU)
- 👥 Managing multi-agent interactions
- 🎯 Designing reward functions for team games
- 📊 Real-time performance monitoring
- 🔧 Debugging and optimizing RL algorithms

---

## 🏗️ Project Structure

```
happyLandlord/
├── 🧠 network.py        # CNN+GRU neural architecture
├── 🤖 agent.py          # DQN agent with team coordination
├── 🎮 environment.py    # Complete Landlord 2V2 simulation  
├── 🚀 trainer.py        # Multi-agent training system
├── 📊 memory.py         # Team-based experience replay
├── 📈 main.py           # Training entry point
├── ⚙️ config.py         # Hyperparameters & settings
├── 🧪 test_happy_landlord.py  # Comprehensive tests
└── 📄 docs/            # Detailed documentation
```

---

## 📊 Performance Metrics

### Expected Training Outcomes
- **Convergence**: 1000-5000 episodes for stable play
- **Win Rate**: >60% for trained teams after 10k episodes  
- **Learning Speed**: Rapid improvement in first 2k episodes
- **Strategy Emergence**: Team coordination develops naturally

### Monitoring Capabilities
- Real-time loss tracking
- Win rate progression
- Action selection patterns
- Team coordination metrics
- Resource utilization statistics

---

## 🎯 For Researchers & Students

### Research Opportunities
- **Coordination Mechanisms**: Study different team communication protocols
- **Transfer Learning**: Apply to other card games
- **Opponent Modeling**: Develop adaptive strategies
- **Scalability**: Extend to more complex multi-agent scenarios

### Educational Benefits
- Hands-on experience with DQN
- Understanding of multi-agent challenges
- Practical implementation of RL theory
- Exposure to complex game AI concepts

---

## 🚀 Getting Started

Ready to dive into advanced reinforcement learning? 

1. **Clone** the repository
2. **Install** dependencies (`pip install -r requirements.txt`)  
3. **Run** training (`python main.py`)
4. **Monitor** progress (`tensorboard --logdir=logs`)
5. **Experiment** with different configurations

### Quick Commands
```bash
# Start training immediately
python main.py

# Monitor your AI learning
tensorboard --logdir=logs

# Run tests to verify functionality
python -m unittest test_happy_landlord
```

---

<div align="center">

## 🌟 Start Your Reinforcement Learning Journey Today!

**Perfect for:** Researchers • Students • Game AI Developers • ML Enthusiasts

[Get Started Now](#installation) • [View Documentation](USER_GUIDE.md) • [Run Tests](#testing)

</div>

---

## 📜 License
This project is open-source and available under the MIT License. Perfect for academic and commercial use.