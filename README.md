### 项目说明

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

### 运行说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动训练：
```bash
python main.py
```

3. 查看TensorBoard：
```bash
tensorboard --logdir=logs
```

4. 查看训练日志：
```bash
tail -f logs/<timestamp>/training.log
```

这个项目设计考虑了工业级实现需求，包括模块化设计、类型注解、详细日志和监控，以及高效的训练机制。TensorBoard和Loguru的结合提供了全面的训练过程可视化能力。