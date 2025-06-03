import torch
import numpy as np
from collections import defaultdict
from loguru import logger
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from environment import LandlordEnv2v2
from agent import DQNAgent
from memory import TeamMemory
from config import Config

class Trainer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.config = Config()
        self.device = device
        
        # 初始化环境
        self.env = LandlordEnv2v2()
        
        # 初始化智能体
        self.agents = [
            DQNAgent(self.config.STATE_SHAPE, 600, self.device) for _ in range(4)
        ]
        
        # 团队记忆
        self.memory = TeamMemory(self.config.MEMORY_CAPACITY)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.TENSORBOARD_DIR)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.epsilon = self.config.EPSILON_START
        
        logger.add(f"{self.config.LOG_DIR}/training.log", rotation="500 MB")
        logger.info("训练系统初始化完成")
        logger.info(f"使用设备: {device}")
        logger.info(f"TensorBoard 日志目录: {self.config.TENSORBOARD_DIR}")
    
    def get_epsilon(self):
        """指数衰减的ε值"""
        epsilon = self.config.EPSILON_END + (self.config.EPSILON_START - self.config.EPSILON_END) * \
                  np.exp(-1. * self.step_count / self.config.EPSILON_DECAY)
        return epsilon
    
    def train_episode(self):
        state = self.env.reset()
        done = False
        total_rewards = defaultdict(float)
        episode_log = []
        
        while not done:
            current_player = self.env.current_player
            epsilon = self.get_epsilon()
            
            # 获取当前状态的合法动作
            legal_actions = self.env.get_legal_actions()
            if not legal_actions:
                logger.warning(f"玩家 {current_player} 没有合法动作，强制PASS")
                legal_actions = [0]  # 0 表示PASS
            
            # 智能体选择动作
            action = self.agents[current_player].select_action(
                state, legal_actions, epsilon=epsilon)
            
            # 记录调试信息
            logger.debug(f"Player {current_player} selecting action {action} from {len(legal_actions)} legal actions")
            try:
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
            except ValueError as e:
                # 捕获非法动作异常
                logger.error(f"非法动作错误: {str(e)}")
                logger.error(f"当前玩家: {current_player}, 手牌数量: {len(self.env.hands[current_player])}")
                logger.error(f"上家玩家: {self.env.last_move_player}, 上家出牌: {self.env.last_move}")
                logger.error(f"选择的动作: {action}, 合法动作列表: {legal_actions}")
                logger.error("强制执行PASS动作")
                next_state, reward, done, info = self.env.step(0)  # 强制PASS
                reward = -1.0  # 给予更大的负奖励以惩罚非法动作
        
            self.step_count += 1
            
            # 记录经验
            team_id = 0 if current_player in self.config.TEAM_A else 1
            self.memory.add_experience(
                state, action, reward, next_state, done, team_id)
            
            # 记录奖励
            total_rewards[current_player] += reward
            
            # 记录步骤日志
            episode_log.append({
                "player": current_player,
                "action": action,
                "reward": reward,
                "epsilon": epsilon,
                "step": self.step_count,
                "legal_actions_count": len(legal_actions)
            })
            
            # 更新状态
            state = next_state
            
            # 团队间策略传递
            if current_player % 2 == 0:  # 队伍A的玩家0或队伍B的玩家1
                teammate = self.env._get_teammate(current_player)
                self.agents[teammate].q_net.load_state_dict(
                    self.agents[current_player].q_net.state_dict())
            
            # 训练步骤
            if len(self.memory.buffer) > self.config.BATCH_SIZE:
                loss = self.train_step()
                self.writer.add_scalar('loss', loss, self.step_count)
                self.writer.add_scalar('legal_actions_count', len(legal_actions), self.step_count)
        
        # 记录团队奖励
        team_rewards = {
            "A": total_rewards[0] + total_rewards[2],
            "B": total_rewards[1] + total_rewards[3]
        }
        
        # 记录TensorBoard
        self.writer.add_scalar('reward/team_A', team_rewards["A"], self.episode_count)
        self.writer.add_scalar('reward/team_B', team_rewards["B"], self.episode_count)
        self.writer.add_scalar('epsilon', epsilon, self.episode_count)
        
        # 记录胜率
        winner = info['winner']
        win_A = 1 if winner == 0 else 0
        win_B = 1 if winner == 1 else 0
        self.writer.add_scalar('win_rate/team_A', win_A, self.episode_count)
        self.writer.add_scalar('win_rate/team_B', win_B, self.episode_count)
        
        # 记录日志
        logger.info(f"Episode {self.episode_count} | "
                   f"Team A: {team_rewards['A']:.2f} | "
                   f"Team B: {team_rewards['B']:.2f} | "
                   f"Winner: {'A' if winner == 0 else 'B'} | "
                   f"Steps: {self.env.step_count} | "
                   f"Multiplier: {self.env.multiplier}")
        
        self.episode_count += 1
        return team_rewards
    
    def train_step(self):
        """执行训练步骤"""
        batch = self.memory.sample(self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones, team_ids = batch
        
        losses = []
        for team_id in [0, 1]:
            team_indices = [i for i, tid in enumerate(team_ids) if tid == team_id]
            if not team_indices:
                continue
                
            team_batch = [
                (states[i], actions[i], rewards[i], next_states[i], dones[i])
                for i in team_indices
            ]
            
            # 更新对应团队的智能体
            agent_idx = self.config.TEAM_A[0] if team_id == 0 else self.config.TEAM_B[0]
            loss = self.agents[agent_idx].update(team_batch)
            losses.append(loss)
            
            # 更新团队策略
            self.memory.update_team_strategy(team_id, 
                self.agents[agent_idx].q_net.state_dict())
        
        # 定期更新目标网络
        if self.step_count % self.config.TARGET_UPDATE == 0:
            for agent in self.agents:
                agent.update_target_net()
        
        return np.mean(losses) if losses else 0.0
    
    def save_models(self):
        """保存所有智能体模型"""
        for i, agent in enumerate(self.agents):
            model_path = f"{self.config.MODEL_DIR}/agent_{i}_episode_{self.episode_count}.pth"
            agent.save(model_path)
        logger.info(f"模型已保存至 {self.config.MODEL_DIR}")
    
    def run_training(self):
        """运行训练循环"""
        logger.info("开始训练...")
        
        for ep in range(self.config.NUM_EPISODES):
            self.train_episode()
            
            if (ep + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_models()
        
        self.writer.close()
        logger.info("训练完成!")