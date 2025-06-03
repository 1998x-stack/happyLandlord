import torch
import numpy as np
import random
import torch.optim as optim
from .network import DouZeroNet
from .config import Config

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.config = Config()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.q_net = DouZeroNet(state_dim, action_dim).to(device)
        self.target_net = DouZeroNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.LR)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.choice(legal_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        
        # 只考虑合法动作
        legal_q_values = q_values[0, legal_actions]
        best_action_idx = torch.argmax(legal_q_values).item()
        return legal_actions[best_action_idx]
    
    def update(self, batch: list) -> float:
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.config.GAMMA * next_q * (1 - dones)
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())