import random
from collections import deque
from .config import Config

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class TeamMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.team_strategies = {}
    
    def add_experience(self, state, action, reward, next_state, done, team_id):
        self.buffer.append((state, action, reward, next_state, done, team_id))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, team_ids = zip(*batch)
        return states, actions, rewards, next_states, dones, team_ids
    
    def update_team_strategy(self, team_id, strategy_vector):
        self.team_strategies[team_id] = strategy_vector
    
    def get_team_strategy(self, team_id):
        return self.team_strategies.get(team_id, None)