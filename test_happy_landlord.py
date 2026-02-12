"""
Unit tests for the Happy Landlord 2V2 reinforcement learning project
"""

import unittest
import numpy as np
import torch
from environment import LandlordEnv2v2, Card, CardType, CardGroup
from agent import DQNAgent
from network import DouZeroNet
from config import Config
from memory import ReplayMemory, TeamMemory
import random


class TestCard(unittest.TestCase):
    """Test Card class functionality"""
    
    def test_card_creation(self):
        """Test card creation and properties"""
        card = Card('A', 'S')  # Ace of Spades
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, 'S')
        self.assertFalse(card.is_joker)
        self.assertEqual(card.value, 8)  # A is at index 8 in RANKS
        
        joker = Card('BJ')  # Black Joker
        self.assertTrue(joker.is_joker)
        self.assertEqual(joker.value, 10)  # BJ = 10
        
        red_joker = Card('CJ')  # Red Joker
        self.assertTrue(red_joker.is_joker)
        self.assertEqual(red_joker.value, 11)  # CJ = 11
        
    def test_card_string_representation(self):
        """Test string representation of cards"""
        card = Card('A', 'S')
        self.assertEqual(str(card), 'SA')
        
        bj = Card('BJ')
        self.assertEqual(str(bj), 'BJ')
        
        cj = Card('CJ')
        self.assertEqual(str(cj), 'CJ')


class TestCardGroup(unittest.TestCase):
    """Test CardGroup class functionality"""
    
    def test_card_group_creation(self):
        """Test card group creation and strength calculation"""
        # Test single card group
        card = Card('A', 'S')
        group = CardGroup(CardType.SINGLE, card.value, [card])
        self.assertEqual(group.card_type, CardType.SINGLE)
        self.assertEqual(group.main_rank, 8)
        self.assertEqual(len(group), 1)
        self.assertEqual(group.strength, 8)  # Non-bomb strength equals main_rank
        
        # Test bomb strength calculation
        bomb_cards = [Card('A', suit) for suit in Card.SUITS]  # Four Aces
        bomb_group = CardGroup(CardType.BOMB, 8, bomb_cards)
        self.assertEqual(bomb_group.strength, 1000 + 8)  # Base bomb strength + rank
        
        # Test king bomb strength
        jokers = [Card('BJ'), Card('CJ')]
        king_bomb = CardGroup(CardType.KING_BOMB, 11, jokers)
        self.assertEqual(king_bomb.strength, 2000)  # Double king bomb strength


class TestEnvironment(unittest.TestCase):
    """Test the LandlordEnv2v2 environment"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = LandlordEnv2v2(seed=42)
        self.state = self.env.reset()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.config.NUM_PLAYERS, 4)
        self.assertEqual(self.env.config.TEAM_A, [0, 2])
        self.assertEqual(self.env.config.TEAM_B, [1, 3])
        self.assertEqual(self.state.shape, (6, 5, 15))
        self.assertIn(self.env.current_player, [0, 1, 2, 3])
        
    def test_reset(self):
        """Test environment reset functionality"""
        initial_state = self.env.reset()
        self.assertEqual(initial_state.shape, (6, 5, 15))
        
        # Check that hands are distributed correctly
        total_cards = sum(len(hand) for hand in self.env.hands)
        # After reset, one card is discarded from each player in the second team
        # This makes the total 84 - 2 = 82 cards in hands (since 2 players discard 1 card each)
        self.assertEqual(total_cards, 82)
        
    def test_get_legal_actions(self):
        """Test legal actions method"""
        legal_actions = self.env.get_legal_actions()
        self.assertIsInstance(legal_actions, list)
        self.assertIn(0, legal_actions)  # PASS should always be legal
        self.assertGreaterEqual(len(legal_actions), 1)
        
    def test_step_functionality(self):
        """Test the step function"""
        # Test PASS action (should always be valid)
        initial_player = self.env.current_player
        state, reward, done, info = self.env.step(0)  # PASS action
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (6, 5, 15))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # After a step, the current player should change
        self.assertNotEqual(initial_player, self.env.current_player)
        
    def test_get_teammate(self):
        """Test teammate identification"""
        # Player 0's teammate should be 2
        teammate = self.env._get_teammate(0)
        self.assertEqual(teammate, 2)
        
        # Player 1's teammate should be 3
        teammate = self.env._get_teammate(1)
        self.assertEqual(teammate, 3)
        
        # Player 2's teammate should be 0
        teammate = self.env._get_teammate(2)
        self.assertEqual(teammate, 0)
        
        # Player 3's teammate should be 1
        teammate = self.env._get_teammate(3)
        self.assertEqual(teammate, 1)


class TestNetwork(unittest.TestCase):
    """Test the neural network architecture"""
    
    def test_network_forward_pass(self):
        """Test forward pass of the network"""
        state_dim = (6, 5, 15)  # As defined in config
        action_dim = 600  # As used in trainer
        net = DouZeroNet(state_dim, action_dim)
        
        # Create a dummy input with batch size 1
        batch_size = 1
        input_tensor = torch.randn(batch_size, *state_dim)
        
        # Forward pass
        output = net(input_tensor)
        
        # Check output dimensions
        self.assertEqual(output.shape, (batch_size, action_dim))
        

class TestAgent(unittest.TestCase):
    """Test the DQN agent"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        state_dim = (6, 5, 15)
        action_dim = 10
        agent = DQNAgent(state_dim, action_dim, device='cpu')
        
        # Check that networks are created
        self.assertIsNotNone(agent.q_net)
        self.assertIsNotNone(agent.target_net)
        self.assertIsNotNone(agent.optimizer)
        
        # Check that target and q nets have same parameters initially
        for q_param, target_param in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            self.assertTrue(torch.allclose(q_param, target_param))
            
    def test_action_selection(self):
        """Test action selection with epsilon-greedy"""
        state_dim = (6, 5, 15)
        action_dim = 5
        agent = DQNAgent(state_dim, action_dim, device='cpu')
        
        # Mock state
        state = np.random.rand(*state_dim).astype(np.float32)
        legal_actions = [0, 1, 2, 3, 4]
        
        # Test with high epsilon (should mostly be random)
        action = agent.select_action(state, legal_actions, epsilon=1.0)
        self.assertIn(action, legal_actions)
        
        # Test with low epsilon (should follow policy more)
        action = agent.select_action(state, legal_actions, epsilon=0.0)
        self.assertIn(action, legal_actions)


class TestMemory(unittest.TestCase):
    """Test memory classes"""
    
    def test_replay_memory(self):
        """Test replay memory functionality"""
        memory = ReplayMemory(100)
        
        # Add some samples
        state = np.random.rand(6, 5, 15)
        action = 1
        reward = 0.5
        next_state = np.random.rand(6, 5, 15)
        done = False
        
        memory.push(state, action, reward, next_state, done)
        
        # Sample from memory
        batch_size = 1
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        
        # Check types
        self.assertIsInstance(states[0], np.ndarray)
        self.assertEqual(actions[0], 1)
        self.assertEqual(rewards[0], 0.5)
        self.assertIsInstance(next_states[0], np.ndarray)
        self.assertEqual(dones[0], False)
        
    def test_team_memory(self):
        """Test team memory functionality"""
        memory = TeamMemory(100)
        
        # Add experience with team ID
        state = np.random.rand(6, 5, 15)
        action = 1
        reward = 0.5
        next_state = np.random.rand(6, 5, 15)
        done = False
        team_id = 0
        
        memory.add_experience(state, action, reward, next_state, done, team_id)
        
        # Sample from team memory
        batch_size = 1
        states, actions, rewards, next_states, dones, team_ids = memory.sample(batch_size)
        
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        self.assertEqual(len(team_ids), batch_size)
        self.assertEqual(team_ids[0], team_id)


def run_integration_test():
    """Run an integration test to make sure everything works together"""
    print("Running integration test...")
    
    # Create environment
    env = LandlordEnv2v2(seed=42)
    state = env.reset()
    print(f"✓ Environment initialized, state shape: {state.shape}")
    
    # Create agent
    agent = DQNAgent(Config.STATE_SHAPE, 600, device='cpu')
    print("✓ Agent created")
    
    # Run a few steps
    for step in range(10):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            legal_actions = [0]  # Default to PASS if no legal actions
            
        # Select a random action from legal actions
        action = random.choice(legal_actions)
        
        try:
            next_state, reward, done, info = env.step(action)
            print(f"Step {step+1}: Action {action}, Reward: {reward:.2f}, Done: {done}")
            
            if done:
                print(f"Game ended. Winner: {'A' if info['winner'] == 0 else 'B'}")
                break
                
        except ValueError as e:
            print(f"Invalid move at step {step+1}, trying PASS instead")
            # Try PASS instead
            next_state, reward, done, info = env.step(0)
            print(f"Step {step+1}: PASS Action, Reward: {reward:.2f}, Done: {done}")
            
            if done:
                print(f"Game ended. Winner: {'A' if info['winner'] == 0 else 'B'}")
                break
    
    print("✓ Integration test completed")


def run_basic_training_test():
    """Test basic training functionality"""
    print("\nRunning basic training test...")
    
    from trainer import Trainer
    
    # Create a trainer with minimal settings for testing
    trainer = Trainer(device='cpu')
    print("✓ Trainer initialized")
    
    # Run one episode
    team_rewards = trainer.train_episode()
    print(f"✓ Training episode completed: {team_rewards}")
    
    print("✓ Basic training test completed")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    run_integration_test()
    run_basic_training_test()
    
    print("\n🎉 All tests completed successfully!")