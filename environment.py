import numpy as np
import random
from collections import deque
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
from .config import Config

class CardType(Enum):
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    TRIPLE_WITH_SINGLE = 4
    TRIPLE_WITH_PAIR = 5
    STRAIGHT = 6
    CONSECUTIVE_PAIRS = 7
    AIRPLANE = 8
    AIRPLANE_WITH_SINGLES = 9
    AIRPLANE_WITH_PAIRS = 10
    FOUR_WITH_TWO_SINGLES = 11
    FOUR_WITH_TWO_PAIRS = 12
    BOMB = 13
    KING_BOMB = 14

class Card:
    RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    SUITS = ['S', 'H', 'C', 'D']
    JOKERS = ['BJ', 'CJ']
    
    def __init__(self, rank: str, suit: str = None):
        self.rank = rank
        self.suit = suit
        self.is_joker = suit is None
        
    def __str__(self):
        return self.rank if self.is_joker else f"{self.suit}{self.rank}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    @property
    def value(self) -> int:
        if self.is_joker:
            return 13 if self.rank == 'BJ' else 14
        return Card.RANKS.index(self.rank) + 6
    
    @staticmethod
    def from_value(value: int) -> 'Card':
        if value >= 13:
            return Card('CJ' if value == 14 else 'BJ')
        rank_idx = value - 6
        rank = Card.RANKS[rank_idx]
        suit = Card.SUITS[random.randint(0, 3)]
        return Card(rank, suit)

class CardGroup:
    def __init__(self, card_type: CardType, main_rank: int, cards: List[Card] = None):
        self.card_type = card_type
        self.main_rank = main_rank
        self.cards = cards or []
        
    def __str__(self):
        return f"{self.card_type.name} (Main: {self.main_rank})"
    
    def __len__(self):
        return len(self.cards)
    
    @property
    def strength(self) -> int:
        if self.card_type == CardType.BOMB:
            return self.main_rank * 10 + len(self.cards) * 100
        if self.card_type == CardType.KING_BOMB:
            return 1000
        return self.main_rank

class LandlordEnv2v2:
    def __init__(self, seed: int = None):
        self.config = Config()
        self.seed = seed
        self.reset()
        
    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        self.hands = [self.deck[i*21:(i+1)*21] for i in range(4)]
        for hand in self.hands:
            hand.sort(key=lambda c: c.value)
        
        self.first_team = random.choice([self.config.TEAM_A, self.config.TEAM_B])
        self.second_team = self.config.TEAM_B if self.first_team == self.config.TEAM_A else self.config.TEAM_A
        
        for player in self.second_team:
            discard_idx = random.randint(0, 20)
            self.hands[player].pop(discard_idx)
        
        self.current_player = self.first_team[0]
        self.last_move = None
        self.last_move_player = -1
        self.history = deque(maxlen=12)
        self.done = False
        self.winner = -1
        self.multiplier = 1
        self.bomb_used = False
        self.played_cards = {i: [] for i in range(4)}
        self.step_count = 0
        
        state = self._get_state()
        return state
    
    def _create_deck(self) -> List[Card]:
        deck = []
        for _ in range(2):
            for suit in Card.SUITS:
                for rank in Card.RANKS:
                    deck.append(Card(rank, suit))
            deck.append(Card('BJ'))
            deck.append(Card('CJ'))
        return deck
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise ValueError("Game has already ended")
        
        reward = 0.0
        if action == 0:
            self.history.append((self.current_player, "PASS"))
        else:
            card_group = self._action_to_card_group(action)
            if not self._is_valid_move(card_group):
                raise ValueError(f"Invalid move: {card_group}")
            
            for card in card_group.cards:
                if card in self.hands[self.current_player]:
                    self.hands[self.current_player].remove(card)
                else:
                    raise ValueError(f"Card {card} not in player's hand")
            
            self.last_move = card_group
            self.last_move_player = self.current_player
            self.history.append((self.current_player, str(card_group)))
            self.played_cards[self.current_player].extend(card_group.cards)
            
            if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:
                self.bomb_used = True
                self.multiplier *= 4
                reward += 0.5
            
            if self._is_helping_teammate():
                reward += 0.2
            
            if len(self.hands[self.current_player]) == 0:
                self.done = True
                team = 0 if self.current_player in self.config.TEAM_A else 1
                self.winner = team
                
                base_reward = 10.0
                final_reward = base_reward * self.multiplier
                if self._is_spring():
                    final_reward *= 2
                    self.multiplier *= 2
                
                reward += final_reward if team == 0 else -final_reward
                reward -= self._calculate_resource_penalty()
        
        reward -= self._calculate_resource_penalty() * 0.1
        
        self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS
        self.step_count += 1
        
        if self._is_round_end():
            self.last_move = None
            self.last_move_player = -1
        
        next_state = self._get_state()
        done = self.done
        
        info = {
            "current_player": self.current_player,
            "multiplier": self.multiplier,
            "winner": self.winner,
            "step": self.step_count
        }
        
        return next_state, reward, done, info
    
    def _is_valid_move(self, card_group: CardGroup) -> bool:
        if card_group.card_type == CardType.PASS:
            return True
        
        if self.last_move is None:
            return True
        
        if self.last_move.card_type == card_group.card_type:
            return card_group.strength > self.last_move.strength
        
        if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:
            return True
        
        return False
    
    def _action_to_card_group(self, action: int) -> CardGroup:
        hand = self.hands[self.current_player]
        
        if action == 0:
            return CardGroup(CardType.PASS, -1)
        
        if len(hand) >= 1:
            return CardGroup(CardType.SINGLE, hand[0].value, [hand[0]])
        
        for i in range(len(hand)-1):
            if hand[i].value == hand[i+1].value:
                return CardGroup(CardType.PAIR, hand[i].value, [hand[i], hand[i+1]])
        
        return CardGroup(CardType.PASS, -1)
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.config.STATE_SHAPE, dtype=np.float32)
        
        self._encode_hand(state[0], self.current_player)
        
        teammate = self._get_teammate(self.current_player)
        self._encode_hand(state[1], teammate)
        
        opp1 = self._get_opponent(self.current_player, 0)
        self._encode_played_cards(state[2], opp1)
        
        opp2 = self._get_opponent(self.current_player, 1)
        self._encode_played_cards(state[3], opp2)
        
        self._encode_history(state[4])
        
        self._encode_game_state(state[5])
        
        return state
    
    def _encode_hand(self, channel: np.ndarray, player: int):
        for card in self.hands[player]:
            value = card.value
            if value < 13:
                idx = value - 6
            else:
                idx = 10 if value == 13 else 11
            channel[0, min(idx, 14)] = 1
    
    def _encode_played_cards(self, channel: np.ndarray, player: int):
        for card in self.played_cards[player]:
            value = card.value
            if value < 13:
                idx = value - 6
            else:
                idx = 10 if value == 13 else 11
            channel[0, min(idx, 14)] += 1
    
    def _encode_history(self, channel: np.ndarray):
        for i, (player, move_str) in enumerate(list(self.history)[-5:]):
            channel[i, 0] = player
            channel[i, 1] = len(move_str)
    
    def _encode_game_state(self, channel: np.ndarray):
        channel[0, 0] = self.current_player
        channel[0, 1] = self.last_move_player
        channel[0, 2] = self.multiplier
        channel[0, 3] = 1 if self.bomb_used else 0
        
        for i in range(4):
            channel[1, i] = len(self.hands[i])
    
    def _get_teammate(self, player: int) -> int:
        team = self.config.TEAM_A if player in self.config.TEAM_A else self.config.TEAM_B
        return team[1] if team[0] == player else team[0]
    
    def _get_opponent(self, player: int, index: int) -> int:
        opponents = self.config.TEAM_B if player in self.config.TEAM_A else self.config.TEAM_A
        return opponents[index]
    
    def _is_round_end(self) -> bool:
        if len(self.history) < 3:
            return False
        last_three = list(self.history)[-3:]
        return all(move[1] == "PASS" for move in last_three)
    
    def _is_spring(self) -> bool:
        winner_team = self.config.TEAM_A if self.winner == 0 else self.config.TEAM_B
        for player in winner_team:
            if len(self.played_cards[player]) > 0:
                player_rounds = set()
                for i, (p, _) in enumerate(self.history):
                    if p == player:
                        player_rounds.add(i // 4)
                if len(player_rounds) > 1:
                    return False
        return True
    
    def _calculate_resource_penalty(self) -> float:
        penalty = 0.0
        for player in range(4):
            for card in self.hands[player]:
                if card.value in [13, 14]:
                    penalty += 0.1
                elif card.value == 12:
                    penalty += 0.05
        return penalty
    
    def _is_helping_teammate(self) -> bool:
        if self.last_move is None or self.last_move_player == self.current_player:
            return False
        if self.last_move_player == self._get_teammate(self.current_player):
            return True
        if len(self.hands[self.current_player]) > len(self.last_move.cards) + 2:
            return True
        return False