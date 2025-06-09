# -*- coding: utf-8 -*-
"""
腾讯欢乐斗地主2V2环境实现
优化点：
1. 重构出牌逻辑支持多种牌型
2. 完善炸弹规则和强度计算
3. 增强状态表示
4. 优化初始化丢牌策略
5. 修复春天判定逻辑
"""

import numpy as np
import random
from collections import deque
from enum import Enum
from typing import List, Tuple

class CardType(Enum):
    """卡牌类型枚举"""
    PASS = 0          # 不出
    SINGLE = 1        # 单张
    PAIR = 2          # 对子
    TRIPLE = 3        # 三张
    TRIPLE_WITH_SINGLE = 4   # 三带一
    TRIPLE_WITH_PAIR = 5     # 三带对
    STRAIGHT = 6      # 顺子
    CONSECUTIVE_PAIRS = 7    # 连对
    AIRPLANE = 8      # 飞机
    AIRPLANE_WITH_SINGLES = 9   # 飞机带单
    AIRPLANE_WITH_PAIRS = 10    # 飞机带对
    FOUR_WITH_TWO_SINGLES = 11  # 四带二单
    FOUR_WITH_TWO_PAIRS = 12    # 四带二对
    BOMB = 13         # 普通炸弹
    KING_BOMB = 14    # 王炸

class Card:
    """卡牌类，表示一张扑克牌"""
    RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']  # 卡牌点数
    SUITS = ['S', 'H', 'C', 'D']  # 卡牌花色（黑桃、红桃、梅花、方块）
    JOKERS = ['BJ', 'CJ']  # 大小王
    
    def __init__(self, rank: str, suit: str = None):
        """
        初始化卡牌
        
        Args:
            rank: 点数
            suit: 花色，None表示大小王
        """
        self.rank = rank
        self.suit = suit
        self.is_joker = suit is None  # 是否为大小王
        
    def __str__(self):
        """卡牌的字符串表示"""
        return self.rank if self.is_joker else f"{self.suit}{self.rank}"
    
    def __eq__(self, other):
        """判断两张卡牌是否相同"""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        """哈希函数，用于在字典中使用"""
        return hash((self.rank, self.suit))
    
    @property
    def value(self) -> int:
        """
        获取卡牌数值（值越大牌越大）
        
        牌值顺序: 6<7<8<9<10<J<Q<K<A<2<小王<大王
        """
        if self.is_joker:
            return 10 if self.rank == 'BJ' else 11  # 小王=10, 大王=11
        return Card.RANKS.index(self.rank)  # 非王牌的牌值等于其在RANKS中的索引
    
    @staticmethod
    def from_value(value: int) -> 'Card':
        """
        从数值创建卡牌
        
        Args:
            value: 卡牌数值
            
        Returns:
            对应的Card对象
        """
        if value >= 10:  # 大小王
            return Card('BJ') if value == 10 else Card('CJ')
        rank = Card.RANKS[value]
        suit = Card.SUITS[random.randint(0, 3)]
        return Card(rank, suit)

class CardGroup:
    """卡牌组合类，表示一组出牌"""
    def __init__(self, card_type: CardType, main_rank: int, cards: List[Card] = None):
        """
        初始化卡牌组合
        
        Args:
            card_type: 组合类型
            main_rank: 主牌点数（用于比较强度）
            cards: 包含的卡牌列表
        """
        self.card_type = card_type
        self.main_rank = main_rank
        self.cards = cards or []
        
    def __str__(self):
        """卡牌组合的字符串表示"""
        cards_str = " ".join(str(card) for card in self.cards)
        return f"{self.card_type.name} (Main: {self.main_rank}): [{cards_str}]"
    
    def __len__(self):
        """组合中卡牌的数量"""
        return len(self.cards)
    
    @property
    def strength(self) -> int:
        """
        获取组合强度（用于比较大小）
        
        炸弹强度计算规则:
          四炸: 1000 + 点数
          双王炸: 2000
          五炸: 3000 + 点数
          六炸: 4000 + 点数
          三王炸: 5000
          七炸: 6000 + 点数
          八炸: 7000 + 点数
          四王炸: 8000
        """
        # 王炸强度
        if self.card_type == CardType.KING_BOMB:
            num_kings = len(self.cards)
            return {
                2: 2000,  # 双王炸
                3: 5000,  # 三王炸
                4: 8000   # 四王炸
            }.get(num_kings, 2000)
        
        # 普通炸弹强度
        if self.card_type == CardType.BOMB:
            bomb_size = len(self.cards)
            base_strength = {
                4: 1000,  # 四炸
                5: 3000,  # 五炸
                6: 4000,  # 六炸
                7: 6000,  # 七炸
                8: 7000   # 八炸
            }.get(bomb_size, 1000)
            return base_strength + self.main_rank
        
        # 非炸弹牌型强度
        return self.main_rank

class Config:
    """游戏配置类"""
    def __init__(self):
        self.NUM_PLAYERS = 4  # 玩家数量
        self.TEAM_A = [0, 2]  # A队玩家ID
        self.TEAM_B = [1, 3]  # B队玩家ID
        # 通道0: 当前玩家手牌
        # 通道1: 队友手牌
        # 通道2: 对手1已出牌
        # 通道3: 对手2已出牌
        # 通道4: 历史记录（最近5步）
        # 通道5: 游戏状态（当前玩家、倍数、炸弹使用等）
        self.STATE_SHAPE = (6, 5, 15)  # 状态张量形状
class LandlordEnv2v2:
    """腾讯欢乐斗地主2V2环境实现"""
    def __init__(self, seed: int = None):
        """
        初始化环境
        
        Args:
            seed: 随机种子
        """
        self.config = Config()  # 加载配置
        self.seed = seed  # 设置随机种子
        self.reset()  # 重置环境
        
    def reset(self) -> np.ndarray:
        """
        重置游戏环境
        
        Returns:
            初始游戏状态
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.deck = self._create_deck()  # 创建牌组
        random.shuffle(self.deck)  # 洗牌
        
        # 发牌：每人21张
        self.hands = [self.deck[i*21:(i+1)*21] for i in range(4)]
        for hand in self.hands:
            hand.sort(key=lambda c: c.value)  # 按牌值排序
        
        # 随机选择先手队伍
        self.first_team = random.choice([self.config.TEAM_A, self.config.TEAM_B])
        self.second_team = self.config.TEAM_B if self.first_team == self.config.TEAM_A else self.config.TEAM_A
        
        # 后手队伍策略性丢牌：丢弃最小牌
        for player in self.second_team:
            min_card = min(self.hands[player], key=lambda c: c.value)
            self.hands[player].remove(min_card)
        
        # 初始化游戏状态
        self.current_player = self.first_team[0]  # 当前玩家
        self.last_move = None  # 最后出的牌组
        self.last_move_player = -1  # 最后出牌的玩家
        self.history = deque(maxlen=20)  # 历史记录
        self.done = False  # 游戏结束标志
        self.winner = -1  # 获胜队伍（0=A队，1=B队）
        self.multiplier = 1  # 游戏倍数
        self.bomb_used = False  # 炸弹使用标志
        self.played_cards = {i: [] for i in range(4)}  # 已出牌记录
        self.step_count = 0  # 步数计数器
        self.round_count = 0  # 回合计数器
        
        return self._get_state()  # 返回初始状态
    
    def _create_deck(self) -> List[Card]:
        """创建两副扑克牌（移除3,4,5）共84张"""
        deck = []
        for _ in range(2):  # 两副牌
            for suit in Card.SUITS:  # 四种花色
                for rank in Card.RANKS:  # 十种点数
                    deck.append(Card(rank, suit))
            deck.append(Card('BJ'))  # 小王
            deck.append(Card('CJ'))  # 大王
        return deck
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步游戏动作
        
        Args:
            action: 动作编号（0=不出，其他=出牌）
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("游戏已结束")
        
        reward = 0.0  # 即时奖励
        
        # 处理"不出"动作
        if action == 0:
            self.history.append((self.current_player, "PASS"))
            # 连续不出导致回合结束
            if self._is_round_end():
                self.last_move = None
                self.last_move_player = -1
                self.round_count += 1
        # 处理出牌动作
        else:
            card_group = self._action_to_card_group(action)
            if not self._is_valid_move(card_group):
                raise ValueError(f"无效出牌: {card_group}")
            
            # 从手牌中移除已出的牌
            for card in card_group.cards:
                self.hands[self.current_player].remove(card)
            
            # 更新游戏状态
            self.last_move = card_group
            self.last_move_player = self.current_player
            self.history.append((self.current_player, str(card_group)))
            self.played_cards[self.current_player].extend(card_group.cards)
            
            # 炸弹处理
            if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:
                self.bomb_used = True
                # 四炸×2，其他炸弹×4
                if card_group.card_type == CardType.BOMB and len(card_group.cards) == 4:
                    self.multiplier *= 2
                    reward += 0.3
                else:
                    self.multiplier *= 4
                    reward += 0.5
            
            # 队友配合奖励
            if self._is_helping_teammate():
                reward += 0.2
            
            # 检查游戏是否结束
            if len(self.hands[self.current_player]) == 0:
                self.done = True
                # 确定获胜队伍
                team = 0 if self.current_player in self.config.TEAM_A else 1
                self.winner = team
                
                # 计算最终奖励
                base_reward = 10.0
                final_reward = base_reward * self.multiplier
                if self._is_spring():
                    final_reward *= 2
                    self.multiplier *= 2
                
                # A队获胜得正分，B队获胜得负分
                reward += final_reward if team == 0 else -final_reward
                reward -= self._calculate_resource_penalty()
        
        # 资源惩罚（手牌中剩余大牌的惩罚）
        reward -= self._calculate_resource_penalty() * 0.1
        
        # 更新当前玩家
        self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS
        self.step_count += 1
        
        # 获取下一状态
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
        """
        检查出牌是否合法
        
        Args:
            card_group: 待出的牌组
            
        Returns:
            bool: 出牌是否合法
        """
        # PASS总是合法
        if card_group.card_type == CardType.PASS:
            return True
        
        # 首轮没有限制
        if self.last_move is None:
            return True
        
        # 上家是队友，允许任意出牌
        if self.last_move_player == self._get_teammate(self.current_player):
            return True
        
        # 上家是对手
        # 情况1：上家出的是炸弹（包括王炸）
        if self.last_move.card_type in [CardType.BOMB, CardType.KING_BOMB]:
            # 自己必须出炸弹且强度大于上家
            if card_group.card_type not in [CardType.BOMB, CardType.KING_BOMB]:
                return False
            return card_group.strength > self.last_move.strength
        
        # 情况2：上家出的是非炸弹
        # 炸弹可以压制非炸弹
        if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:
            return True
        
        # 同类型牌比较强度
        if self.last_move.card_type == card_group.card_type:
            return card_group.strength > self.last_move.strength
        
        return False
    
    def _action_to_card_group(self, action: int) -> CardGroup:
        """
        重新设计的智能出牌策略
        
        动作编号定义：
        0: 不出
        1-10: 根据当前局面智能选择最佳牌型
        11-20: 特定牌型策略（如炸弹、连牌等）
        """
        hand = self.hands[self.current_player]
        hand_values = [card.value for card in hand]
        hand_values.sort()
        
        # 0: 不出
        if action == 0:
            return CardGroup(CardType.PASS, -1)
        
        # 1: 最小单牌（基础策略）
        if action == 1:
            min_card = min(hand, key=lambda c: c.value)
            return CardGroup(CardType.SINGLE, min_card.value, [min_card])
        
        # 2: 最小对子
        if action == 2:
            for value in set(hand_values):
                if hand_values.count(value) >= 2:
                    pair_cards = [card for card in hand if card.value == value][:2]
                    return CardGroup(CardType.PAIR, value, pair_cards)
        
        # 3: 最小三张
        if action == 3:
            for value in set(hand_values):
                if hand_values.count(value) >= 3:
                    triple_cards = [card for card in hand if card.value == value][:3]
                    return CardGroup(CardType.TRIPLE, value, triple_cards)
        
        # 4: 三带一
        if action == 4:
            # 先找三张
            triple_value = None
            for value in set(hand_values):
                if hand_values.count(value) >= 3:
                    triple_value = value
                    break
            
            if triple_value:
                # 再找最小单牌（非三张牌的点数）
                single_value = min([v for v in set(hand_values) if v != triple_value])
                triple_cards = [card for card in hand if card.value == triple_value][:3]
                single_card = [card for card in hand if card.value == single_value][0]
                return CardGroup(CardType.TRIPLE_WITH_SINGLE, triple_value, triple_cards + [single_card])
        
        # 5: 三带对
        if action == 5:
            # 先找三张
            triple_value = None
            for value in set(hand_values):
                if hand_values.count(value) >= 3:
                    triple_value = value
                    break
            
            if triple_value:
                # 再找最小对子（非三张牌的点数）
                for value in set(hand_values):
                    if value != triple_value and hand_values.count(value) >= 2:
                        triple_cards = [card for card in hand if card.value == triple_value][:3]
                        pair_cards = [card for card in hand if card.value == value][:2]
                        return CardGroup(CardType.TRIPLE_WITH_PAIR, triple_value, triple_cards + pair_cards)
        
        # 6: 最小顺子（5张）
        if action == 6:
            # 寻找连续5张牌
            for start in range(len(hand_values) - 4):
                if hand_values[start+4] - hand_values[start] == 4:
                    # 确认连续
                    if all(hand_values[start+i] + 1 == hand_values[start+i+1] 
                        for i in range(4)):
                        # 取出这5张牌
                        straight_cards = []
                        for value in range(hand_values[start], hand_values[start]+5):
                            # 取最小花色的牌
                            cards_of_value = [c for c in hand if c.value == value]
                            cards_of_value.sort(key=lambda c: c.suit)
                            straight_cards.append(cards_of_value[0])
                        return CardGroup(CardType.STRAIGHT, hand_values[start], straight_cards)
        
        # 7: 最小连对（3连对）
        if action == 7:
            # 获取所有对子点数
            pair_values = [v for v in set(hand_values) if hand_values.count(v) >= 2]
            pair_values.sort()
            
            # 寻找连续3个对子
            for i in range(len(pair_values) - 2):
                if pair_values[i] + 2 == pair_values[i+2]:
                    # 取出这些对子
                    consecutive_pairs = []
                    for value in pair_values[i:i+3]:
                        cards_of_value = [c for c in hand if c.value == value][:2]
                        consecutive_pairs.extend(cards_of_value)
                    return CardGroup(CardType.CONSECUTIVE_PAIRS, pair_values[i], consecutive_pairs)
        
        # 8: 最小飞机（2个连续三张）
        if action == 8:
            # 获取所有三张点数
            triple_values = [v for v in set(hand_values) if hand_values.count(v) >= 3]
            triple_values.sort()
            
            # 寻找连续2个三张
            for i in range(len(triple_values) - 1):
                if triple_values[i] + 1 == triple_values[i+1]:
                    # 取出这些三张
                    airplane_cards = []
                    for value in triple_values[i:i+2]:
                        cards_of_value = [c for c in hand if c.value == value][:3]
                        airplane_cards.extend(cards_of_value)
                    return CardGroup(CardType.AIRPLANE, triple_values[i], airplane_cards)
        
        # 9: 炸弹（最小四张炸）
        if action == 9:
            for value in set(hand_values):
                if hand_values.count(value) >= 4:
                    bomb_cards = [card for card in hand if card.value == value][:4]
                    return CardGroup(CardType.BOMB, value, bomb_cards)
        
        # 10: 王炸（如果有）
        if action == 10:
            jokers = [card for card in hand if card.is_joker]
            if len(jokers) >= 2:
                return CardGroup(CardType.KING_BOMB, 11, jokers[:2])
        
        # 11: 顶牌策略（出比对手大的最小牌）
        if action == 11 and self.last_move:
            # 获取对手牌型
            opp_type = self.last_move.card_type
            opp_strength = self.last_move.strength
            
            # 寻找比对手大的最小牌
            for card in sorted(hand, key=lambda c: c.value):
                # 单牌比较
                if opp_type == CardType.SINGLE and card.value > opp_strength:
                    return CardGroup(CardType.SINGLE, card.value, [card])
                
                # 对子比较
                if opp_type == CardType.PAIR:
                    # 找同点数的对子
                    same_value_cards = [c for c in hand if c.value == card.value]
                    if len(same_value_cards) >= 2 and card.value > opp_strength:
                        return CardGroup(CardType.PAIR, card.value, same_value_cards[:2])
        
        # 12: 拆牌策略（拆大牌管小牌）
        if action == 12 and self.last_move:
            # 当无法直接压制时，考虑拆大牌
            if self.last_move.card_type == CardType.SINGLE:
                # 找最小的大于对手牌的单牌
                for card in sorted(hand, key=lambda c: c.value):
                    if card.value > self.last_move.main_rank:
                        return CardGroup(CardType.SINGLE, card.value, [card])
            
            # 拆对子压制单牌
            if self.last_move.card_type == CardType.SINGLE:
                # 寻找比对手牌大的最小对子
                for value in set(hand_values):
                    if value > self.last_move.main_rank and hand_values.count(value) >= 2:
                        pair_cards = [card for card in hand if card.value == value][:2]
                        # 拆出一张单牌
                        return CardGroup(CardType.SINGLE, value, [pair_cards[0]])
        
        # 13: 过牌策略（队友出牌后选择过牌）
        if action == 13 and self.last_move_player == self._get_teammate(self.current_player):
            return CardGroup(CardType.PASS, -1)
        
        # 14: 压牌策略（对手出牌后尽量压制）
        if action == 14 and self.last_move and self.last_move_player != self._get_teammate(self.current_player):
            # 尝试用同类型牌压制
            if self.last_move.card_type != CardType.PASS:
                # 寻找同类型但更大的牌
                for value in set(hand_values):
                    if value > self.last_move.main_rank:
                        # 单牌
                        if self.last_move.card_type == CardType.SINGLE:
                            same_value_cards = [c for c in hand if c.value == value]
                            if same_value_cards:
                                return CardGroup(CardType.SINGLE, value, [same_value_cards[0]])
                        
                        # 对子
                        if self.last_move.card_type == CardType.PAIR:
                            same_value_cards = [c for c in hand if c.value == value]
                            if len(same_value_cards) >= 2:
                                return CardGroup(CardType.PAIR, value, same_value_cards[:2])
        
        # 15: 保存实力策略（保留大牌和炸弹）
        if action == 15:
            # 找最小牌出，避免出大牌
            min_value = min(hand_values)
            min_cards = [card for card in hand if card.value == min_value]
            return CardGroup(CardType.SINGLE, min_value, [min_cards[0]])
        
        # 默认策略：出最小单牌
        min_card = min(hand, key=lambda c: c.value)
        return CardGroup(CardType.SINGLE, min_card.value, [min_card])

    
    def _get_state(self) -> np.ndarray:
        """
        获取当前游戏状态
        
        Returns:
            np.ndarray: 状态张量
        """
        state = np.zeros(self.config.STATE_SHAPE, dtype=np.float32)
        
        # 通道0: 当前玩家手牌
        self._encode_hand(state[0], self.current_player)
        
        # 通道1: 队友手牌
        teammate = self._get_teammate(self.current_player)
        self._encode_hand(state[1], teammate)
        
        # 通道2: 对手1已出牌
        opp1 = self._get_opponent(self.current_player, 0)
        self._encode_played_cards(state[2], opp1)
        
        # 通道3: 对手2已出牌
        opp2 = self._get_opponent(self.current_player, 1)
        self._encode_played_cards(state[3], opp2)
        
        # 通道4: 历史记录
        self._encode_history(state[4])
        
        # 通道5: 游戏状态
        self._encode_game_state(state[5])
        
        return state
    
    def _encode_hand(self, channel: np.ndarray, player: int):
        """编码玩家手牌到状态通道"""
        for card in self.hands[player]:
            channel[0, card.value] = 1
    
    def _encode_played_cards(self, channel: np.ndarray, player: int):
        """编码玩家已出牌到状态通道"""
        for card in self.played_cards[player]:
            channel[0, card.value] += 1
    
    def _encode_history(self, channel: np.ndarray):
        """编码历史记录到状态通道"""
        # 记录最近5步历史
        for i, (player, move_str) in enumerate(list(self.history)[-5:]):
            channel[i, 0] = player  # 玩家ID
            channel[i, 1] = len(move_str)  # 动作长度
    
    def _encode_game_state(self, channel: np.ndarray):
        """编码游戏状态到状态通道"""
        # 第0行: 基础状态
        channel[0, 0] = self.current_player  # 当前玩家
        channel[0, 1] = self.last_move_player  # 最后出牌玩家
        channel[0, 2] = self.multiplier  # 当前倍数
        channel[0, 3] = 1 if self.bomb_used else 0  # 是否使用过炸弹
        channel[0, 4] = self.last_move.card_type.value if self.last_move else 0  # 最后牌型
        channel[0, 5] = self.last_move.main_rank if self.last_move else -1  # 最后牌组主点数
        
        # 第1行: 玩家手牌数量
        for i in range(4):
            channel[1, i] = len(self.hands[i])
    
    def _get_teammate(self, player: int) -> int:
        """获取队友玩家ID"""
        team = self.config.TEAM_A if player in self.config.TEAM_A else self.config.TEAM_B
        return team[1] if team[0] == player else team[0]
    
    def _get_opponent(self, player: int, index: int) -> int:
        """获取对手玩家ID"""
        opponents = self.config.TEAM_B if player in self.config.TEAM_A else self.config.TEAM_A
        return opponents[index]
    
    def _is_round_end(self) -> bool:
        """检查回合是否结束（连续3个PASS）"""
        if len(self.history) < 3:
            return False
        last_three = list(self.history)[-3:]
        return all(move[1] == "PASS" for move in last_three)
    
    def _is_spring(self) -> bool:
        """检查是否是春天（对手未出牌或仅出一轮即败）"""
        if self.winner == -1:
            return False
        
        # 确定失败队伍
        losing_team = self.config.TEAM_B if self.winner == 0 else self.config.TEAM_A
        
        # 检查失败队伍出牌轮次
        losing_rounds = set()
        for i, (player, _) in enumerate(self.history):
            if player in losing_team:
                # 根据回合计数确定轮次
                losing_rounds.add(i // 4)
        
        # 失败队伍出牌轮次不超过1轮
        return len(losing_rounds) <= 1
    
    def _calculate_resource_penalty(self) -> float:
        """计算资源惩罚（手牌中剩余大牌的惩罚）"""
        penalty = 0.0
        for player in range(4):
            for card in self.hands[player]:
                if card.value in [10, 11]:  # 大小王
                    penalty += 0.1
                elif card.value == 9:  # 2
                    penalty += 0.05
        return penalty
    
    def _is_helping_teammate(self) -> bool:
        """检查出牌是否帮助队友"""
        if self.last_move is None or self.last_move_player == self.current_player:
            return False
        
        # 上家是队友
        if self.last_move_player == self._get_teammate(self.current_player):
            return True
        
        # 为队友创造接牌机会
        if len(self.hands[self.current_player]) > len(self.last_move.cards) + 2:
            return True
        
        return False

# 环境使用示例
if __name__ == "__main__":
    print("=== 腾讯欢乐斗地主2V2环境测试 ===")
    env = LandlordEnv2v2(seed=42)
    state = env.reset()
    print("游戏初始化完成")
    print(f"初始状态形状: {state.shape}")
    print(f"当前玩家: {env.current_player}")
    
    # 打印初始手牌
    for i, hand in enumerate(env.hands):
        print(f"玩家 {i} 手牌 ({len(hand)}张):")
        hand_str = " ".join(str(card) for card in sorted(hand, key=lambda c: c.value))
        print(hand_str)
    
    # 进行10步测试
    print("\n=== 开始测试游戏步骤 ===")
    for step in range(10):
        print(f"\n--- 步骤 {step+1} ---")
        print(f"当前玩家: {env.current_player}")
        
        # 随机选择动作 (0=不出, 1=出最小牌)
        action = random.choice([0, 1])
        print(f"选择动作: {'PASS' if action == 0 else '出牌'}")
        
        state, reward, done, info = env.step(action)
        print(f"奖励: {reward:.2f}, 结束: {done}, 倍数: {env.multiplier}")
        
        if done:
            winner_team = "A队" if info['winner'] == 0 else "B队"
            print(f"游戏结束! 获胜队伍: {winner_team}")
            break