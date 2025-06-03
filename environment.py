# 导入必要的库
import numpy as np  # 导入numpy用于数值计算
import random  # 导入random用于随机数生成
from collections import deque  # 导入deque用于实现双端队列
from enum import Enum  # 导入Enum用于创建枚举类型
from typing import List, Tuple, Dict, Optional, Any  # 导入类型提示
from config import Config  # 导入配置类

class CardType(Enum):  # 定义卡牌类型枚举
    PASS = 0  # 不出
    SINGLE = 1  # 单张
    PAIR = 2  # 对子
    TRIPLE = 3  # 三张
    TRIPLE_WITH_SINGLE = 4  # 三带一
    TRIPLE_WITH_PAIR = 5  # 三带对
    STRAIGHT = 6  # 顺子
    CONSECUTIVE_PAIRS = 7  # 连对
    AIRPLANE = 8  # 飞机
    AIRPLANE_WITH_SINGLES = 9  # 飞机带单
    AIRPLANE_WITH_PAIRS = 10  # 飞机带对
    FOUR_WITH_TWO_SINGLES = 11  # 四带二单
    FOUR_WITH_TWO_PAIRS = 12  # 四带二对
    BOMB = 13  # 炸弹
    KING_BOMB = 14  # 王炸

class Card:  # 定义卡牌类
    RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']  # 定义卡牌点数
    SUITS = ['S', 'H', 'C', 'D']  # 定义卡牌花色（黑桃、红桃、梅花、方块）
    JOKERS = ['BJ', 'CJ']  # 定义大小王
    
    def __init__(self, rank: str, suit: str = None):  # 初始化卡牌
        self.rank = rank  # 设置点数
        self.suit = suit  # 设置花色
        self.is_joker = suit is None  # 判断是否为大小王
        
    def __str__(self):  # 字符串表示
        return self.rank if self.is_joker else f"{self.suit}{self.rank}"  # 返回卡牌字符串表示
    
    def __eq__(self, other):  # 相等比较
        return self.rank == other.rank and self.suit == other.suit  # 比较两张卡牌是否相同
    
    @property
    def value(self) -> int:  # 获取卡牌值
        if self.is_joker:  # 如果是大小王
            return 13 if self.rank == 'BJ' else 14  # 返回大小王的值
        return Card.RANKS.index(self.rank) + 6  # 返回普通卡牌的值
    
    @staticmethod
    def from_value(value: int) -> 'Card':  # 从值创建卡牌
        if value >= 13:  # 如果是大小王
            return Card('CJ' if value == 14 else 'BJ')  # 返回对应的大小王
        rank_idx = value - 6  # 计算点数索引
        rank = Card.RANKS[rank_idx]  # 获取点数
        suit = Card.SUITS[random.randint(0, 3)]  # 随机选择花色
        return Card(rank, suit)  # 返回新卡牌

class CardGroup:  # 定义卡牌组合类
    def __init__(self, card_type: CardType, main_rank: int, cards: List[Card] = None):  # 初始化卡牌组合
        self.card_type = card_type  # 设置组合类型
        self.main_rank = main_rank  # 设置主牌点数
        self.cards = cards or []  # 设置卡牌列表
        
    def __str__(self):  # 字符串表示
        return f"{self.card_type.name} (Main: {self.main_rank})"  # 返回组合字符串表示
    
    def __len__(self):  # 获取长度
        return len(self.cards)  # 返回卡牌数量
    
    @property
    def strength(self) -> int:  # 获取组合强度
        if self.card_type == CardType.BOMB:  # 如果是炸弹
            return self.main_rank * 10 + len(self.cards) * 100  # 计算炸弹强度
        if self.card_type == CardType.KING_BOMB:  # 如果是王炸
            return 1000  # 返回王炸强度
        return self.main_rank  # 返回普通组合强度

class LandlordEnv2v2:  # 定义斗地主环境类
    def __init__(self, seed: int = None):  # 初始化环境
        self.config = Config()  # 加载配置
        self.seed = seed  # 设置随机种子
        self.reset()  # 重置环境
        
    def reset(self):  # 重置环境
        if self.seed is not None:  # 如果设置了种子
            random.seed(self.seed)  # 设置随机种子
            np.random.seed(self.seed)  # 设置numpy随机种子
        
        self.deck = self._create_deck()  # 创建牌组
        random.shuffle(self.deck)  # 洗牌
        
        self.hands = [self.deck[i*21:(i+1)*21] for i in range(4)]  # 发牌
        for hand in self.hands:  # 对每个玩家的手牌
            hand.sort(key=lambda c: c.value)  # 按值排序
        
        self.first_team = random.choice([self.config.TEAM_A, self.config.TEAM_B])  # 随机选择先手队伍
        self.second_team = self.config.TEAM_B if self.first_team == self.config.TEAM_A else self.config.TEAM_A  # 确定后手队伍
        
        for player in self.second_team:  # 对后手队伍的玩家
            discard_idx = random.randint(0, 20)  # 随机选择一张牌
            self.hands[player].pop(discard_idx)  # 移除该牌
        
        self.current_player = self.first_team[0]  # 设置当前玩家
        self.last_move = None  # 初始化最后一步
        self.last_move_player = -1  # 初始化最后一步的玩家
        self.history = deque(maxlen=12)  # 初始化历史记录
        self.done = False  # 初始化游戏结束标志
        self.winner = -1  # 初始化获胜者
        self.multiplier = 1  # 初始化倍数
        self.bomb_used = False  # 初始化炸弹使用标志
        self.played_cards = {i: [] for i in range(4)}  # 初始化已出牌记录
        self.step_count = 0  # 初始化步数计数
        
        state = self._get_state()  # 获取初始状态
        return state  # 返回初始状态
    
    def _create_deck(self) -> List[Card]:  # 创建牌组
        deck = []  # 初始化牌组
        for _ in range(2):  # 创建两副牌
            for suit in Card.SUITS:  # 遍历花色
                for rank in Card.RANKS:  # 遍历点数
                    deck.append(Card(rank, suit))  # 添加卡牌
            deck.append(Card('BJ'))  # 添加小王
            deck.append(Card('CJ'))  # 添加大王
        return deck  # 返回牌组
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:  # 执行一步
        if self.done:  # 如果游戏已结束
            raise ValueError("Game has already ended")  # 抛出错误
        
        reward = 0.0  # 初始化奖励
        if action == 0:  # 如果选择不出
            self.history.append((self.current_player, "PASS"))  # 记录历史
        else:  # 如果选择出牌
            card_group = self._action_to_card_group(action)  # 获取卡牌组合
            if not self._is_valid_move(card_group):  # 如果移动无效
                raise ValueError(f"Invalid move: {card_group}")  # 抛出错误
            
            for card in card_group.cards:  # 遍历卡牌
                if card in self.hands[self.current_player]:  # 如果卡牌在手牌中
                    self.hands[self.current_player].remove(card)  # 移除卡牌
                else:  # 如果卡牌不在手牌中
                    raise ValueError(f"Card {card} not in player's hand")  # 抛出错误
            
            self.last_move = card_group  # 更新最后一步
            self.last_move_player = self.current_player  # 更新最后一步的玩家
            self.history.append((self.current_player, str(card_group)))  # 记录历史
            self.played_cards[self.current_player].extend(card_group.cards)  # 更新已出牌记录
            
            if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:  # 如果使用了炸弹
                self.bomb_used = True  # 设置炸弹使用标志
                self.multiplier *= 4  # 更新倍数
                reward += 0.5  # 增加奖励
            
            if self._is_helping_teammate():  # 如果帮助队友
                reward += 0.2  # 增加奖励
            
            if len(self.hands[self.current_player]) == 0:  # 如果手牌为空
                self.done = True  # 设置游戏结束
                team = 0 if self.current_player in self.config.TEAM_A else 1  # 确定获胜队伍
                self.winner = team  # 设置获胜者
                
                base_reward = 10.0  # 设置基础奖励
                final_reward = base_reward * self.multiplier  # 计算最终奖励
                if self._is_spring():  # 如果是春天
                    final_reward *= 2  # 加倍奖励
                    self.multiplier *= 2  # 加倍倍数
                
                reward += final_reward if team == 0 else -final_reward  # 更新奖励
                reward -= self._calculate_resource_penalty()  # 减去资源惩罚
        
        reward -= self._calculate_resource_penalty() * 0.1  # 减去资源惩罚
        
        self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS  # 更新当前玩家
        self.step_count += 1  # 更新步数
        
        if self._is_round_end():  # 如果回合结束
            self.last_move = None  # 重置最后一步
            self.last_move_player = -1  # 重置最后一步的玩家
        
        next_state = self._get_state()  # 获取下一个状态
        done = self.done  # 获取游戏结束标志
        
        info = {  # 创建信息字典
            "current_player": self.current_player,  # 当前玩家
            "multiplier": self.multiplier,  # 倍数
            "winner": self.winner,  # 获胜者
            "step": self.step_count  # 步数
        }
        
        return next_state, reward, done, info  # 返回结果
        
    def _is_valid_move(self, card_group: CardGroup) -> bool:
        # PASS 总是合法
        if card_group.card_type == CardType.PASS:
            return True
        
        # 首轮没有限制
        if self.last_move is None:
            return True
        
        # 如果上家是队友，允许出任意牌
        if self.last_move_player == self._get_teammate(self.current_player):
            return True
        
        # 压制上家牌型
        if self.last_move.card_type == card_group.card_type:
            return card_group.strength > self.last_move.strength
        
        # 炸弹可以压制任何非炸弹牌型
        if card_group.card_type in [CardType.BOMB, CardType.KING_BOMB]:
            return True
        
        return False
    
    def _action_to_card_group(self, action: int) -> CardGroup:  # 将动作转换为卡牌组合
        hand = self.hands[self.current_player]
        
        if action == 0:
            return CardGroup(CardType.PASS, -1)
        
        # 确保总是能出牌（即使无法压制）
        if len(hand) >= 1:
            # 尝试出最小牌（保证合法）
            min_card = min(hand, key=lambda c: c.value)
            return CardGroup(CardType.SINGLE, min_card.value, [min_card])
        
        # 如果没有任何牌可出（理论上不可能）
        return CardGroup(CardType.PASS, -1)
    
    def _get_state(self) -> np.ndarray:  # 获取状态
        state = np.zeros(self.config.STATE_SHAPE, dtype=np.float32)  # 初始化状态数组
        
        self._encode_hand(state[0], self.current_player)  # 编码当前玩家手牌
        
        teammate = self._get_teammate(self.current_player)  # 获取队友
        self._encode_hand(state[1], teammate)  # 编码队友手牌
        
        opp1 = self._get_opponent(self.current_player, 0)  # 获取对手1
        self._encode_played_cards(state[2], opp1)  # 编码对手1已出牌
        
        opp2 = self._get_opponent(self.current_player, 1)  # 获取对手2
        self._encode_played_cards(state[3], opp2)  # 编码对手2已出牌
        
        self._encode_history(state[4])  # 编码历史记录
        
        self._encode_game_state(state[5])  # 编码游戏状态
        
        return state  # 返回状态
    
    def _encode_hand(self, channel: np.ndarray, player: int):  # 编码手牌
        for card in self.hands[player]:  # 遍历手牌
            value = card.value  # 获取卡牌值
            if value < 13:  # 如果是普通卡牌
                idx = value - 6  # 计算索引
            else:  # 如果是大小王
                idx = 10 if value == 13 else 11  # 设置索引
            channel[0, min(idx, 14)] = 1  # 设置通道值
    
    def _encode_played_cards(self, channel: np.ndarray, player: int):  # 编码已出牌
        for card in self.played_cards[player]:  # 遍历已出牌
            value = card.value  # 获取卡牌值
            if value < 13:  # 如果是普通卡牌
                idx = value - 6  # 计算索引
            else:  # 如果是大小王
                idx = 10 if value == 13 else 11  # 设置索引
            channel[0, min(idx, 14)] += 1  # 增加通道值
    
    def _encode_history(self, channel: np.ndarray):  # 编码历史记录
        for i, (player, move_str) in enumerate(list(self.history)[-5:]):  # 遍历最近5步历史
            channel[i, 0] = player  # 设置玩家
            channel[i, 1] = len(move_str)  # 设置移动长度
    
    def _encode_game_state(self, channel: np.ndarray):  # 编码游戏状态
        channel[0, 0] = self.current_player  # 设置当前玩家
        channel[0, 1] = self.last_move_player  # 设置最后一步的玩家
        channel[0, 2] = self.multiplier  # 设置倍数
        channel[0, 3] = 1 if self.bomb_used else 0  # 设置炸弹使用标志
        
        for i in range(4):  # 遍历所有玩家
            channel[1, i] = len(self.hands[i])  # 设置手牌数量
    
    def _get_teammate(self, player: int) -> int:  # 获取队友
        team = self.config.TEAM_A if player in self.config.TEAM_A else self.config.TEAM_B  # 确定队伍
        return team[1] if team[0] == player else team[0]  # 返回队友
    
    def _get_opponent(self, player: int, index: int) -> int:  # 获取对手
        opponents = self.config.TEAM_B if player in self.config.TEAM_A else self.config.TEAM_A  # 确定对手队伍
        return opponents[index]  # 返回对手
    
    def _is_round_end(self) -> bool:  # 检查回合是否结束
        if len(self.history) < 3:  # 如果历史记录不足3步
            return False  # 返回未结束
        last_three = list(self.history)[-3:]  # 获取最后3步
        return all(move[1] == "PASS" for move in last_three)  # 检查是否都选择不出
    
    def _is_spring(self) -> bool:  # 检查是否是春天
        winner_team = self.config.TEAM_A if self.winner == 0 else self.config.TEAM_B  # 确定获胜队伍
        for player in winner_team:  # 遍历获胜队伍玩家
            if len(self.played_cards[player]) > 0:  # 如果已出牌
                player_rounds = set()  # 初始化回合集合
                for i, (p, _) in enumerate(self.history):  # 遍历历史记录
                    if p == player:  # 如果是该玩家
                        player_rounds.add(i // 4)  # 添加回合
                if len(player_rounds) > 1:  # 如果回合数大于1
                    return False  # 返回不是春天
        return True  # 返回是春天
    
    def _calculate_resource_penalty(self) -> float:  # 计算资源惩罚
        penalty = 0.0  # 初始化惩罚
        for player in range(4):  # 遍历所有玩家
            for card in self.hands[player]:  # 遍历手牌
                if card.value in [13, 14]:  # 如果是大小王
                    penalty += 0.1  # 增加惩罚
                elif card.value == 12:  # 如果是2
                    penalty += 0.05  # 增加惩罚
        return penalty  # 返回惩罚
    
    def _is_helping_teammate(self) -> bool:  # 检查是否帮助队友
        if self.last_move is None or self.last_move_player == self.current_player:  # 如果没有最后一步或是自己
            return False  # 返回否
        if self.last_move_player == self._get_teammate(self.current_player):  # 如果是队友
            return True  # 返回是
        if len(self.hands[self.current_player]) > len(self.last_move.cards) + 2:  # 如果手牌数量足够
            return True  # 返回是
        return False  # 返回否