import numpy as np
from numba import int32, float32, uint8, types, typed, int64, njit, jit, prange, boolean
from numba.experimental import jitclass
import matplotlib.pyplot as plt

# 0: 파괴가능
# 1: 재생가능 빈칸
# 2: 재생불가능 (불변) 빈칸
# 3: 혼돈 블럭
# -------------------------------특수타일
# 4: 정령 교체 가능 횟수 증가
# 5: 정령 소환 횟수 증가하지 않음
# 7: 남은 정령이 분출(120) or 세계수 공명(121)으로 변환
# 9: 남은 정령이 한 단계 강화 (유적의 신비 2종 및 최고 단계 정령은 강화 불가)
# 10: 남은 정령이 소환한 정령으로 복제
# 11: 남아 있는 모든 석판이 재배치

# Order must be:
# future 2 -> future 1 -> future 0

# 0 ~ 19: 타일 종류
# 20 ~ 39: 남은 기회
# 40 ~ 59: 무기 교체 가능 횟수
# 100 ~ 200: 무기 종류
# Max index = 200

# Action range: 0 ~ boardsize**2 * 2 + 1
# 0 ~ boardsize**2 - 1: 무기 0 board_idx
# boardsize**2 : 무기 0 교체
# boardsize**2 + 1 ~ boardsize**2 * 2: 무기 1 board_idx
# boardsize**2 * 2 + 1: 무기 1 교체



CHANCE_OFFSET = 20
CHANGE_WEAPON_OFFSET = 40
TILE_BREAK_PROB = 10000

# 타일 파괴 종류
NORMAL = 0
SHUFFLE_BOARD = -1
TRIGGERED_CHAOS = -2

@jitclass([
    ('shuffle_on_reset', boolean),
    ('random_chogibaek', boolean),
    ('boardsize', int32),
    ('chogibaek', int32),
    ('max_chogibaek', int32),
    ('min_chogibaek', int32),
    ('change_weapon_chance', int32),
    ('max_chance', int32),
    ('current_chance', int32),
    ('chaos_idx_list', int32[:]),
    ('_original_board', int32[:]),
    ('board', int32[:]),
    ('special_tile_ids', int32[:]),
    ('special_tile_cumsum', int32[:]),
    ('mutable_idx_list', int32[:]),
    ('future_weapon_list', int32[:]),
    ('current_weapons', int32[:]),
    ('normal_weapon_ids', int32[:]),
    ('upgrade_id_change', int32),
    ('normal_weapon_cumsum', int32[:]),
    ('special_weapon_ids', int32[:]),
    ('state_history', types.List(types.Array(int32, 1, 'C'))),
    ('action_history', types.List(int64)),
])
class GameState():
    def __init__(
        self, 
        boardsize, 
        max_chance, 
        immutable_idx_list, 
        chaos_idx_list, 
        chogibaek:int=0,
        min_chogibaek:int=0,
        shuffle_on_reset:bool=False,
        random_chogibaek:bool=False
    ):
        """
        !Important: if immutable_idx_list or chaos_idx_list is empty, 
        !           it should be None, not np.array([])
        !Important: If random_chogibaek is true, then parameter 'chogibaek' is the maximum of chogibaek [0, chogibaek] (Inclusive)
        """
        self.shuffle_on_reset = shuffle_on_reset
        self.random_chogibaek = random_chogibaek
        self.boardsize = boardsize
        self.chogibaek = chogibaek
        self.max_chogibaek = chogibaek
        self.min_chogibaek = min_chogibaek
        if self.random_chogibaek:
            self.chogibaek = np.random.randint(self.min_chogibaek, self.max_chogibaek+1)
        self.change_weapon_chance = 2+self.chogibaek
        self.max_chance = max_chance
        self.current_chance = max_chance
        if chaos_idx_list is not None:
            self.chaos_idx_list = chaos_idx_list
        else:
            self.chaos_idx_list = np.zeros(0, dtype=np.int32)
        self._original_board = np.zeros(boardsize**2, dtype=np.int32)
        if immutable_idx_list is not None:
            self._original_board[immutable_idx_list] = 2
        self.board = self._original_board.copy()

        if self.chaos_idx_list.shape[0] > 0:
            sampled_chaos_idx_list = np.random.choice(
                self.chaos_idx_list, 
                self.chaos_idx_list.shape[0]-self.chogibaek, 
                replace=False
            )
            self.board[sampled_chaos_idx_list] = 3
        if self.shuffle_on_reset:
            self.shuffle_board()
        self.special_tile_ids = np.array([9,10,7,4,11,5], dtype=np.int32)
        self.special_tile_cumsum = np.cumsum(np.array([
            1600,
            1600,
            1600,
            2350,
            1700,
            1150
        ])).astype(np.int32)
        self.mutable_idx_list = np.nonzero(self.board != 2)[0].astype(np.int32)
        self.future_weapon_list = np.zeros(3, dtype=np.int32)
        #! always sort current_weapons
        self.current_weapons = np.zeros(2, dtype=np.int32)
        self.normal_weapon_ids = np.array([100,101,102,103,104,105,109,110,114,116], dtype=np.int32)
        self.upgrade_id_change = 30
        normal_weapon_ratio = np.array([
            1500,
            1150,
            950,
            550,
            1050,
            700,
            900,
            700,
            1000,
            1500,
        ])
        self.normal_weapon_cumsum = np.cumsum(normal_weapon_ratio).astype(np.int32)
        self.special_weapon_ids = np.array([120,121], dtype=np.int32)

        # initialize weapons
        self.current_weapons[0] = self.get_new_normal_weapon_id()
        self.current_weapons[1] = self.get_new_normal_weapon_id()
        self.future_weapon_list[0] = self.get_new_normal_weapon_id()
        self.future_weapon_list[1] = self.get_new_normal_weapon_id()
        self.future_weapon_list[2] = self.get_new_normal_weapon_id()

        # check duplication
        self.check_dup()

        # history
        start_state, _ = self.state()
        self.state_history = [start_state]
        self.action_history = [-1] # dummy

    def reset(self):
        if self.random_chogibaek:
            self.chogibaek = np.random.randint(self.min_chogibaek, self.max_chogibaek+1)
        self.change_weapon_chance = 2+self.chogibaek
        self.current_chance = self.max_chance
        self.board = self._original_board.copy()
        if self.chaos_idx_list.shape[0] > 0 and self.chaos_idx_list.shape[0] > self.chogibaek:
            sampled_chaos_idx_list = np.random.choice(
                self.chaos_idx_list, 
                self.chaos_idx_list.shape[0]-self.chogibaek, 
                replace=False
            )
            self.board[sampled_chaos_idx_list] = 3
        if self.shuffle_on_reset:
            self.shuffle_board()
        self.current_weapons[0] = self.get_new_normal_weapon_id()
        self.current_weapons[1] = self.get_new_normal_weapon_id()
        self.future_weapon_list[0] = self.get_new_normal_weapon_id()
        self.future_weapon_list[1] = self.get_new_normal_weapon_id()
        self.future_weapon_list[2] = self.get_new_normal_weapon_id()
        self.check_dup()
        start_state, _ = self.state()
        self.state_history = [start_state]
        self.action_history = [-1] # dummy

    def get_new_normal_weapon_id(self):
        return self.normal_weapon_ids[np.searchsorted(self.normal_weapon_cumsum, np.random.randint(0, self.normal_weapon_cumsum[-1]))]
    
    def get_new_special_weapon_id(self):
        return np.random.choice(self.special_weapon_ids)
    
    def check_dup(self):
        if np.all(self.current_weapons>=130):
            raise Exception('Current weapons cannot both be upgraded weapons')
        self.current_weapons.sort()
        while (
            (self.current_weapons[0] == self.current_weapons[1]) or
            (self.current_weapons[0] + self.upgrade_id_change == self.current_weapons[1]) or
            (self.current_weapons[0] - self.upgrade_id_change == self.current_weapons[1])
        ):
            self.current_weapons[0] = np.max(self.current_weapons) + self.upgrade_id_change
            self.current_weapons[1] = self.future_weapon_list[0]
            self.current_weapons.sort()
            self.future_weapon_list[0] = self.future_weapon_list[1]
            self.future_weapon_list[1] = self.future_weapon_list[2]
            self.future_weapon_list[2] = self.get_new_normal_weapon_id()
    
    def check_definitive(self):
        pass

    def state(self):
        purifiable = np.array([114, 121, 144, 174])
        raw_state = np.logical_and(
            self.board != 1,
            self.board != 2
        )
        selectable_board_mask_0 = raw_state.copy()
        if np.all(self.current_weapons[0] != purifiable):
            # Not purifiable
            selectable_board_mask_0 = np.logical_and(
                selectable_board_mask_0,
                self.board != 3
            )
        selectable_board_mask_1 = raw_state.copy()
        if np.all(self.current_weapons[1] != purifiable):
            # Not purifiable
            selectable_board_mask_1 = np.logical_and(
                selectable_board_mask_1,
                self.board != 3
            )
        
        weapon_changable = self.change_weapon_chance > 0
        action_avail_mask = np.concatenate((
            selectable_board_mask_0,
            np.array([weapon_changable]),
            selectable_board_mask_1,
            np.array([weapon_changable]),
        ))
        return (np.concatenate((
                self.board, 
                np.array([self.current_chance+CHANCE_OFFSET, 
                          self.change_weapon_chance+CHANGE_WEAPON_OFFSET], 
                        dtype=np.int32),
                self.future_weapon_list, 
                self.current_weapons, 
            )).astype(np.int32),
            action_avail_mask)


    def check_valid_weapon_usage(self, board_idx, purifiable):
        if board_idx < 0 or board_idx >= self.boardsize**2:
            raise Exception('Invalid board_idx')
        if self.board[board_idx] == 1:
            raise Exception('Broken tile')
        if self.board[board_idx] == 2:
            raise Exception('Immutable tile')
        if self.board[board_idx] == 3 and not purifiable:
            raise Exception('Chaos tile, but not purifiable')

    def successed(self):
        # only 1, 2, 3 left
        return np.all(np.logical_or(self.board==1, 
                                    np.logical_or(self.board==2, self.board==3)))

    def finished(self):
        no_more_chance = ((self.current_chance == 0) and np.all(self.board != 5)) or self.current_chance < 0
        return no_more_chance or self.successed()

    def _end_act(self, action_idx):
        new_state = self.state()
        self.state_history.append(new_state[0])
        self.action_history.append(action_idx)
        return new_state

    def act(self, action_idx):
        if action_idx<0 or action_idx > self.boardsize**2 * 2 + 1:
            raise Exception('Invalid action_idx')
        # Tried to act even though the game is over
        if self.finished():
            raise Exception('Game is over')

        weapon_changed = False
        current_weapon_idx = 3
        if action_idx < self.boardsize**2:
            current_weapon_idx = 0
            weapon_idx = self.current_weapons[0]
            board_idx = action_idx
        elif action_idx == self.boardsize**2:
            self.weapon_change(0)
            weapon_changed = True
        elif action_idx < self.boardsize**2 * 2 + 1:
            current_weapon_idx = 1
            weapon_idx = self.current_weapons[1]
            board_idx = action_idx - self.boardsize**2 - 1
        else:
            self.weapon_change(1)
            weapon_changed = True

        if weapon_changed:
            return self._end_act(action_idx)      
        if current_weapon_idx == 3:
            raise Exception('current_weapon_idx not set')
        purifiable = np.any(weapon_idx == np.array([114, 121, 144, 174]))
        self.check_valid_weapon_usage(board_idx, purifiable)
        self.current_chance -= 1
        self.weapons(board_idx, current_weapon_idx)
        self.weapon_update(current_weapon_idx)
        self.renew_special_tile()
        return self._end_act(action_idx)

    def get_history(self, include_last_state=False):
        # Do not need dummy action
        valid_action_history = self.action_history[1:]
        if include_last_state:
            valid_state_history = self.state_history
            if len(valid_state_history) != len(valid_action_history)+1:
                raise Exception('Invalid history - state and action length not equal')
        else:
            valid_state_history = self.state_history[:-1]
            if len(valid_state_history) != len(valid_action_history):
                raise Exception('Invalid history - state and action length not equal')

        
        return valid_state_history, valid_action_history

    def renew_special_tile(self):
        # Delete all left special tiles
        special_tile_indices = np.nonzero(self.board>=4)[0]
        if special_tile_indices.shape[0] > 1:
            raise Exception('More than one special tile exists')
        self.board[special_tile_indices] = 0
        unbroken_indices = np.nonzero(self.board==0)[0]
        if unbroken_indices.shape[0] == 0:
            return
        special_tile_id = self.special_tile_ids[np.searchsorted(self.special_tile_cumsum, np.random.randint(0, self.special_tile_cumsum[-1]))]
        self.board[np.random.choice(unbroken_indices)] = special_tile_id

    def weapon_update(self, current_weapon_idx):
        if current_weapon_idx < 0 or current_weapon_idx > 1:
            raise Exception('Invalid current_weapon_idx')
        
        self.current_weapons[current_weapon_idx] = self.future_weapon_list[0]
        self.future_weapon_list[0] = self.future_weapon_list[1]
        self.future_weapon_list[1] = self.future_weapon_list[2]
        self.future_weapon_list[2] = self.get_new_normal_weapon_id()
        self.check_dup()

    def weapon_change(self, weapon_idx):
        if self.change_weapon_chance <= 0:
            raise Exception('No weapon change chance left')
        self.change_weapon_chance -= 1
        self.current_weapons[weapon_idx] = self.future_weapon_list[0]
        self.future_weapon_list[0] = self.future_weapon_list[1]
        self.future_weapon_list[1] = self.future_weapon_list[2]
        self.future_weapon_list[2] = self.get_new_normal_weapon_id()
        self.check_dup()

    def process_broken_tile(self, board_idx, current_weapon_idx, is_max_upgrade, is_purifiable):
        """
        !Note: Call this function only when the tile is broken
        !Note: It does not raise exception when the board_idx is out of range or the tile is immutable/broken
        """
        # 0: 파괴가능
        # 1: 재생가능 빈칸
        # 2: 재생불가능 (불변) 빈칸
        # 3: 혼돈 블럭
        # -------------------------------특수타일
        # 4: 정령 교체 가능 횟수 증가
        # 5: 정령 소환 횟수 증가하지 않음
        # 7: 남은 정령이 분출(120) or 세계수 공명(121)으로 변환
        # 9: 남은 정령이 한 단계 강화 (유적의 신비 2종 및 최고 단계 정령은 강화 불가)
        # 10: 남은 정령이 소환한 정령으로 복제
        # 11: 남아 있는 모든 석판이 재배치

        if self.board[board_idx] == 1 or self.board[board_idx] == 2:
            return NORMAL
        another_weapon_idx = (current_weapon_idx+1)%2
        result = NORMAL
        if board_idx < 0 or board_idx >= self.boardsize**2:
            return 0
        if self.board[board_idx] == 0:
            self.board[board_idx] = 1
        elif self.board[board_idx] == 3: # 왜곡된 타일
            if is_purifiable:
                self.board[board_idx] = 1
            elif is_max_upgrade:
                # 0% if max upgrade (except purifiable)
                pass
            else:
                result = TRIGGERED_CHAOS
        elif self.board[board_idx] == 4:
            self.change_weapon_chance += 1
            self.board[board_idx] = 1
        elif self.board[board_idx] == 5:
            self.current_chance += 1
            self.board[board_idx] = 1
        elif self.board[board_idx] == 7:
            special_weapon_id = self.get_new_special_weapon_id()
            self.current_weapons[another_weapon_idx] = special_weapon_id
            self.board[board_idx] = 1
        elif self.board[board_idx] == 9:
            if ((self.current_weapons[another_weapon_idx] < 160) and 
                (self.current_weapons[another_weapon_idx] != 120) and 
                (self.current_weapons[another_weapon_idx] != 121)):
                self.current_weapons[another_weapon_idx] += self.upgrade_id_change
            self.board[board_idx] = 1
        elif self.board[board_idx] == 10:
            self.current_weapons[another_weapon_idx] = self.current_weapons[current_weapon_idx]
            self.board[board_idx] = 1
        elif self.board[board_idx] == 11:
            self.board[board_idx] = 1
            result = SHUFFLE_BOARD
        return result

    def regenerate_block(self, N):
        for _ in range(N):
            if np.all(self.board != 1):
                return
            regeneratable_board_indices = np.nonzero(self.board == 1)[0]
            self.board[np.random.choice(regeneratable_board_indices)] = 0

    def shuffle_board(self):
        shufflable_indices = np.nonzero(self.board != 2)[0]
        shuffled_shufflable_indices = np.random.permutation(shufflable_indices)
        self.board[shufflable_indices] = self.board[shuffled_shufflable_indices]

    def break_tiles(self, board_idx, x, y, prob, current_weapon_idx, is_max_upgrade, is_purifiable):
        # Safety
        if self.board[board_idx] == 1:
            raise Exception('Clicked tile is Broken tile')
        if self.board[board_idx] == 2:
            raise Exception('Clicked tile is Immutable tile')
        if self.board[board_idx] == 3 and not is_purifiable:
            raise Exception('Clicked tile is Chaos tile, but not purifiable')
        if not (x.shape[0] == y.shape[0] == prob.shape[0]):
            raise Exception('Invalid shape - x, y, prob all should have same shape')

        # Clicked tile always breaks
        result = self.process_broken_tile(
            board_idx, current_weapon_idx, is_max_upgrade, is_purifiable)
        
        random_probs = np.random.randint(0, TILE_BREAK_PROB, size=prob.shape)
        successed = random_probs < prob
        triggered_chaos_num = 0
        # Clicked tile never triggers chaos
        need_to_shuffle = result==SHUFFLE_BOARD

        board_idx_x = board_idx % self.boardsize
        board_idx_y = board_idx // self.boardsize
        for i in range(x.shape[0]):
            if (
                successed[i] and
                board_idx_x + x[i] >= 0 and
                board_idx_x + x[i] < self.boardsize and
                board_idx_y + y[i] >= 0 and
                board_idx_y + y[i] < self.boardsize
            ):
                result = self.process_broken_tile(
                    board_idx + x[i] + y[i]*self.boardsize, 
                    current_weapon_idx, is_max_upgrade, is_purifiable)
                if result == TRIGGERED_CHAOS:
                    triggered_chaos_num += 1
                elif result == SHUFFLE_BOARD:
                    need_to_shuffle = True
                elif result == NORMAL:
                    pass
                else:
                    raise Exception(f'Unhandled result: {result}')
        
        if triggered_chaos_num > 0:
            self.regenerate_block(triggered_chaos_num*3)
        if need_to_shuffle:
            self.shuffle_board()
            
        

    def weapons(self, board_idx, current_weapon_idx):
        weapon_idx = self.current_weapons[current_weapon_idx]
        is_max_upgrade = weapon_idx >= 160
        is_purifiable = (weapon_idx == 114) or\
                        (weapon_idx == 121) or\
                        (weapon_idx == 144) or\
                        (weapon_idx == 174)
        
        if weapon_idx == 100:
            x = np.array([1, 0, 0, -1])
            y = np.array([0, 1, -1, 0])
            prob = np.array([5000, 5000, 5000, 5000])
        elif weapon_idx == 101:
            x = np.array([0, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, 0])
            y = np.array([2, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -2])
            prob = np.array([5000]*12)
        elif weapon_idx == 102:
            x = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
            y = np.array([1, 1, 1, 0, 0, -1, -1, -1])
            prob = np.array([7500]*8)
        elif weapon_idx == 103:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7,
                          0, 0, 0, 0, 0, 0, 0])
            y = np.array([0, 0, 0, 0, 0, 0, 0,
                          1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000])
        elif weapon_idx == 104:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7,
                          1, 2, 3, 4, 5, 6, 7])
            prob = np.array([8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000])
        elif weapon_idx == 105:
            x = np.array([0]*14)
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000])
        elif weapon_idx == 109:
            # 벼락
            self.lightning(board_idx, current_weapon_idx, max_n=2)
            return
        elif weapon_idx == 110:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([0]*14)
            prob = np.array([8500, 7000, 5500, 4000, 2500, 1000, 1000,
                             8500, 7000, 5500, 4000, 2500, 1000, 1000])
        elif weapon_idx == 114:
            x = np.array([1, -1])
            y = np.array([0, 0])
            prob = np.array([5000,5000])
        elif weapon_idx == 116:
            x = np.array([1, -1, 1, -1])
            y = np.array([1, 1, -1, -1])
            prob = np.array([5000, 5000, 5000, 5000])

        # 특수 2종
        elif weapon_idx == 120:
            # 분출: 클릭한 곳 말고는 없음
            x = np.array([0])
            y = np.array([10])
            prob = np.array([1])
        elif weapon_idx == 121:
            x = np.array([1, 2, -1, -2, 0, 0, 0, 0])
            y = np.array([0, 0, 0, 0, 1, 2, -1, -2])
            prob = np.array([10000]*8)

        # 2단계
        elif weapon_idx == 130:
            x = np.array([1, 0, 0, -1])
            y = np.array([0, 1, -1, 0])
            prob = np.array([10000]*4)
        elif weapon_idx == 131:
            x = np.array([0, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, 0])
            y = np.array([2, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -2])
            prob = np.array([10000]*12)
        elif weapon_idx == 132:
            x = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
            y = np.array([1, 1, 1, 0, 0, -1, -1, -1])
            prob = np.array([10000]*8)
        elif weapon_idx == 133:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7,
                          0, 0, 0, 0, 0, 0, 0])
            y = np.array([0, 0, 0, 0, 0, 0, 0,
                          1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([10000]*28)
        elif weapon_idx == 134:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7,
                          1, 2, 3, 4, 5, 6, 7])
            prob = np.array([10000]*28)
        elif weapon_idx == 135:
            x = np.array([0]*14)
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([10000]*14)
        elif weapon_idx == 139:
            # 벼락 2단계
            self.lightning(board_idx, current_weapon_idx, max_n=4)
            return
        elif weapon_idx == 140:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([0]*14)
            prob = np.array([10000]*14)
        elif weapon_idx == 144:
            x = np.array([1, -1])
            y = np.array([0, 0])
            prob = np.array([10000]*2)
        elif weapon_idx == 146:
            x = np.array([1, -1, 1, -1])
            y = np.array([1, 1, -1, -1])
            prob = np.array([10000]*4)

        #3단계
        elif weapon_idx == 160:
            x = np.array([1, 0, 0, -1])
            y = np.array([0, 1, -1, 0])
            prob = np.array([10000]*4)
        elif weapon_idx == 161:
            x = np.array([0, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, 0])
            y = np.array([2, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -2])
            prob = np.array([10000]*12)
        elif weapon_idx == 162:
            x = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
            y = np.array([1, 1, 1, 0, 0, -1, -1, -1])
            prob = np.array([10000]*8)
        elif weapon_idx == 163:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7,
                          0, 0, 0, 0, 0, 0, 0])
            y = np.array([0, 0, 0, 0, 0, 0, 0,
                          1, 2, 3, 4, 5, 6, 7,
                          0, 0, 0, 0, 0, 0, 0,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([10000]*28)
        elif weapon_idx == 164:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7,
                          -1, -2, -3, -4, -5, -6, -7,
                          1, 2, 3, 4, 5, 6, 7])
            prob = np.array([10000]*28)
        elif weapon_idx == 165:
            x = np.array([0]*14)
            y = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            prob = np.array([10000]*14)
        elif weapon_idx == 169:
            # 벼락 3단계
            self.lightning(board_idx, current_weapon_idx, max_n=6)
            return
        elif weapon_idx == 170:
            x = np.array([1, 2, 3, 4, 5, 6, 7,
                          -1, -2, -3, -4, -5, -6, -7])
            y = np.array([0]*14)
            prob = np.array([10000]*14)
        elif weapon_idx == 174:
            x = np.array([1, 0, -1, 0])
            y = np.array([0, 1, 0, -1])
            prob = np.array([10000]*4)
        elif weapon_idx == 176:
            x = np.array([1, -1, 1, -1])
            y = np.array([1, 1, -1, -1])
            prob = np.array([10000]*4)

        self.break_tiles(board_idx, x, y, prob,
                        current_weapon_idx=current_weapon_idx, 
                        is_max_upgrade=is_max_upgrade, is_purifiable=is_purifiable)

    def lightning(self, board_idx, current_weapon_idx, max_n):
        # Safety
        if self.board[board_idx] == 1:
            raise Exception('Broken tile')
        if self.board[board_idx] == 2:
            raise Exception('Immutable tile')
        if self.board[board_idx] == 3:
            raise Exception('Chaos tile, but not purifiable')
        
        # Clicked tile always breaks
        result = self.process_broken_tile(
            board_idx, current_weapon_idx, False, False)
        need_to_shuffle = result==SHUFFLE_BOARD
        if result == TRIGGERED_CHAOS:
            raise Exception('Lightning cannot trigger chaos')

        additional_break = np.random.randint(-1, max_n+1)
        if additional_break == -1:
            self.regenerate_block(1)
        elif additional_break > 0:
            for _ in range(additional_break):
                if np.all(
                    np.logical_or(
                        np.logical_or(
                            self.board==1, self.board==2), self.board==3)):
                    break
                else:
                    breakable_board_indices = np.nonzero(
                        np.logical_and(
                            np.logical_and(
                                self.board!=1, self.board!=2), self.board!=3))[0]
                    break_index = np.random.choice(breakable_board_indices)
                    result = self.process_broken_tile(
                        break_index, current_weapon_idx, False, False)
                    if result == TRIGGERED_CHAOS:
                        raise Exception('Lightning cannot trigger chaos')
                    elif result == SHUFFLE_BOARD:
                        need_to_shuffle = True
                    elif result == NORMAL:
                        pass
                    else:
                        raise Exception(f'Unhandled result: {result}')
        if need_to_shuffle:
            self.shuffle_board()


def visualize_state(state, action=None, last_state=None):
    weapons = {
        100: '낙뢰',
        101: '업화',
        102: '충격파',
        103: '해일',
        104: '대폭발',
        105: '폭풍우',
        109: '벼락',
        110: '지진',
        114: '정화',
        116: '용오름',
        120: '분출',
        121: '세계수의 공명',
        130: '낙뢰(2단계)',
        131: '업화(2단계)',
        132: '충격파(2단계)',
        133: '해일(2단계)',
        134: '대폭발(2단계)',
        135: '폭풍우(2단계)',
        139: '벼락(2단계)',
        140: '지진(2단계)',
        144: '정화(2단계)',
        146: '용오름(2단계)',
        160: '낙뢰(3단계)',
        161: '업화(3단계)',
        162: '충격파(3단계)',
        163: '해일(3단계)',
        164: '대폭발(3단계)',
        165: '폭풍우(3단계)',
        169: '벼락(3단계)',
        170: '지진(3단계)',
        174: '정화(3단계)',
        176: '용오름(3단계)',
    }
    special_tiles = {
        4: '정령 교체 가능 횟수 증가',
        5: '정령 소환 횟수 증가하지 않음',
        7: '남은 정령이 분출(120) or 세계수 공명(121)으로 변환',
        9: '남은 정령이 한 단계 강화 (유적의 신비 2종 및 최고 단계 정령은 강화 불가)',
        10: '남은 정령이 소환한 정령으로 복제',
        11: '남아 있는 모든 석판이 재배치',
    }
    state, action_avail_mask = state
    board = state[:-7]
    current_chance = state[-7] - CHANCE_OFFSET
    change_weapon_chance = state[-6] - CHANGE_WEAPON_OFFSET
    future_weapon_list = state[-5:-2]
    current_weapons = state[-2:]
    
    boardsize = int(np.sqrt(len(board)))
    board = board.reshape((boardsize, boardsize))
    board_img = np.ones((boardsize, boardsize, 3), dtype=np.uint8)*255
    board_img[board==1] = np.array([125,125,125], dtype=np.uint8)
    board_img[board==2] = np.array([0,0,0], dtype=np.uint8)
    board_img[board==3] = np.array([128,0,128], dtype=np.uint8)
    board_img[board>=4] = np.array([0,0,255], dtype=np.uint8)
    if np.any(board>=4):
        print('특수 타일:', end=' ')
        for special_tile_id in np.unique(board[board>=4]):
            print(special_tiles[special_tile_id], end=' ')
        print('\n')
    if action is not None:
        assert last_state is not None, 'last_state should be given when action is given'
        last_current_weapons = last_state[-2:]
        if action == boardsize**2:
            print('무기 0 교체')
        elif action == boardsize**2*2+1:
            print('무기 1 교체')
        elif action < boardsize**2:
            print(f'무기 0: {weapons[last_current_weapons[0]]}')
            board_img[action//boardsize, action%boardsize] = np.array([255,255,0], dtype=np.uint8)
        else:
            action = action - boardsize**2 - 1
            print(f'무기 1: {weapons[last_current_weapons[1]]}')
            board_img[action//boardsize, action%boardsize] = np.array([255,255,0], dtype=np.uint8)

    plt.imshow(board_img)
    plt.show()
    print('남은 기회:', current_chance)
    print('무기 교체 가능 횟수:', change_weapon_chance)
    print('현재 무기:', weapons[current_weapons[0]], weapons[current_weapons[1]])
    print('다음 무기:', weapons[future_weapon_list[2]], weapons[future_weapon_list[1]], weapons[future_weapon_list[0]])
    print('-'*20)