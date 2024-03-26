import random
import numpy as np
import pandas as pd

from environment.data import DataGenerator


class Plate:
    def __init__(self, id, arrival_date, retrieval_date):
        self.id = id
        self.arrival_date = arrival_date
        self.retrieval_date = retrieval_date


class Stacking:
    def __init__(self, data_src, num_piles=4, max_height=4, reward_mode=1):
        self.data_src = data_src
        self.num_piles = num_piles
        self.max_height = max_height
        self.reward_mode = reward_mode

        if type(self.data_src) is DataGenerator:
            self.df = self.data_src.generate()
        else:
            self.df = pd.read_excel(data_src, engine='openpyxl')

        self.piles = [[] for _ in range(num_piles)]
        self.plates = []
        for i, row in self.df.iterrows():
            id = row["plate id"]
            arrival_date = row["arrival date"]
            retrieval_date = row["retrieval date"]
            plate = Plate(id, arrival_date, retrieval_date)
            self.plates.append(plate)
        self.num_plates = len(self.plates)

        self.state_size = (self.max_height, self.num_piles + 1, 1)
        self.action_size = self.num_piles
        self.current_date = 0
        self.crane_move = 0

    def step(self, action):
        done = False
        plate = self.plates.pop(0)

        self.piles[action].append(plate)
        reward = self._calculate_reward(action)

        if len(self.plates) == 0:
            done = True
        elif int(self.plates[0].arrival_date) != self.current_date:
            self.current_date = int(self.plates[0].arrival_date)
            self._retrieve_plates()

        next_state, mask = self._get_state()

        if done:
            self._retrieve_all_plates()

        return next_state, reward, done, mask

    def reset(self):
        if type(self.data_src) is DataGenerator:
            self.df = self.data_src.generate()
        else:
            self.df = pd.read_excel(self.data_src, engine='openpyxl')

        self.piles = [[] for _ in range(self.num_piles)]
        self.plates = []
        for i, row in self.df.iterrows():
            id = row["plate id"]
            arrival_date = row["arrival date"]
            retrieval_date = row["retrieval date"]
            plate = Plate(id, arrival_date, retrieval_date)
            self.plates.append(plate)

        self.num_plates = len(self.plates)
        self.current_date = min(self.plates, key=lambda x: x.arrival_date).arrival_date
        self.crane_move = 0

        initial_state, mask = self._get_state()

        return initial_state, mask

    def _calculate_reward(self, action):
        pile = self.piles[action]

        if self.reward_mode == 1:
            max_move = 0

            if len(pile) == 1:
                return 0

            for i, plate in enumerate(pile[:-1]):
                move = 0
                if i + 1 + max_move >= len(pile):
                    break
                for upper_plate in pile[i + 1:]:
                    if int(plate.retrieval_date) < int(upper_plate.retrieval_date):
                        move += 1
                if move > max_move:
                    max_move = move

            if max_move != 0:
                reward = 1 / max_move
            else:
                reward = 2
        elif self.reward_mode == 2:
            retrieval_dates = np.array([int(plate.retrieval_date) for plate in pile])
            retrieval_dates_sorted = np.array(sorted(retrieval_dates))
            diff = np.sum(np.abs(retrieval_dates_sorted - retrieval_dates))
            reward = 1 / diff if diff != 0 else 1
        elif self.reward_mode == 3:
            if len(pile) == 1:
                new_plate = pile[-1]
                reward = 1 / (int(new_plate.retrieval_date) - self.current_date)
            else:
                new_plate = pile[-1]
                old_plate = pile[-2]
                if int(old_plate.retrieval_date) > int(new_plate.retrieval_date):
                    reward = 1 / (old_plate.retrieval_date - new_plate.retrieval_date)
                elif int(old_plate.retrieval_date) == int(new_plate.retrieval_date):
                    reward = 0
                else:
                    reward = -1

        return reward

    def _get_state(self):
        state = np.full([self.max_height, self.num_piles + 1], -1.0)
        mask = np.ones(self.num_piles)

        inbound_plates = [plate for plate in self.plates[:self.max_height] if int(plate.arrival_date) == self.current_date]
        target_plates = [inbound_plates[::-1]] + self.piles[:]

        for i, pile in enumerate(target_plates):
            if i > 0 and len(pile) == self.max_height:
                mask[i-1] = 0
            for j, plate in enumerate(pile):
                state[j, i] = int(plate.retrieval_date) - self.current_date

        state = np.flipud(state).copy()
        state = state[np.newaxis, :, :]
        # state = state / np.max(state)

        return state, mask

    def _retrieve_plates(self):
        for pile in self.piles:
            plates_retrieved = []

            for i, plate in enumerate(pile):
                if int(plate.retrieval_date) <= self.current_date:
                    plates_retrieved.append(i)

            if len(plates_retrieved) > 0:
                self.crane_move += (len(pile) - plates_retrieved[0] - len(plates_retrieved))

            for index in plates_retrieved[::-1]:
                del pile[index]

    def _retrieve_all_plates(self):
        while True:
            next_retrieval_date = int(min(sum(self.piles, []), key=lambda x: x.retrieval_date).retrieval_date)
            self.current_date = next_retrieval_date
            self._retrieve_plates()

            if not sum(self.piles, []):
                break


if __name__ == "__main__":
    data_src = DataGenerator()
    num_piles = 10
    max_height = 20

    env = Stacking(data_src, num_piles, max_height)
    s = env.reset()

    while True:
        a = random.choice([i for i in range(num_piles)])
        s_next, r, done = env.step(a)
        print(r)
        s = s_next

        if done:
            break