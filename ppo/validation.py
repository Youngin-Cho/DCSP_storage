import os
import torch
import numpy as np

from environment.env import Stacking


def evaluate(agent, args):
    agent.policy.eval()
    val_paths = os.listdir(args.load_val_dir)
    with torch.no_grad():
        move_lst = []
        for path in val_paths:
            data_src = args.load_val_dir + path
            test_env = Stacking(data_src, args.num_piles, args.max_height, args.reward_mode)

            state, mask = test_env.reset()
            h_in = np.zeros((1, args.n_units))
            c_in = np.zeros((1, args.n_units))

            while True:
                action, h_out, c_out = agent.get_action(state, mask, h_in, c_in, train=False)
                next_state, reward, done, mask = test_env.step(action)

                state = next_state
                h_in = h_out
                c_in = c_out

                if done:
                    break

            move_lst.append(test_env.crane_move)
        move_avg = sum(move_lst) / len(move_lst)

        return move_avg