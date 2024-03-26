import os
import json
import vessl
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from environment.data import DataGenerator
from environment.env import Stacking
from a2c.agent import Agent
from a2c.validation import evaluate


def train(args):
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)

    with open(args.save_log_dir + "parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(args.save_log_dir + "train_log.csv", 'w') as f:
        f.write('episode, mean value, reward, move\n')

    data_generator = DataGenerator(args.num_plates)
    env = Stacking(data_generator, num_piles=args.num_piles, max_height=args.max_height, reward_mode=args.reward_mode)
    agent = Agent(env.state_size, env.action_size, args)

    if not bool(args.vessl):
        writer = SummaryWriter(args.save_log_dir)

    for e in range(1, args.num_episodes + 1):
        agent.policy.train()
        state, mask = env.reset()

        h_in = np.zeros((1, args.n_units))
        c_in = np.zeros((1, args.n_units))
        agent.h = h_in
        agent.c = c_in

        ep_reward = 0
        ep_value = 0

        step = 0
        while True:
            action, action_logprob, state_value, h_out, c_out \
                = agent.get_action(state, mask, h_in, c_in, train=True)
            next_state, reward, done, mask = env.step(action)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.masks.append(mask)
            agent.buffer.rewards.append(reward)
            agent.buffer.state_values.append(state_value)

            state = next_state
            ep_reward += reward
            ep_value += state_value
            step += 1

            h_in = h_out
            c_in = c_out

            if step % args.num_steps == 0 or done:
                agent.update(state, mask, h_in, c_in, done)

            if done:
                break
            else:
                agent.h = h_in
                agent.c = c_in

        print("episode: %d | mean value: %.4f | reward: %.4f | move: %d"
              % (e, ep_value / step, ep_reward, env.crane_move))
        with open(args.save_log_dir + "train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %d\n' % (e, ep_value / step, ep_reward, env.crane_move))

        if bool(args.vessl):
            vessl.log(payload={"Train/MeanValue": ep_value / step,
                               "Train/Reward": ep_reward,
                               "Train/Move": env.crane_move}, step=e)
        else:
            writer.add_scalar("Train/MeanValue", ep_value / step, e)
            writer.add_scalar("Train/Reward", ep_reward, e)
            writer.add_scalar("Train/Move", env.crane_move, e)

        if e == 1 or e % args.eval_every == 0:
            average_move = evaluate(agent, args)

            with open(args.save_log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.2f\n' % (e, average_move))

            if bool(args.vessl):
                vessl.log(payload={"Validation/Move": average_move}, step=e)
            else:
                writer.add_scalar("Validation/Move", average_move, e)

        if e % args.save_every == 0:
            agent.save(e, args.save_model_dir)