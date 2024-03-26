import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c.network import Network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.masks = []
        self.rewards = []
        self.state_values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.masks[:]
        del self.rewards[:]
        del self.state_values[:]


class Agent:
    def __init__(self, state_size, action_size, args):
        self.gamma = args.gamma
        self.V_coef = args.V_coef
        self.E_coef = args.E_coef

        self.h = None
        self.c = None

        self.buffer = RolloutBuffer()
        self.policy = Network(state_size, action_size, args.n_units).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)

    def get_action(self, state, mask, h_in, c_in, train=True):
        state = torch.from_numpy(state).float().to(device)
        mask = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        h_in = torch.from_numpy(h_in).float().to(device)
        c_in = torch.from_numpy(c_in).float().to(device)

        with torch.no_grad():
            action, action_logprob, state_value, h_out, c_out = self.policy.act(state, mask, h_in, c_in)

        if train:
            return action.item(), action_logprob.item(), state_value.item(), h_out.cpu().numpy(), c_out.cpu().numpy()
        else:
            return action.item(), h_out.cpu().numpy(), c_out.cpu().numpy()

    def update(self, last_state, last_mask, last_h, last_c, done):
        if done:
            self.buffer.rewards.append(0.0)
        else:
            last_state = torch.from_numpy(last_state).float().to(device)
            last_mask = torch.from_numpy(last_mask).bool().unsqueeze(0).to(device)
            last_h = torch.from_numpy(last_h).float().to(device)
            last_c = torch.from_numpy(last_c).float().to(device)
            _, _, last_state_value, _, _ = self.policy.act(last_state, last_mask, last_h, last_c)
            self.buffer.rewards.append(last_state_value.item())

        n_step_returns = []
        temp = self.buffer.rewards[-1]
        for reward in reversed(self.buffer.rewards[:-1]):
            temp = reward + self.gamma * temp
            n_step_returns.insert(0, temp)
        n_step_returns = torch.tensor(n_step_returns, dtype=torch.float32).to(device)

        old_states = torch.from_numpy(np.array(self.buffer.states)).float().to(device)
        old_actions = torch.from_numpy(np.array(self.buffer.actions)).long().to(device)
        old_masks = torch.from_numpy(np.array(self.buffer.masks)).bool().to(device)
        old_state_values = torch.from_numpy(np.array(self.buffer.state_values)).float().to(device)
        h_in = torch.from_numpy(self.h).float().to(device)
        c_in = torch.from_numpy(self.c).float().to(device)

        advantages = n_step_returns.detach() - old_state_values.detach()
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks, h_in, c_in)

        policy_loss = logprobs * advantages.unsqueeze(1)
        value_loss = F.huber_loss(state_values, n_step_returns.unsqueeze(1), reduction='none')
        loss = - policy_loss + self.V_coef * value_loss - self.E_coef * dist_entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.buffer.clear()

    def save(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)