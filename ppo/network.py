import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


class Network(nn.Module):
    def __init__(self, state_size, action_size, n_units):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_flatten = state_size[0] * state_size[1] * 32

        self.conv1 = nn.Conv2d(in_channels=state_size[-1], out_channels=16, kernel_size=2, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding="same")

        self.fc = nn.Linear(self.n_flatten, n_units)
        self.lstm = nn.LSTM(n_units, n_units)

        self.actor = nn.Linear(n_units, action_size)
        self.critic = nn.Linear(n_units, 1)

        # self.actor.weight.data = norm_col_init(self.actor.weight.data, 0.01)
        # self.critic.weight.data = norm_col_init(self.critic.weight.data, 1.0)

    def act(self, state, mask, h_in, c_in, greedy=False):
        x = F.elu(self.conv1(state))
        x = F.elu(self.conv2(x))
        x = x.contiguous().view(-1, self.n_flatten)

        x = F.elu(self.fc(x))
        x, (h_out, c_out) = self.lstm(x, (h_in, c_in))

        logits = self.actor(x)
        logits[~mask] = float('-inf')
        dist = Categorical(logits=logits)

        if greedy:
            action = torch.argmax(logits)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        state_value = self.critic(x)

        return action.detach(), action_logprob.detach(), state_value.detach(), h_out.detach(), c_out.detach()

    def evaluate(self, batch_state, batch_action, batch_mask, h_in, c_in):
        x = F.elu(self.conv1(batch_state))
        x = F.elu(self.conv2(x))
        x = x.contiguous().view(-1, self.n_flatten)

        x = F.elu(self.fc(x))
        x, (h_out, c_out) = self.lstm(x, (h_in, c_in))

        batch_logits = self.actor(x)
        batch_logits[~batch_mask] = float('-inf')

        batch_dist = Categorical(logits=batch_logits)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        batch_state_values = self.critic(x)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys