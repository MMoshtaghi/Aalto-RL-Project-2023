from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time


class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)  # look at the attributes of the BaseAgent class
        self.alpha_entropy_bonus = 0.01

    def ppo_update(self, states, actions, rewards, next_states, dones, action_log_probs, return_estimates):
        '''
        - For the 1st batch of epoch, the old policy and new_policy are the same, ratio=1,
        but for the others, we compare the new policy with the policy we took actions with during episode.
        - also the ruturn_estimates are computed with the value network we used during episode, but we compute the loss between the returns and the new updated value network.
        - the same for advantge
        '''
        # print(f'{states.shape=}') # state : (bs, state_dim)
        new_policy_action_dists, new_values = self.policy(states)
        # print(f'{new_values.shape=}')
        new_values = new_values.squeeze()  # (bs,)
        # print(f'{return_estimates.shape=}') # (bs,)

        # value_loss = (new_values - return_estimates).pow(2).mean() # (1)
        value_loss = F.smooth_l1_loss(new_values, return_estimates, reduction="mean")  # (1)
        # print(f'{value_loss=}')

        advantages = return_estimates - new_values.detach()  # (bs,) detach values to block gradient backprop
        ## Smilar to BatchNorm in DL, Normalization of the returns is employed to make training more stable
        # TODO mark as improvement and put into ppo_extension
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # (bs,)
        # print(f'{advantages.shape=}')

        # print(f'{actions.shape=}') # (bs, act_dim)
        # Calculate the actor/policy loss: make new_policy_action_probs similar to advantage distribution to an extent
        new_policy_action_log_probs = new_policy_action_dists.log_prob(
            actions)  # (bs,) log prob of joint distribution of all dimensions Pr(x,y)
        ratio = torch.exp(new_policy_action_log_probs - action_log_probs)  # exp(logx - logy) = x/y :)
        # print(f'{ratio.shape=}')  # (bs,)

        clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)  # (bs,)
        # print(f'{clipped_ratio.shape=}')

        # for each action sample in the batch, 1st sum of the dimensions, 2nd the min between clipped and unclipped will be returned, 3rd mean
        policy_objective = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # calculate entropy, for the entropy bonus
        entropy = new_policy_action_dists.entropy().mean()

        loss = 1.0 * policy_objective + 1.0 * value_loss - self.alpha_entropy_bonus * entropy

        # update alpha_entropy_bonus using exponential day
        self.alpha_entropy_bonus *= 0.999

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
