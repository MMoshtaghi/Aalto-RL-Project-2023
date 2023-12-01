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
        new_values = new_values.squeeze() # (bs,)
        # print(f'{return_estimates.shape=}') # (bs,)

        # value_loss = (new_values - return_estimates).pow(2).mean() # (1)
        value_loss = F.smooth_l1_loss(new_values, return_estimates, reduction="mean") # (1)
        # print(f'{value_loss=}')

        advantages = return_estimates - new_values.detach() # (bs,) detach values to block gradient backprop
        # Similar to BatchNorm in DL, Normalization of the returns is employed to make training more stable
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # (bs,)
        # print(f'{advantages.shape=}')

        # print(f'{actions.shape=}') # (bs, act_dim)
        # Calculate the actor/policy loss: make new_policy_action_probs similar to advantage distribution to an extent
        new_policy_action_log_probs = new_policy_action_dists.log_prob(actions)  # (bs,) log prob of joint distribution of all dimensions Pr(x,y)
        ratio = torch.exp(new_policy_action_log_probs - action_log_probs) # exp(logx - logy) = x/y :)
        # print(f'{ratio.shape=}')  # (bs,)

        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)  # (bs,)
        # print(f'{clipped_ratio.shape=}')

        # for each action sample in the batch, 1st sum of the dimensions, 2nd the min between clipped and unclipped will be returned, 3rd mean
        policy_objective = -torch.min( ratio*advantages , clipped_ratio*advantages ).mean()

        # IMPROVEMENT: Entropy bonus
        # calculate entropy, for the entropy bonus
        entropy = new_policy_action_dists.entropy().mean()

        loss = 1.0 * policy_objective + 1.0 * value_loss - self.alpha_entropy_bonus * entropy
        # IMPROVEMENT OFF
        # loss = 1.0 * policy_objective + 1.0 * value_loss

        # update alpha_entropy_bonus using exponential day
        self.alpha_entropy_bonus *= 0.999
        # IMPROVEMENT

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)


        # 1. the self.policy returns a normal distribution, check the PyTorch document to see
        #    how to calculate the log_prob of an action and how to sample.
        # 2. if evaluating the policy, return policy mean, otherwise, return a sample
        # 3. the returned action and the act_logprob should be torch.Tensors.
        #    Please always make sure the shape of variables is as you expected.

        # Return mean if evaluation, else sample from the distribution
        # Pass state x through the policy network (T1)
        act_distr, _ = self.policy(x) #  act_distr, values
        if evaluation:
            # Return mean if evaluation, else sample from the distribution
            action = act_distr.mean
        else:
            # Consider the sanding area's dimensions of 100 units in both width and height when sampling x, y coordinates.
            '''
            Robot Characteristics:
            The robot is visualized as a purple circle with a radius of 10, operating on a 2D plane. The x and y coordinates range from -50 to 50.
            Sanding & No-Sanding Areas:
            There are sanding (green) and no-sanding (red) areas, each with a radius of 10. Their configurations vary based on the task.
            
            A state ( s ) is defined as:
            𝑠=[(𝑥ROBOT,𝑦ROBOT),(𝑥SAND,𝑦SAND)1,…,(𝑥SAND,𝑦SAND)𝑁,(𝑥NOSAND,𝑦NOSAND)1,…,(𝑥NOSAND,𝑦NOSAND)𝑀)]

            𝑁
              is the number of sanding areas (circles)
            𝑀
              is the number of no-sanding areas (circles)
            observation[0], observation[1] : (𝑥ROBOT,𝑦ROBOT): Robot's current location 
            observation[2(1+n)], observation[2(1+n)+1] : (𝑥SAND,𝑦SAND)𝑖 : Location of the  𝑖th sanding area
            observation[2(1+n+m)], observation[2(1+n+m)+1] : (𝑥NOSAND,𝑦NOSAND)𝑗 : Location of the  𝑗th no-sanding area
            
            $$$$$$$ Note: each spot that gets sanded by the robot is replaced by -70.0 !!!!!
            
            $$$$$$ The NN action_mean output must be between (-1, 1), then in the sanding.py, it gets multiplied by 50 to match the size of the environment
            '''
            action = act_distr.sample()

        action = action.flatten()
        x = x.flatten()

        # IMPROVEMENT
        '''
        we can hit a sanding spot when the distance between the robot position and the sanding spot is less than 0.2 or 10, so (-0.8, 0.8) or (-0.4, 0.4) can be a good limit for action.
        Also, the pd controller overshoots proportional to the distance between current robot position and the target position.
        
        so we can first clamp the mean action (-0.82 ,0.82) or (-41, 41),  
        '''
        # print(f'{action=}, {action.shape=}')
        # print(f'{x=}, {x.shape=}')
        action = 0.82 * action
        overshoot_brake = 0.3 # the higher brake, the more brake when higher distance from target
        action = torch.clamp(input=action, min=-0.9 + overshoot_brake*torch.abs(input=action-x[:2])/50.0 , max=0.9 - overshoot_brake*torch.abs(input=action-x[:2])/50.0 )
        # assert False
        # IMPROVEMENT

        
        
        action = torch.clamp(input=action, min=-1.0, max=1.0 )#
        # action = torch.clamp(input=action, min=-45.0 + 0.3*torch.abs(input=action-x[:2]) , max=45.0 - 0.3*torch.abs(input=action-x[:,:2]) )
        # 45 - |target - current position| ( 45 - torch.abs(input=action-x[:2]) )
        # -45 + |target - current position|
        # Calculate the log probability of the action (T1)
        action_log_prob = act_distr.log_prob(action)
        # print(f'{action_log_prob=}, {action_log_prob.shape=}')

        return action, action_log_prob