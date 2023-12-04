from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOExtension(PPOAgent):
    # 2.4.1.2
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        self.entropy_coef = 0.01
        
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
        ## Smilar to BatchNorm in DL, Normalization of the returns is employed to make training more stable
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
        
        entropy = new_policy_action_dists.entropy().mean()  ########################### improvement
        
        loss = 1.0*policy_objective + 0.5*value_loss - self.entropy_coef*entropy   ########################### improvement

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        
#     # 2.4.1.1
#     @torch.no_grad()
#     def compute_returns(self):
#         _, next_state_values = self.policy(self.next_states) # act_distr , values
#         next_state_values = next_state_values.squeeze() # (bs, )

#         # TD1_return_estimate = self.rewards + (1-self.dones)*self.gamma*next_state_values # (bs,)
#         # return TD1_return_estimate # (bs,)

#         # Generalized Advantage Estimation TD(n)
#         _, values = self.policy(self.states) # act_distr , values
#         values = values.squeeze()
        
#         nx_robot = self.next_states[:,:2].unsqueeze(dim=1) # (bs, 1, 2)
#         # print(f'{nx_robot=}, {nx_robot.shape=}')
        
#         n_sanding = self.cfg.env_config['n_sanding']
#         n_no_sanding = self.cfg.env_config['n_no_sanding']
#         # print(f'{n_sanding=}, {n_no_sanding=}')
        
#         # print(f'{self.states=}, {self.states.shape=}')
#         sandings = self.states[:,2:2+2*n_sanding]
#         sandings = sandings.view(-1, n_sanding, 2) # (bs, n_sand, 2)
#         # print(f'{sandings=}, {sandings.shape=}')
        
#         sandings_distance = ((sandings - nx_robot)**2) # (bs, n_no_sand, 2)
#         # print(f'{sandings_distance=}, {sandings_distance.shape=}')
#         # remove -70s or -1.4s
#         sandings_distance[sandings==-1.4] = 0.0  # (bs, n_no_sand, 2)
#         # print(f'{sandings_distance=}, {sandings_distance.shape=}')
        
#         sandings_distance = (sandings_distance.sum(dim=-1)**0.5).sum(dim=-1) # (bs)
#         # print(f'{sandings_distance=}, {sandings_distance.shape=}')
        
#         sandings_reward = (4.0 - sandings_distance)/4.0
#         sandings_reward[sandings_reward==1.0] = 0.0
#         # print(f'{sandings_reward=}')
        
#         # sandings = sandings[sandings>-1.05].view(-1,2) # remove -70s or -1.4s
#         # print(f'{sandings=}, {sandings.shape=}')
        
        
#         no_sandings = self.states[:,2+2*n_sanding:]
#         no_sandings = no_sandings.view(-1, n_no_sanding, 2) # (bs, n_no_sand, 2)
#         # print(f'{no_sandings=}, {no_sandings.shape=}')
        
#         # no_sandings = no_sandings[no_sandings>-1.05]# remove -70s or -1.4s
#         # print(f'{no_sandings=}, {no_sandings.shape=}')
#         no_sandings_distance = ((no_sandings - nx_robot)**2) # (bs, n_no_sand, 2)
#         # print(f'{no_sandings_distance=}, {no_sandings_distance.shape=}')
#         # remove -70s or -1.4s
#         no_sandings_distance[no_sandings==-1.4] = 0.0  # (bs, n_no_sand, 2)
#         # print(f'{no_sandings_distance=}, {no_sandings_distance.shape=}')
        
#         no_sandings_distance = (no_sandings_distance.sum(dim=-1)**0.5).sum(dim=-1) # (bs)
#         # print(f'{no_sandings_distance=}, {no_sandings_distance.shape=}')
        
#         no_sandings_reward = (4.0 - no_sandings_distance)/4.0
#         no_sandings_reward[no_sandings_reward==1.0] = 0.0
        
#         # assert False
#         s_reward = sandings_reward - no_sandings_reward
        
        
        
        
#         TDn_return_estimate = []
#         gaes = torch.zeros(1)
#         timesteps = len(self.rewards)
#         for t in range(timesteps-1, -1, -1):
#             deltas = (self.rewards[t] + 0.05*s_reward[t]) + self.gamma * next_state_values[t] * (1-self.dones[t]) - values[t]
#             gaes = deltas + self.gamma*self.tau*(1-self.dones[t])*gaes
#             TDn_return_estimate.append(gaes + values[t])

#         return torch.Tensor(list(reversed(TDn_return_estimate))) # (bs,)
    
    # 2.1
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1:
            observation = observation[None] # add the batch dimension
        
        if evaluation:
            observation = observation/50.0 # input to the NN : (-1, 1)
            observation[observation==-1.4] = 0.0
        
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
            action = act_distr.sample()
        
        action = action.flatten()
        
        ########################### improvement
        '''
        we can hit a sanding spot when the distance between the robot position and the sanding spot is less than 0.2 or 10, so (-0.8, 0.8) or (-0.4, 0.4) can be a good limit for action.
        Also, the pd controller overshoots proportional to the distance between current robot position and the target position.
        
        so we can first clamp the mean action (-0.82 ,0.82) or (-41, 41),  
        '''
        robot = x[:,:2].squeeze()
        
        act_clamp = 0.85 # we do not need touch the boarder, so (-0.85, 0.85) is enough
        
        # action = act_clamp * action
        
        overshoot_brake = 0.5 # only go the (1-overshoot_brake)% of the path, the higher brake, the more brake when higher distance from target 
        
        action = torch.clamp( input=action,
                             min= -act_clamp + overshoot_brake * torch.abs(input= action - robot) ,
                             max= act_clamp - overshoot_brake * torch.abs(input= action - robot) )
        
        # action = action - overshoot_brake*(action-robot) # NOPE
        
        # action = torch.clamp( input=action, min= robot - overshoot_brake , max= robot + overshoot_brake )
        # action = torch.clamp( input=action, min= -act_clamp , max= act_clamp )
        
        ########################### improvement
        
        # print(f'{action=}, {action.shape=}')
        # print(f'{x=}, {x.shape=}')
        # assert False
        # action = torch.clamp(input=action, min=-1.0, max=1.0 )
        # action = torch.clamp(input=action, min=-45.0 + 0.3*torch.abs(input=action-x[:2]) , max=45.0 - 0.3*torch.abs(input=action-x[:,:2]) )
        # 45 - |target - current position| ( 45 - torch.abs(input=action-x[:2]) )
        # -45 + |target - current position|
        # Calculate the log probability of the action (T1)
        action_log_prob = act_distr.log_prob(action)
        # print(f'{action_log_prob=}, {action_log_prob.shape=}')
        
        return action, action_log_prob

        
        # 2
    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()
        observation = observation/50.0
        
        observation[observation==-1.4] = 0.0
        
        # print(f'{observation=}')
        # print(f'{observation[:2]=}')
        # assert False
        # $$$$$ remember to remove the seed when not debugging

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)

            # Perform the action on the environment, get new state and reward
            next_observation, reward, done, _, _ = self.env.step(action)
            next_observation = next_observation/50.0
            next_observation[next_observation==-1.4] = 0.0
            # print(f'{next_observation=}')
            
            # self.compute_distance(next_observation)
            
            # Store action's outcome (so that the agent can improve its policy)
            self.store_outcome(state=observation, action=action, next_state=next_observation, reward=reward,
                               action_log_prob=action_log_prob, done=done)

            # Store total episode reward
            reward_sum += reward
            episode_length += 1
            
            # update observation
            observation = next_observation.copy()

            # Update the policy, if we have enough data
            if len(self.states) > self.cfg.min_update_samples:
                # assert False
                self.update_policy()
                num_updates += 1

                # this is for the extension
                # Update policy randomness
                # self.policy.set_logstd_ratio(ratio_of_episodes)
                # self.entropy_coef = 0.01 * ratio_of_episodes**6

        # Return stats of training
        update_info = {'episode_length': episode_length, 'ep_reward': reward_sum}
        return update_info