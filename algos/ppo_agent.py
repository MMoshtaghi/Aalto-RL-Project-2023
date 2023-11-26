from .agent_base import BaseAgent
from .ppo_utils import Policy
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOAgent(BaseAgent):
    def __init__(self, config=None):
        super(PPOAgent, self).__init__(config) # look at the attributes of the BaseAgent class
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy= Policy(state_space=self.observation_space_dim, # from BaseAgent : 6
                            action_space=self.action_space_dim, # from BaseAgent : 2
                            hidden_size=32,
                            device=self.device)
        self.lr=float(self.cfg.lr)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.clip = self.cfg.clip
        self.epochs = self.cfg.epochs
        self.running_mean = None
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []
        self.silent = self.cfg.silent

    # 2.4.1.2
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
        
        value_loss = F.smooth_l1_loss(new_values, return_estimates, reduction="mean") # (1)
        # print(f'{value_loss=}')

        advantages = return_estimates - new_values.detach() # (bs,) detach values to block gradient backprop
        ## Smilar to BatchNorm in DL, Normalization of the returns is employed to make training more stable
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # (bs,)
        # print(f'{advantages.shape=}')
        
        # print(f'{actions.shape=}') # (bs, act_dim)
        # Calculate the actor/policy loss: make new_policy_action_probs similar to advantage distribution to an extent
        new_policy_action_log_probs = new_policy_action_dists.log_prob(actions)  # (bs,) log prob of joint distribution of all dimensions Pr(x,y)
        # print(f'{action_log_probs.shape=}')
        # print(f'{new_policy_action_log_probs.shape=}')
        
        ratio = torch.exp(new_policy_action_log_probs - action_log_probs) # exp(logx - logy) = x/y :)
        # print(f'{ratio.shape=}')  # (bs,)
        
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)  # (bs,)
        # print(f'{clipped_ratio.shape=}')
        
        # for each action sample in the batch, 1st sum of the dimensions, 2nd the min between clipped and unclipped will be returned, 3rd mean
        policy_objective = -torch.min( ratio*advantages , clipped_ratio*advantages ).mean()
        
        # entropy = action_dists.entropy().mean()
        
        loss = 1.0*policy_objective + 1.0*value_loss # - 0.01*entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    # 2.4.1.1
    def compute_returns(self):
        with torch.no_grad():
            _, values = self.policy(self.states) # act_distr , values
            _, next_state_values = self.policy(self.next_states) # act_distr , values
            values = values.squeeze()
            next_values = next_values.squeeze()
        
        # TD1_return_estimate = rewards + not_dones*self.gamma*next_state_values # (bs,)
        # return TD1_return_estimate # (bs,)
        
        # Generalized Advantage Estimation TD(n)
        TDn_return_estimate = []
        gaes = torch.zeros(1)
        timesteps = len(self.rewards)
        for t in range(timesteps-1, -1, -1):
            deltas = self.rewards[t] + self.gamma * next_state_values[t] * (1-self.dones[t]) - values[t]
            gaes = deltas + self.gamma*self.tau*(1-self.dones[t])*gaes
            TDn_return_estimate.append(gaes + values[t])

        
        return torch.Tensor(list(reversed(TDn_return_estimate))) # (bs,)
    
    # 2.4.1
    def ppo_epoch(self):
        indices = list(range(len(self.states)))
        return_estimates = self.compute_returns()
        '''
        - we batchify the episode in order to have multiple updates of the value and policy networks in one episode.
        This way we easily have 2 versions of policy (old and new) to compare in one episode, instead of 2 versions from 2 episodes. 
        - For the 1st batch of epoch, the old policy and new_policy are the same, ratio=1,
        but for the others, we compare the new policy with the policy we took actions with during episode
        '''
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size, replace=False)

            # Do the update
            self.ppo_update(states=self.states[batch_indices], actions=self.actions[batch_indices],
                rewards=self.rewards[batch_indices], next_states=self.next_states[batch_indices],
                dones=self.dones[batch_indices], action_log_probs=self.action_log_probs[batch_indices],
                return_estimates=return_estimates[batch_indices])

            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]
    
    # 2.4
    def update_policy(self):
        if not self.silent:
            print("Updating the policy...")
        
        # print(f'{self.states.shape=}')
        self.states = torch.stack(self.states, dim=0).to(self.device).squeeze(-1) # (bs, state_dim)
        # print(f'{self.states.shape=}')
        # print(f'{self.next_states.shape=}')
        self.next_states = torch.stack(self.next_states, dim=0).to(self.device).squeeze(-1)  # (bs, state_dim)
        # print(f'{self.next_states.shape=}')
        # print(f'{self.actions.shape=}')
        self.actions = torch.stack(self.actions, dim=0).to(self.device).squeeze()  # (bs, act_dim)
        # print(f'{self.actions.shape=}')
        # print(f'{self.action_log_probs.shape=}')
        self.action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.device).squeeze()  # (bs, act_dim)
        # print(f'{self.action_log_probs.shape=}')
        # print(f'{self.rewards.shape=}')
        self.rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1) # (bs,)
        # print(f'{self.rewards.shape=}')
        # print(f'{self.dones.shape=}')
        self.dones = torch.stack(self.dones, dim=0).to(self.device).squeeze(-1) # (bs,)
        # print(f'{self.dones.shape=}')
        
        for e in range(self.epochs):
            self.ppo_epoch()

        # Clear the episode buffer
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []

        if not self.silent:
            print("Updating finished!")
        
        return {}
    
    # 2.1
    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)


        # 1. the self.policy returns a normal distribution, check the PyTorch document to see 
        #    how to calculate the log_prob of an action and how to sample.
        # 2. if evaluating the policy, return policy mean, otherwise, return a sample
        # 3. the returned action and the act_logprob should be torch.Tensors.
        #    Please always make sure the shape of variables is as you expected.
        
        if evaluation:
            with torch.no_grad():
                # Pass state x through the policy network (T1)
                act_distr, _ = self.policy(x) #  act_distr, values

                # Return mean if evaluation, else sample from the distribution
                action = act_distr.mean
        else:
            # Pass state x through the policy network (T1)
            act_distr, _ = self.policy(x) #  act_distr, values
            # Return mean if evaluation, else sample from the distribution
            action = act_distr.sample()
        
        # Calculate the log probability of the action (T1)
        action_log_prob = act_distr.log_prob(action)
        
        return action, action_log_prob

    # 2
    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)

            # Perform the action on the environment, get new state and reward
            next_observation, reward, done, _, _ = self.env.step(self.to_numpy(action))
            
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
                self.update_policy()
                num_updates += 1

                # this is for the extension
                # Update policy randomness
                self.policy.set_logstd_ratio(ratio_of_episodes)

        # Return stats of training
        update_info = {'episode_length': episode_length, 'ep_reward': reward_sum}
        return update_info

# 1
    def train(self):
        if self.cfg.save_logging: 
            L = cu.Logger() # create a simple logger to record stats
        total_step=0
        run_episode_reward=[]
        start = time.perf_counter()

        for ep in range(self.cfg.train_episodes+1):
            ratio_of_episodes = (self.cfg.train_episodes - ep) / self.cfg.train_episodes # begins with 1 -> goes to 0
            train_info = self.train_iteration(ratio_of_episodes)
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            logstd = self.policy.actor_logstd
            
            if total_step%self.cfg.log_interval==0:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return} ({train_info['episode_length']} episode_length, {logstd} logstd)")

                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]

        # Save the model
        if self.cfg.save_model:
            self.save_model()

        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()
        
        end = time.perf_counter()
        train_time = (end-start)/60
        print("------Training finished.------")
        print(f'Total traning time is {train_time}mins')
    
    def load_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        state_dict = torch.load(filepath)
        self.policy.load_state_dict(state_dict)
    
    def save_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        torch.save(self.policy.state_dict(), filepath)
        print("Saved model to", filepath, "...")

    # 2.2
    def to_numpy(self, tensor):
        return tensor.cpu().numpy().flatten()

    # 2.3
    def store_outcome(self, state, action, next_state, reward, action_log_prob, done): # added
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob.detach())
        self.rewards.append(torch.Tensor([reward]).float())
        self.dones.append(torch.Tensor([done]))
        self.next_states.append(torch.from_numpy(next_state).float())
