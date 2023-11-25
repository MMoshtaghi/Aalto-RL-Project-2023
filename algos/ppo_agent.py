from .agent_base import BaseAgent
from .ppo_utils import Policy
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOAgent(BaseAgent):
    def __init__(self, config=None):
        super(PPOAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy= Policy(state_space=self.cfg.action_space_dim, # based on Setup in project file
                            action_space=self.cfg.observation_space_dim,
                            hidden_size=32,
                            device=self.device)
        self.lr=self.cfg.lr

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

        
    def update_policy(self):
        if not self.silent:
            print("Updating the policy...")

        states = torch.stack(self.states, dim=0).to(device).squeeze(-1) # (bs, state_dim)
        next_states = torch.stack(self.next_states, dim=0).to(device).squeeze(-1)  # (bs, state_dim)
        actions = torch.stack(self.actions, dim=0).to(device).squeeze(-1)  # (bs, act_dim)
        action_log_probs = torch.stack(self.action_probs, dim=0).to(device).squeeze(-1)  # (bs, act_dim)
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # (bs,)
        dones = torch.stack(self.dones, dim=0).to(device).squeeze(-1) # (bs,)
        # clear buffer
        self.states, self.action_probs, self.rewards, self.dones, self.next_states = [], [], [], [], []

        # Advantage actor critic
        # calculate the TD return estimate
        
        # n_step_TD_return_estimate = u.discount_rewards(next_state_values, self.gamma)
    
        next_state_values = self.value(next_states).detach()
        # print(f'{next_state_values=}, {next_state_values.shape=}')
        not_dones = torch.logical_not(input=dones, out=torch.empty_like(dones, dtype=torch.int16))
        TD_return_estimate = rewards + not_dones*self.gamma*next_state_values # (bs,)
        # print(f'{not_dones=}, {dones=}')
        # print(f'{n_step_TD_return_estimate=}, {n_step_TD_return_estimate.shape=}')
        state_values = self.value(states) # (bs,)
        # print(f'{state_values=}, {state_values.shape=}')
        advantages = TD_return_estimate - state_values # (bs,)
        
        # critic/value loss
        critic_loss = advantages.pow(2).mean()
        
        ## Smilar to BatchNorm in DL, Normalization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is added to the standard deviation of the returns to avoid division by zero
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps) # (bs,)
        # # print(f'{advantages=}, {advantages.shape=}')
        # # print(f'{act_logprobs=}, {act_logprobs.shape=}, {act_logprobs.sum(dim=1)=}, {act_logprobs.sum(dim=1).shape=}')
        
        # Calculate the actor/policy loss: Negative log likelihood : make the output distribution similar to advantage distribution
        # I = torch.zeros_like(advantages)
        # for idx in range(advantages.shape[0]):
        #     I[idx] = self.gamma**idx
        # actor_loss = -( act_logprobs.sum(dim=1) * I * advantages ).mean()
        actor_loss = -( act_logprobs.sum(dim=1) * advantages ).mean()
        
        # print(f'{actor_loss=}, {critic_loss=}')
        
        # Compute the optimization term 
        loss = 1*actor_loss + 1*critic_loss

        # perform backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Update network parameters using self.optimizer and zero gradients 
        self.optimizer.step()
        if not self.silent:
            print("Updating finished!")
        
        rreturn {}

    
    def ppo_epoch(self):
        indices = list(range(len(self.states)))
        returns = self.compute_returns()
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)

            # Do the update
            self.ppo_update(self.states[batch_indices], self.actions[batch_indices],
                self.rewards[batch_indices], self.next_states[batch_indices],
                self.dones[batch_indices], self.action_log_probs[batch_indices],
                returns[batch_indices])

            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]
    
    
    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)


        # 1. the self.policy returns a normal distribution, check the PyTorch document to see 
        #    how to calculate the log_prob of an action and how to sample.
        # 2. if evaluating the policy, return policy mean, otherwise, return a sample
        # 3. the returned action and the act_logprob should be torch.Tensors.
        #    Please always make sure the shape of variables is as you expected.
        
        # Pass state x through the policy network (T1)
        act_distr, _ = self.policy(x) #  act_distr, value
        
        # Return mean if evaluation, else sample from the distribution
        action = act_distr.mean if evaluation else act_distr.sample()
        
        # Calculate the log probability of the action (T1)
        action_log_prob = act_distr.log_prob(action)
        return action, action_log_prob

        
    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)
            previous_observation = observation.copy()

            # Perform the action on the environment, get new state and reward
            observation, reward, done, _, _ = self.env.step(action)
            
            # Store action's outcome (so that the agent can improve its policy)
            self.store_outcome(previous_observation, action, observation,
                                reward, action_log_prob, done)

            # Store total episode reward
            reward_sum += reward
            episode_length += 1

            # Update the policy, if we have enough data
            if len(self.states) > self.cfg.min_update_samples:
                self.update_policy()
                num_updates += 1

                # Update policy randomness
                self.policy.set_logstd_ratio(ratio_of_episodes)

        # Return stats of training
        update_info = {'episode_length': episode_length,
                    'ep_reward': reward_sum}
        return update_info


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
