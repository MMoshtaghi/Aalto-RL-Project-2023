import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32, device):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
       

        # implement the rest
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space), std=0.01),
        )
        # Use log of std to make sure std (standard deviation) of the policy doesn't become negative during training
        self.actor_logstd = torch.zeros(action_dim, device=device)
        # Extend:
        # self.register_parameter(name='actor_logstd',
        #                         param=nn.parameter.Parameter(data=torch.zeros(action_space,
        #                                                                       requires_grad=True,
        #                                                                       device=device)) )
        
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1))
        )

    def forward(self, x):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(x)
        # Make sure action_logstd matches dimensions of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)
        # TODO: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        act_distr = Normal(loc=action_mean , scale=action_std) 
        
        
        self.value(x).squeeze(1) # output shape [batch,]

        return act_distr, value

    
    def set_logstd_ratio(self, ratio_of_episodes):
        pass # will be implemented in extension