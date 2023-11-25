import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
       

        # implement the rest
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space), std=0.01),
        )
        
        # Use log of std to make sure std (standard deviation) of the policy
        # doesn't become negative during training
        self.register_parameter(name='actor_logstd',
                                param=nn.parameter.Parameter(data=torch.zeros(action_dim, requires_grad=True, device=device)) )


    def forward(self, x):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimensions of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        # TODO: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        distr = Normal(loc=action_mean , scale=action_std) 

        return distr #probs