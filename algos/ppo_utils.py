import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent #, MultivariateNormal
import numpy as np
import  torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, device, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
       

        # implement the rest (similar to ex1 configuration as requested in the instruction)
        self.device = device
        
        # self.actor_mean = nn.Sequential(
        #     nn.Linear(state_space, hidden_size),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, action_space),
        #     nn.Tanh(), #The NN action_mean output must be between (-1, 1), then in the sanding.py, it gets multiplied by 50 to match the size of the environment
        # )
        '''
        As TA said on Zulip:"
        - For the NN setting , yes you can use the same architecture [as the ex1, with same depth and width]
        - It is typical to add the tanh layer at the end of NN such that action is in the proper range
        for the previous activation layer you can use any
        '''
        # IMPROVEMENT
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space), std=0.01),
            nn.Tanh(), #The NN action_mean output must be between (-1, 1), then in the sanding.py, it gets multiplied by 50 to match the size of the environment
        )
            
        # Use log of std to make sure std (standard deviation) of the policy doesn't become negative during training
        # Consider the sanding area's dimensions of 100 units in both width and height when sampling x, y coordinates.
        '''
        Robot Characteristics:
        The robot is visualized as a purple circle with a radius of 10, operating on a 2D plane. The x and y coordinates range from -50 to 50.
        Sanding & No-Sanding Areas:
        There are sanding (green) and no-sanding (red) areas, each with a radius of 10. Their configurations vary based on the task.
        '''
        # IMPROVEMENT
        # self.actor_logstd = torch.log( 0.2*torch.ones(self.action_space, device=self.device) )
        self.actor_logstd = torch.log( 0.001*torch.ones(self.action_space, device=self.device) )
        
        
        # self.actor_logstd = torch.ones(self.action_space, device=self.device)
        # Extend:
        # self.register_parameter(name='actor_logstd',
        #                         param=nn.parameter.Parameter(data=torch.zeros(action_space,
        #                                                                       requires_grad=True,
        #                                                                       device=device)) )
        
        # self.value = nn.Sequential(
        #     nn.Linear(state_space, hidden_size),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, 1)
        # )
        
        #IMPROVEMENT
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1)),
        )
            
        # self.init_weights()

        
    # def init_weights(self):
    #     for m in self.modules():
    #         if type(m) is torch.nn.Linear:
    #             torch.nn.init.normal_(m.weight, 0, 1e-1)
    #             torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        # print(x.shape)
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(x) # (bs, act_dim)
        # print(f'{action_mean.shape=}')
        # assert action_mean.shape == torch.Size([2])
        # Make sure action_logstd matches dimensions of action_mean
        # action_logstd = self.actor_logstd.expand_as(action_mean) # (bs, act_dim)
        
        # Exponentiate the log std to get actual std
        # print("self.actor_logstd", self.actor_logstd)
        action_std = torch.exp(self.actor_logstd) # (act_dim,)
        # torch.diag(action_std) : (act_dim, act_dim)
        
        # Gaussian policy with an isotropic covariance matrix:
        # A covariance matrix C is called isotropic, or spherical, if it is proportionate to the identity matrix
        # so Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        # act_distr = MultivariateNormal( loc=action_mean, scale_tril=torch.diag(action_std) )
        
        act_normal_distr = Normal(loc=action_mean, scale=action_std)
        act_distr = Independent(base_distribution=act_normal_distr, reinterpreted_batch_ndims=1)
        
        value = self.value(x) # output shape [bs,]

        return act_distr, value

    
    def set_logstd_ratio(self, ratio_of_episodes):
        self.actor_logstd = torch.log( 0.2*torch.ones(self.action_space, device=self.device) )
        
    def set_logstd_ratio_normalized(self, ratio_of_episodes):
        # IMPROVEMENT
        self.actor_logstd = torch.log( 0.5*(ratio_of_episodes+0.001)*torch.ones(self.action_space, device=self.device) )
        # print("ratio_of_episodes", ratio_of_episodes) 
        # print("self.actor_logstd", self.actor_logstd)
        # self.actor_logstd = .05 * np.exp(ratio_of_episodes) * torch.ones(self.action_space, device=self.device)