import sys, os
sys.path.insert(0, os.path.abspath(".."))
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
from common import helper as h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ReLU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i+1]), act_fn()]
    # the output layer
    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)



    

class DDQNAgent(object):
    def __init__(self, state_shape, n_actions,
                 batch_size=32, hidden_dims=[12], shared_layer_dim=256, dual_layers_dims= [256,256], 
                 gamma=0.98, lr=1e-3, grad_clip_norm=1000, tau=0.001):
        self.n_actions = n_actions
        self.state_dim = state_shape[0]
        self.outdim = shared_layer_dim 
        self.dual_layers = dual_layers_dims 

        self.policy_net_shared = mlp(self.state_dim, hidden_dims, self.outdim).to(device) #n_actions = shared layer dimension
        
        self.state_val = mlp(self.outdim, self.dual_layers, 1).to(device) # state value
        self.advantage = mlp(self.outdim, self.dual_layers, self.n_actions).to(device) # for each action
        
     
        self.optimizer = optim.Adam([
            {'params': self.policy_net_shared.parameters()},
            {'params': self.state_val.parameters()},
            {'params': self.advantage.parameters()}],
            lr=lr)
            
        
        self.target_net_shared = copy.deepcopy(self.policy_net_shared).to(device) 
        self.target_net_sv = copy.deepcopy(self.state_val).to(device) 
        self.target_net_ad = copy.deepcopy(self.advantage).to(device)
        
        self.target_net_shared.eval()
        self.target_net_sv.eval()
        self.target_net_ad.eval()

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        
        self.counter = 0
        
    

    def update(self, buffer):
        """ One gradient step, update the policy net."""
        self.counter += 1
        # Do one step gradient update
        batch = buffer.sample(self.batch_size, device=device)
        
        # TODO: Task 3: Finish the DQN implementation.
        ########## You code starts here #########
        # Hints: 1. You can use torch.gather() to gather values along an axis specified by dim. 
        #        2. torch.max returns a namedtuple (values, indices) where values is the maximum 
        #           value of each row of the input tensor in the given dimension dim.
        #           And indices is the index location of each maximum value found (argmax).
        #        3.  batch is a namedtuple, which has state, action, next_state, not_done, reward
        #           you can access the value be batch.<name>, e.g, batch.state
        #        4. check torch.nn.utils.clip_grad_norm_() to know how to clip grad norm
        #        5. You can go throught the PyTorch Tutorial given on MyCourses if you are not familiar with it.

        f_state = batch.state.to(device)
        f_next_state = batch.next_state.to(device)
        actions = batch.action.to(device)
        rewards = batch.reward.to(device)
        not_done = batch.not_done.to(device)
        q_tar = 0 

        # calculate the q(s,a)
        shared = self.policy_net_shared(f_state)
        
        sv_stream = self.state_val(shared)
        ad_stream = self.advantage(shared)
        
        max_a, _ = torch.max(ad_stream, axis=1)
        
        # calculate according to eqn 8
        qs_state = torch.add(sv_stream, torch.subtract(ad_stream, max_a.reshape((len(max_a), 1))))
        
        actions = actions.to(dtype=torch.int64)
        qs = torch.gather(qs_state, 1, actions)

#         # predict the q values for the next state
        shared_next = self.target_net_shared(f_next_state)
        sv_next = self.target_net_sv(shared_next)
        ad_next = self.target_net_ad(shared_next)
        max_a_next, _ = torch.max(ad_next, 1)
        q_values_next =  torch.add(sv_next, torch.subtract(ad_next, max_a_next.reshape((len(max_a_next), 1))))
       
        # take the max over all possible q values per observation
        best_q_values_next_state, ind = torch.FloatTensor(q_values_next.cpu()).max(axis=1)
        best_q_values_next_state = torch.reshape(best_q_values_next_state, (len(best_q_values_next_state), 1))        

        best_q_values_next_state = torch.mul(not_done, best_q_values_next_state.to(device))

        # calculate q target (check q-learning)
        q_tar = rewards + self.gamma * best_q_values_next_state
    
        # detach q_tar to avoid grad computation
        q_tar = q_tar.detach()       

        #calculate the loss 
        loss= torch.mean((qs - q_tar)**2)

        self.optimizer.zero_grad()
        
        loss.backward()
        
        # clip grad norm and perform the optimization step
#         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        ########## You code ends here #########

        h.soft_update_params(self.policy_net_shared, self.target_net_shared, self.tau)
        h.soft_update_params(self.state_val, self.target_net_sv, self.tau)      
        h.soft_update_params(self.advantage, self.target_net_ad, self.tau) 
        
        return {'loss': loss.item(), 
                'q_mean': qs.mean().item(),
                'num_update': self.counter}


    @torch.no_grad()
    def get_action(self, state, epsilon=0.5):
        # TODO:  Task 3: implement epsilon-greedy action selection
        ########## You code starts here #########
        
        shared = self.policy_net_shared(torch.tensor(state).to(device))#.cpu()
        sv_stream = self.state_val(shared)
        ad_stream = self.advantage(shared)
               
        max_a = torch.max(ad_stream) #batch_size=1
       
        qs_state = (torch.add(sv_stream, torch.subtract(ad_stream, max_a))).cpu()
      
        
        random_num = np.random.sample(1)[0]

        if random_num <= epsilon:
            # explore
            action = np.random.choice(list(range(self.n_actions)))
            return action
        else:
            greedy_action = torch.argmax(qs_state).detach().numpy()
            return greedy_action

       
        ########## You code ends here #########


    def save(self, fp):
        path = fp
        torch.save({
#             'policy': self.policy_net.state_dict(),
#             'policy_target': self.target_net.state_dict()
            'policy_net_shared': self.policy_net_shared.state_dict(),
            'policy_state_val': self.state_val.state_dict(),
            'policy_advantage': self.advantage.state_dict(),
            
            'target_net_shared': self.target_net_shared.state_dict(),
            'target_state_val': self.target_net_sv.state_dict(),
            'target_advantage': self.target_net_ad.state_dict()
        }, path)

    def load(self, fp):
        path = fp
        #d = torch.load(path)
        d = torch.load(fp,map_location=torch.device('cpu'))
#         self.policy_net.load_state_dict(d['policy'])
#         self.target_net.load_state_dict(d['policy_target'])
        self.policy_net_shared.load_state_dict(d['policy_net_shared'])
        self.state_val.load_state_dict(d['policy_state_val'])
        self.advantage.load_state_dict(d['policy_advantage'])
        
        self.target_net_shared.load_state_dict(d['target_net_shared'])
        self.target_net_sv.load_state_dict(d['target_state_val'])
        self.target_net_ad.load_state_dict(d['target_advantage'])