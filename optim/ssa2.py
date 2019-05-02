import torch
from .optimizer import Optimizer, required


class SSA2(Optimizer):

    def __init__(self, params, lr=required, k=5, q=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, k=k, q=q)
        super(SSA2, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['theta'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(SSA2, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3) #update inertial parameter

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data = state['theta'] + lr * beta * state['v'] #compute value under gradient

        loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            q = group['q']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_p = p.grad.data
                state['theta'] = state['theta'] + lr * (1 - lr * beta) * state['v'] #update information
                state['v'] = beta ** k *((1 - lr * beta) ** q * state['v'] - lr * d_p) #update velocity


        return loss
