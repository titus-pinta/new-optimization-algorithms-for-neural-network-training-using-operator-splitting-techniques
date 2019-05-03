import torch
from .optimizer import Optimizer, required


class SSA(Optimizer):

    def __init__(self, params, lr=required, k=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, k=k)
        super(SSA, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['theta'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(SSA, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3) #updte inertial parameter

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
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_p = p.grad.data
                state['v'] = beta ** k * ((1 - lr * beta) * state['v'] - lr * d_p)
                state['theta'] = state['theta'] + (1 - lr * beta) / beta * (p.data - state['theta']) - lr ** 2 * d_p

        return loss
