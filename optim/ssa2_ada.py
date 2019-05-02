import torch
from .optimizer import Optimizer, required


class SSA2Ada(Optimizer):

    def __init__(self, params, lr=required, k=5, q=5, rho=0.9, eps=1e-6):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, k=k, q=q, rho=rho, eps=eps)
        super(SSA2Ada, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['theta'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)
                self.state[p]['avg'] = torch.zeros_like(p.data)
                self.state[p]['delta_avg'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(SSA2Ada, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3) #update inertial parameter

        for group in self.param_groups:
            rho = group['rho']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['avg'] = rho * state['avg'] + (1 - rho) * p.grad.data ** 2 #accumulate gradient
                std = torch.sqrt(state['avg'] + eps) #root mean square of f
                delta = torch.sqrt(state['delta_avg'] + eps) / (std) * p.grad.data #compute update
                lr = delta * group['lr'] #compute adaptative step size
                state['delta_avg'] = rho * state['delta_avg'] + (1 - rho) * delta ** 2 #accumultate updates
                p.data = state['theta'] + lr * beta * state['v'] #compute aditional iteration

        loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            q = group['q']
            k = group['k']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                std = torch.sqrt(state['avg'] + eps)
                delta = torch.sqrt(state['delta_avg'] + eps) / std #compute adaptative stepsize
                lr = delta * group['lr']
                d_p = p.grad.data
                state['theta'] = state['theta'] + lr * (1 - lr * beta) * state['v'] #update information
                state['v'] = beta ** k *((1 - lr * beta) ** q * state['v'] - lr * d_p) #update velocity
        return loss
