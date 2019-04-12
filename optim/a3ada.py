import torch
from .optimizer import Optimizer, required
import pdb

class A3Ada(Optimizer):

    def __init__(self, params, lr=required, k=5, rho=0.9, eps=1e-6):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, k=k, rho=rho, eps=eps)
        super(A3Ada, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['u'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)
                self.state[p]['avg'] = torch.zeros_like(p.data)
                self.state[p]['delta_avg'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super(A3Ada, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3)

        for group in self.param_groups:
            rho = group['rho']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['avg'].mul_(rho).addcmul_(1 - rho, p.grad.data, p.grad.data)
                std = state['avg'].add(eps).sqrt_()
                delta = state['delta_avg'].add(eps).sqrt_().div_(std).mul(p.grad.data)
                lr = delta.mul(group['lr'])
                state['delta_avg'].mul_(rho).addcmul_(1 - rho, delta, delta)
                p.data = state['u'].add(state['v'].mul(lr.mul(beta)))

        loss = closure()
        for group in self.param_groups:
            k = group['k']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                std = state['avg'].add(eps).sqrt_()
                delta = state['delta_avg'].add(eps).sqrt_().div_(std)
                lr = delta.mul(group['lr'])
                d_p = p.grad.data
                state['v'] = state['v'].mul(1 - lr * beta).sub(d_p.mul(lr)).mul(beta ** k)
                state['u'].add_(p.data.sub(state['u']).mul((1 - lr * beta) * beta).sub(d_p.mul(lr ** 2)))

        return loss
