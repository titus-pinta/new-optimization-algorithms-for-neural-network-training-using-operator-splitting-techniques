import torch
from .optimizer import Optimizer, required


class A11(Optimizer):

    def __init__(self, params, lr=required, k=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        print(k)
        defaults = dict(lr=lr, k=k)
        super(A11, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['v'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(A11, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3)

        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.add_(state['v'].mul(k * beta))

        loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_p = p.grad.data
                state['v'] = state['v'].mul(beta ** k).sub(d_p.mul(lr * beta ** (k - 1)))

        return loss
