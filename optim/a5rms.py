import torch
from .optimizer import Optimizer, required


class A5RMS(Optimizer):

    def __init__(self, params, lr=required, k=5, q=5, alpha=0.9, eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, k=k, q=q, alpha=alpha, eps=eps)
        super(A5RMS, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['u'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)
                self.state[p]['avg'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(A5RMS, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1
        beta = self.state['n_iter'] / (self.state['n_iter'] + 3)

        for group in self.param_groups:
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['avg'].mul_(alpha).addcmul_(1 - alpha, p.grad.data, p.grad.data)
                lr = state['avg'].sqrt().add(group['eps']).pow(-1).mul(group['lr'])
                p.data = state['u'].add(state['v'].mul(lr * beta))

        loss = closure()
        for group in self.param_groups:
            k = group['k']
            q = group['q']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                d_p = p.grad.data
                lr = ((state['avg'].sqrt().add(group['eps'])).pow(-1)).mul(group['lr'])
                state['v'] = state['v'].mul((1 - lr * beta) ** q).sub(d_p.mul(lr * beta ** k))
                state['u'].add_(p.data.sub(state['u']).mul((1 - lr * beta) / beta).sub(d_p.mul(lr ** 2)))

        return loss
