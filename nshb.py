import torch
from torch.optim import Optimizer


class NSHB(Optimizer):
  def __init__(self, params, lr, momentum=0, weight_decay=0):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
    super(NSHB, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(NSHB, self).__setstate__(state)

  def step(self, closure=None, itr=0):
    loss = None
    if closure is not None:
      loss = closure()

    if itr != 0:
      dp_list = []
      
    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        if weight_decay != 0:
          d_p.add_(weight_decay, p.data)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            buf.mul_(momentum).add_(d_p, alpha=(1-momentum))
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(d_p, alpha=(1-momentum))

          d_p = buf
          if itr != 0:
            dp_list.append(d_p)
            
        p.data.add_(d_p, alpha=-group['lr'])

    if itr != 0:
        return dp_list
    else:
      return loss
