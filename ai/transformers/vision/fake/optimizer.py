import torch.optim as optim

class LayerDecayOptimizer:
    def __init__(self, _optimizer, _layerwise_decay_rate):
        self.optimizer = _optimizer
        self.layerwise_decay_rate = _layerwise_decay_rate

    def step(self, *args, **kwargs):
        for i, group in enumerate(self.optimizer.param_groups):
            #group['lr'] *= self.layerwise_decay_rate[i]
            print(group['lr'])
        self.optimizer.step(*args, **kwargs)
        
    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)