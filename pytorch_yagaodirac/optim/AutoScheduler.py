import torch

class AutoScheduler:
    def __init__(self, optimizer, distance = 100, gamma = 0.9):
        self.dist = distance# assuming loss > lr. So, this distance is define as some value near loss/lr
        self.gamma = gamma
        self.reverse_gamma = 1/gamma

        # The real core:
        self.sdl = torch.optim.lr_scheduler.LambdaLR(optimizer, self.sdl_fn)
        self.init_lr = self.sdl.get_last_lr()[0]
        #self.sdl = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, self.sdl_fn)
        pass

    def sdl_fn(self, n):
        if 0 == n:
            return 1
            pass

        result = 1

        last_lr = self.sdl.get_last_lr()[0]
        ratio = self.loss / (last_lr * self.dist)
        if ratio < 1:
            result = min(ratio, self.gamma)
            pass
        if ratio > 10:
            result = self.reverse_gamma
            pass

        return (result*last_lr)/self.init_lr
        pass
    def step(self, loss):
        self.loss = loss
        self.sdl.step()
        pass
    pass