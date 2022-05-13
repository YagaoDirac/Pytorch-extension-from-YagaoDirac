import torch

class Gaussian_functional(torch.nn.Module):
    def __init__(self, *, name = None):
        super(Gaussian_functional, self).__init__()
        self.name = name
        pass
    def forward(self, x):
        #https://en.wikipedia.org/wiki/Gaussian_function
        x = -x*x
        x = torch.exp(x)
        return x
        pass
    def __str__(self):
        return F'{self.name} Stateless Gaussian like activation.'
        pass
    pass#class
