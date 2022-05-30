import torch

class Gaussian_functional(torch.nn.Module):
    def __init__(self, *, name = None):
        '''
            https://en.wikipedia.org/wiki/Gaussian_function
            Also I have a stateful version of Gaussian function in pytorch_yagaodirac.nn.Gaussian_simple
            '''
        super(Gaussian_functional, self).__init__()
        self.name = name
        pass
    def forward(self, x):
        return Gaussian(x)
        pass
    def __str__(self):
        return F'{self.name} Stateless Gaussian like activation.'
        pass
    pass#class

def Gaussian(x):
    '''
    https://en.wikipedia.org/wiki/Gaussian_function
    Also I have a stateful version of Gaussian function in pytorch_yagaodirac.nn.Gaussian_simple
    '''
    x = -x * x
    x = torch.exp(x)
    return x
    pass