import torch

class Gaussian_like_activation(torch.nn.Module):
    '''This is a simplified version of Gaussian.
    The formula might be wrong..'''
    def __init__(self, size, * , w = 1., l_like = 1., sigma_like = 2., epi = 0.0001):
        super(Gaussian_like_activation, self).__init__()
        self.w          = torch.nn.Parameter(torch.full((size, ), w         , dtype = torch.float32), True)
        self.l_like     = torch.nn.Parameter(torch.full((size, ), l_like    , dtype = torch.float32), True)
        self.sigma_like = torch.nn.Parameter(torch.full((size, ), sigma_like, dtype = torch.float32), True)
        self.epi = epi
        pass
    def forward(self, x):
        with torch.no_grad():
            flags = torch.abs(self.sigma_like)<self.epi
            minus_me = self.sigma_like*flags
            add_me = self.epi*flags
            self.sigma_like = self.sigma_like-minus_me+add_me
            pass

        x = x * self.w
        numerator = -torch.pow(x * x, self.l_like)
        result = torch.exp(numerator / (self.sigma_like * self.sigma_like))
        return result
        pass
    def __str__(self):
        return F'Gaussian like activation. w: {self.w}  l_like: {self.l_like}  sigma_like: {self.sigma_like}'
    pass
