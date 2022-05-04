import torch

class Gaussian_simple(torch.nn.Module):
    '''This is a simplified version of Gaussian.
    I simplified the formula as f(x) === e^(  -((x-b)/c)^2  )
    If bias = False, then the formula is f(x) === e^(  -(x/c)^2  )'''
    def __init__(self, size, bias = False, *, init_shrink_factor = 1., epi = 0.0001 , name = None):
        super(Gaussian_simple, self).__init__()
        self.name = name
        self.one_over_c = torch.nn.Parameter(torch.full((size, ), init_shrink_factor, dtype = torch.float32))
        self.b = None
        if bias:
            self.b = torch.nn.Parameter(torch.zeros_like(self.one_over_c))
            pass
        self.epi = epi
        pass
    def forward(self, x):
        #https://en.wikipedia.org/wiki/Gaussian_function
        if None != self.b:
            x = x-self.b
            pass
        x = x * self.one_over_c
        x = -x*x
        x = torch.exp(x)
        return x
        pass
    def __str__(self):
        return F'{self.name} Gaussian like activation. b: {self.b}  one_over_c: {self.one_over_c}'
    pass



if 0:
    layer = Gaussian_simple(3,bias = True)
    in1 = torch.rand(3)
    out1 = layer(in1)
    in2 = torch.rand(1, 3)
    out2 = layer(in2)
    in3 = torch.rand(5, 3)
    out3 = layer(in3)



    pass