import torch

def Gaussian(x):
    '''
    https://en.wikipedia.org/wiki/Gaussian_function
    Also I have a stateful version of Gaussian function in pytorch_yagaodirac.nn.Gaussian_simple
    '''
    x = -x * x
    x = torch.exp(x)
    return x
    pass

#def make_Linear(in_features, out_features, bias = True, ):
#    result = torch.nn.Linear(in_features, out_features, bias)
#
#    return