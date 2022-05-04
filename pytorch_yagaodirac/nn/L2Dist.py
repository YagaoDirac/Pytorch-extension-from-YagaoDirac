import torch

class L2Dist(torch.nn.Module):
    '''
    This layer is designed to work like
    x = Gaussian(self.L2Dist_layer(x))
    It gives out non negative results.
    The activation function after it better gives out non zero if given something near zero. But gives out 0 if given super big numbers.
    '''
    def __init__(self, in_dim, out_dim,* , name = None):
        super(L2Dist, self).__init__()
        self.name = name
        self.points = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        pass
    def forward(self, x):
        '''This method accepts the same shape as nn.Linear'''
        if len(list(x.shape)) == 1:
            x = x.view(1, 1, -1)
            pass
        else:
            x = x.view(x.shape[0], 1, -1)
            pass

        x = x-self.points
        x = x*x
        x = x.sum(dim = -1)
        x = x.view(x.shape[0],-1)
        return x
        pass

    def __str__(self):
        return F'{self.name} L2 distance layer. Points: {self.points}'
    pass#class


# test
if 0:
    layer = L2Dist(2, 3)
    layer.points = torch.nn.Parameter(torch.tensor([[2., 2.], [2., 3.], [2., 4.]]))
    in1 =  torch.tensor([2., 2.])
    out1 = layer(in1)
    in2 = torch.tensor([[2., 2.], [2., 2.01]])
    out2 = layer(in2)
    pass