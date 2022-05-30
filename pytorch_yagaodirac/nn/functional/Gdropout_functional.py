import torch

class Gdropout_functional(torch.autograd.Function):
    '''See pytorch_yagaodirac.nn.Gdropout for doc.'''
    @staticmethod
    def forward(ctx, x, p):
        p_as_tensor = torch.tensor([p],dtype = torch.float32)
        ctx.save_for_backward(p_as_tensor)
        return x
        pass
    @staticmethod
    def backward(ctx, g):
        p_as_tensor, = ctx.saved_tensors
        p = p_as_tensor.item()
        mul_factor = 1/(1-p)

        flag = torch.rand_like(g)>p
        flag = flag * mul_factor
        g = g*flag
        return g, None
    pass#class