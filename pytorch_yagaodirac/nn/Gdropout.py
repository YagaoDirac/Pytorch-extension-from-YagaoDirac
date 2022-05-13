import torch

#from .functional.Gdropout_functional import Gdropout_functional as Gdropout_functional

class Gdropout_functional(torch.autograd.Function):
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



class Gdropout(torch.nn.Module):
    def __init__(self, *, p = 0.5):
        '''I don't like anything that messes with the forward propagation.
        Since GBN works better than expected, I think a dropout affects only the gradient would also works great.
        All the details are copied from the torch.nn.Dropout. Find doc in official doc web.'''
        super(Gdropout, self).__init__()
        assert p>=0.0001, "p is too small"
        assert p<=0.9999, "p is too big"
        self.p = p
        pass
    def forward(self, x):
        x = Gdropout_functional.apply(x, self.p)
        return x
        pass#def forward
    def get_mul_factor(self):
        return 1/(1-self.p)
        pass
    pass#class



if 1:
    layer1 = Gdropout(p = 0.2)
    ddd = layer1.get_mul_factor()
    in1 = torch.tensor([1,2,3], dtype = torch.float32, requires_grad= True)
    in1_2 = torch.tensor([1,2,3], dtype = torch.float32)
    out1 = layer1(in1)*in1_2
    out1.mean().backward()
    #Then, in1.grad is prepared. It's 12.2. If the scale is 1 by default, the grad is 1.22.

    layer2 = Gdropout(p = 0.2).cuda()
    in2 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True).cuda()
    in2.retain_grad()#I don't know the reason. But this line helps.
    in2_2 = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
    out2 = layer2(in2) * in2_2
    out2.mean().backward()
    pass
