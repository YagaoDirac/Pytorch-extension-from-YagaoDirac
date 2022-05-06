import torch

from functional.GBN_functional import GBN_functional as GBN_functional

#class GBN_functional(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, training, epi_as_tensor = torch.tensor([1e-5])):
#        ctx.save_for_backward(torch.tensor([training], dtype= torch.bool), epi_as_tensor)
#        return x
#        pass
#    @staticmethod
#    def backward(ctx, g):
#        training, epi_as_tensor = ctx.saved_tensors
#        epi = epi_as_tensor.item()
#        if not training:
#            return g, None
#            pass
#
#        mean = g.mean(dim=0, keepdim=True)
#        _centralized = g - mean
#        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
#        std_too_small = std < epi
#        std = (std - std * std_too_small) + std_too_small * epi
#        _normalized = _centralized / std
#        return _normalized, None
#        pass
#    pass#class

class GBN(torch.nn.Module):
    def __init__(self, *, scale = 1.):
        '''Since BN messes up with the forward propagation, Now we have Gradient Batch Normalization.
        Notice, to replace x = relu(self.Linear_0(BN(x))),
        I recommend x = relu(GBN(self.Linear_0(x))).
        Reason is that, BN makes x in a range which helps the w to learn in a prefered speed.
        This GBN directly calculated prefered update strength, it helps both w and b to learn in a proper speed.
        Modify the scale value to scale the gradient.
        If you prefer the traditional BN but you also need the scale feature, try x = relu(self.Linear_0( BN(x)*scale ))
        Warning: this gradient adjusting flows backward all the way, it may mess up with the earlier layers.'''
        super(GBN, self).__init__()
        self.scale = scale
        pass
    def forward(self, x):
        x = GBN_functional.apply(x, self.training)
        if self.scale != 1.:
            return x*self.scale
        else:
            return x
        pass#def forward
    pass#class



if 0:
    layer1 = GBN()
    in1 = torch.tensor([1,2,3], dtype = torch.float32, requires_grad= True)
    in1_2 = torch.tensor([1,2,3], dtype = torch.float32)
    out1 = layer1(in1)*in1_2
    out1.mean().backward()

    layer2 = GBN().cuda()
    in2 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True).cuda()
    in2.retain_grad()#I don't know the reason. But this line helps.
    in2_2 = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
    out2 = layer2(in2) * in2_2
    out2.mean().backward()

    layer3 = GBN()
    in3 = torch.rand((3,2), dtype=torch.float32, requires_grad=True)
    in3_2 = torch.tensor([[1,2],[3,4],[5,6]], dtype=torch.float32)
    out3 = layer3(in3) * in3_2
    out3.mean().backward()
    pass
