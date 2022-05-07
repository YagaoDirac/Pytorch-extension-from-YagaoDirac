import torch

from .functional.GBN_functional import GBN_functional as GBN_functional

#class GBN_functional(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, scale, epi_as_tensor = torch.tensor([1e-5])):
#        scale_as_tensor = torch.tensor([scale],dtype = torch.float32)
#        ctx.save_for_backward(scale_as_tensor, epi_as_tensor)
#        return x
#        pass
#    @staticmethod
#    def backward(ctx, g):
#        scale_as_tensor, epi_as_tensor = ctx.saved_tensors
#        scale = scale_as_tensor.item()
#        epi = epi_as_tensor.item()
#
#        mean = g.mean(dim=0, keepdim=True)
#        _centralized = g - mean
#        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
#        std_too_small = std < epi
#        std = (std - std * std_too_small) + std_too_small * epi
#        _normalized = _centralized / std
#        if scale != 1.:
#            return _normalized*scale, None
#            pass
#        else:
#            return _normalized, None
#            pass
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
        x = GBN_functional.apply(x, self.scale)
        return x
        pass#def forward
    pass#class



if 0:
    layer1 = GBN(scale= 10)
    in1 = torch.tensor([1,2,3], dtype = torch.float32, requires_grad= True)
    in1_2 = torch.tensor([1,2,3], dtype = torch.float32)
    out1 = layer1(in1)*in1_2
    out1.mean().backward()
    #Then, in1.grad is prepared. It's 12.2. If the scale is 1 by default, the grad is 1.22.

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
