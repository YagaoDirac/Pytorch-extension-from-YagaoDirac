import torch

class GBN_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epi=torch.tensor([1e-12]), suppression_factor = torch.tensor([1e3])):
        ctx.save_for_backward(scale, epi, suppression_factor)
        return x
        pass

    @staticmethod
    def backward(ctx, g):
        scale, epi, suppression_factor = ctx.saved_tensors
        #scale, epi = ctx.saved_tensors

        mean = g.mean(dim=0, keepdim=True)
        _centralized = g - mean
        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
        std_too_small = std < epi
        std = (std - std * std_too_small) + std_too_small * (epi* suppression_factor)
        _normalized = _centralized / std
        if scale != 1.:
            return _normalized * scale, None
            pass
        else:
            return _normalized, None, None, None
            pass
        pass
    pass  # class


class GBN(torch.nn.Module):
    def __init__(self, scale=1., *, epi=1e-12, suppression_factor=1e3):  # , *, lr = 1e-2):
        super(GBN, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor([scale], dtype=torch.float32))
        self.dynamic_scale = torch.nn.Parameter(torch.ones(1, ))
        self.epi = torch.nn.Parameter(torch.tensor([epi], dtype=torch.float32))
        self.suppression_factor = torch.nn.Parameter(torch.tensor([suppression_factor], dtype=torch.float32))
        pass

    def forward(self, x):
        if 1 == x.shape[0]:
            raise ValueError("expected anything greater than 1 for the first dimension (got 1 for the first dim)")
            pass
        if 1 == len(list(x.shape)):
            raise ValueError("expected 2D or higher dimension input (got 1D input)")
            pass
        x = GBN_functional.apply(x, self.get_working_scale(), self.epi, self.suppression_factor)
        return x
        pass  # def forward

    def set_dynamic_scale(self, dynamic_scale):
        '''set scale keeps the lr.'''
        self.dynamic_scale = dynamic_scale
        pass

    def get_working_scale(self):
        return self.scale * self.dynamic_scale
        pass

    pass  # class


if 1:
    gbn1 = GBN(epi=1e-12)
    gbn2 = GBN(epi=1e-12)
    W1 = torch.nn.Parameter(torch.tensor([-1.]))
    W2 = torch.nn.Parameter(torch.tensor([2.]))
    x = torch.tensor([[0.3],[0.1]])


    x = W1 * x
    x = gbn1(x)
    x = W2 * x
    x = gbn2(x)

    x = x*torch.tensor([[0.0005],[0.0002]])
    x.backward(torch.ones_like(x))

    d1 = W1.grad
    d2 = W2.grad

    jklsdf=890456

    pass