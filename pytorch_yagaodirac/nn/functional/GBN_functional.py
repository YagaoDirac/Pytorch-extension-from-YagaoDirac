import torch

class GBN_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training, epi_as_tensor = torch.tensor([1e-5])):
        ctx.save_for_backward(torch.tensor([training], dtype= torch.bool), epi_as_tensor)
        return x
        pass
    @staticmethod
    def backward(ctx, g):
        training, epi_as_tensor = ctx.saved_tensors
        epi = epi_as_tensor.item()
        if not training:
            return g, None
            pass

        mean = g.mean(dim=0, keepdim=True)
        _centralized = g - mean
        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
        std_too_small = std < epi
        std = (std - std * std_too_small) + std_too_small * epi
        _normalized = _centralized / std
        return _normalized, None
        pass
    pass#class