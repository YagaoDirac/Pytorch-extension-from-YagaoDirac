import torch

class GBN_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epi_as_tensor = torch.tensor([1e-5])):
        scale_as_tensor = torch.tensor([scale],dtype = torch.float32)
        ctx.save_for_backward(scale_as_tensor, epi_as_tensor)
        return x
        pass
    @staticmethod
    def backward(ctx, g):
        scale_as_tensor, epi_as_tensor = ctx.saved_tensors
        scale = scale_as_tensor.item()
        epi = epi_as_tensor.item()

        mean = g.mean(dim=0, keepdim=True)
        _centralized = g - mean
        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
        std_too_small = std < epi
        std = (std - std * std_too_small) + std_too_small * epi
        _normalized = _centralized / std
        if scale != 1.:
            return _normalized*scale, None
            pass
        else:
            return _normalized, None
            pass
        pass
    pass#class
