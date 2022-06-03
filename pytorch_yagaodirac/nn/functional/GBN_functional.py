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
            return _normalized * scale, None, None, None
            pass
        else:
            return _normalized, None, None, None
            pass
        pass
    pass  # class







class XXXXXXXXXXXXXXXXXX_GBN_V2_functional(torch.autograd.Function):
    r"""
    This version use a simple l2 distance to do the trick.
    It modifies the length of the g vector to 1, for each batch.
    """
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


        g_g:torch.Tensor = g*g
        length = g_g.sum
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



