import torch

'''OK, this part is not well designed. 
The torch.std automatically subtracts the mean from the data, and maintains the derivative
'''

class plain_mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False, dtype=None):
        if None == dim:
            dim = 0
            pass
        ctx.save_for_backward(torch.tensor(x.shape), torch.tensor([dim]), torch.tensor([keepdim]))
        return x.mean(dim = dim, keepdim = keepdim, dtype = dtype)
        pass
    @staticmethod
    def backward(ctx, g:torch.Tensor):
        shape, dim_as_tensor, keepdim_as_tensor = ctx.saved_tensors
        dim = dim_as_tensor.item()
        keepdim = keepdim_as_tensor.item()
        ratio = 1 / shape[dim]
        g = g * ratio
        if keepdim:
            repeat_shape = torch.ones_like(shape)
            pass
        else:
            repeat_shape = torch.ones_like(shape+1)
            pass
        repeat_shape[dim] = shape[dim]
        g = g.repeat(repeat_shape)
        return g, None, None, None
        pass

    pass#class




if 1:
    a1 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True).cuda()
    m1 = plain_mean.apply(a1, 0)
    m1.backward()

    a2 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    m2 = plain_mean.apply(a2)
    c2 = a2 - m2
    c2.backward(torch.tensor([1, 1, 1], dtype=torch.float32))

    a3 = torch.tensor([[1, 2, 3],[1, 2, 3]], dtype=torch.float32, requires_grad=True)
    b3 = plain_mean.apply(a3)
    b3.backward()

    a4 = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32, requires_grad=True)
    b4 = plain_mean.apply(a4, 1)
    b4.backward()

    a5 = torch.rand(3,4,5)
    plain_mean.apply(a5,1, keepdim = True).mean().backward()
    pass

if 1:
    a5 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    m5 = plain_mean(a5)
    c5 = a5-m5
    std5 = c5.std(unbiased = False)
    res5 = c5/std5
    pass


