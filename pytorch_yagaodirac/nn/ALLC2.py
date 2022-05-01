import torch

class ALLC2_functional_part(torch.autograd.Function):
    '''Abs Linear leaky clamp 2
    The forward and backward hebavior differently.
    '''
    @staticmethod
    #def forward(ctx, x, a, b, factor, negative_slope):
    def forward(ctx, x, a, b, negative_slope):#factor, negative_slope):
        #assert factor.item()>=1.
        #assert negative_slope.item()>0. and negative_slope.item()<1.
        #ctx.save_for_backward(x, a, b, torch.tensor([factor]), torch.tensor([negative_slope]))
        ctx.save_for_backward(x, a, b, torch.tensor([negative_slope]))
        with torch.no_grad():
            x = torch.abs(x)
            x = a * x + b
            # Then, double relu to make the result almost inside 0 to 1
            x = torch.relu(x)
            x = 1 - torch.relu(1 - x)
            pass# with torch.no_grad():
        return x
        pass
    @staticmethod
    def backward(ctx, g):
        #x, a, b, factor, negative_slope = ctx.saved_tensors
        x, a, b, negative_slope = ctx.saved_tensors
        with torch.no_grad():
            #negative_slope = negative_slope.item()

            x_abs = torch.abs(x)
            ori_ax_b = a * x_abs + b
            leaky_flags = (ori_ax_b > 1.) + (ori_ax_b < 0.)
            non_leaky_flags = leaky_flags.logical_not()
            flags = non_leaky_flags + leaky_flags*negative_slope
            b_grad = g * flags
            a_grad = g * flags * x_abs
            x_less_than_0 = x < 0.
            x_less_than_0 = x_less_than_0 * -2 + 1
            x_grad = g * flags * a * x_less_than_0
            pass
        return x_grad, a_grad, b_grad, None#, None
        pass
    def __str__(self):
        return F'LLC2_functional. It\'s designed only for the LLC2 layer. It behaviors differently in forward propagation and backward propagation.'
        pass
    pass
class ALLC2(torch.nn.Module):
    '''Linear leaky clamp version 2.0
    The forward pass works with ReLU like algorithm.
    While the backward pass works with Leaky ReLU like algorithm.
    '''
    def __init__(self, size, * , a = -1., b = 2., negative_slope=0.01, epi = 1e-8):
        super(ALLC2, self).__init__()
        assert a < 0
        assert b >= 0.5
        assert negative_slope > 0
        assert epi > 0
        self.a = torch.nn.Parameter(torch.full((size,), a))
        b_like = torch.sqrt(torch.tensor([b - 0.5]))
        self.b_like = torch.nn.Parameter(torch.full((size,), b_like.item()))
        self.negative_slope = negative_slope
        self.epi = torch.abs(torch.tensor([epi]))
        pass
    def forward(self, x):
        with torch.no_grad():
            temp = torch.abs(self.a)
            temp = temp - self.epi
            temp = torch.relu(temp)
            temp = temp + self.epi
            self.a = torch.nn.Parameter(-temp) # self.a < 0
            pass
        b = self.b_like * self.b_like + 0.5  # b >= 0.5
        x = ALLC2_functional_part.apply(x, self.a, b, self.negative_slope)
        # The parameters for the specific function are: x, a, b, negative_slope
        return x
        pass
    def __str__(self):
        return F'Absolute linear leaky clamp ver 2.0. a: {self.a}  b: {self.b}  the boundary: {torch.abs(self.b/self.a)} boundary_factor: {self.boundary_factor:.1f}  negative_slope: {self.negative_slope:.4f}'
    pass


if 1:
    layer1 = ALLC2(4)
    layer1.a = torch.nn.Parameter(torch.full((4,), 1.))
    in1 = torch.tensor([0.9, 1.1, 1.9, 2.1])
    out1 = layer1(in1)
    out1.mean().backward()

    layer2 = ALLC2(2)
    in2 = torch.tensor([[1., 2], [3, 4]])
    out2 = layer2(in2)
    out2.mean().backward()

    pass