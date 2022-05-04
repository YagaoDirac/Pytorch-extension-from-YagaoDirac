import torch

class ALLC2_functional_part(torch.autograd.Function):
    '''Abs Linear leaky clamp 2
    The forward and backward hebavior differently.
    '''
    @staticmethod
    #def forward(ctx, x, a, b, factor, negative_slope):
    def forward(ctx, x, a, b, negative_slope):#, device_index):#factor, negative_slope):
        #assert factor.item()>=1.
        #assert negative_slope.item()>0. and negative_slope.item()<1.
        #ctx.save_for_backward(x, a, b, torch.tensor([factor]), torch.tensor([negative_slope]))
        ctx.save_for_backward(x, a, b, negative_slope)#, torch.tensor([device_index]))
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
        x, a, b, negative_slope = ctx.saved_tensors
        with torch.no_grad():
            device_index = a.get_device()
            x_abs = torch.abs(x)
            ori_ax_b = a * x_abs + b
            leaky_flags = (ori_ax_b > 1.) + (ori_ax_b < 0.)
            non_leaky_flags = leaky_flags.logical_not()
            if device_index<0:#cpu
                leaky_flags.cpu()
                non_leaky_flags.cpu()
                negative_slope.cpu()
                pass
            else:
                device = torch.device('cuda', index = device_index)
                leaky_flags.to(device)
                non_leaky_flags.to(device)
                negative_slope.to(device)
                pass
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
    def __init__(self, size, * , shrink_factor = 1., shape_factor = 1.5, negative_slope=0.01, epi = 1e-8, name = None):
        """Param:size, * , a = -1., b = 2., negative_slope=0.01, epi = 1e-8
        Linear leaky clamp version 2.0
            The forward pass works with ReLU like algorithm.
            While the backward pass works with Leaky ReLU like algorithm.
            """
        super(ALLC2, self).__init__()
        self.name = name
        #a = float(a)
        #b = float(b)
        #negative_slope = float(negative_slope)
        #epi = float(epi)
        assert shrink_factor >= epi
        assert shape_factor >= 0.5
        assert negative_slope > 0
        assert epi > 0
        a = float(shrink_factor)
        b = float(shape_factor)
        a_like = torch.sqrt(torch.tensor([a-epi]))
        self.a_like = torch.nn.Parameter(torch.full((size,), a_like.item()))
        b_like = torch.sqrt(torch.tensor([b - 0.5]))
        self.b_like = torch.nn.Parameter(torch.full((size,), b_like.item()))
        self.negative_slope = torch.nn.Parameter(torch.tensor([negative_slope]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.abs(torch.tensor([epi])), requires_grad=False)
        pass
    def forward(self, x):
        #with torch.no_grad():
        #temp1 = torch.abs(self.a)
        #temp2 = temp1 - self.epi
        #temp3 = torch.relu(temp2)
        #temp4 = temp3 + self.epi
        #self.a = torch.nn.Parameter(-temp4) # self.a < 0
            #pass
        a = self.a_like * self.a_like + self.epi  # a > 0
        b = self.b_like * self.b_like + 0.5       # b >= 0.5
        x = ALLC2_functional_part.apply(x, a, b, self.negative_slope)
        return x
        pass
    #def parameters(self, recurse: bool = True):# -> Iterator[Parameter]:
    #    return iter([self.a, self.b_like])
    #    pass
    def __str__(self):
        a = self.a_like * self.a_like + self.epi  # a > 0
        b = self.b_like * self.b_like + 0.5  # b >= 0.5
        return F'{self.name} Absolute linear leaky clamp ver 2.0. a: {self.a_like}  b: {self.b_like}  the boundary: {torch.abs(b/a):.4f}  negative_slope: {self.negative_slope:.4f}'
    pass


if 0:
    layer1 = ALLC2(4)
    layer1.a_like = torch.nn.Parameter(torch.full((4,), 1.))
    in1 = torch.tensor([0.9, 1.1, 1.9, 2.1])
    out1 = layer1(in1)
    out1.mean().backward()

    opt = torch.optim.SGD(layer1.parameters(), lr = 1)
    opt.step()

    layer2 = ALLC2(2).cuda()
    in2 = torch.tensor([[1., 2], [3, 4]]).cuda()
    out2 = layer2(in2)
    out2.mean().backward()

    pass
if 1:
    class testModel(torch.nn.Module):
        def __init__(self):
            super(testModel, self).__init__()
            self.layer = ALLC2(1)
            pass
        def forward(self, x):
            x = self.layer(x)
            return x
            pass
        pass
    m = testModel().float()
    opt = torch.optim.SGD(m.parameters(recurse = True), lr = 0.123)
    opt.zero_grad()
    pred = m(torch.tensor([1.]))
    pred.backward()
    opt.step()
    #print(m.layer.a_like)
    #print(m.layer.b_like)
    jfklds = 5465
    pass

