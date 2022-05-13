import torch

class parameter_gen:
    def __init__(self, signed = True, length = 1., *, const = None):
        '''If const is specified, the left 2 parameters don't do anything.
        Also notice, the signed and unsigned don't share the same formula.'''
        self.signed = signed
        self.length = length
        self.const = const
        pass#def
    def gen1d(self, size):
        if None != self.const:
            return torch.nn.Parameter( torch.full((size, ), self.const, dtype = torch.float32))
            pass
        if self.signed:
            t = torch.randn(size, dtype=torch.float32)
            t = t/(torch.sqrt(torch.tensor([size])))
            '''
            The reason for t = t/(torch.sqrt(torch.tensor([size])))  is shown as below:
            a = torch.randn(1000, 1024)
            a = a*a
            a = a.abs()
            a = a.sum(dim = -1)
            print(a.shape)
            a = a.mean()
            print(torch.sqrt(a))
            The output is 
            torch.Size([1000])
            tensor(31.9938)  
            Run this and you'll see the last result jitters near sqrt(1024). 
            If you modify the 1024, you'll see the final result is always near to its sqrt.
            '''
            t = t * self.length
            return torch.nn.Parameter(t)
            pass
        else:#unsigned
            t = torch.randn(size, dtype = torch.float32)
            t = torch.exp(t)
            length = torch.sqrt((t*t).sum())
            t = t / length
            t = t * self.length
            return torch.nn.Parameter(t)
            pass
        pass#def
    def gen2d(self, dim_0, dim_1):
        t = []
        [t.append(self.gen1d(dim_1).view(1, -1)) for _ in range(dim_0)]
        t = torch.cat(t)
        return torch.nn.Parameter(t)
        pass
    pass#class

if 0:
    const_gen = parameter_gen()
    const_gen.const = 3
    t_const = const_gen.gen1d(3)

    gen1 = parameter_gen(True)
    t1_1 = gen1.gen1d(3)
    t1_2 = gen1.gen2d(2, 3)

    gen2 = parameter_gen(False)
    t2_1 = gen2.gen1d(3)
    t2_2 = gen2.gen2d(2, 3)

    gen3 = parameter_gen(True, 2)
    t3_1 = gen3.gen1d(3)
    t3_2 = gen3.gen2d(2, 3)

    gen4 = parameter_gen(False, 2)
    t4_1 = gen4.gen1d(3)
    t4_2 = gen4.gen2d(2, 3)

    put_a_breakpoint_here = 424242
    pass





class linear_layer_gen:
    '''
    This tool helps initialize a linear layer for either real development or test.
    The basic idea is to make abs(wx+b) near to 1.
    The math is basically about the distance.
    If length of x and w[i] are both near to 1, then, I don't know, I think the result would be near 1 or -1
    Also, if 0<c<1, then cw is shorter than w, which makes room for the bias.
    A 1 is devided into 2 parts, and calculate the sqrt respectively. Then one is applied to w and the other is for c.
    For clarity,
    m*m+n*n == 1.
    w[i] = rand(...).normalize()# then length of w is 1.
    weight = m*w  #shorten a bit with m
    bias = rand(...).normalize()*n  #not with m but with n.
    '''
    def __init__(self, weight_gen:parameter_gen = parameter_gen(), bias_gen:parameter_gen = None, *, proportion_of_weight = 1.,
                 scale = 1., bias_for_bias = 0.):
        assert proportion_of_weight>=0.
        assert proportion_of_weight<=1.
        self.weight_gen = weight_gen
        self.scale = scale

        if None == bias_gen or 1. == proportion_of_weight:
            bias_gen = None
            proportion_of_weight = 1.
            bias_for_bias = 0.
            pass
        self.bias_gen = bias_gen
        self.proportion_of_weight = torch.tensor([proportion_of_weight])
        self.bias_for_bias = bias_for_bias
        pass#def
    def gen(self, in_dim, out_dim):
        if 1 != self.proportion_of_weight:
            result = torch.nn.Linear(in_dim, out_dim)
            b = self.bias_gen.gen1d(out_dim)
            b = b * torch.sqrt(1. - self.proportion_of_weight)
            b = b * self.scale
            b = b + self.bias_for_bias
            result.bias = torch.nn.Parameter(b)
            pass
        else:
            result = torch.nn.Linear(in_dim, out_dim, bias=False)
            pass
        w = self.weight_gen.gen2d(out_dim, in_dim)
        w = w*torch.sqrt(self.proportion_of_weight)
        w = w*self.scale
        result.weight = torch.nn.Parameter(w)
        return result
        pass#def
    pass#class

if 1:
    s_gen = parameter_gen(True, 10)
    u_gen = parameter_gen(False, 10)

    gen1 = linear_layer_gen()
    layer1 = gen1.gen(3, 2)

    gen2 = linear_layer_gen(s_gen)
    layer2 = gen2.gen(3, 2)

    gen3 = linear_layer_gen(s_gen, u_gen, proportion_of_weight = 0.8)
    layer3 = gen3.gen(3, 2)

    gen4 = linear_layer_gen(parameter_gen(const = 10), parameter_gen(const = 3), proportion_of_weight = 0.8)
    layer4 = gen4.gen(3, 2)

    gen5 = linear_layer_gen(parameter_gen(const = 10), parameter_gen(const = 3),
                            proportion_of_weight = 0.8, scale = 2)
    layer5 = gen5.gen(3, 2)

    gen6 = linear_layer_gen(parameter_gen(const=10), parameter_gen(const=3),
                            proportion_of_weight=0.8,  scale = 2, bias_for_bias=10)
    layer6 = gen6.gen(3, 2)

    put_a_breakpoint_here = 424242

    pass