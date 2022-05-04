import torch
#import importlib.util
#spec = importlib.util.spec_from_file_location("SelMul_functional",
#                                              "functional/SelMul_functional Dont use at the moment.py")
#util = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(util)
#test_var = util.SelMul_functional()

class SelMul(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias = False, * , name = None):
        # If you don't like this design, you can simply replace this layer with a x*(x.mT)
        super(SelMul, self).__init__()
        self.name = name
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.halfway_dim = int((input_dim+1)*input_dim/2)
        self.Linear = torch.nn.Linear(self.halfway_dim, output_dim, bias = bias)
        #self.Linear.weight = torch.nn.Parameter(torch.zeros_like(self.Linear.weight))
        #if bias:
        #    self.Linear.bias = torch.nn.Parameter(torch.zeros_like(self.Linear.bias))
        #    pass

        self.flags = torch.nn.Parameter(torch.ones(input_dim, input_dim).triu().to(torch.bool).view(1, input_dim, input_dim), requires_grad=False)
        pass#def __init__

    def forward(self, x):
        if len(list(x.shape)) == 1:
            x = x.view(1, 1, -1)
            batch_size = None
            pass
        else:
            x = x.view(x.shape[0], 1, -1)
            batch_size = x.shape[0]
            pass
        x = (x.mT)@x
        flags = self.flags.repeat(x.shape[0], 1, 1)
        x = x[flags]
        if None != batch_size:
            x = x.view(batch_size, self.halfway_dim)
            pass
        x = self.Linear(x)
        return x
        pass#def forward

    def __str__(self):
        return F"{self.name} Self Multiplication Layer. In dim:({self.in_dim}), Half-way dim:({self.halfway_dim}), Out dim:({self.out_dim}), with a Linear layer: {self.Linear}"
        pass
    pass



if 0:
    layer = SelMul(2, 3, True)
    in1 = torch.tensor([2., 3])
    out1 = layer(in1)
    in2 = torch.tensor([[1., 2], [2., 3]])
    out2 = layer(in2)

    layer = SelMul(2, 3, True).cuda()
    in1 = torch.tensor([2., 3]).cuda()
    out1 = layer(in1)
    in2 = torch.tensor([[1., 2], [2., 3]]).cuda()
    out2 = layer(in2)
    pass

if 0:
    layer = SelMul(2,16)
    in1 = torch.rand(2)
    out1 = layer(in1)
    in1 = torch.rand(1, 2)
    out1 = layer(in1)
    in1 = torch.rand(5, 2)
    out1 = layer(in1)
    pass