import torch
#import importlib.util
#spec = importlib.util.spec_from_file_location("SelMul_functional",
#                                              "functional/SelMul_functional Dont use at the moment.py")
#util = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(util)
#test_var = util.SelMul_functional()

class SelMul(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias = False):
        super(SelMul, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.halfway_dim = int((input_dim+1)*input_dim/2)
        self.Linear = torch.nn.Linear(self.halfway_dim, output_dim, bias = bias)
        self.Linear.weight = torch.nn.Parameter(torch.zeros_like(self.Linear.weight))
        if bias:
            self.Linear.bias = torch.nn.Parameter(torch.zeros_like(self.Linear.bias))
            pass

        self.flags = torch.ones(input_dim, input_dim).triu().to(torch.bool).view(1, input_dim, input_dim)
        # If you don't like this design, you can simply replace this layer with a x*(x.mT)
        self._correct_device()

        pass#def __init__

    def _correct_device(self):
        i = self.Linear.weight.get_device()
        if i<0:#cpu
            self.flags.cpu()
            pass
        else:
            dev = torch.device('cuda', index = i)
            self.flags.to(dev)
            pass
        pass

    def forward(self, x):
        if len(list(x.shape)) == 1:
            x = x.view(1, 1, -1)
            pass
        else:
            x = x.view(x.shape[0], 1, -1)
            pass
        x = (x.mT)@x
        self._correct_device()
        flags = self.flags.repeat(x.shape[0], 1, 1)
        x = x[flags]
        x = x.view(-1, self.out_dim)
        return self.Linear(x)
        pass#def forward

    def __str__(self):
        return F"Self Multiplication Layer. In({self.in_dim}), Out({self.out_dim}), with a Linear layer: {self.Linear}"
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

