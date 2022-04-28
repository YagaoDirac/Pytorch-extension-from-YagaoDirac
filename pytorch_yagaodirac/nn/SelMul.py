import torch
import importlib.util
spec = importlib.util.spec_from_file_location("SelMul_functional", "./functional/SelMul_functional.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
#test_var = util.SelMul_functional()

class SelMul(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_bias = False):
        super(SelMul, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.halfway_dim = int((input_dim+1)*input_dim/2)
        self.Linear = torch.nn.Linear(self.halfway_dim, output_dim, use_bias = use_bias)
        pass

    def forward(self, x):
        smf = util.SelMul_functional()
        return self.Linear((x))
        pass

    def __str__(self):
        return F"Self Multiplication Layer. In({self.in_dim}), Out({self.out_dim}), with a Linear layer: {self.Linear}"
        pass
    pass
