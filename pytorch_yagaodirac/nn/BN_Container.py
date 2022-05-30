import torch

torch.nn.BatchNorm1d

class BN_Container(torch.nn.Module):
    r"""
    This small tool interpolates the bn result with origin tensor, according to the epoch.
    When epoch is very small, which means it's the early time of a training process,
    the result of this layer is more of bn result than the input.
    When the epoch is super big, which means it's the late training, the result of this layer
    is the input itself.
    Only before 'no_BN_from'th epoch, the bn is interpolated into the result of this layer.

    Args:
        max_ratio: the max ratio of bn. No more bn than (bn_result*max_ratio+input*(1-max_ratio))
        pure_BN_before: before this, result is (bn_result*max_ratio+input*(1-max_ratio))
        no_BN_from: after this, returns input.
        x_like_range: small number for smoother, big number for not very smooth.
                Basically I don't know the difference.
    Shape:
        - Input: any
        - Output: the same. The first dim is deemed as batch dimention on which the normalization occurs.
    """
    def __init__(self, max_ratio = 1., pure_BN_before = 100, no_BN_from = 1000, *, x_like_range = 5.):
        super(BN_Container, self).__init__()
        self.max_ratio = torch.nn.Parameter(torch.tensor([max_ratio], dtype = torch.float32), requires_grad=False)
        self.pure_BN_before = pure_BN_before
        self.no_BN_from = no_BN_from
        self.x_like_range = torch.nn.Parameter(torch.tensor([x_like_range], dtype = torch.float32), requires_grad=False)
        self.epoch = 0
        pass

    def set_epoch(self, epoch):
        self.epoch = epoch
        pass

    def _calc_pure_ratio(self)->torch.Tensor:
        if self.epoch <= self.pure_BN_before:
            return torch.ones(1)
            pass
        if self.epoch >= self.no_BN_from:
            return torch.zeros(1)
            pass
        temp = self.x_like_range*(((self.epoch-self.pure_BN_before)/(self.no_BN_from-self.pure_BN_before))*2 -1)
        result = 1 - torch.sigmoid(temp)
        return result
        pass

    def forward(self, x):
        if not self.training:
            return x
            pass
        _temp = self._calc_pure_ratio()
        _temp = _temp.to(self.max_ratio.device)
        BN_ratio:torch.Tensor = _temp*self.max_ratio
        if 0. == BN_ratio.item():#the .item() is actually not needed.
            return x
            pass
        _s, _m = torch.std_mean(x, unbiased= False)
        bn = (x-_m)/_s
        return bn*BN_ratio+x*(1-BN_ratio)
        pass
    pass#class



if 0:
    cont = BN_Container(0.5, 5, 10, x_like_range = 2)
    in1 = torch.tensor([[1],[3]], dtype = torch.float32)
    out1 = cont(in1)
    cont.on_epoch(4)
    out1 = cont(in1)
    cont.on_epoch(5)
    out1 = cont(in1)
    cont.on_epoch(6)
    out1 = cont(in1)
    cont.on_epoch(9)
    out1 = cont(in1)
    cont.on_epoch(10)
    out1 = cont(in1)
    cont.on_epoch(11)
    out1 = cont(in1)
    cont.on_epoch(6)
    out1 = cont(in1)

    cont.cuda()
    in2 = torch.tensor([[1], [3]], dtype = torch.float32).cuda()
    cont.on_epoch(6)
    out2 = cont(in2)

    sdfjkl=789456
    pass

