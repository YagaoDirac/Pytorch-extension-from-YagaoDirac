import torch
from pytorch_yagaodirac.linear_layer_tool.linear_layer_gen import linear_layer_gen, parameter_gen
from pytorch_yagaodirac.nn.BN_Container import BN_Container
from pytorch_yagaodirac.nn.GBN import GBN
from pytorch_yagaodirac.linear_layer_tool.indicator import Linear_indicator
from pytorch_yagaodirac.Dragger import Dragger


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 *,bn_cont = True) -> None:
        '''This layer needs 3 on the fly update.
        1, If the 'bn_cont' is True when constructs, call the on_epoch() function once every epoch.
            It updates BN container and automatically calls on_new_loss() function once.
        2, call on_opt_stepped() function only in training. It updates the scale of GBN, gradually.
            I haven't tested it yet, I don't know how often should you call it.
            I personally recommend you call on_new_loss() every batch, or literally whenever a new loss is calculated.
        I'm not very sure, pytorch seems to apply lr directly to the final grad variable inside each tensor. Thus, the backward
        process has nothing to do with the lr. Which also means, before opt.step() is called, the whole nn has no idea of the lr.
        If I were wrong, emmm....GBN probably breaks, and then this whole thing.
        If the BN is turned off too early with default param, modify the parameters in BN container.
        If the update step length is too big or too small, modify the formula in on_new_loss() and comment in my Github.
        '''
        super(Linear, self).__init__()

        if bn_cont:
            self.BN_Cont = BN_Container(1, 15, 500)
            pass
        else:
            self.BN_Cont = None
            pass

        _param_gen = parameter_gen()
        if bias:
            _gen = linear_layer_gen(_param_gen, _param_gen, proportion_of_weight=0.9)
            pass
        else:
            _gen = linear_layer_gen()
            pass
        self.l0 = _gen.gen(in_features, out_features)
        self.l1 = _gen.gen(in_features, out_features)
        self.indicator0 = Linear_indicator(self.l0)
        self.indicator1 = Linear_indicator(self.l1)
        self._to_train_mode()#behind the init of indicators.

        self.gbn = GBN()
        self.gbn_score_dragger = Dragger(size = 32, init_val=0.)
        pass

    def on_epoch(self, epoch):
        self.BN_Cont.on_epoch(epoch)
        self.on_new_loss()
        pass
    def on_opt_stepped(self):
        if not self.training:
            return

        score = (self.indicator0.update().score+self.indicator1.update().score)/2.
        if score>1.:
            score = 1.
            pass
        if score <0.:
            score  = 0.
            pass
        self.gbn_score_dragger.set(score)
        score = self.gbn_score_dragger.get()
        scale = 10**(0.5-5*score)# I didn't study on this formula. It's only intuition.
        self.gbn.set_scale(scale)
        pass
    #def set_lr(self, lr):
    #    self.gbn.set_lr(lr, keep_inner_behavior= False)
    #    pass

    #def eval(self):
    #    if self.training:
    #        self._to_eval_mode()
    #        pass
    #    super(Linear, self).eval()
    #    pass
    def train(self, mode: bool = True):
        if mode and (not self.training):
            self._to_train_mode()
            pass
        if (not mode) and self.training:
            self._to_eval_mode()
            pass
        super(Linear, self).train(mode)
        pass

    @torch.no_grad()
    def _to_train_mode(self):
        self.l0.weight = torch.nn.Parameter(self.l0.weight/float(2.))
        self.l1.weight = torch.nn.Parameter(self.l0.weight.clone().detach())
        if None != self.l0.bias:
            self.l0.bias = torch.nn.Parameter(self.l0.bias/float(2.))
            self.l1.bias = torch.nn.Parameter(self.l0.bias.clone().detach())
            pass

        self.indicator0.update()
        self.indicator1.update()
        pass

    @torch.no_grad()
    def _to_eval_mode(self):
        if None != self.l0.bias:
            new_bias = self.l0.weight.sum() + self.l0.bias - self.l1.weight.sum() + self.l1.bias
            new_bias = torch.nn.Parameter(new_bias.view(-1))
            self.l0.bias = new_bias
            self.l1.bias = None
            pass
        new_weight = self.l0.weight + self.l1.weight
        new_weight = torch.nn.Parameter(new_weight)
        self.l0.weight = new_weight
        self.l1.weight = None
        pass

    def forward(self, x):
        if not self.training:# evaluating mode:
            return self.l0(x)
            pass

        if self.BN_Cont:
            if 1 == x.shape[0]:
                raise Exception("The first dimention is only 1. It's not possible to do BN."
                                " Consider turn off BN feature when construct this layer.")
                pass
            if 1 == len(list(x.shape)):
                raise Exception("The input has only 1 dimention. It's not possible to do BN."
                                " Consider turn off BN feature when construct this layer.")
                pass
            x = self.BN_Cont(x)
            pass
        h0 = self.l0(x)
        h1 = self.l1(x)
        x = h0 + h1
        x = self.gbn(x)
        return x
        pass#def forward

    pass#class


if 0:
    X = torch.tensor([1.,2.])
    m = Linear(2, 3, bias= True, bn_cont=False)
    pred1 = m(X)
    m.eval()
    pred2 = m(X)
    m.train()
    pred3 = m(X)

    X2 = torch.tensor([1., 2.]).cuda()
    m2 = Linear(2, 3, bias=True, bn_cont=False).cuda()
    pred22 = m2(X2)

    X3 = torch.tensor([[1.],[-1.]]).cuda()
    X3 = X3+torch.rand(1).cuda()
    m3 = Linear(1, 1, bias=True).cuda()#BN by defaut is True.
    pred33 = m3(X3)

    dfshjkl=456890
    pass


