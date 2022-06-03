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

        _gen = linear_layer_gen()#this mode doesn't gen bias.
        self.l0 = _gen.gen(in_features, out_features)#No bias is generated.
        self.l1 = _gen.gen(in_features, out_features)#No bias is generated.
        self.indicator0 = Linear_indicator(self.l0)
        self.indicator1 = Linear_indicator(self.l1)
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(out_features,), requires_grad=True)
            pass
        else:
            self.bias = None
            pass
        self._to_train_mode()#behind the init of indicators.

        self.gbn = GBN(base_scale=100)
        self.gbn_score_dragger = Dragger(size = 32, init_val=0.)
        pass
    @torch.no_grad()
    def on_epoch(self, epoch):
        if None != self.BN_Cont:
            self.BN_Cont.on_epoch(epoch)
            pass
        self.on_new_loss()
        pass
    @torch.no_grad()
    def on_opt_stepped(self):
        r"""
        :return: dragged score. Or the smoothened score value.
        """
        if not self.training:
            return

        score = (self.indicator0.update().score+self.indicator1.update().score)/2.
        if score<0:
            score = 0
            pass
        if score > 1:
            score = 1
            pass
        self.gbn_score_dragger.set(score)
        score = self.gbn_score_dragger.get()
        scale = 10**(0.5-5*score)# I didn't study on this formula. It's only intuition.
        self.gbn.set_dynamic_scale(scale)

        return score
        pass
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
        self.l0.weight = torch.nn.Parameter(self.l0.weight/float(2.), requires_grad=True)
        self.l1.weight = torch.nn.Parameter(self.l0.weight.clone().detach(), requires_grad=True)
        #if None != self.bias:
        #    self.l0.bias = torch.nn.Parameter(self.l0.bias/float(2.))
        #    self.l1.bias = torch.nn.Parameter(self.l0.bias.clone().detach())
        #    pass

        self.indicator0.update()
        self.indicator1.update()
        pass

    @torch.no_grad()
    def _to_eval_mode(self):
        if None != self.bias:
            new_bias = self.l0.weight.sum() - self.l1.weight.sum() + self.bias#+ self.l1.bias + self.l0.bias
            self.bias = torch.nn.Parameter(new_bias.view(-1), requires_grad=True)
            #self.l0.bias = new_bias
            #self.l1.bias = None
            pass
        new_weight = self.l0.weight + self.l1.weight
        new_weight = torch.nn.Parameter(new_weight, requires_grad=True)
        self.l0.weight = new_weight
        self.l1.weight = None
        pass

    def forward(self, x):
        if not self.training:# evaluating mode:
            return self.l0(x) + self.bias
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
        h0 = self.l0(x+1)
        h1 = self.l1(x-1)
        x = h0 + h1
        if None != self.bias:
            x = x+self.bias
            pass
        x = self.gbn(x)
        return x
        pass#def forward

    pass#class


if 0:
    X = torch.tensor([[1.,2.],[1.,4.]])
    m = Linear(2, 1, bias= True, bn_cont=False)
    #m.l0.weight = torch.nn.Parameter(torch.ones_like(m.l0.weight))
    #m.l1.weight = torch.nn.Parameter(torch.zeros_like(m.l1.weight)+1.46)
    #m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
    pred1 = m(X)
    m.eval()
    pred2 = m(X)
    m.train()
    pred3 = m(X)

    X2 = torch.tensor([[1.,2.],[1.,3.]]).cuda()
    m2 = Linear(2, 3, bias=True, bn_cont=False).cuda()
    pred22 = m2(X2)

    X3 = torch.tensor([[1.],[-1.],[2.],[3.]]).cuda()
    X3 = X3+torch.rand(1).cuda()
    m3 = Linear(1, 1, bias=True).cuda()#BN by defaut is True.
    pred33 = m3(X3)

    dfshjkl=456890
    pass




if 0:
    #L1Loss doesn't work well enough for this test.
    layer1 = Linear(1,1,bn_cont=False)
    layer2 = Linear(1,1,bn_cont=False)
    loss_fn = torch.nn.MSELoss()
    tensors = [*layer1.parameters(), *layer2.parameters()]
    opt = torch.optim.SGD(tensors, lr = 0.2)

    X = torch.tensor([[1.],[2.]])
    Y = torch.tensor([[0.9],[1.]])

    while 1:
        opt.zero_grad()
        x = layer1(X)
        pred = layer2(x)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

        l0w__1  = layer1.l0.weight     .item()
        l1w__1  = layer1.l1.weight     .item()
        l0b__1  = layer1.l0.bias       .item()
        l1b__1  = layer1.l1.bias       .item()
        gl0w_1 = layer1.l0.weight.grad.item()
        gl1w_1 = layer1.l1.weight.grad.item()
        gl0b_1 = layer1.l0.bias  .grad.item()
        gl1b_1 = layer1.l1.bias  .grad.item()
        l0w__2  = layer2.l0.weight.item()
        l1w__2  = layer2.l1.weight.item()
        l0b__2  = layer2.l0.bias.item()
        l1b__2  = layer2.l1.bias.item()
        gl0w_2 = layer2.l0.weight.grad.item()
        gl1w_2 = layer2.l1.weight.grad.item()
        gl0b_2 = layer2.l0.bias.grad.item()
        gl1b_2 = layer2.l1.bias.grad.item()
        dfs=345
        pass


    pass




if 0:
    loss_fn = torch.nn.L1Loss()

    a = torch.nn.Parameter(torch.ones(1,1), requires_grad=True)
    b = torch.nn.Parameter(torch.ones(1,1), requires_grad=True)
    x = torch.nn.Parameter(torch.tensor([[1.],  [1.],  [1.5]]), requires_grad=False)
    Y = torch.nn.Parameter(torch.tensor([[-11.], [11.], [12.5]]), requires_grad=False)
    gbn = GBN(base_scale=1)
    c = gbn(a*(x + 1) +b*(x - 1))
    c = loss_fn(c, Y)
    c.backward()
    jkldfs=456890

    a = torch.nn.Linear(1,1,bias =False)
    b = torch.nn.Linear(1,1,bias =False)
    gbn = GBN(base_scale=1)
    c = gbn(a (x + 1) + b (x - 1))
    c = loss_fn(c, Y)
    c.sum().backward()

    jkldfs=456890
    pass





if 0:
    #Does the mirror work at all? Probably yes, imo mathmatically yes.
    #This test probably works better with L1Loss and maybe RMSProp or only L1Loss.
    layer0:torch.nn.Linear = torch.nn.Linear(1,1, bias=False)
    layer1:torch.nn.Linear = torch.nn.Linear(1,1, bias=False)
    layer1.weight = torch.nn.Parameter(layer0.weight.clone().detach())
    bias = torch.nn.Parameter(torch.zeros(1,), requires_grad=True)
    gbn = GBN()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD([*layer0.parameters(), *layer1.parameters(), bias], lr=0.1)

    X = torch.tensor([[0.], [0.1], [0.2]])#when this is 0, or super near to 0.
    Y = torch.tensor([[1.], [1.1], [1.5]])

    while 1:
        opt.zero_grad()
        pred = layer0(X+1) + layer1(X-1)#+1 or not
        #pred = layer0(X+1) + layer1(X-1)#+1 or not
        pred = gbn(pred/2.+bias)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

        l0w_ = layer0.weight.item()
        l1w_ = layer1.weight.item()
        bias = bias
        gl0w = layer0.weight.grad.item()
        gl1w = layer1.weight.grad.item()
        gbia = bias.grad#.item()

        dfs = 345
        pass


