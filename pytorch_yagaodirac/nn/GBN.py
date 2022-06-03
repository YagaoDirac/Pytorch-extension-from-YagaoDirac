import torch

from pytorch_yagaodirac.nn.functional.GBN_functional import GBN_functional as GBN_functional

'''
Also, in test, or in specific cases of real usage, if the gradient is sent back by something like:
x.mean().backward()   or
x.sum().backward()
All elements in the g is the same, which causes the return value to be zeros.
This hanppens in test a lot.
A simple way is 
x = x*x
x.mean().backward()   or
x = x*torch.rand_like(x)
x.mean().backward()
If x is very small in shape, try manually specify some dummy data to it:
x.backward(torch.tensor([1.,2,3]))
or make sure the dummy data covers both direction:
X = torch.tensor([1.,2,3])#don't remove the dot after 1 in code.
Y = torch.tensor([-999., 999, some small number])


Loss functions:
MSE is recommended. But I'm probably newer than you, so you know what to do.

When using L1Loss along with GBN, such as:
pred = gbn(x)
loss = L1Loss_fn(pred, Y)#Y is the target, or label.
loss.backward()
Say a case, the batch size is only 2, the input for backward of gbn can only be [0.5, 0.5] or any 
signed combination of it. When 2 elements are the same sign, the result is [0, 0], otherwise, positive 
to 1 and negative to -1, [1, -1] or [-1, 1]. 
If batch size is small enough, L1Loss might mess a bit.
If you don't know how to deal with this issue and you really want to use gbn, simply try MSE out.








The sequence:(this part is just based on a rookie's experience, trust it at your own risk)
Now we have a lot tricks to make a STACKED linear(or dence or fully connected) layer trainable.

BN--------------
BN: modifies input for a layer, or protect the layer behind it.
        BN                                 layer
input >>>>>> friendly but differnet input >>>>>>> (other parts)
BN actually messes the forward propagation. A trick is that, prepare an avg pseudo data with 
the real data to do the BN, makes the behavior stable. This protects the forward propagation.
I personally don't like it.
I made a tool to turn off BN halfway in training. Its name is BN_Container.

PE----------------------------------------------------------------
PE: makes input bigger(slightly or drastically), makes trainability chain longer.
legend: Trainability 1 ~ 9, the greater the trainable. Not trainable - . PE |
example: ---29(last layer is super trainable(9), but the second last is only slightly trainable(2)
        all the 3 layers in the from is not trainable at all.
normal model:                                  ...--39
add pe to the input:                |--1479
pe for input and 4th layer:         |--14|799999
pe for input and 4, 8th layer:      |---|-147|999999999
pe for input and 4, 8, 12th layer:  |1479|9999|9999|9999

According to what I saw, PE seems to protect 3 to 6 layers each, backward. 
before:      at input        at layer1    at layer2
------39     |-----369       -|--2589     --|146899
diffenece:         ^^            ^^^         ^^^^^

!!!!!!!!!!All data is evaluated manually, trust it at your own risk!!!!!!!!!!!
I've seen different PE algorithms, trigonometric functions, sawtooth, triangle, binary.
I think the only thing that matters is the magnitude. PE stables the the distribution 
of the input. With out PE, the input can be all close to a certain number or super 
variant, but with PE, it's mean and std are stabler. 
PE also helps void 0 input at like 1 or 2 or even more layers later, so it protects the 
trainability from backwards.
I personally don't like it.

Dropout---------------------------------------------




Weight Decay---------------------------



SET and SOUP------------------------------------------------------------



Mirror



GBN
Doesn't works normally if any sum(), mean(), or L1Loss is the only thing behind it.
Not tested enough.



'''

class GBN(torch.nn.Module):
    '''According to my test, the working scale should be at least greater than 1e-4 and less than 1e18 for torch.float32.
      If it's greater than 1e-3, it works fine.
      Farther test needed.
      Since BN messes up with the forward propagation, Now we have Gradient(only)Batch Normalization.
      Notice, to replace x = relu(self.Linear_0(BN(x))),
      I recommend x = relu(GBN(self.Linear_0(x))).
      Or I also recommend my invention in which I combined a lot useful tools for you. It's pytorch_yagaodirac.nn.Linear
      Reason is that, BN makes x in a range which helps the w to learn in a prefered speed.
      This GBN directly calculated prefered update strength, it helps both w and b to learn in a proper speed.
      Modify the scale value to scale the gradient.
      If you prefer the traditional BN but you also need the scale feature, try x = relu(self.Linear_0( BN(x)*scale ))
      Warning: this gradient adjusting flows backward all the way, it may mess up with the earlier layers.'''

    def __init__(self, base_scale=1., *, epi=1e-12, suppression_factor = 1e3):  # , *, lr = 1e-2):
        super(GBN, self).__init__()
        self.base_scale = torch.nn.Parameter(torch.tensor([base_scale], dtype= torch.float32), requires_grad=False)
        self.dynamic_scale = torch.nn.Parameter(torch.ones(1, ), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([epi], dtype= torch.float32), requires_grad=False)
        self.suppression_factor = torch.nn.Parameter(torch.tensor([suppression_factor]
                                                                  , dtype= torch.float32), requires_grad=False)
        pass

    def forward(self, x):
        if 1 == x.shape[0]:
            raise ValueError("expected anything greater than 1 for the first dimension (got 1 for the first dim)")
            pass
        if 1 == len(list(x.shape)):
            raise ValueError("expected 2D or higher dimension input (got 1D input)")
            pass
        x = GBN_functional.apply(x, self.get_working_scale(), self.epi, self.suppression_factor)
        return x
        pass  # def forward

    def set_dynamic_scale(self, dynamic_scale):
        '''set scale keeps the lr.'''
        self.dynamic_scale = torch.nn.Parameter(torch.tensor([dynamic_scale], dtype= torch.float32))
        pass

    def get_working_scale(self):
        return self.base_scale * self.dynamic_scale
        pass

    pass  # class






if 0:
    gbn = GBN()
    W1 = torch.nn.Parameter(torch.tensor([-1.]))
    W2 = torch.nn.Parameter(torch.tensor([2.]))
    x = torch.tensor([[1.],[3.]])


    x = W1 * x
    x = gbn(x)
    x1 = x
    x = W2 * x
    x = gbn(x)

    x = x*torch.tensor([[1.],[10.]])
    x2 = x
    x.backward(torch.ones_like(x))

    d1 = W1.grad
    d2 = W2.grad

    jklsdf=890456

    pass



if 0:
    layer0 = GBN(3)
    layer0.set_dynamic_scale(33)
    layer0.set_dynamic_scale(3)


    layer1 = GBN(scale= 10)
    in1 = torch.tensor([1,2,3], dtype = torch.float32, requires_grad= True).view(3,1)
    in1_2 = torch.tensor([1,2,3], dtype = torch.float32).view(3,1)
    out1 = layer1(in1)*in1_2
    out1.mean().backward()
    #Then, in1.grad is prepared. It's 12.2. If the scale is 1 by default, the grad is 1.22.

    layer2 = GBN().cuda()
    in2 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True).cuda().view(3,1)
    in2.retain_grad()#I don't know the reason. But this line helps.
    in2_2 = torch.tensor([1, 2, 3], dtype=torch.float32).cuda().view(3,1)
    out2 = layer2(in2) * in2_2
    out2.mean().backward()

    layer3 = GBN()
    in3 = torch.rand((3,2), dtype=torch.float32, requires_grad=True)
    in3_2 = torch.tensor([[1,2],[3,4],[5,6]], dtype=torch.float32)
    out3 = layer3(in3) * in3_2
    out3.mean().backward()

    sdfjkl=356980
    pass
