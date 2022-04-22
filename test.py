'''
Notice, some of the files contains a line of
    channel_first = (torch.sigmoid((channel_first)) - 0.5) * 7 +0.5
The purpose is to visulization the values out of 0 to 1. But this is not the designed way to use the model defined in this proj.
When the model outputs black or while image, try this out.
'''

'''
Loss:  0.007108466234058142
Loss:  0.011661529541015625
Loss:  0.004659366328269243
Loss:  0.0017400456126779318
Loss:  0.00929915439337492
Loss:  0.008106804452836514
Loss:  0.00420359056442976
'''



import numpy as np
import numpy.random

from scipy import signal, special
from PIL import Image

import torch
import torchvision
from torchvision.utils import save_image

#My customized part.
from pytorch_yagaodirac.util import Counter
from pytorch_yagaodirac.datagen.nerf2d_datagen_no_pe import nerf2d_datagen_no_pe
from pytorch_yagaodirac.optim.AutoScheduler import AutoScheduler
from pytorch_yagaodirac.nn.LLC2 import LLC2


device = torch.device('cuda')


#class Counter:
#    def __init__(self, start_from = 5, every = -1):
#        self.next = start_from
#        self.every = every
#        if every<=0:
#            self.every = start_from
#            pass
#        pass
#    def get(self, current):
#        if current>= self.next:
#            self.next += self.every
#            return True
#            pass
#        return False
#        pass
#    pass


#class NoPEDateGen:
#    def __init__(self, path, file_name):
#        super(NoPEDateGen, self).__init__()
#        self.file_name = file_name
#        im = Image.open(path + file_name)
#        im2arr = np.array(im)
#        im2arr = im2arr[:, :, 0:3]
#        color = torch.from_numpy(im2arr).to(device=device)
#        # for png, removes the alpha channel.
#        color = (color.clone().detach() / 255.0).to(dtype=torch.float32)
#        self.W = color.shape[1]
#        self.H = color.shape[0]
#        X = torch.linspace(-1, 1, self.W, dtype=torch.float32, device=device)  # device=None,)
#        X = X.view(1, -1, 1).repeat(self.H, 1, 1)
#        Y = torch.linspace(-1, 1, self.H, dtype=torch.float32, device=device)  # device=None,)
#        Y = Y.view(-1, 1, 1).repeat(1, self.W, 1)
#        self.original_data = torch.cat((X, Y, color), dim=-1)
#        self.original_data = self.original_data.flatten(0, 1)
#        self.data_length = self.original_data.shape[0]
#        self.epoch = 0
#        [self.shuffle() for i in range(10)]
#        self.epoch = 0  # again.
#        self.shuffle_counter = Counter(20, 20)
#        # init for def get_pure_coord
#        self._make_pure_coord()
#        pass  # def __init__
#
#    def shuffle(self):
#        self.shuffled_data = self.original_data.clone().detach()
#        l = self.shuffled_data.shape[0]
#
#        pos2 = numpy.random.randint(l / 4, l * 3 / 4, size=(1,))[0]
#        pos1 = numpy.random.randint(pos2 / 4, pos2 * 3 / 4, size=(1,))[0]
#        right_side_length = l - pos2
#        pos3 = numpy.random.randint(pos2 + right_side_length / 4, pos2 + right_side_length * 3 / 4, size=(1,))[0]
#
#        data = self.shuffled_data
#        self.shuffled_data = torch.cat((data[pos2:pos3], data[:pos1], data[pos3:], data[pos1:pos2]))
#
#        self.dataGenPos = 0
#        self.epoch += 1
#        pass  # def
#
#    def get_data(self, batch_size=32):
#        left = batch_size
#        l = self.shuffled_data.shape[0]
#        coord = self.shuffled_data[0:0,:2]
#        color = self.shuffled_data[0:0,2:]
#
#        while 1:
#            to_add = min(left, l-self.dataGenPos)
#            cell = self.shuffled_data[self.dataGenPos:self.dataGenPos + to_add]
#            coord = torch.cat((coord, cell[:,:2]))
#            color = torch.cat((color, cell[:,2:]))
#            self.dataGenPos = self.dataGenPos + to_add
#            if l == self.dataGenPos:
#                self.dataGenPos = 0
#                self.epoch += 1
#                if self.shuffle_counter.get(self.epoch):
#                    self.shuffle()
#                    pass
#                pass
#            left -= to_add
#            if 0==left:
#                return coord, color
#                pass
#            pass#while 1
#        pass#def
#
#    def get_pure_coord(self, batch_size=128):
#        # print(F"batch size   {batch_size}")
#        if self.pure_coord_ind + batch_size >= self.data_length:
#            # the last part
#            result = self.pure_coords[self.pure_coord_ind:]
#            self.pure_coord_ind = 0
#            return {'coords': result, 'is_last': True}
#            pass
#        last_ind = self.pure_coord_ind + batch_size
#        result = self.pure_coords[self.pure_coord_ind:last_ind]
#        self.pure_coord_ind = last_ind
#        return {'coords': result, 'is_last': False}
#        pass  # def
#
#    def _make_pure_coord(self):
#        X = torch.linspace(-1, 1, self.W, dtype=torch.float32)  # device=None,)
#        X = X.view(1, -1, 1).repeat(self.H, 1, 1)
#        Y = torch.linspace(-1, 1, self.H, dtype=torch.float32)  # device=None,)
#        Y = Y.view(-1, 1, 1).repeat(1, self.W, 1)
#        self.pure_coords = torch.cat((X, Y), dim=-1).view(-1, 2).cuda()
#        self.pure_coord_ind = 0
#        pass  # def
#
#    pass


if 0:
    data_gen = NoPEDateGen('dataset/', 'test.png')
    #X, Y = data_gen.get_data(99)
    #print(X.device)
    #X, Y = data_gen.get_data(99)

    pass
if 0:
    X, Y = data_gen.get_data(3)
    print(X)
    print(Y)
    X = data_gen.get_pure_coord(7)
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")
    X = data_gen.get_pure_coord(7)
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")
    X = data_gen.get_pure_coord(7)
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")
    print(X)
    pass
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")

    pass


#class AutoScheduler:
#    def __init__(self, optimizer, distance = 100, gamma = 0.9):
#        self.dist = distance# assuming loss > lr. So, this distance is define as some value near loss/lr
#        self.gamma = gamma
#        self.reverse_gamma = 1/gamma
#
#        # The real core:
#        self.sdl = torch.optim.lr_scheduler.LambdaLR(optimizer, self.sdl_fn)
#        self.init_lr = self.sdl.get_last_lr()[0]
#        #self.sdl = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, self.sdl_fn)
#        pass
#
#    def sdl_fn(self, n):
#        if 0 == n:
#            return 1
#            pass
#
#        result = 1
#
#        last_lr = self.sdl.get_last_lr()[0]
#        ratio = self.loss / (last_lr * self.dist)
#        if ratio < 1:
#            result = min(ratio, self.gamma)
#            pass
#        if ratio > 10:
#            result = self.reverse_gamma
#            pass
#
#        return (result*last_lr)/self.init_lr
#        pass
#    def step(self, loss):
#        self.loss = loss
#        self.sdl.step()
#        pass
#    pass


#class Gaussian_like_activation(torch.nn.Module):
#    '''This is a simplified version of '''
#    def __init__(self, size, * , w = 1., l_like = 1., sigma_like = 2.):
#        super(Gaussian_like_activation, self).__init__()
#        self.w          = torch.nn.Parameter(torch.full((size, ), w         , dtype = torch.float32), True)
#        self.l_like     = torch.nn.Parameter(torch.full((size, ), l_like    , dtype = torch.float32), True)
#        self.sigma_like = torch.nn.Parameter(torch.full((size, ), sigma_like, dtype = torch.float32), True)
#        pass
#    def forward(self, x):
#        x = x * self.w
#        numerator = -torch.pow(x * x, self.l_like)
#        result = torch.exp(numerator / (self.sigma_like * self.sigma_like))
#        return result
#        pass
#    def __str__(self):
#        return F'Gaussian like activation. w: {self.w}  l_like: {self.l_like}  sigma_like: {self.sigma_like}'
#    pass


#class Gaussian_like_activation_Parabola_Leaky_ReLU(torch.nn.Module):
#    def __init__(self, size, * , sqrt_of_a = 1., negative_slope=0.01):
#        super(Gaussian_like_activation_Parabola_Leaky_ReLU, self).__init__()
#        self.sqrt_of_a = torch.nn.Parameter(torch.full((size, ), sqrt_of_a, dtype = torch.float32), True)
#        self.negative_slope = negative_slope
#        pass
#    def forward(self, x):
#        x = x * self.sqrt_of_a
#        x = 1 - x * x # 1 - (x*sqrt(a))2 === -ax2 + 1. This is the parabola for the name.
#        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
#        return x
#        pass
#    def __str__(self):
#        a = self.sqrt_of_a*self.sqrt_of_a
#        return F'Gaussian like activation. sqrt of a: {self.sqrt_of_a}  a: {a}  the boundary: {torch.abs(1/self.sqrt_of_a)}'
#    pass


#class PDLR(torch.nn.Module):
#    '''Parabola double leady relu'''
#    def __init__(self, size, * , sqrt_of_a = 2., c = 2., negative_slope=0.01):
#        super(PDLR, self).__init__()
#        self.sqrt_of_a = torch.nn.Parameter(torch.full((size, ), sqrt_of_a, dtype = torch.float32), True)
#        self.c         = torch.nn.Parameter(torch.full((size, ), c        , dtype = torch.float32), True)
#        self.negative_slope = negative_slope
#        pass
#    def forward(self, x):
#        x = x * self.sqrt_of_a
#        x = self.c - x * x # c - (x*sqrt(a))2 === -ax2 + c. This is the parabola for the name.
#        # Then, double relu to make the result almost inside 0 to 1
#        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
#        x = 1 - torch.nn.functional.leaky_relu(1 - x, negative_slope=self.negative_slope)
#        return x
#        pass
#    def __str__(self):
#        a = self.sqrt_of_a*self.sqrt_of_a
#        return F'Parabola double leady relu. sqrt of a: {self.sqrt_of_a}  a: {a}  c: {self.c}  the boundary: {torch.abs(torch.sqrt(self.c)/self.sqrt_of_a)}'
#    pass


#class LLC(torch.nn.Module):
#    '''Linear leaky clamp'''
#    def __init__(self, size, * , a = -1., b = 2., negative_slope=0.01):
#        super(LLC, self).__init__()
#        self.a = torch.nn.Parameter(torch.full((size, ), a, dtype = torch.float32), True)
#        self.b = torch.nn.Parameter(torch.full((size, ), b, dtype = torch.float32), True)
#        self.negative_slope = negative_slope
#        pass
#    def forward(self, x):
#        x = torch.abs(x)
#        x = self.a*x+self.b
#        # Then, double relu to make the result almost inside 0 to 1
#        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
#        x = 1 - torch.nn.functional.leaky_relu(1 - x, negative_slope=self.negative_slope)
#        return x
#        pass
#    def __str__(self):
#        return F'Linear leaky clamp. a: {self.a}  b: {self.b}  the boundary: {torch.abs(self.b/self.a)}'
#    pass


#class LLC2_functional_part(torch.autograd.Function):
#    '''Linear leaky clamp 2
#    The forward and backward hebavior differently.
#    '''
#    @staticmethod
#    def forward(ctx, x, a, b, factor, negative_slope):
#        #assert factor.item()>=1.
#        #assert negative_slope.item()>0. and negative_slope.item()<1.
#        ctx.save_for_backward(x, a, b, torch.tensor([factor]), torch.tensor([negative_slope]))
#        x = torch.abs(x)
#        x = a * x + b
#        # Then, double relu to make the result almost inside 0 to 1
#        x = torch.relu(x)
#        x = 1 - torch.relu(1 - x)
#        return x
#        pass
#    @staticmethod
#    def backward(ctx, g):
#        x, a, b, factor, negative_slope = ctx.saved_tensors
#        factor = factor.item()
#        negative_slope = negative_slope.item()
#        boundary = torch.abs(b/a)*factor
#        x_abs = torch.abs(x)
#        out_flags = x_abs>boundary
#        ori_ax_b = a * torch.abs(x) + b
#        in_flags = (ori_ax_b < 1.) * (ori_ax_b > 0.)
#        flags = in_flags + negative_slope * out_flags
#        b_grad = g*flags
#        a_grad = g*flags*x_abs
#        x_less_than_0 = x<0.
#        x_less_than_0 = x_less_than_0 * -2 + 1
#        x_grad = g*flags*a*x_less_than_0
#        return x_grad, a_grad, b_grad, None, None
#        pass
#    def __str__(self):
#        return F'LLC2_functional. It\'s designed only for the LLC2 layer. It behaviors differently in forward propagation and backward propagation.'
#        pass
#    pass
#class LLC2(torch.nn.Module):
#    '''Linear leaky clamp version 2.0
#    The forward pass is pretty simple.
#    '''
#    def __init__(self, size, * , a = -1., b = 2., boundary_factor = 2., negative_slope=0.01):
#        super(LLC2, self).__init__()
#        self.a = torch.nn.Parameter(torch.full((size, ), a, dtype = torch.float32), True)
#        self.b = torch.nn.Parameter(torch.full((size, ), b, dtype = torch.float32), True)
#        assert boundary_factor>=1.
#        self.boundary_factor = boundary_factor
#        self.negative_slope = negative_slope
#        pass
#    def forward(self, x):
#        self.a = torch.nn.Parameter(-torch.abs(self.a))
#        x = LLC2_functional_part.apply(x, self.a, self.b, self.boundary_factor, self.negative_slope)
#        # The parameters for the specific function are: x, a, b, factor, negative_slope
#        return x
#        pass
#    def __str__(self):
#        return F'Linear leaky clamp ver 2.0. a: {self.a}  b: {self.b}  the boundary: {torch.abs(self.b/self.a)} boundary_factor: {self.boundary_factor:.1f}  negative_slope: {self.negative_slope:.4f}'
#    pass


class L2distance(torch.nn.Module):
    def __init__(self, units, width):
        super(L2distance, self).__init__()
        self.anchors = torch.nn.Parameter(torch.tensor.randn(units, width), requires_grad=True)
        pass
    def forward(self, x):
        x = x - self.anchors
        x = x * x
        x = x.sum(dim = 0)
        return x
        pass
    pass






class Model(torch.nn.Module):
    def __init__(self, output_dims=3):
        super().__init__()
        # model . Model part must be in front of the optim part, since the init of optimizer relies on the registered layers.
        units = 256
        self.Lin0 = torch.nn.Linear(2, units)
        self.Lin1 = torch.nn.Linear(units, units)
        #self.Lin2 = torch.nn.Linear(units, units)
        #self.Lin3 = torch.nn.Linear(units, units)
        #self.Lin4 = torch.nn.Linear(units, units)
        #self.Lin5 = torch.nn.Linear(units, units)
        #self.Lin6 = torch.nn.Linear(units, units)
        #self.Lin7 = torch.nn.Linear(units, units)
        a = -10.
        b = 5
        boundary_factor = 1.
        self.LLC2_0 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        self.LLC2_1 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_2 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_3 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_4 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_5 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_6 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)
        #self.LLC2_7 = LLC2(units, a = a, b = b, boundary_factor = boundary_factor)

        self.Output = torch.nn.Linear(units, output_dims)
        # dropout
        self.dropout_5 = torch.nn.Dropout(0.5)
        # optim:
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.RMSprop(self.parameters(),
                                       lr=1e-2)  # I personally prefer RMSprop, but the original proj used Adam. Probably doesn't affect too much.
        self.sdl = AutoScheduler(self.opt, distance=100)
        pass

    def forward(self, x):
        x = self.LLC2_0(self.Lin0(x))
        #if self.training:
        #    print("=======================================================")
        #    print("=======================================================")
        #    print("=======================================================")
        #    print(x)
        #    pass
        x = self.LLC2_1(self.Lin1(x))
        #x = self.LLC2(self.Lin2(x))
        #x = self.LLC3(self.Lin3(x))
        #x = torch.sigmoid(self.Lin3(x))
        #x = self.LLC4(self.Lin4(x))
        #x = self.LLC5(self.Lin5(x))
        #x = self.LLC6(self.Lin6(x))
        #x = self.LLC7(self.Lin7(x))
        #if self.training:
        #    print(x)
        #    print("---------------------------------------------------")
        #    pass
        return self.Output(x)
        pass


    #def parameters(self, recurse: bool = True): #-> Iterator[Parameter]:
    #    result = [super(Model, self).parameters()]
    #    result.append(self.GLA0.parameters())
    #    pass
    pass


model = Model().float().cuda()

#data_gen = NoPEDateGen('dataset/', 'dot second version.png')
#data_gen = NoPEDateGen('dataset/', 'dot 3.0.png')
data_gen = NoPEDateGen('dataset/', 'glasses.jpg')

#####################################################################################

epochs = 1000000
save_counter = Counter(100)
save_format = 'jpg'
#save_format = 'png'

####################################################

batch_size = 1024  # 1024

if 0:
    data_gen = NoPEDateGen('dataset/', 'test.png')
    epochs = 4
    batch_size = 16  # 1024
    save_counter = Counter(1, 1)

while data_gen.epoch < epochs:
    model.train()
    X, Y = data_gen.get_data(batch_size)    # X is coord(x,y), Y is color(R,G,B)
    model.opt.zero_grad()
    pred = model(X)
    loss = model.loss_fn(pred, Y)
    loss.backward()
    model.opt.step()
    model.sdl.step(loss.item())
    if save_counter.get(data_gen.epoch):
        print(F"Loss:  {loss.item()}")
        #print(model.LLC0)
        #print(model.LLC7)
        model.eval()
        with torch.no_grad():
            coords_ = data_gen.get_pure_coord(256)
            channel_last = model(coords_['coords'])
            while not coords_['is_last']:
                coords_ = data_gen.get_pure_coord(256)
                channel_last = torch.cat((channel_last, model(coords_['coords'])))
                #print(F"channel_last length   {channel_last.shape}")
                pass  # while 1
            # print(channel_last)
            channel_first = torch.cat((channel_last[:, 0].view(1, data_gen.H, data_gen.W),
                                       channel_last[:, 1].view(1, data_gen.H, data_gen.W),
                                       channel_last[:, 2].view(1, data_gen.H, data_gen.W)  # ,
                                       #     torch.ones(1,data_gen.H,data_gen.W)
                                       ))
            #channel_first = (torch.sigmoid((channel_first)) - 0.5) * 7 +0.5#This line converts the data to false color to show the numbers out of 0 to 1
            file_name = F'output/{data_gen.file_name} training_evolution_ {data_gen.epoch:04d} .{save_format}'
            save_image(channel_first, file_name)
            # save_image(torch.rand(3,50,50), F'file_name{torch.rand(1).item:.4f}')
            # save_image(torch.rand(3,50,50), F'11111111.png')

            pass  # with
        pass  # if
    pass  # while