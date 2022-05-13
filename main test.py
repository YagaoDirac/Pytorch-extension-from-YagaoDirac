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



from termcolor import colored
#print(colored('hello', 'red'), colored('world', 'green'))
#print(colored("hello red world", 'red'))

import numpy as np
import numpy.random

from scipy import signal, special
from PIL import Image

import torch
import torchvision
from torchvision.utils import save_image

#My customized part.
import pytorch_yagaodirac as yd

device = torch.device('cuda')


class Model(torch.nn.Module):
    def __init__(self, output_dims=3):
        super().__init__()
        # model . Model part must be in front of the optim part, since the init of optimizer relies on the registered layers.
        units = 1
        self.Lin0 = torch.nn.Linear(2, units)
        # self.Lin0.weight = torch.nn.Parameter(torch.tensor([[5, 0]]*units,dtype = torch.float32))
        # self.Lin0.bias = torch.nn.Parameter(torch.tensor([0]*units,dtype = torch.float32))

        # self.Lin1 = torch.nn.Linear(units, units)
        self.Gaussian0 = yd.nn.Gaussian_simple(units, True, init_shrink_factor=1)

        # self.Gaussian1 = yd.nn.Gaussian_simple(units)

        ################Don't modify anything under this line unless you know what you are doing.
        self.Output = torch.nn.Linear(units, output_dims)
        self.Output.weight = torch.nn.Parameter(torch.zeros((output_dims, units), dtype=torch.float32))
        self.Output.bias = torch.nn.Parameter(torch.zeros(output_dims, dtype=torch.float32))

        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.RMSprop(self.parameters(),
                                       lr=1e-2)  # I personally prefer RMSprop, but the original proj used Adam. Probably doesn't affect too much.
        self.sdl = yd.optim.AutoScheduler(self.opt, distance=10)
        self.printing = False

        # self.sparser = yd.sparser.Sparser_torch_nn_Linear(abs_dist=0.002, rel_dist=1)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.gbn = yd.nn.GBN(scale=1)
        pass

    def forward(self, x):
        x = self.dropout(x)
        # x = self.ALLC2_0(self.L2D0(self.SelMul0(x)))
        x = self.Lin0(x)
        x = self.gbn(x)
        x = self.Gaussian0(x)
        x = self.gbn(x)

        # debug_string = "hidden layer 1:"
        # if self.printing:
        #    if len(list(x.shape)) == 1:
        #        print(F"{debug_string}{x}")
        #        pass
        #    else:
        #        print(F"{debug_string}{x}")
        #        pass
        #    pass
        return self.Output(x)
        pass  # def forward

    pass  # class


model = Model().float().cuda()

# data_gen = NoPEDateGen('dataset/', 'dot second version.png')
# data_gen = NoPEDateGen('dataset/', 'dot 3.0.png')
# data_gen = yd.datagen.nerf2d_datagen_no_pe('dataset/', 'glasses.jpg')
data_gen = yd.datagen.nerf2d_datagen_no_pe('dataset/', 'dot 3.0.png').cuda()
save_format = 'jpg'
save_format = 'png'

batch_size = 1024  # 1024
########################################################################################################################################################################
epochs = 20
save_counter = yd.Counter(epochs / 5)
########################################################################################################################################################################
if 0:
    epochs = 4
    batch_size = 16  # 1024
    save_counter = Counter(1)
    pass

while data_gen.epoch < epochs:
    model.train()
    model.printing = False
    X, Y = data_gen.get_data(batch_size)  # X is coord(x,y), Y is color(R,G,B)
    model.opt.zero_grad()
    pred = model(X)
    loss = model.loss_fn(pred, Y)
    loss.backward()
    model.opt.step()

    # break

    if save_counter.get(data_gen.epoch):
        with torch.no_grad():
            print(colored(F"-----------------   {data_gen.epoch}   ------------------", 'yellow'))
            print(colored(F"Lin0-------------------------", 'red'))
            print(model.Lin0.weight.data.clone().detach().cpu().numpy())
            print(model.Lin0.bias.data.clone().detach().cpu().numpy())
            print(colored(F"Gaussian-------------------------", 'red'))
            print(model.Gaussian0.one_over_c.data.clone().detach().cpu().numpy())
            if None != model.Gaussian0.bias:
                print(model.Gaussian0.bias.data.clone().detach().cpu().numpy())
                pass
            print(colored(F"Output-------------------------", 'red'))
            print(model.Output.weight.data.clone().detach().cpu().numpy())
            print(model.Output.bias.data.clone().detach().cpu().numpy())
            model.printing = True
            # model(torch.tensor([-1, 0], dtype=torch.float32, device=device))
            # model(torch.tensor([0, 0], dtype=torch.float32, device=device))
            # model(torch.tensor([0, -1], dtype=torch.float32, device=device))
            model.printing = False
            pass
        model.sdl.step(loss.item())
        temp = 0
        # temp = model.sparser.apply(model.Lin0)
        # print(F"sparsed: {temp} -----------")

        print(F"Loss:  {loss.item()}")
        model.eval()
        with torch.no_grad():
            # coords_ = data_gen.get_pure_coord(256)

            # channel_last = model(coords_['data'])
            channel_last = torch.empty(0, 3).cuda()
            is_last = False
            while not is_last:
                coords_ = data_gen.get_pure_coord(256)
                channel_last = torch.cat((channel_last, model(coords_['data'])))
                # raise(Exception("STOP!!!!!!!!"))
                is_last = coords_['is_last']
                # print(F"channel_last length   {channel_last.shape}")
                pass  # while 1
            # print(channel_last)
            channel_first = torch.cat((channel_last[:, 0].view(1, data_gen.H, data_gen.W),
                                       channel_last[:, 1].view(1, data_gen.H, data_gen.W),
                                       channel_last[:, 2].view(1, data_gen.H, data_gen.W)  # ,
                                       #     torch.ones(1,data_gen.H,data_gen.W)
                                       ))
            # channel_first = (torch.sigmoid((channel_first)) - 0.5) * 7 +0.5#This line converts the data to false color to show the numbers out of 0 to 1
            file_name = F'output/{data_gen.file_name} training_evolution_ {data_gen.epoch:04d} .{save_format}'
            save_image(channel_first, file_name)
            # save_image(torch.rand(3,50,50), F'11111111.png')
            pass  # with
        pass  # if
    pass  # while





