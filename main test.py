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
        in_dim = 4
        out_put = 3
        width = 16
        units = [in_dim]

        init_lr = 1e-2
        factor = 1  # this is test only. Set to 1.
        scale = 1  # this is test only. Set to 1.

        i = 0
        units.append(width)
        self.Lin0 = torch.nn.Linear(units[i], units[i + 1])
        self.gbn0 = yd.nn.GBN(scale=scale, lr=init_lr)
        self.Lin0.weight = torch.nn.Parameter(self.Lin0.weight / units[i] * factor)
        self.Indicator0 = yd.Linear_indicator(self.Lin0)
        # self.Lin0.weight = torch.nn.Parameter(torch.tensor([[1, 0.8]]*units[i+1], dtype = torch.float32))
        # self.Lin0.bias = torch.nn.Parameter(torch.full((units[i+1],), 0.666* factor, dtype = torch.float32), requires_grad = True)

        i = 1
        units.append(width)
        self.Lin1 = torch.nn.Linear(units[i], units[i + 1])
        self.gbn1 = yd.nn.GBN(scale=scale, lr=init_lr)
        self.Lin1.weight = torch.nn.Parameter(self.Lin1.weight / units[i] * factor)
        self.Indicator1 = yd.Linear_indicator(self.Lin1)
        # self.Lin1.weight = torch.nn.Parameter(torch.full((units[i+1], units[i]), 0.444/units[i], dtype = torch.float32), requires_grad = True)
        # self.Lin1.bias = torch.nn.Parameter(torch.full((units[i+1],), 0.444* factor, dtype = torch.float32), requires_grad = True)

        # i = 2
        # units.append(width)
        # self.Lin2 = torch.nn.Linear(units[i], units[i+1])
        # i = 3
        # units.append(width)
        # self.Lin3 = torch.nn.Linear(units[i], units[i+1])

        ################Don't modify anything under this line unless you know what you are doing.
        units.append(out_put)
        self.Output = torch.nn.Linear(units[-2], units[-1])
        self.gbnOut = yd.nn.GBN(scale=1e-3, lr=init_lr)  # Don't know why it doesn't work at all.
        # self.Output.weight = torch.nn.Parameter(torch.full((units[-1], units[-2]), 0.123/units[-2]* factor, dtype = torch.float32), requires_grad = True)
        self.Output.weight = torch.nn.Parameter(self.Output.weight / units[-2] * factor)
        # self.Output.bias = torch.nn.Parameter(torch.full((units[-1], ), 0.35* factor, dtype = torch.float32), requires_grad = True)
        # self.Output.bias = torch.nn.Parameter(torch.full((units[-1], ), 0.35* factor, dtype = torch.float32), requires_grad = True)

        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.RMSprop(self.parameters(),
                                       lr=init_lr)  # I personally prefer RMSprop, but the original proj used Adam. Probably doesn't affect too much.
        self.sdl = yd.optim.AutoScheduler(self.opt, distance=10)
        self.printing = False

        self.sparser = yd.sparser.Sparser_torch_nn_Linear(abs_dist=0.02, rel_dist=0.1)
        self.dropout_small = torch.nn.Dropout(p=0.2)
        self.dropout_big = torch.nn.Dropout(p=0.4)
        pass

    def forward(self, x):
        h1 = x + 2.
        h2 = x - 2.
        x = torch.cat((h1, h2), dim=-1)

        # x = self.dropout_small(x)

        x = self.Lin0(x)
        x = self.gbn0(x)  ###########
        x = yd.Gaussian(x)
        # x = torch.sin(x)
        # x = torch.relu(x)
        # x = self.dropout_big(x)

        x = self.Lin1(x)
        x = self.gbn1(x)
        x = yd.Gaussian(x)
        # x = self.dropout_big(x)

        # debug_string = "hidden layer 1:"
        # if self.printing:
        #    if len(list(x.shape)) == 1:
        #        print(F"{debug_string}{x}")
        #        pass
        #    else:
        #        print(F"{debug_string}{x}")
        #        pass
        #    pass
        x = self.Output(x)
        x = self.gbnOut(x)
        return x
        pass  # def forward

    def on_batch_begin(self):
        self.opt.zero_grad()
        pass

    def on_batch_end(self):
        self.opt.step()
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_checkpoint_begin(self):
        pass

    def on_checkpoint_end(self):
        pass

    pass  # class


model = Model().float().cuda()

# data_gen = NoPEDateGen('dataset/', 'dot second version.png').cuda()
# data_gen = NoPEDateGen('dataset/', 'dot 3.0.png').cuda()
# data_gen = yd.datagen.nerf2d_datagen_no_pe('dataset/', 'glasses.jpg').cuda()
data_gen = yd.datagen.nerf2d_datagen_no_pe('dataset/', 'compound dot.png').cuda()
save_format = 'jpg'
save_format = 'png'

batch_size = 1024  # 1024
########################################################################################################################################################################
save_every = 1
total_save = 5
epochs = save_every * total_save
save_counter = yd.Counter(save_every)

epochs = 20
save_counter = yd.Counter_log(min_step_length=10)
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
    model.on_batch_begin()  # model.opt.zero_grad()
    pred = model(X)
    loss = model.loss_fn(pred, Y)
    loss.backward()
    model.on_batch_end()  # model.opt.step()

    # break
    # print(data_gen.epoch)
    if save_counter.get(data_gen.epoch):
        with torch.no_grad():
            _r = model.Indicator0.update()
            if _r.valid:
                print(F"Lin0 converge score: {_r.score}")
                pass
            #_r = model.Indicator1.update()
            #if _r.valid:
            #    print(F"Lin1 converge score: {_r.score}")
            #    pass
            # print(colored(F"-----------------   {data_gen.epoch}   ------------------", 'green'))
            # print(colored(F"Lin0[0][:4]-------------------------", 'yellow'))
            # print(model.Lin0.weight.data.clone().detach().cpu().numpy()[0][:4])
            # print(model.Lin0.bias.data.clone().detach().cpu().numpy()[0])
            # print(colored(F"Lin1[0][:4]-------------------------", 'yellow'))
            # print(model.Lin1.weight.data.clone().detach().cpu().numpy()[0][:4])
            # print(model.Lin1.bias.data.clone().detach().cpu().numpy()[0])
            print(colored(F"Output[0][:4]-------------------------", 'yellow'))
            print(model.Output.weight.data.clone().detach().cpu().numpy()[0][:4])
            print(model.Output.bias.data.clone().detach().cpu().numpy()[0])

            # model.printing = True
            # model(torch.tensor([-1, 0], dtype=torch.float32, device=device))
            # model(torch.tensor([0, 0], dtype=torch.float32, device=device))
            # model(torch.tensor([0, -1], dtype=torch.float32, device=device))
            # model.printing = False
            pass
        model.sdl.step(loss.item())
        last_lr = model.sdl.get_last_lr()[0]
        print(F"last lr = {last_lr}")
        model.gbn0.set_lr(last_lr)  ###################################################################################
        model.gbn1.set_lr(last_lr)  ###################################################################################
        model.gbnOut.set_lr(
            last_lr)  ###################################################################################
        temp = 0
        # temp = temp + model.sparser.apply(model.Lin0)
        # temp = temp + model.sparser.apply(model.Lin1)
        # temp = temp + model.sparser.apply(model.Lin2)
        # temp = temp + model.sparser.apply(model.Lin3)
        if temp > 0:
            print(colored(F"sparsed: {temp} -----------", 'red'))
            pass
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





