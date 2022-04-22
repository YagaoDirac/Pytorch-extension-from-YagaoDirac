from PIL import Image
import numpy
import torch

#https://stackoverflow.com/questions/67631/how-do-i-import-a-module-given-the-full-path
#https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
import importlib.util
spec = importlib.util.spec_from_file_location("util", "../util.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
#test_var = util.Counter()


device = torch.device('cuda')

class nerf2d_datagen_no_pe:
    def __init__(self, path, file_name):
        super(nerf2d_datagen_no_pe, self).__init__()
        self.file_name = file_name
        im = Image.open(path + file_name)
        im2arr = numpy.array(im)
        im2arr = im2arr[:, :, 0:3]
        color = torch.from_numpy(im2arr).to(device=device)
        # for png, removes the alpha channel.
        color = (color.clone().detach() / 255.0).to(dtype=torch.float32)
        self.W = color.shape[1]
        self.H = color.shape[0]
        X = torch.linspace(-1, 1, self.W, dtype=torch.float32, device=device)  # device=None,)
        X = X.view(1, -1, 1).repeat(self.H, 1, 1)
        Y = torch.linspace(-1, 1, self.H, dtype=torch.float32, device=device)  # device=None,)
        Y = Y.view(-1, 1, 1).repeat(1, self.W, 1)
        self.original_data = torch.cat((X, Y, color), dim=-1)
        self.original_data = self.original_data.flatten(0, 1)
        self.data_length = self.original_data.shape[0]
        self.epoch = 0
        [self.shuffle() for i in range(10)]
        self.epoch = 0  # again.
        self.shuffle_counter = util.Counter(20, 20)
        # init for def get_pure_coord
        self._make_pure_coord()
        pass  # def __init__

    def shuffle(self):
        self.shuffled_data = self.original_data.clone().detach()
        l = self.shuffled_data.shape[0]

        pos2 = numpy.random.randint(l / 4, l * 3 / 4, size=(1,))[0]
        pos1 = numpy.random.randint(pos2 / 4, pos2 * 3 / 4, size=(1,))[0]
        right_side_length = l - pos2
        pos3 = numpy.random.randint(pos2 + right_side_length / 4, pos2 + right_side_length * 3 / 4, size=(1,))[0]

        data = self.shuffled_data
        self.shuffled_data = torch.cat((data[pos2:pos3], data[:pos1], data[pos3:], data[pos1:pos2]))

        self.dataGenPos = 0
        self.epoch += 1
        pass  # def

    def get_data(self, batch_size=32):
        left = batch_size
        l = self.shuffled_data.shape[0]
        coord = self.shuffled_data[0:0,:2]
        color = self.shuffled_data[0:0,2:]

        while 1:
            to_add = min(left, l-self.dataGenPos)
            cell = self.shuffled_data[self.dataGenPos:self.dataGenPos + to_add]
            coord = torch.cat((coord, cell[:,:2]))
            color = torch.cat((color, cell[:,2:]))
            self.dataGenPos = self.dataGenPos + to_add
            if l == self.dataGenPos:
                self.dataGenPos = 0
                self.epoch += 1
                if self.shuffle_counter.get(self.epoch):
                    self.shuffle()
                    pass
                pass
            left -= to_add
            if 0==left:
                return coord, color
                pass
            pass#while 1
        pass#def

    def get_pure_coord(self, batch_size=128):
        # print(F"batch size   {batch_size}")
        if self.pure_coord_ind + batch_size >= self.data_length:
            # the last part
            result = self.pure_coords[self.pure_coord_ind:]
            self.pure_coord_ind = 0
            return {'coords': result, 'is_last': True}
            pass
        last_ind = self.pure_coord_ind + batch_size
        result = self.pure_coords[self.pure_coord_ind:last_ind]
        self.pure_coord_ind = last_ind
        return {'coords': result, 'is_last': False}
        pass  # def

    def _make_pure_coord(self):
        X = torch.linspace(-1, 1, self.W, dtype=torch.float32)  # device=None,)
        X = X.view(1, -1, 1).repeat(self.H, 1, 1)
        Y = torch.linspace(-1, 1, self.H, dtype=torch.float32)  # device=None,)
        Y = Y.view(-1, 1, 1).repeat(1, self.W, 1)
        self.pure_coords = torch.cat((X, Y), dim=-1).view(-1, 2).cuda()
        self.pure_coord_ind = 0
        pass  # def

    pass



if 1:
    data_gen = nerf2d_datagen_no_pe('dataset/', 'test.png')
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