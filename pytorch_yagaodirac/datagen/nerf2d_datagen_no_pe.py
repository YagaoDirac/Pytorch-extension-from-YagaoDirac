from PIL import Image
import numpy
import torch

#https://stackoverflow.com/questions/67631/how-do-i-import-a-module-given-the-full-path
#https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
#import importlib.util
#spec = importlib.util.spec_from_file_location("util", "../Counter.py")
#util = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(util)
#test_var = util.Counter()


device = torch.device('cuda')


#I copied the Counter class from out side. The jupyterLab doesn't run the import code above.
class Counter:
    def __init__(self, start_from = 5, every = -1):
        self.next = start_from
        self.every = every
        if every<=0:
            self.every = start_from
            pass
        pass
    def get(self, current):
        if current>= self.next:
            self.next += self.every
            return True
            pass
        return False
        pass
    pass



class nerf2d_datagen_no_pe(torch.nn.Module):
    def forward(self):
        raise 'This is data generator, not a real neural network. Inherition only servers for the dtype and device management'
        pass
    def __init__(self, path, file_name):
        '''Tips: datagen = nerf2d_datagen_no_pe(...), then, call datagen.cuda() to move it to gpu.
        Also notice, because the original data is never used, it's always on cpu.
        '''
        super(nerf2d_datagen_no_pe, self).__init__()
        #self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.file_name = file_name
        im = Image.open(path + file_name)
        im2arr = numpy.array(im)
        im2arr = im2arr[:, :, 0:3]
        color = torch.nn.Parameter(torch.from_numpy(im2arr), requires_grad= False)
        # for png, removes the alpha channel.
        color = (color.clone().detach() / 255.0)
        self.W = color.shape[1]
        self.H = color.shape[0]
        self._make_pure_coord()
        self.original_data = torch.cat((self.pure_coords, color.view(-1,3)), dim=-1)
        self.X = None
        self.Y = None
        self.data_length = self.original_data.shape[0]
        self.epoch = 0
        self.shuffled_data = torch.nn.Parameter(self.original_data, requires_grad=False)
        [self.shuffle() for i in range(10)]
        self.shuffle_counter = Counter(1)
        pass  # def __init__

    def where_is_original_data(self):
        '''This is gonna be cpu at all time.'''
        return self.original_data.device;
        pass

    def shuffle(self):
        l = self.shuffled_data.shape[0]
        pos2 = numpy.random.randint(l / 4, l * 3 / 4, size=(1,))[0]
        pos1 = numpy.random.randint(pos2 / 4, pos2 * 3 / 4, size=(1,))[0]
        right_side_length = l - pos2
        pos3 = numpy.random.randint(pos2 + right_side_length / 4, pos2 + right_side_length * 3 / 4, size=(1,))[0]

        data = self.shuffled_data
        self.shuffled_data = torch.nn.Parameter(torch.cat((data[pos2:pos3], data[:pos1], data[pos3:], data[pos1:pos2])), requires_grad= False)

        self.dataGenPos = 0
        pass  # def

    def get_data(self, batch_size=32):
        left = batch_size
        l = self.data_length
        coord = self.shuffled_data[0:0,:2]
        color = self.shuffled_data[0:0,2:]

        while 1:
            to_add = min(left, l-self.dataGenPos)
            cell = torch.nn.Parameter(self.shuffled_data[self.dataGenPos:self.dataGenPos + to_add], requires_grad=False)
            cell = cell.to(self.shuffled_data.device)
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
                #coord = torch.nn.Parameter(coord, requires_grad=False).to(self.dummy.device)
                #color = torch.nn.Parameter(color, requires_grad=False).to(self.dummy.device)
                return coord, color
                pass
            pass#while 1
        pass#def

    def get_pure_coord(self, batch_size=128):
        # print(F"batch size   {batch_size}")
        result = {}
        if self.pure_coord_ind + batch_size >= self.data_length:
            # the last part
            data = self.pure_coords[self.pure_coord_ind:]
            self.pure_coord_ind = 0
            result['data'] = data
            result['is_last'] = True
            return result
            pass
        last_ind = self.pure_coord_ind + batch_size
        data = self.pure_coords[self.pure_coord_ind:last_ind]
        self.pure_coord_ind = last_ind
        result['data'] = data
        result['is_last'] = False
        return result
        pass  # def

    def _make_pure_coord(self):
        X = torch.nn.Parameter(torch.linspace(-1, 1, self.W, dtype=torch.float32), requires_grad=False)
        self.X = X.view(1, -1, 1).repeat(self.H, 1, 1)
        Y = torch.nn.Parameter(torch.linspace(-1, 1, self.H, dtype=torch.float32), requires_grad=False)
        self.Y = Y.view(-1, 1, 1).repeat(1, self.W, 1)
        self.pure_coords = torch.nn.Parameter(torch.cat((self.X, self.Y), dim=-1).view(-1, 2), requires_grad=False)
        self.pure_coord_ind = 0
        pass  # def

    pass



if 0:
    data_gen = nerf2d_datagen_no_pe('dataset/', 'test.png')
    data_gen.cuda()
    data_gen.double()
    X,Y = data_gen.get_data(3)
    #print(X.device)
    #print(Y.device)
    data_gen.get_data(21)
    data_gen.get_data(21)

    X = data_gen.get_pure_coord(15)['data']
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")
    print(X.device)
    X = data_gen.get_pure_coord(7)['data']
    print(F"data_gen.pure_coord_ind   {data_gen.pure_coord_ind}")
    print(X.device)
    pass