import torch

'''
to do:
Since different layers don't share the same parameters, this tool probably needs to fit the layers, 
especially the mostly used, built in.
'''

class Sparser_torch_nn_Linear:
    '''
    This tool helps take max use of torch.nn.Linear. It works both with bias or w/o bias.
    Also I recommend you use this with dropout.
    Dropout force as many neutrons to be significant which always push some vectors to the same point. Then sparse them.
    The mechanism is like, if any 2 or more points of the weight( or with the bias) are too close, one of them
    is moved to a random place.
    How:
    sps = Sparser() # or Sparser(rel_dist = 0.001)
    layer = torch.nn.Linear(...)
    #also, in the forward function in the model, do something like: x = torch.dropout(layer)
    for epoch in range(total_epochs):
        #some training code.
        if epoch%1000 == 1000-1:
            sps.apply(layer)
            pass
    I personally recommend you call this tool every 10 ~ 1000 epochs, but the frequency is not tested.
    '''
    def __init__(self, abs_dist = 0.002,* ,rel_dist = 0.01, epi = 1e-5):
        self.abs_dist = abs_dist
        self.rel_dist = rel_dist
        self.epi = epi
        pass

    def apply(self, LinearLayer):
        length = LinearLayer.weight.shape[0]
        if length < 2:
            raise(Exception("This sparser works only with at least 2 dim for the output of a nn.Linear."))
            pass
        with torch.no_grad():
            if LinearLayer.bias == None:
                combined = LinearLayer.weight
                pass
            else:
                combined = torch.cat((LinearLayer.weight, LinearLayer.bias.view(-1, 1)), dim=1)
                pass

            mean = combined.mean(dim=0, keepdim=True)
            _centralized = combined - mean
            std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is very important here!!!
            std_too_small = std<self.epi
            std = (std-std*std_too_small)+std_too_small*self.epi
            _normalized = _centralized / std
            #length = _normalized.shape[0]
            seg_lists = _normalized.split([1 for i in range(length)])
            seg_lists = list(seg_lists)

            dist = self.rel_dist/torch.pow(torch.tensor([length],dtype = torch.float32), torch.tensor([1/combined.shape[1]],dtype = torch.float32))#可能不对。
            dist = dist.item()
            count = 0
            for i1 in range(length-1):
                for i2 in range(i1+1, length):
                    _dif = combined[i1] - combined[i2]
                    _dis = (_dif * _dif).mean()
                    if _dis.item() < self.abs_dist:
                        seg_lists[i2] = torch.rand_like(seg_lists[i2]) * (self.abs_dist * 10/std)
                        # mean + 10 * abs_distance.
                        # Also, I assumed this is not near any other points.
                        combined[i2] = seg_lists[i2]*std+mean
                        count += 1
                        continue
                        pass

                    _dif = seg_lists[i1] - seg_lists[i2]
                    _dis = (_dif*_dif).mean()
                    if _dis.item()<dist:
                        seg_lists[i2] = torch.rand_like(seg_lists[i2])*2
                        # mean + 2std. This is already nearly 95%, I don't remember.
                        #Also, I assumed this is not near any other points.
                        combined[i2] = seg_lists[i2]*std+mean
                        count += 1
                        pass
                    pass
                pass

            _combined2 = torch.cat(seg_lists, dim = 0)
            _combined2 = _combined2*std+mean
            if LinearLayer.bias == None:
                LinearLayer.weight = torch.nn.Parameter(_combined2)
                pass
            else:
                split_segs = _combined2.split((_combined2.shape[1]-1,1), dim = 1)
                LinearLayer.weight = torch.nn.Parameter(split_segs[0])
                LinearLayer.bias = torch.nn.Parameter(split_segs[1].view(-1))
                pass

            pass#with torch.no_grad():
            return count
        pass#def

    pass#class


#test
if 0:
    sps1 = Sparser_torch_nn_Linear(abs_dist=0.1)
    l1 = torch.nn.Linear(2,2,bias=False)
    l1.weight = torch.nn.Parameter(torch.tensor([[1., 1.],[1., 1.]]))
    sps1.apply((l1))

    sps2 = Sparser_torch_nn_Linear(abs_dist=0.00001, rel_dist=0.1)
    l2 = torch.nn.Linear(2,3,bias=True)
    l2.weight = torch.nn.Parameter(torch.tensor([[1., 1.],[1., 1.1],[2., 2]]))
    l2.bias = torch.nn.Parameter(torch.tensor([[1.],[1.],[2.]]))
    sps2.apply((l2))

    sps3 = Sparser_torch_nn_Linear()
    l3 = torch.nn.Linear(1,3,bias=False)
    l3.weight = torch.nn.Parameter(torch.tensor([[1.],[1.],[2.]]))
    sps3.apply((l3))

    sps4 = Sparser_torch_nn_Linear()
    l4 = torch.nn.Linear(1, 3, bias=False)
    l4.weight = torch.nn.Parameter(torch.tensor([[1.], [1.], [2.]]))
    l4.cuda()
    sps4.apply((l4))
    pass

