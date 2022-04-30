import torch

class Sparser:
    '''
    This tool helps take max use of torch.nn.Linear. It works both with bias or w/o bias.
    The mechanism is like, if any 2 or more points of the weight( or with the bias) are too close, one of them
    is moved to a random place.
    How:
    sps = Sparser() # or Sparser(rel_dist = 0.001)
    layer = torch.nn.Linear(...)
    for epoch in range(total_epochs):
        #some training code.
        if epoch%1000 == 1000-1:
            sps.apply(layer)
            pass
    I personally recommend you call this tool every 10 ~ 1000 epochs, but the frequency is not tested.
    '''
    #@staticmethod
    #def get_recommend_rel_distance(dim0):
    #    '''This function helps decide the hyperparameter of relative distance'''
    #    return 0.05/dim0
    #    pass

    def __init__(self, rel_dist = 0.01):
        self.rel_dist = rel_dist
        pass

    def apply(self, LinearLayer):
        with torch.no_grad():
            if LinearLayer.bias == None:
                combined = LinearLayer.weight
                pass
            else:
                combined = torch.cat((LinearLayer.weight, LinearLayer.bias.view(-1, 1)), dim=1)
                pass
            mean = combined.mean(dim=0, keepdim=True)
            _centrilized = combined - mean
            std = _centrilized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is very important here!!!
            _normalized = _centrilized / std
            length = _normalized.shape[0]
            seg_lists = _normalized.split([1 for i in range(length)])
            seg_lists = list(seg_lists)

            dist = self.rel_dist/length
            count = 0
            for i1 in range(length-1):
                for i2 in range(i1+1, length):
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
                LinearLayer.bias = torch.nn.Parameter(split_segs[1])
                pass

            pass#with torch.no_grad():
            return count
        pass#def

    pass#class

sps = Sparser()

if 0:
    l1 = torch.nn.Linear(2,3,bias=False)
    l1.weight = torch.nn.Parameter(torch.tensor([[1., 1.],[1., 1.],[2., 2.]]))
    sps.apply((l1))
    pass

l2 = torch.nn.Linear(1,3,bias=True)
l2.weight = torch.nn.Parameter(torch.tensor([[1., 1.],[1., 1.],[2., 2.]]))
l2.bias = torch.nn.Parameter(torch.tensor([[1.],[1.],[2.]]))
sps.apply((l2))

l3 = torch.nn.Linear(2,3,bias=False)
l3.weight = torch.nn.Parameter(torch.tensor([[1.],[1.],[2.]]))
sps.apply((l3))

