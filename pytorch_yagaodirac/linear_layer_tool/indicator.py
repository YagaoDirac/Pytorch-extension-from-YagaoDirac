import torch

class indicator_result():
    def __init__(self, valid = False, info = 'For some reason, this result is invalid.', score = 0., name = 'invalid name'):
        self.valid = valid
        self.info = info
        self.score = score
        self.name = name
        pass
    def set_name(self, name):
        self.name = name
        pass
    def __str__(self):
        if not self.valid:
            return F"{self.name} {self.info}"
            pass
        return F"{self.name} {self.info} Converge score: {self.score:.2f}"
        pass
    def to_gbn_scale(self):
        score = self.score #this is [0,1](math ly)


        pass
    pass#class




class Linear_indicator():
    #def forward(self):
    #    raise Exception("This is not a layer.")
    #    pass
    @torch.no_grad()
    def __init__(self, layer:torch.nn.Linear, slice = [4, 4], *, name = '', epi = 1e-5):
        '''example: slice = [4, 4]. Or, slice can be torch.size object. Only slice[0] and [1] are used.'''
        #super(Linear_indicator, self).__init__()
        self.data = {}
        self.data['layer'] = layer#To not modify the position of this layer.
        self.slice = slice
        if None == self.slice:
            self.slice = self.data['layer'].weight.shape
            pass
        self.weight_save1 = None
        self.weight_save2 = None
        self._update_save()
        self.name = name
        self.epi = epi
        pass

    @torch.no_grad()
    def _update_save(self):
        #new to save1, save1 to save2
        self.weight_save2 = self.weight_save1
        self.weight_save1 = self._get_slice()
        pass

    @torch.no_grad()
    def _get_slice(self):
        temp:torch.Tensor = self.data['layer'].weight[:self.slice[0], :self.slice[1]].clone()#.detach()
        temp = temp.view(1, temp.shape[0], temp.shape[1])
        temp = temp.to(self.data['layer'].weight.device)
        return temp
        pass

    @torch.no_grad()
    def update(self):
        if (None == self.weight_save2):
            #for the first time, returns invalid result.
            self._update_save()
            result = indicator_result(False, 'One more update needed to generate valid result.', name = self.name)
            return result
            pass
        device = self.data['layer'].weight.device
        self.weight_save1 = self.weight_save1.to(self.data['layer'].weight.device)
        self.weight_save2 = self.weight_save2.to(self.data['layer'].weight.device)

        combined = torch.cat((self.weight_save1, self.weight_save2))
        the_min = combined.min(dim = 0)[0]
        the_max = combined.max(dim = 0)[0]
        new_slice = self._get_slice()

        _temp = the_max-the_min
        _temp = _temp - self.epi
        _temp = torch.relu(_temp)
        _temp = _temp + self.epi

        #new to save1, save1 to save2
        score = 4*(new_slice-the_min)*(the_max-new_slice)/(_temp*_temp)
        score = torch.relu(score)
        score = score.mean()

        #new to save1, save1 to save2
        self.weight_save2 = self.weight_save1
        self.weight_save1 = new_slice

        result = indicator_result(True, 'OK', score.item(), name = self.name)
        return result
        pass




if 0:
    layer = torch.nn.Linear(1,2,bias = False)
    indicator = Linear_indicator(layer)
    layer.weight = torch.nn.Parameter(layer.weight+1)
    score = indicator.update()
    layer.weight = torch.nn.Parameter(layer.weight+1)
    score = indicator.update()
    layer.weight = torch.nn.Parameter(layer.weight - 0.25)
    score = indicator.update()
    layer.weight = torch.nn.Parameter(layer.weight + 0.125)
    score = indicator.update()

    layer2 = torch.nn.Linear(1, 2, bias=False).cuda()
    indicator2 = Linear_indicator(layer)
    score2 = indicator2.update()
    score2 = indicator2.update()
    score2 = indicator2.update()


    sdfhjkl  = 345789
    pass






