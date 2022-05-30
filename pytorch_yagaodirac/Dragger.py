import torch

class Dragger_simple():
    def __init__(self, volume = 5, init_val = 0.):
        self.volume = volume
        self.one_over_size = 1 / volume
        self.one_minus_one_over_size = 1-self.one_over_size
        self.value = init_val
        pass
    def set(self, value):
        self.value = value*self.one_over_size+self.value*self.one_minus_one_over_size
        pass
    def get(self):
        return self.value
        pass
    pass#class


if 0:
    d = Dragger_simple()
    d.set(1)
    d.set(1)
    d.set(1)
    pass










class Dragger():
    def __init__(self, size = 50, init_val = 0.):
        self.data = torch.full((size,), init_val, requires_grad= False)
        self.ind = 0
        self.size = size
        self._dirty = False
        self._result = init_val
        pass
    def set(self, value):
        self.data[self.ind] = value
        self.ind += 1
        self.ind %= self.size
        self._dirty = True
        pass
    def get(self):
        if self._dirty:
            self._result = self.data.mean()
            self._dirty = False
            pass
        return self._result
        pass
    pass#class



if 0:
    d = Dragger(3)
    d.set(1)
    out1 = d.get()
    d.set(1)
    out1 = d.get()
    d.set(2)
    out1 = d.get()
    d.set(2)
    out1 = d.get()
    d.set(3)
    out1 = d.get()
    d.set(3)
    out1 = d.get()
    d.set(4)
    out1 = d.get()
    d.set(4)
    out1 = d.get()
    pass

