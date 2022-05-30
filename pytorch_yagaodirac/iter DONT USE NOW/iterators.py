import torch

r"""
Why and what.
Iterators are old conceptions for probably over 40 years. 
Why do I need a iter layer between real nn layers and some algorithms in my lib?
Because, torch.nn.Linear has weight and bias, 
    my pytorch_yagaodirac.nn.Linear has l0.weight, l0.bias, l1.weight and l1.bias.
Also some activation layers have their own tensor or anything. 
Sometimes some algo can also be applied to grad.
Algorithms in this lib are mainly indicators and sparsers. Generally at least both indicators 
    and sparsers should work for almost all the stateful layers.

Caveats :
Before using this iter system, make sure you know that indirectly connection like iter system 
    separates containers and algorithm which weakens the expressiveness. This cons causes 
    missleading and also real mistakes if the cont-iter-algo combination is not designed. 
    Make sure check a bit before using. 



"""



def make_iter(layer):
    if isinstance(layer, torch.nn.Linear)



class _iter_base:
    def set_layer(layer)
        pass










