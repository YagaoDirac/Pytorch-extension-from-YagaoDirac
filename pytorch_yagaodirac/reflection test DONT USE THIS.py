from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from collections import OrderedDict
import torch
Don't use this at the moment.

from .nn.GBN import GBN

class TEST():
    def __init__(self):
        self._gbn_layers: Dict[str, Optional[GBN]] = OrderedDict()
        #self._need_refresh: Dict[str, Optional[GBN]] = OrderedDict()

    def __setattr__(self, name: str, value):
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                        pass
                    else:
                        d.discard(name)
                        pass
                    pass
                pass
            pass

        gbn_layers = self.__dict__.get('_gbn_layers')
        if isinstance(value, GBN):
            if gbn_layers is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
                pass
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            gbn_layers[name]= value
            return
        elif (gbn_layers is not None) and (name in gbn_layers):
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
                pass
            self.register_parameter
            return


            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
        pass#def
t = TEST()








