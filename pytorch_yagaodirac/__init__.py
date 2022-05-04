##https://stackoverflow.com/questions/67631/how-do-i-import-a-module-given-the-full-path
##https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
#import importlib.util
#spec = importlib.util.spec_from_file_location("util", "./Counter.py")
#util = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(util)
##test_var = util.Counter()
#
from .Counter import Counter as Counter
from . import datagen
from . import nn
from . import optim
from . import sparser


