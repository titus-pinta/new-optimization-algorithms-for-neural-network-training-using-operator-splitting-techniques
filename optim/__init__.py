"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .sparse_adam import SparseAdam
from .adamax import Adamax
from .asgd import ASGD
from .sgd import SGD
from .rprop import Rprop
from .rmsprop import RMSprop
from .optimizer import Optimizer
from .lbfgs import LBFGS
from .a3 import A3
from .a5 import A5
from .a11 import A11
from .a3rms import A3RMS
#from .a3titus import A3Titus
from .a3ada import A3Ada
from .a5rms import A5RMS
from .a5ada import A5Ada
from . import lr_scheduler

del adadelta
del adagrad
del adam
del sparse_adam
del adamax
del asgd
del sgd
del rprop
del rmsprop
del optimizer
del lbfgs
del a3
del a5
del a11
del a3rms
del a3ada
del a5rms
del a5ada
#del a3titus
