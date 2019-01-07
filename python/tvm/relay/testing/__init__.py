"""Utilities for testing and benchmarks"""
from __future__ import absolute_import as _abs

from . import mlp
from . import resnet
from . import dqn
from . import dcgan
from . import mobilenet
from . import lstm
from . import inception_v3
from . import squeezenet
from . import vgg
from . import densenet

from .config import ctx_list
from .init import create_workload
