#!/usr/bin/env python
# coding: utf-8
"""NNVM python API for ease of use and help new framework establish python API. """
from __future__ import absolute_import as _abs
import warnings

from . import _base
from . import symbol as sym
from . import symbol
from ._base import NNVMError
from . import frontend

__version__ = _base.__version__

warnings.warn("NNVM is deprecated and will be removed in a future version. Use Relay instead.",
              FutureWarning)
