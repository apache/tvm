# pylint: disable=wildcard-import
"""Generic declaration and schedules.

This is a recommended way of using TOPI API.
To use the generic schedule function, user must set
the current target scope using with block. See also :any:`tvm.target`

Example
-------
.. code-block:: python

  # create schedule that dispatches to topi.cuda.schedule_injective
  with tvm.target.create("cuda"):
    s = tvm.generic.schedule_injective(outs)
"""
from __future__ import absolute_import as _abs

from .nn import *
from .injective import *
from .extern import *
from .vision import *
from .sort import *
