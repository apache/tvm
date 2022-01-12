# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=wildcard-import
"""Generic declaration and schedules.

This is a recommended way of using TOPI API.
To use the generic schedule function, user must set
the current target scope using with block. See also :any:`tvm.target`

Example
-------
.. code-block:: python

  # create schedule that dispatches to topi.cuda.schedule_injective
  with tvm.target.Target("cuda"):
    s = tvm.tir.generic.schedule_injective(outs)
"""
from __future__ import absolute_import as _abs

from .nn import *
from .injective import *
from .extern import *
from .vision import *
from .sort import *
from .search import *
from .image import *
from .math import *
