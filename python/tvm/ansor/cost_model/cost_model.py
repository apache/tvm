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
# pylint: disable=unused-import
""" ... """
import ctypes
import numpy as np

import tvm._ffi
from tvm.runtime import Object

from .. import _ffi_api


@tvm._ffi.register_object("ansor.CostModel")
class CostModel(Object):
    pass


@tvm._ffi.register_object("ansor.RandomModel")
class RandomModel(Object):
    """
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.RandomModel)

# A random number generator func for c++'s RandomModel
@tvm._ffi.register_func("ansor.cost_model.random_number")
def random_number(n, return_ptr):
    if n == 0:
        return
    return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
    array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
    array_wrapper[:] = np.random.uniform(0, 1, (n,))
