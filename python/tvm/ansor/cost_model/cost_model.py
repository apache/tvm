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

""" Cost model that estimates the performance of programs """
import ctypes
import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .. import _ffi_api


@tvm._ffi.register_object("ansor.CostModel")
class CostModel(Object):
    """The base class for cost model"""
    pass


@tvm._ffi.register_object("ansor.RandomModel")
class RandomModel(Object):
    """A model returns random estimation for all inputs"""
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.RandomModel)


@tvm._ffi.register_func("ansor.cost_model.random_number")
def random_number(n, return_ptr):
    """ A random number generator func for c++'s RandomModel """
    if n == 0:
        return
    return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
    array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
    array_wrapper[:] = np.random.uniform(0, 1, (n,))


@tvm._ffi.register_object("ansor.PythonBasedModel")
class PythonBasedModel(CostModel):
    """Base class for cost models implemented in python"""
    def __init__(self):
        def update_func(inputs, results):
            self.update(inputs, results)

        def predict_func(task, states, return_ptr):
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(len(states),))
            array_wrapper[:] = self.predict(task, states)

        def predict_stage_func(task, states, return_ptr):
            ret = self.predict_stages(task, states)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=ret.shape)
            array_wrapper[:] = ret

        self.__init_handle_by_constructor__(_ffi_api.PythonBasedModel, update_func,
                                            predict_func, predict_stage_func)

    def update(self, inputs, results):
        raise NotImplementedError

    def predict(self, task, states):
        raise NotImplementedError

    def predict_stages(self, task, states):
        raise NotImplementedError
