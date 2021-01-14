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

""" Cost models that estimate the performance of programs """
import ctypes
import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .. import _ffi_api


@tvm._ffi.register_object("auto_scheduler.CostModel")
class CostModel(Object):
    """The base class for cost model"""


@tvm._ffi.register_object("auto_scheduler.RandomModel")
class RandomModel(CostModel):
    """A model that returns random estimation for all inputs"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.RandomModel)

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        _ffi_api.CostModelUpdate(self, inputs, results)

    def predict(self, search_task, states):
        """Predict the scores of states

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all states
        """
        return [x.value for x in _ffi_api.CostModelPredict(self, search_task, states)]


@tvm._ffi.register_func("auto_scheduler.cost_model.random_fill_float")
def random_fill_float(size, return_ptr):
    """Fills a c++ float array with random numbers in [0, 1]

    Parameters
    ----------
    size: int
        The size of the array
    return_ptr:
        A pointer to a c++ float array
    """
    if size == 0:
        return
    return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
    array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(size,))
    array_wrapper[:] = np.random.uniform(0, 1, (size,))


@tvm._ffi.register_object("auto_scheduler.PythonBasedModel")
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

        self.__init_handle_by_constructor__(
            _ffi_api.PythonBasedModel, update_func, predict_func, predict_stage_func
        )

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        raise NotImplementedError

    def predict(self, task, states):
        """Predict the scores of states

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all states
        """
        raise NotImplementedError

    def predict_stages(self, task, states):
        """Predict the scores of all stages in states. This is the breakdown version of `predict`.

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        states : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all stages in all states in the packed format

        Note
        ----
        For faster data copy between c++ and python, the python part returns scores in a
        single flatten array using a packed format. The c++ part then unpacks the flatten array.

        The packed format is:
        {
          float  scores[N];                 // scores[i] is the score for states[i].
          int    n_stage_0;                 // the number of stages in states[0]
          float  stage_scores_0[[n_stage_0] // the scores for all stages in states[0]
          int    n_stage_1;                 // the number of stages in states[1]
          float  stage_scores_1[n_stage_1]; // the scores for all stages in states[1]
          ...
          int    n_stage_i;                 // the number of stages in states[i]
          float  stage_scores_1[n_stage_i]; // the scores for all stages in states[i]
          ...  // until i == N - 1
        }
        To implement this format, we also store int as float, so we can store all numbers
        into a single float array.
        """
        raise NotImplementedError
