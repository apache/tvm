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
"""Meta Schedule CostModel."""
import ctypes
from typing import List

import numpy as np  # type: ignore
from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext
from ..utils import _get_hex_address, check_override


@register_object("meta_schedule.CostModel")
class CostModel(Object):
    """Cost model."""

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        _ffi_api.CostModelLoad(self, path)  # type: ignore # pylint: disable=no-member

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        _ffi_api.CostModelSave(self, path)  # type: ignore # pylint: disable=no-member

    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """
        _ffi_api.CostModelUpdate(self, context, candidates, results)  # type: ignore # pylint: disable=no-member

    def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted normalized score.
        """
        n = len(candidates)
        results = np.zeros(shape=(n,), dtype="float64")
        _ffi_api.CostModelPredict(  # type: ignore # pylint: disable=no-member
            self,
            context,
            candidates,
            results.ctypes.data_as(ctypes.c_void_p),
        )
        return results


@register_object("meta_schedule.PyCostModel")
class PyCostModel(CostModel):
    """An abstract CostModel with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, CostModel)
        def f_load(path: str) -> None:
            self.load(path)

        @check_override(self.__class__, CostModel)
        def f_save(path: str) -> None:
            self.save(path)

        @check_override(self.__class__, CostModel)
        def f_update(
            context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            self.update(context, candidates, results)

        @check_override(self.__class__, CostModel)
        def f_predict(context: TuneContext, candidates: List[MeasureCandidate], return_ptr) -> None:
            n = len(candidates)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_double))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
            array_wrapper[:] = self.predict(context, candidates)
            assert (
                array_wrapper.dtype == "float64"
            ), "ValueError: Invalid data type returned from CostModel Predict!"

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.CostModelPyCostModel,  # type: ignore # pylint: disable=no-member
            f_load,
            f_save,
            f_update,
            f_predict,
            f_as_string,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({_get_hex_address(self.handle)})"
