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
"""Meta Schedule FeatureExtractor."""
from typing import Callable, List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.runtime.ndarray import NDArray

from .. import _ffi_api
from ..utils import _get_default_str
from ..tune_context import TuneContext
from ..search_strategy import MeasureCandidate


@register_object("meta_schedule.FeatureExtractor")
class FeatureExtractor(Object):
    """Extractor for features from measure candidates for use in cost model."""

    def extract_from(
        self, context: TuneContext, candidates: List[MeasureCandidate]
    ) -> List[NDArray]:
        """Extract features from the given measure candidate.

        Parameters
        ----------
        context : TuneContext
            The tuning context for feature extraction.
        candidates : List[MeasureCandidate]
            The measure candidates to extract features from.

        Returns
        -------
        features : List[NDArray]
            The feature tvm ndarray extracted.
        """
        result = _ffi_api.FeatureExtractorExtractFrom(  # type: ignore # pylint: disable=no-member
            self, context, candidates
        )
        return result


@register_object("meta_schedule.PyFeatureExtractor")
class _PyFeatureExtractor(FeatureExtractor):
    """
    A TVM object feature extractor to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyFeatureExtractor
    """

    def __init__(self, f_extract_from: Callable, f_as_string: Callable = None):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.FeatureExtractorPyFeatureExtractor,  # type: ignore # pylint: disable=no-member
            f_extract_from,
            f_as_string,
        )


class PyFeatureExtractor:
    """
    An abstract feature extractor with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyFeatureExtractor,
        "methods": ["extract_from", "__str__"],
    }

    def extract_from(
        self, context: TuneContext, candidates: List[MeasureCandidate]
    ) -> List[NDArray]:
        """Extract features from the given measure candidate.

        Parameters
        ----------
        context : TuneContext
            The tuning context for feature extraction.
        candidates : List[MeasureCandidate]
            The measure candidates to extract features from.

        Returns
        -------
        features : List[NDArray]
            The feature tvm ndarray extracted.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return _get_default_str(self)
