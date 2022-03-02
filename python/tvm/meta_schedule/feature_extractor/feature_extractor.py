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
from typing import List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.runtime.ndarray import NDArray

from .. import _ffi_api
from ..utils import _get_hex_address, check_override
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
            The feature numpy ndarray extracted.
        """
        result = _ffi_api.FeatureExtractorExtractFrom(  # type: ignore # pylint: disable=no-member
            self, context, candidates
        )
        return result


@register_object("meta_schedule.PyFeatureExtractor")
class PyFeatureExtractor(FeatureExtractor):
    """An abstract feature extractor with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, FeatureExtractor)
        def f_extract_from(
            context: TuneContext, candidates: List[MeasureCandidate]
        ) -> List[NDArray]:
            features = self.extract_from(context, candidates)
            return features

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.FeatureExtractorPyFeatureExtractor,  # type: ignore # pylint: disable=no-member
            f_extract_from,
            f_as_string,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({_get_hex_address(self.handle)})"
