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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import re
from typing import List

import numpy as np
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.feature_extractor import PyFeatureExtractor
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.utils import derived_object
from tvm.runtime.ndarray import array


def test_meta_schedule_feature_extractor():
    @derived_object
    class FancyFeatureExtractor(PyFeatureExtractor):
        def extract_from(
            self,
            context: TuneContext,  # pylint: disable = unused-argument
            candidates: List[MeasureCandidate],  # pylint: disable = unused-argument
        ) -> List[np.ndarray]:
            return [array(np.random.rand(4, 5))]

    extractor = FancyFeatureExtractor()
    features = extractor.extract_from(TuneContext(), [])
    assert len(features) == 1
    assert features[0].shape == (4, 5)


def test_meta_schedule_feature_extractor_as_string():
    @derived_object
    class NotSoFancyFeatureExtractor(PyFeatureExtractor):
        def extract_from(
            self,
            context: TuneContext,  # pylint: disable = unused-argument
            candidates: List[MeasureCandidate],  # pylint: disable = unused-argument
        ) -> List[np.ndarray]:
            return []

    feature_extractor = NotSoFancyFeatureExtractor()
    pattern = re.compile(r"meta_schedule.NotSoFancyFeatureExtractor\(0x[a-f|0-9]*\)")
    assert pattern.match(str(feature_extractor))


if __name__ == "__main__":
    test_meta_schedule_feature_extractor()
    test_meta_schedule_feature_extractor_as_string()
