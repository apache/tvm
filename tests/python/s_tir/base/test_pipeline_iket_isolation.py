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
import inspect

import pytest

from tvm.s_tir import pipeline
from tvm.s_tir.backend.adreno import pipeline as adreno_pipeline


@pytest.mark.parametrize(
    "factory", [pipeline.default_s_tir_pipeline, adreno_pipeline.default_tir_pipeline]
)
def test_pipeline_does_not_lower_iket(factory):
    assert "LowerIket" not in inspect.getsource(factory)
