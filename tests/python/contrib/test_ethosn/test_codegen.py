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

"""NPU codegen tests"""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


@requires_ethosn
def test_compile_with_unsupported_variant():
    """Test compilation with unsupported variant."""
    dtype = "int8"
    input_shape = (1, 2, 2, 2)

    x = relay.var("x", shape=input_shape, dtype=dtype)
    y = relay.reshape(x, newshape=(1, 1, 1, 8))
    mod = tei.make_ethosn_partition(y)

    additional_config_args = {
        "variant": "foo",
        "inline_non_compute_intensive_partitions": False,
    }

    inputs = {
        "x": np.random.randint(
            low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=input_shape, dtype=dtype
        )
    }

    with pytest.raises(tvm.TVMError, match=r"Unknown NPU type"):
        tei.build_and_run(mod, inputs, 1, {}, True, additional_config_args=additional_config_args)
