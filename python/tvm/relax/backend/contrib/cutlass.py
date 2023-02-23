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

"""Pattern table for CUTLASS backend"""

from ..pattern_registry import register_patterns
from ..patterns import make_fused_bias_activation_pattern, make_matmul_pattern

register_patterns(
    [
        (
            "cutlass.conv2d",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d",
                with_bias=False,
                activation=None,
            ),
        ),
        (
            "cutlass.conv2d_bias_relu",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d",
                with_bias=True,
                activation="relax.nn.relu",
            ),
        ),
        (
            "cutlass.matmul",
            make_matmul_pattern(
                with_bias=False,
            ),
        ),
        (
            "cutlass.matmul_bias",
            make_matmul_pattern(
                with_bias=True,
            ),
        ),
        (
            "cutlass.matmul_bias_relu",
            make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
        ),
        (
            "cutlass.matmul_bias_gelu",
            make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
            ),
        ),
    ]
)
