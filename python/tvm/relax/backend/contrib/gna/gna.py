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
"""Pattern table for GNA backend"""

from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns


def _check_default(context: PatternCheckContext) -> bool:  # pylint: disable=unused-argument
    return True


def linear_patterns():
    """
    Returns a list of linear/dense patterns in GNA BYOC backend.
    """

    def _make_linear_pattern():
        input0 = wildcard()
        weight = wildcard()
        out = is_op("relax.matmul")(input0, weight)
        annotations = {"input": input0, "weight": weight, "root": out}
        return out, annotations

    def _linear_pattern(pattern_name):
        return (pattern_name, *_make_linear_pattern(), _check_default)

    return [_linear_pattern("gna.dense")]


def conv1d_patterns():
    """
    Returns a list of conv1d patterns in GNA BYOC backend.
    """

    def _make_conv1d_pattern():
        input0 = wildcard()
        weight = wildcard()
        out = is_op("relax.nn.conv1d")(input0, weight)
        annotations = {"input": input0, "weight": weight, "root": out}
        return out, annotations

    def _conv1d_pattern(pattern_name):
        return (pattern_name, *_make_conv1d_pattern(), _check_default)

    return [_conv1d_pattern("gna.conv1d")]


def activation_patterns():
    """
    Returns a list of activation patterns in GNA BYOC backend.
    """

    def _make_activation_pattern():
        input0 = wildcard()
        out = is_op("relax.nn.relu")(input0)
        annotations = {"input": input0, "root": out}
        return out, annotations

    def _activation_pattern(pattern_name):
        return (pattern_name, *_make_activation_pattern(), _check_default)

    return [_activation_pattern("gna.relu")]


register_patterns(
    [
        *linear_patterns(),
        *conv1d_patterns(),
        *activation_patterns(),
    ]
)
