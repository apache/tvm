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
"""Relay graph patterns for the my_ai_hw accelerator"""

from tvm.relay.dataflow_pattern import is_op, wildcard, has_attr


def conv2d_pattern():
    pattern = is_op("nn.conv2d")(wildcard(), wildcard())
    pattern = pattern.has_attr({"strides": [1, 1]})
    return pattern


def dense_pattern():
    pattern = is_op("nn.dense")(wildcard(), wildcard())
    pattern = pattern.optional(
        lambda x: is_op("nn.bias_add")(x, wildcard()) | is_op("add")(x, wildcard())
    )
    pattern = pattern.optional(lambda x: is_op("nn.relu")(x))
    return pattern
