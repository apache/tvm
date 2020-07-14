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
# pylint: disable=unused-argument
"""
A pass for simplifying the Relay expression.
"""
from . import transform
from ..dataflow_pattern import wildcard, is_op, DFPatternCallback, rewrite
from .. import op as _op

class SimplifyReshapeCallback(DFPatternCallback):
    """Callback to merge consecutive reshape ops"""
    def __init__(self):
        self.x = wildcard()
        reshape1 = is_op("reshape") | is_op("contrib_reverse_reshape")
        reshape2 = is_op("reshape") | is_op("contrib_reverse_reshape")
        self.pattern = reshape1(reshape2(self.x))

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        return _op.reshape(x, newshape=pre.checked_type.shape)


@transform.function_pass(opt_level=0, required=["InferType"])
class SimplifyExpr:
    """ A pass to simplify the Relay expression."""
    def __init__(self):
        self.callbacks = [SimplifyReshapeCallback()]

    def transform_function(self, func, mod, _):
        return rewrite(self.callbacks, func)
