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
"""Unit test for pattern table registry (BYOC)."""
from tvm.relay.op.contrib import get_pattern_table, register_pattern_table
from tvm import relay


@register_pattern_table("test_pattern_table")
def pattern_table():
    def _make_add_relu_pattern():
        x = relay.var('x')
        y = relay.var('y')
        add_node = relay.add(x, y)
        r = relay.nn.relu(add_node)
        return r

    def _check_add_relu_pattern():
        return True

    return [
        ("test_pattern_table.add_relu", _make_add_relu_pattern(), _check_add_relu_pattern)
    ]


def test_retrieve_pattern_table():
    table = get_pattern_table("test_pattern_table")
    assert table[0][0] == "test_pattern_table.add_relu"


if __name__ == "__main__":
    test_retrieve_pattern_table()
