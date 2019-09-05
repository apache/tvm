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
import pytest

import tvm
from tvm import relay
from tvm.relay import Function, transform
from tvm.relay.analysis import pointer_analysis
from tvm.relay.op import log, add, equal, subtract


def test_ref_create():
    expr = relay.RefCreate(relay.const(5))
    result = pointer_analysis(expr)
    assert expr in result.spawn


def test_ref_write():
    ref = relay.RefCreate(relay.const(5))
    new_val = relay.const(4)
    write = relay.RefWrite(ref, new_val)
    result = pointer_analysis(write)
    store = set()
    for loc in result.contain[ref]:
        for expr in result.store[loc]:
            store.add(expr)
    assert new_val in store

if __name__ == "__main__":
    pytest.main([__file__])
