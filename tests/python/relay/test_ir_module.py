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
"""Tests for module functionality."""
import tvm
from tvm import relay
from tvm.relay import Module
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions

def constructor_list(p):
    return [p.nil, p.cons, p.rose, p.some, p.none, p.z, p.s]


def test_constructor_tag_round_trip():
    mod1 = Module()
    p1 = Prelude(mod1)
    add_nat_definitions(p1)
    mod2 = Module()
    p2 = Prelude(mod2)
    add_nat_definitions(p2)

    # ensure hashes match across modules
    ctors1 = constructor_list(p1)
    ctors2 = constructor_list(p2)

    for i in range(len(ctors1)):
        tag = ctors1[i].tag
        ctor = mod2.get_constructor(tag)
        assert ctor == ctors2[i]
        assert ctor.name_hint == ctors1[i].name_hint
