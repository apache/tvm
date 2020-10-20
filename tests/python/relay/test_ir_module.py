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
from tvm import te
from tvm import relay
from tvm.relay.prelude import Prelude


def constructor_list(p):
    list_ctors = p.mod.get_type("List")
    optional_ctors = p.mod.get_type("Option")
    nat_ctors = p.mod.get_type("nat")
    rose_ctors = p.mod.get_type("Tree")
    return list_ctors[1:] + optional_ctors[1:] + nat_ctors[1:] + rose_ctors[1:]


def adt_list(p):
    list_ctors = p.mod.get_type("List")
    optional_ctors = p.mod.get_type("Option")
    nat_ctors = p.mod.get_type("nat")
    rose_ctors = p.mod.get_type("Tree")
    return list_ctors[:1] + optional_ctors[:1] + nat_ctors[:1] + rose_ctors[:1]


def test_constructor_tag_round_trip():
    mod1 = tvm.IRModule()
    p1 = Prelude(mod1)
    p1.mod.import_from_std("nat.rly")

    mod2 = tvm.IRModule()
    p2 = Prelude(mod2)
    p2.mod.import_from_std("nat.rly")

    # ensure hashes match across modules
    ctors1 = constructor_list(p1)
    ctors2 = constructor_list(p2)

    for i in range(len(ctors1)):
        tag = ctors1[i].tag
        ctor = mod2.get_constructor(tag)
        assert ctor == ctors2[i]
        assert ctor.name_hint == ctors1[i].name_hint


def test_constructor_tag_differences():
    # ensure that if we have the type data for a given ADT, the tags
    # for the constructors of the *same ADT* are simple offsets from
    # each other
    mod = tvm.IRModule()
    p = Prelude(mod)
    p.mod.import_from_std("nat.rly")

    adts = adt_list(p)
    for adt in adts:
        data = mod[adt]
        for i in range(len(data.constructors) - 1):
            ctor1 = data.constructors[i]
            ctor2 = data.constructors[i + 1]
            assert ctor2.tag - ctor1.tag == 1
            # make sure there is something present at the MSB
            assert ctor1.tag - i != 0
            assert ctor2.tag - (i + 1) != 0
