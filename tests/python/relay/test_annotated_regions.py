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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end


def check_region(region_set, target, args, nodes, rets):
    region = region_set.get_region(args[0])
    assert region
    assert target == region.target
    assert set(args) == set(region.args)
    assert set(nodes) == set(region.nodes)
    assert set(rets) == set(region.rets)


def test_region_set_creator_diamond():
    data = relay.var("data", shape=(10, 10))
    cb_1 = compiler_begin(data, "test_target")
    O_1 = relay.abs(cb_1)
    ce_1 = compiler_end(O_1, "test_target")
    ce_2 = compiler_end(O_1, "test_target")
    cb_2 = compiler_begin(ce_1, "test_target")
    O_2 = relay.nn.relu(cb_2)
    ce_3 = compiler_end(O_2, "test_target")
    cb_d = compiler_begin(ce_2, "default")
    X = relay.tanh(cb_d)
    ce_d = compiler_end(X, "default")
    cb_3 = compiler_begin(ce_3, "test_target")
    cb_4 = compiler_begin(ce_d, "test_target")
    O_3 = relay.add(cb_3, cb_4)
    ce_4 = compiler_end(O_3, "test_target")
    diamond = relay.Function([data], ce_4)

    region_set = relay.analysis.AnnotatedRegionSet(
        diamond, relay.op.get("annotation.compiler_begin"), relay.op.get("annotation.compiler_end")
    )
    assert len(region_set) == 4
    check_region(
        region_set,
        "test_target",
        [cb_1],
        [cb_1, O_1, ce_1, ce_2],
        [ce_1, ce_2],
    )
    check_region(
        region_set,
        "test_target",
        [cb_2],
        [cb_2, O_2, ce_3],
        [ce_3],
    )
    check_region(
        region_set,
        "default",
        [cb_d],
        [cb_d, X, ce_d],
        [ce_d],
    )
    check_region(
        region_set,
        "test_target",
        [cb_3, cb_4],
        [cb_3, cb_4, O_3, ce_4],
        [ce_4],
    )


def test_region_set_creator_merged():
    data = relay.var("data", shape=(10, 10))
    cb_1 = compiler_begin(data, "test_target")
    O_1 = relay.abs(cb_1)
    ce_2 = compiler_end(O_1, "test_target")
    O_2 = relay.nn.relu(O_1)
    ce_3 = compiler_end(O_2, "test_target")
    cb_d = compiler_begin(ce_2, "default")
    X = relay.tanh(cb_d)
    ce_d = compiler_end(X, "default")
    cb_3 = compiler_begin(ce_3, "test_target")
    cb_4 = compiler_begin(ce_d, "test_target")
    O_3 = relay.add(cb_3, cb_4)
    O_4 = relay.add(cb_3, cb_4)
    O_5 = relay.Tuple([O_3, O_4])
    ce_4 = compiler_end(O_5, "test_target")
    merged = relay.Function([data], ce_4)

    region_set = relay.analysis.AnnotatedRegionSet(
        merged, relay.op.get("annotation.compiler_begin"), relay.op.get("annotation.compiler_end")
    )
    assert len(region_set) == 3
    check_region(
        region_set,
        "test_target",
        [cb_1],
        [cb_1, O_1, O_2, ce_2, ce_3],
        [ce_2, ce_3],
    )
    check_region(
        region_set,
        "default",
        [cb_d],
        [cb_d, X, ce_d],
        [ce_d],
    )
    check_region(
        region_set,
        "test_target",
        [cb_3, cb_4],
        [cb_3, cb_4, O_3, O_4, O_5, ce_4],
        [ce_4],
    )


if __name__ == "__main__":
    test_region_set_creator_diamond()
    test_region_set_creator_merged()
