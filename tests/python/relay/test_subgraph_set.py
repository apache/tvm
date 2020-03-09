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
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end


def check_subgraph(subgraph_set, args, nodes, rets):
    subgraph = subgraph_set.get_subgraph(args[0])
    assert subgraph
    assert set(args) == set(subgraph.args)
    assert set(nodes) == set(subgraph.nodes)
    assert set(rets) == set(subgraph.rets)


def test_subgraph_set_creator_diamond():
    data = relay.var('data', shape=(10, 10))
    cb_1 = compiler_begin(data, 'test_target')
    O_1 = relay.abs(cb_1)
    ce_1 = compiler_end(O_1, 'test_target')
    ce_2 = compiler_end(O_1, 'test_target')
    cb_2 = compiler_begin(ce_1, 'test_target')
    O_2 = relay.nn.relu(cb_2)
    ce_3 = compiler_end(O_2, 'test_target')
    cb_d = compiler_begin(ce_2, "default")
    X = relay.tanh(cb_d)
    ce_d = compiler_end(X, 'default')
    cb_3 = compiler_begin(ce_3, 'test_target')
    cb_4 = compiler_begin(ce_d, 'test_target')
    O_3 = relay.add(cb_3, cb_4)
    ce_4 = compiler_end(O_3, 'test_target')
    diamond = relay.Function([data], ce_4)

    subgraph_set = relay.analysis.SubgraphSet(diamond,
                                              relay.op.get("annotation.compiler_begin"),
                                              relay.op.get("annotation.compiler_end"))
    assert len(subgraph_set) == 4
    check_subgraph(
        subgraph_set,
        [cb_1],
        [cb_1, O_1, ce_1, ce_2],
        [ce_1, ce_2],
    )
    check_subgraph(
        subgraph_set,
        [cb_2],
        [cb_2, O_2, ce_3],
        [ce_3],
    )
    check_subgraph(
        subgraph_set,
        [cb_d],
        [cb_d, X, ce_d],
        [ce_d],
    )
    check_subgraph(
        subgraph_set,
        [cb_3, cb_4],
        [cb_3, cb_4, O_3, ce_4],
        [ce_4],
    )


def test_subgraph_set_creator_merged():
    data = relay.var('data', shape=(10, 10))
    cb_1 = compiler_begin(data, 'test_target')
    O_1 = relay.abs(cb_1)
    ce_2 = compiler_end(O_1, 'test_target')
    O_2 = relay.nn.relu(O_1)
    ce_3 = compiler_end(O_2, 'test_target')
    cb_d = compiler_begin(ce_2, "default")
    X = relay.tanh(cb_d)
    ce_d = compiler_end(X, 'default')
    cb_3 = compiler_begin(ce_3, 'test_target')
    cb_4 = compiler_begin(ce_d, 'test_target')
    O_3 = relay.add(cb_3, cb_4)
    ce_4 = compiler_end(O_3, 'test_target')
    merged = relay.Function([data], ce_4)

    subgraph_set = relay.analysis.SubgraphSet(merged,
                                              relay.op.get("annotation.compiler_begin"),
                                              relay.op.get("annotation.compiler_end"))
    assert len(subgraph_set) == 3
    check_subgraph(
        subgraph_set,
        [cb_1],
        [cb_1, O_1, O_2, ce_2, ce_3],
        [ce_2, ce_3],
    )
    check_subgraph(
        subgraph_set,
        [cb_d],
        [cb_d, X, ce_d],
        [ce_d],
    )
    check_subgraph(
        subgraph_set,
        [cb_3, cb_4],
        [cb_3, cb_4, O_3, ce_4],
        [ce_4],
    )


if __name__ == "__main__":
    test_subgraph_set_creator_diamond()
    test_subgraph_set_creator_merged()

