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
"""Example end-to-end graph partitioning flow test."""
import tvm
import tvm.relay.testing
from tvm import relay
import tvm.relay.op as reg


def test_graph_partitioning_flow():
    def before():
        in_1 = relay.var('in_1', shape=(10, 10), dtype='float32')
        in_2 = relay.var('in_2', shape=(10, 10), dtype='float32')
        in_3 = relay.var('in_3', shape=(10, 10), dtype='float32')
        in_4 = relay.var('in_4', shape=(10, 10), dtype='float32')
        in_5 = relay.var('in_5', shape=(10, 10), dtype='float32')
        in_6 = relay.var('in_6', shape=(10, 10), dtype='float32')
        in_7 = relay.var('in_7', shape=(10, 10), dtype='float32')
        in_8 = relay.var('in_8', shape=(10, 10), dtype='float32')
        in_9 = relay.var('in_9', shape=(10, 10), dtype='float32')
        in_10 = relay.var('in_10', shape=(10, 10), dtype='float32')

        node0 = relay.add(in_1, in_2)
        node1 = relay.add(in_3, in_4)
        node2 = relay.add(node0, node1)

        node3 = relay.subtract(in_5, in_6)
        node4 = relay.subtract(in_7, node3)

        node5 = relay.add(node2, node4)
        node6 = relay.subtract(in_8, node5)
        node7 = relay.add(in_9, node5)

        node8 = relay.add(node6, node7)
        node9 = relay.add(in_10, node8)

        f = relay.Function([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10], node9)
        mod = tvm.IRModule.from_expr(f)
        return mod

    annotated = relay.transform.AnnotateTarget('test')(before())
    merged = relay.transform.MergeSupported()(annotated)
    partitioned = relay.transform.PartitionGraph()(merged)


@reg.register("add", "target.test")
def add(attrs, args):
    return True


if __name__ == "__main__":
    test_graph_partitioning_flow()
