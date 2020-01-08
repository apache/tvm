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
import tvm
import numpy as np
import re
import topi
from tvm.contrib import tedd


def findany(pattern, str):
    matches = re.findall(pattern, str)
    assert (len(matches) >
            0), 'Pattern not found.\nPattern: ' + pattern + '\nString:  ' + str


def test_dfg():
    A = tvm.placeholder((1024, 4096), dtype='float32', name='A')
    B = topi.nn.softmax(A)
    # confirm lower works
    s = tvm.create_schedule([B.op])

    def verify():
        str = tedd.viz_dataflow_graph(s, False, '', True)
        # Check all edges are available
        findany("digraph \"Dataflow Graph\"", str)
        findany("A0x[\da-f]+:O_0 -> T_softmax_maxelem0x[\da-f]+:I_0", str)
        findany("A0x[\da-f]+:O_0 -> T_softmax_exp0x[\da-f]+:I_0", str)
        findany(
            "T_softmax_maxelem0x[\da-f]+:O_0 -> T_softmax_exp0x[\da-f]+:I_1",
            str)
        findany(
            "T_softmax_exp0x[\da-f]+:O_0 -> T_softmax_expsum0x[\da-f]+:I_0",
            str)
        findany("T_softmax_exp0x[\da-f]+:O_0 -> T_softmax_norm0x[\da-f]+:I_0",
                str)
        findany(
            "T_softmax_expsum0x[\da-f]+:O_0 -> T_softmax_norm0x[\da-f]+:I_1",
            str)

    verify()


def test_itervar_relationship_graph():
    n = tvm.var("n")
    m = tvm.var("m")
    A = tvm.placeholder((n, m), name='A')
    k = tvm.reduce_axis((0, m), "k")
    B = tvm.compute((n, ), lambda i: tvm.sum(A[i, k], axis=k), name="B")

    s = tvm.create_schedule(B.op)
    s[B].split(B.op.reduce_axis[0], factor=16)

    def verify():
        str = tedd.viz_itervar_relationship_graph(s, False, '', True)
        findany("digraph \"IterVar Relationship Graph\"", str)
        findany("subgraph cluster_legend", str)
        # Check subgraphs for stages
        findany("subgraph cluster_A0x[\da-f]+", str)
        findany("subgraph cluster_B0x[\da-f]+", str)
        # Check itervars and their types
        findany("i0x[\da-f]+.+\>0\</TD\>.+\>i\(kDataPar\)", str)
        findany("k0x[\da-f]+.+\>-1\</TD\>.+\>k\(kCommReduce\)", str)
        # Check the split node
        findany("B_rel_00x[\da-f]+.+\>Split", str)
        # Check all edges to/from the split node
        findany("k0x[\da-f]+:itervar -> B_rel_00x[\da-f]+:Input", str)
        findany("B_rel_00x[\da-f]+:Outer -> k_outer0x[\da-f]+:itervar", str)
        findany("B_rel_00x[\da-f]+:Inner -> k_inner0x[\da-f]+:itervar", str)

    verify()


def test_schedule_tree():
    block_x = tvm.thread_axis('blockIdx.x')
    thread_x = tvm.thread_axis('threadIdx.x')
    n = tvm.var("n")
    m = tvm.var("m")
    l = tvm.var("l")
    A = tvm.placeholder((n, m, l), name='A')
    B = tvm.compute((n, m, l), lambda bi, bj, bk: A[bi, bj, bk] + 1, name='B')
    r = tvm.reduce_axis((0, m), "r")
    C = tvm.compute((n, m,),
                    lambda ci, cj: tvm.sum(B[ci, cj, r], axis=r),
                    name="C")
    s = tvm.create_schedule(C.op)
    BS = s.cache_read(A, 'shared', [B])
    s[B].vectorize(B.op.axis[-1])
    s[C].reorder(C.op.reduce_axis[0], C.op.axis[0])
    ko, ki = s[C].split(C.op.reduce_axis[0], factor=16)
    Cr = s.rfactor(C, ki)
    s[Cr].compute_at(s[C], s[C].op.axis[-1])
    s[C].bind(s[C].op.axis[0], block_x)
    s[C].bind(s[C].op.axis[1], thread_x)

    def verify():
        str = tedd.viz_schedule_tree(s, False, '', True)
        findany("digraph \"Schedule Tree\"", str)
        findany("subgraph cluster_legend", str)
        # Check the A_shared stage, including memory scope, itervars, 
        # and compute
        findany("A_shared0x[\da-f]+.*A\.shared<br/>Scope: shared.+>0.+>" \
            "ax0\(kDataPar\).+>1.+ax1\(kDataPar\).+>2.+>ax2\(kDataPar\).+>" \
            "\[A\(ax0, ax1, ax2\)\]", str)
        # Check itervars of types different from KDataPar
        findany("bk\(kVectorized\)", str)
        findany("r.outer\(kCommReduce\)", str)
        findany("label=ROOT", str)
        # Check the compute_at edge
        findany("C_rf0x[\da-f]+:stage -> C_repl0x[\da-f]+:ax10x[\da-f]+", str)

    verify()


if __name__ == "__main__":
    test_dfg()
    test_itervar_relationship_graph()
    test_schedule_tree()
