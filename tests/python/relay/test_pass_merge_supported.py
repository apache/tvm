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
"""Unit tests for merge supported."""
import os
import sys
import numpy as np
import pytest

import tvm
from tvm import te
import tvm.relay.testing
import tvm.relay.transform as transform
from tvm import relay
from tvm import runtime
from tvm.contrib import util
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.testing import run_opt_pass


def test_diamond_graph():
    """
    This tests that the data dependencies present in a diamond-shaped
    graph are correctly resolved by the merging pass.

    O = supported by target
    X = not supported by target

       O         O
      / \       /               \
     O   X --> O    +       +    X
      \ /              \ /
       O                O

    Note that we can't just merge the three supported operators together,
    otherwise both subgraphs would depend on the other.
    """

    def diamond_graph_default_anno():
        data = relay.var('data', shape=(10, 10))
        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_1 = compiler_end(O_1, 'test_target')
        ce_2 = compiler_end(O_1, 'test_target')
        cb_2 = compiler_begin(ce_1, 'test_target')
        O_2 = relay.nn.relu(cb_2)
        ce_3 = compiler_end(O_2, 'test_target')

        cb_x = compiler_begin(ce_2, 'default')
        X = relay.tanh(cb_x)
        ce_x = compiler_end(X, 'default')

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(ce_x, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')
        diamond = relay.Function([data], ce_4)
        return diamond

    def diamond_graph():
        data = relay.var('data', shape=(10, 10))
        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_1 = compiler_end(O_1, 'test_target')
        ce_2 = compiler_end(O_1, 'test_target')
        cb_2 = compiler_begin(ce_1, 'test_target')
        O_2 = relay.nn.relu(cb_2)
        ce_3 = compiler_end(O_2, 'test_target')

        X = relay.tanh(ce_2)

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(X, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')
        diamond = relay.Function([data], ce_4)
        return diamond

    def expected():
        data = relay.var('data', shape=(10, 10))
        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_2 = compiler_end(O_1, 'test_target')
        O_2 = relay.nn.relu(O_1)
        ce_3 = compiler_end(O_2, 'test_target')

        cb_x = compiler_begin(ce_2, 'default')
        X = relay.tanh(cb_x)
        ce_x = compiler_end(X, 'default')

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(ce_x, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')
        func = relay.Function([data], ce_4)
        return func

    golden = expected()
    print(golden)
    result = run_opt_pass(diamond_graph(), relay.transform.MergeSupported())
    print(result)


def test_diamond_graph_fanouts():
    def diamond_graph_fanouts():
        data = relay.var('data', shape=(10, 10))
        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_1 = compiler_end(O_1, 'test_target')
        ce_2 = compiler_end(O_1, 'test_target')
        cb_2 = compiler_begin(ce_1, 'test_target')
        O_2 = relay.nn.relu(cb_2)
        ce_3 = compiler_end(O_2, 'test_target')

        X = relay.tanh(ce_2)

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(X, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')

        O_4 = relay.add(ce_4, X)

        diamond = relay.Function([data], O_4)
        return diamond

    def expected():
        data = relay.var('data', shape=(10, 10))
        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_2 = compiler_end(O_1, 'test_target')
        O_2 = relay.nn.relu(O_1)
        ce_3 = compiler_end(O_2, 'test_target')

        cb_x = compiler_begin(ce_2, 'default')
        X = relay.tanh(cb_x)
        ce_x1 = compiler_end(X, 'default')
        ce_x2 = compiler_end(X, 'default')

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(ce_x1, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')

        cb_o4_i1 = compiler_begin(ce_4, 'default')
        cb_o4_i2 = compiler_begin(ce_x2, 'default')
        O_4 = relay.add(cb_o4_i1, cb_o4_i2)
        ce_o4 = compiler_end(O_4, 'default')

        func = relay.Function([data], ce_o4)
        return func

    golden = expected()
    print("input :")
    print(diamond_graph_fanouts())
    print("golden :")
    print(golden)
    result = run_opt_pass(diamond_graph_fanouts(), relay.transform.MergeSupported())
    print("result :")
    print(result)



def test_split():
    data = relay.var('data', shape=(10, 10))
    out = relay.split(data, 2, axis=1)
    func = relay.Function([data], out)
    mod = tvm.IRModule()
    mod["main"] = func
    print(mod)


if __name__ == "__main__":
    #test_diamond_graph()
    # test_split()
    test_diamond_graph_fanouts()