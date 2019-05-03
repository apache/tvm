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
import numpy as np

import tvm
from tvm import relay

from tvm._ffi.function import _init_api
_init_api("tvm.relay.build_module")

class BuildModule(object):
    def __init__(self):
        self.mod = relay.build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._set_opt_level = self.mod["set_opt_level"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]

  
    def build(self, func, target, target_host, params):
        tgts = []
        for kv in target.items():
            tgts.append(kv[0])
            tgts.append(kv[1])
        self._set_params(params)
        self._build(func, tgts, target_host)

    def get_json(self):
        return self._get_graph_json()

    def get_module(self):
        return self._get_module()

    def set_opt_level(self, level):
        self._set_opt_level(level)

    def _set_params(self, params):
        inputs = {}
        for name, param in params.items():
            inputs[name] = relay.Constant(param)
        self._set_params_func(inputs)

    def get_params(self):
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret


def test_build():
    m_bld = BuildModule()
    tgt_name = "llvm"
    tgt = "llvm"
    ctx = tvm.cpu()
    # func
    a = relay.var("a", dtype="float32", shape=(16, 8))
    b = relay.var("b", dtype="float32", shape=(8, 8))
    c = relay.var("c", dtype="float32", shape=(16, 8))
    x = relay.nn.dense(a, b)
    y = relay.nn.relu(x)
    z = y + c
    func = relay.Function([a, b, c], z)
    A = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), ctx=ctx)
    B = tvm.nd.array(np.random.uniform(-1, 1, (8, 8)).astype("float32"), ctx=ctx)
    C = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), ctx=ctx)
    params = {
        "b" : B,
        "c" : C
    }
    # build
    targets = {
        tgt: tgt
    }
    m_bld.set_opt_level(3)
    m_bld.build(func, targets, "llvm -mcpu=sse3", params=params)
    g_json = m_bld.get_json()
    mmod = m_bld.get_module()
    params = m_bld.get_params()
   
    # test
    rt = tvm.contrib.graph_runtime.create(g_json, mmod, ctx)
    rt.set_input("a", A)
    rt.load_params(relay.save_param_dict(params))
    rt.run()
    out = rt.get_output(0)
   
    np.testing.assert_allclose(out.asnumpy(),
        np.maximum(np.dot(A.asnumpy(), B.asnumpy().T), 0) + C.asnumpy(), atol=1e-5, rtol=1e-5)
  
