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

   def build(self, func, target, target_host):
      tgts = []
      for kv in target.items():
         tgts.append(kv[0])
         tgts.append(kv[1])
      self._build(func, tgts, target_host)

   def get_json(self):
      return self._get_graph_json()

   def get_module(self):
      return self._get_module()

def test_build():
   m_bld = BuildModule()
   # func
   a = relay.var("a", dtype="float32", shape=(16, 8))
   b = relay.var("b", dtype="float32", shape=(8, 8))
   c = relay.var("c", dtype="float32", shape=(16, 8))
   x = relay.nn.dense(a, b)
   y = relay.nn.relu(x)
   z = y + c
   func = relay.Function([a, b, c], z)
   # build
   targets = {
       "cpu": "llvm -mcpu=sse3"
   }
   m_bld.build(func, targets, "llvm -mcpu=sse3")
   g_json = m_bld.get_json()
   mmod = m_bld.get_module()
   
   
   # test
   A = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"))
   B = tvm.nd.array(np.random.uniform(-1, 1, (8, 8)).astype("float32"))
   C = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"))

   rt = tvm.contrib.graph_runtime.create(g_json, mmod, tvm.cpu())
   rt.set_input("a", A)
   rt.set_input("b", B)
   rt.set_input("c", C)
   rt.run()
   out = rt.get_output(0)
   
   np.testing.assert_allclose(out.asnumpy(),
        np.maximum(np.dot(A.asnumpy(), B.asnumpy().T), 0) + C.asnumpy(), atol=1e-5, rtol=1e-5)