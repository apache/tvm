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

import mmap
import os
import tvm
import tvm.relay
import tvm.relay.op.contrib.android_nnapi


class RPCTestingTracker:
    def request(self, key):
        return RPCTestingSession()


class RPCTestingSession:
    def __init__(self):
        self._remote_fs = {}

    def cpu(self, *args, **kwargs):
        return tvm.cpu(*args, **kwargs)

    def load_module(self, remote_fpath):
        return RPCTestingModule(self._remote_fs[remote_fpath])

    def upload(self, local_fpath):
        self._remote_fs[os.path.basename(local_fpath)] = local_fpath


class RPCTestingModule:
    def __init__(self, module_fpath):
        self._module_fpath = module_fpath

    def time_evaluator(self, fname, *args, **kwargs):
        return RPCTestingFunction(self._module_fpath, fname)


class RPCTestingFunction:
    def __init__(self, module_fpath, fname):
        fname = fname.lower()
        fd = os.open(module_fpath, os.O_RDONLY)
        with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mcontent:
            assert mcontent.find(fname.encode()) != -1
            if mcontent.find(b"ANEURALNETWORKS") != -1:  # mod is built with android nnapi
                # this cost structure should put nn.conv2d on android nnapi and add on tvm
                if mcontent.find(b"CONV_2D") != -1:
                    self.mean = 10
                else:
                    self.mean = 1
            else:
                if mcontent.find(b"nn_conv2d") != -1:
                    self.mean = 100
                else:
                    self.mean = 10
        os.close(fd)

    def __call__(self, *args, **kwargs):
        return self


def test_rpc_partition():
    def _scope():
        data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
        data = tvm.relay.var("data", data_t)
        weight_t = tvm.relay.TensorType((1, 1, 2, 2), "float32")
        weight = tvm.relay.var("weight", weight_t)
        conv = tvm.relay.nn.conv2d(data=data, weight=weight)
        bias_t = tvm.relay.TensorType((1,), "float32")
        bias = tvm.relay.var("bias", bias_t)
        func_body = conv + bias
        func = tvm.relay.Function([data, weight, bias], func_body)
        mod = tvm.IRModule({"main": func})
        mod, _ = tvm.relay.op.contrib.android_nnapi.rpc_partition_for_android_nnapi(
            mod=mod, params={}, tracker=RPCTestingTracker(), options={}
        )
        return mod

    res = _scope()

    def _scope():
        data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
        data = tvm.relay.var("data", data_t)
        data_a = tvm.relay.annotation.compiler_begin(data, "android_nnapi")
        weight_t = tvm.relay.TensorType((1, 2, 2, 1), "float32")
        weight = tvm.relay.var("weight", weight_t)
        weight_a = tvm.relay.annotation.compiler_begin(weight, "android_nnapi")
        conv = tvm.relay.nn.conv2d(data=data_a, weight=weight_a, kernel_layout="OHWI")
        conv_a = tvm.relay.annotation.compiler_end(conv, "android_nnapi")
        bias_t = tvm.relay.TensorType((1,), "float32")
        bias = tvm.relay.var("bias", bias_t)
        func_body = conv_a + bias
        func = tvm.relay.Function([data, weight, bias], func_body)
        mod = tvm.IRModule({"main": func})
        mod = tvm.relay.transform.PartitionGraph()(mod)
        gvs = mod.get_global_vars()
        for gv in gvs:
            fn = mod[gv]
            if getattr(fn.attrs, "Compiler", None) == "android_nnapi":
                fn = fn.with_attr("NnapiTargetVersion", 29)
            mod[gv] = fn
        return mod

    ans = _scope()

    tvm.ir.assert_structural_equal(ans, res, map_free_vars=True)


if __name__ == "__main__":
    test_rpc_partition()
