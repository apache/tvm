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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition

import logging

logging.basicConfig(level=logging.ERROR)

import tvm
from tvm import relay
from tvm import transform
from tvm.relay import testing
from tvm.testing.aot import AOTTestModel, compile_and_run
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER


def benchmark_dense_int8_acc32(tgt, opt):

    m = 1024
    n = 1024
    k = 1024

    # Gops for gemm
    gops_per_mm = 2.0 * (n * m * k) / 1e9

    def verify(tgt, opt):

        target = tvm.target.Target("llvm -mcpu=ivybridge -keys=cpu,fast-math")

        ##
        ## GRAPH
        ##

        # network graph
        dat = relay.var("data", shape=(m, k), dtype="uint8")
        weight = relay.var("weight", shape=(n, k), dtype="int8")
        out = relay.nn.dense(dat, weight, out_dtype="int32")

        # convert to relay IR
        f = relay.Function(relay.analysis.free_vars(out), out)
        mod, params = testing.create_workload(f)

        ##
        ## EVAL
        ##

        with relay.build_config(opt_level=3):
            with tvm.transform.PassContext(opt_level=opt):
                # build relay module
                lib = relay.build(mod, target=target, params=None)

        tensorized = False
        if "@llvm.x86." in lib.lib.get_source():
            tensorized = True

        import numpy as np

        np.random.seed(seed=None)
        d_dtype = dat.type_annotation
        w_dtype = weight.type_annotation
        from tvm.topi.utils import get_const_tuple

        X = np.random.randint(
            low=0, high=127, size=get_const_tuple(d_dtype.shape), dtype=d_dtype.dtype
        )
        W = np.random.randint(
            low=-63, high=63, size=get_const_tuple(w_dtype.shape), dtype=w_dtype.dtype
        )

        # build runtime module
        dev = tvm.device(str(target), 0)
        import tvm.contrib.graph_executor as runtime

        module = runtime.GraphModule(lib["default"](dev))
        module.set_input("data", tvm.nd.array(X))
        params = {"weight": tvm.nd.array(W)}
        module.set_input(**params)

        # evaluate performance
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=10)
        result = np.array(ftimer().results)
        gops_per_sec = gops_per_mm / np.mean(result)
        print(
            "Task tensorized: {%-5s} [%-45s], running time: %.3f ms, %.2f Gops/s"
            % (tensorized, tgt, np.mean(result) * 1000, gops_per_sec)
        )

        # evaluate results
        module.run()
        module.get_output(0).asnumpy()
        O = module.get_output(0).asnumpy()
        tvm.testing.assert_allclose(O, np.dot(X.astype("int32"), W.T.astype("int32")), rtol=0)

        return

    verify(tgt, opt)


@tvm.testing.requires_x86_vnni
def test_fc_int8_acc32_x86_vnni():
    benchmark_dense_int8_acc32("llvm -mcpu=cascadelake", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=cascadelake", opt=2)


@tvm.testing.requires_x86_avx512
def test_fc_int8_acc32_x86_avx512():
    benchmark_dense_int8_acc32("llvm -mcpu=skylake-avx512", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=skylake-avx512 -keys=cpu,fast-math", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=skylake-avx512", opt=2)


@tvm.testing.requires_x86
def test_fc_int8_acc32_x86_simd():
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge -keys=cpu,fast-math", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell -keys=cpu,fast-math", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge", opt=2)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell", opt=2)


if __name__ == "__main__":
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge -keys=cpu,fast-math", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell -keys=cpu,fast-math", opt=3)
    benchmark_dense_int8_acc32("llvm -mcpu=ivybridge", opt=2)
    benchmark_dense_int8_acc32("llvm -mcpu=haswell", opt=2)
