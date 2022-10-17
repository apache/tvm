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
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import utils
import vta.testing
from vta.testing import simulator


def test_gemm():
    def run_gemm_packed(env, remote, batch_size, channel, block):
        data_shape = (batch_size // env.BATCH, channel // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
        weight_shape = (
            channel // env.BLOCK_OUT,
            channel // env.BLOCK_IN,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        res_shape = (batch_size // env.BATCH, channel // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT)
        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * channel * channel * batch_size

        ko = te.reduce_axis((0, channel // env.BLOCK_IN), name="ko")
        ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")

        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)
        data_buf = te.compute(data_shape, lambda *i: data(*i), "data_buf")
        weight_buf = te.compute(weight_shape, lambda *i: weight(*i), "weight_buf")
        res_gem = te.compute(
            res_shape,
            lambda bo, co, bi, ci: te.sum(
                data_buf[bo, ko, bi, ki].astype(env.acc_dtype)
                * weight_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki],
            ),
            name="res_gem",
        )
        res_shf = te.compute(res_shape, lambda *i: res_gem(*i) >> 8, name="res_shf")
        res_max = te.compute(res_shape, lambda *i: tvm.te.max(res_shf(*i), 0), "res_max")  # relu
        res_min = te.compute(
            res_shape, lambda *i: tvm.te.min(res_max(*i), (1 << (env.INP_WIDTH - 1)) - 1), "res_min"
        )  # relu
        res = te.compute(res_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

        def verify(s):
            mod = vta.build(
                s,
                [data, weight, res],
                tvm.target.Target("ext_dev", host=env.target_host),
                name="gemm",
            )
            temp = utils.tempdir()
            mod.save(temp.relpath("gemm.o"))
            remote.upload(temp.relpath("gemm.o"))
            f = remote.load_module("gemm.o")
            # verify
            dev = remote.ext_dev(0)
            # Data in original format
            data_orig = np.random.randint(-128, 128, size=(batch_size, channel)).astype(data.dtype)
            weight_orig = np.random.randint(-128, 128, size=(channel, channel)).astype(weight.dtype)
            data_packed = data_orig.reshape(
                batch_size // env.BATCH, env.BATCH, channel // env.BLOCK_IN, env.BLOCK_IN
            ).transpose((0, 2, 1, 3))
            weight_packed = weight_orig.reshape(
                channel // env.BLOCK_OUT, env.BLOCK_OUT, channel // env.BLOCK_IN, env.BLOCK_IN
            ).transpose((0, 2, 1, 3))
            res_np = np.zeros(res_shape).astype(res.dtype)
            data_arr = tvm.nd.array(data_packed, dev)
            weight_arr = tvm.nd.array(weight_packed, dev)
            res_arr = tvm.nd.array(res_np, dev)
            res_ref = np.zeros(res_shape).astype(env.acc_dtype)
            for b in range(batch_size // env.BATCH):
                for i in range(channel // env.BLOCK_OUT):
                    for j in range(channel // env.BLOCK_IN):
                        res_ref[b, i, :] += np.dot(
                            data_packed[b, j, :].astype(env.acc_dtype),
                            weight_packed[i, j].T.astype(env.acc_dtype),
                        )
            res_ref = np.right_shift(res_ref, 8)
            res_ref = np.clip(res_ref, 0, (1 << (env.INP_WIDTH - 1)) - 1).astype(res.dtype)
            time_f = f.time_evaluator("gemm", dev, number=20)
            if env.TARGET in ["sim", "tsim"]:
                simulator.clear_stats()
            cost = time_f(data_arr, weight_arr, res_arr)
            if env.TARGET in ["sim", "tsim"]:
                stats = simulator.stats()
                print("Execution statistics:")
                for k, v in stats.items():
                    print("\t{:<16}: {:>16}".format(k, v))
            res_unpack = res_arr.numpy().reshape(
                batch_size // env.BATCH, channel // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT
            )
            return cost

        def run_schedule(load_inp, load_wgt, gemm, alu, store_out, print_ir):
            s = te.create_schedule(res.op)
            s[data_buf].set_scope(env.inp_scope)
            s[weight_buf].set_scope(env.wgt_scope)
            s[res_gem].set_scope(env.acc_scope)
            s[res_shf].set_scope(env.acc_scope)
            s[res_min].set_scope(env.acc_scope)
            s[res_max].set_scope(env.acc_scope)

            if block:
                bblock = block // env.BATCH
                iblock = block // env.BLOCK_IN
                oblock = block // env.BLOCK_OUT
                xbo, xco, xbi, xci = s[res].op.axis
                xb1, xco1, xb2, xco2 = s[res].tile(xbo, xco, bblock, oblock)
                store_pt = xb2

                s[res_gem].compute_at(s[res], xco1)
                s[res_shf].compute_at(s[res], xco1)
                s[res_min].compute_at(s[res], xco1)
                s[res_max].compute_at(s[res], xco1)

                xbo, xco, xbi, xci = s[res_gem].op.axis
                # Compute one line at a time
                ko1, ko2 = s[res_gem].split(ko, iblock)
                s[res_gem].reorder(ko1, ko2, xbo, xco, xbi, xci, ki)
                s[data_buf].compute_at(s[res_gem], ko1)
                s[weight_buf].compute_at(s[res_gem], ko1)
                # Use VTA instructions
                s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
                s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
                s[res_gem].tensorize(xbi, gemm)
                s[res_shf].pragma(s[res_shf].op.axis[0], alu)
                s[res_min].pragma(s[res_min].op.axis[0], alu)
                s[res_max].pragma(s[res_max].op.axis[0], alu)
                s[res].pragma(store_pt, store_out)
            else:
                xbo, xco, xbi, xci = s[res_gem].op.axis
                s[res_gem].reorder(ko, xbo, xco, xbi, xci, ki)
                # Use VTA instructions
                s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
                s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
                s[res_gem].tensorize(xbi, gemm)
                s[res_shf].pragma(s[res_shf].op.axis[0], alu)
                s[res_min].pragma(s[res_min].op.axis[0], alu)
                s[res_max].pragma(s[res_max].op.axis[0], alu)
                s[res].pragma(s[res].op.axis[0], store_out)

            if print_ir:
                print(tvm.lower(s, [data, weight, res], simple_mode=True))
            return verify(s)

        def gemm_normal(print_ir):
            mock = env.mock
            print("----- GEMM GOPS End-to-End Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    env.dma_copy,
                    env.dma_copy,
                    env.gemm,
                    env.alu,
                    env.dma_copy,
                    print_ir,
                )
                gops = (num_ops / cost.mean) / float(10**9)
                print(header)
                print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

            with vta.build_config():
                run_test("NORMAL", print_ir)

        def gemm_unittest(print_ir):
            mock = env.mock
            print("----- GEMM Unit Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    mock.dma_copy, mock.dma_copy, env.gemm, mock.alu, mock.dma_copy, print_ir
                )
                gops = (num_ops / cost.mean) / float(10**9)
                print(header)
                print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

            with vta.build_config():
                run_test("NORMAL", print_ir)

        def alu_unittest(print_ir):
            mock = env.mock
            print("----- ALU Unit Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    mock.dma_copy, mock.dma_copy, mock.gemm, env.alu, mock.dma_copy, print_ir
                )
                gops = (num_ops / cost.mean) / float(10**9)
                print(header)
                print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

            with vta.build_config():
                run_test("NORMAL", print_ir)
            print("")

        def load_inp_unittest(print_ir):
            mock = env.mock
            print("----- LoadInp Unit Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    env.dma_copy, mock.dma_copy, mock.gemm, mock.alu, mock.dma_copy, print_ir
                )
                gops = (num_ops / cost.mean) / float(10**9)
                bandwith = (batch_size * channel * env.INP_WIDTH / cost.mean) / float(10**9)
                print(header)
                print(
                    "\tTime cost = %g sec/op, %g GOPS, bandwidth=%g Gbits"
                    % (cost.mean, gops, bandwith)
                )

            with vta.build_config():
                run_test("NORMAL", print_ir)
            print("")

        def load_wgt_unittest(print_ir):
            mock = env.mock
            print("----- LoadWgt Unit Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    mock.dma_copy, env.dma_copy, mock.gemm, mock.alu, mock.dma_copy, print_ir
                )
                gops = (num_ops / cost.mean) / float(10**9)
                bandwith = (channel * channel * env.WGT_WIDTH / cost.mean) / float(10**9)
                print(header)
                print(
                    "\tTime cost = %g sec/op, %g GOPS, bandwidth=%g Gbits"
                    % (cost.mean, gops, bandwith)
                )

            with vta.build_config():
                run_test("NORMAL", print_ir)
            print("")

        def store_out_unittest(print_ir):
            mock = env.mock
            print("----- StoreOut Unit Test-------")

            def run_test(header, print_ir):
                cost = run_schedule(
                    mock.dma_copy, mock.dma_copy, mock.gemm, mock.alu, env.dma_copy, print_ir
                )
                gops = (num_ops / cost.mean) / float(10**9)
                bandwith = (batch_size * channel * env.OUT_WIDTH / cost.mean) / float(10**9)
                print(header)
                print(
                    "\tTime cost = %g sec/op, %g GOPS, bandwidth=%g Gbits"
                    % (cost.mean, gops, bandwith)
                )

            with vta.build_config():
                run_test("NORMAL", print_ir)
            print("")

        gemm_normal(False)
        gemm_unittest(False)
        alu_unittest(False)

    def _run(env, remote):
        print("========GEMM 128=========")
        run_gemm_packed(env, remote, 128, 128, 128)

    vta.testing.run(_run)


if __name__ == "__main__":
    test_gemm()
