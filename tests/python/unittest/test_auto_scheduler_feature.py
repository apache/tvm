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

"""Test feature extraction"""

import math
import tempfile

import tvm
from tvm import te, auto_scheduler, relay
from tvm.script import tir as T

from tvm.testing.auto_scheduler import matmul_auto_scheduler_test


def fequal(a, b):
    return math.fabs(a - b) < 1e-6


def test_cpu_matmul():
    dag = auto_scheduler.ComputeDAG(matmul_auto_scheduler_test(512, 512, 512))
    s = dag.get_init_state()
    C = s.stage_ops[2]

    i, j, k = s[C].iters
    io, ii = s.split(C, i, [16])
    jo, ji = s.split(C, j, [8])
    s.reorder(C, [io, jo, k, ji, ii])
    s.vectorize(C, ji)
    s.parallel(C, io)
    s.parallel(C, jo)
    s.unroll(C, k)

    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(compute_dag=dag, workload_key="test", target=target)
    names = auto_scheduler.feature.get_per_store_feature_names()
    fea = auto_scheduler.feature.get_per_store_features_from_states([s], task)[0]

    stage_0 = fea[0]
    assert len(stage_0) == len(names), "%d vs %d" % (len(stage_0), len(names))
    fea_dict = {}
    for name, value in zip(names, stage_0):
        fea_dict[name] = value

    for name in ["B0", "B1", "B2"]:
        if fequal(fea_dict[name + ".acc_type.kReadWrite"], 1.0):
            c_name = name
        if fequal(fea_dict[name + ".acc_type.kRead"], 1.0):
            if fequal(fea_dict[name + ".stride"], 0.0):
                b_name = name
            else:
                a_name = name

    """
    lowered IR:

    Placeholder: A, B
    parallel i.0 (0,32)
      parallel j.0 (0,64)
        unroll k (0,512)
          vectorize j.1 (0,8)
            for i.1 (0,16)
              C...] = A[...] * B[...]
    """

    # check touched memory in bytes, touched unique memory in bytes, reuse distance, etc.
    assert fequal(fea_dict[c_name + ".bytes"], math.log2(512**3 * 4 + 1))
    assert fequal(fea_dict[b_name + ".unique_bytes"], math.log2(512**2 * 4 + 1))
    assert fequal(fea_dict[c_name + ".reuse_dis_iter"], math.log2(8 * 16 + 1))
    assert fequal(fea_dict[c_name + ".reuse_dis_bytes"], math.log2((8 * 16 + 8 + 16) * 4 + 1))
    assert fequal(fea_dict[c_name + ".reuse_ct"], math.log2(512 + 1))

    # check annotations
    assert fequal(fea_dict["unroll_num"], math.log2(1 + 1))
    # assert fequal(fea_dict["unroll_type.kPosInnerReduce"], 1.0)
    assert fequal(fea_dict["vec_num"], math.log2(1 + 1))
    assert fequal(fea_dict["parallel_num"], math.log2(2 + 1))
    assert fequal(fea_dict["parallel_prod"], math.log2((512 * 512 / 16 / 8) + 1))


def test_cpu_fusion():
    def fusion_test(N, M):
        A = te.placeholder((N, M), name="A")
        B = te.compute((N, M), lambda i, j: A[i][j], name="B")
        C = te.compute((N, M), lambda i, j: B[i][j], name="C")
        return [A, B, C]

    dag = auto_scheduler.ComputeDAG(fusion_test(64, 32))
    s = dag.get_init_state()
    s.compute_at(1, 2, s.stages[2].iters[1])

    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(compute_dag=dag, workload_key="test", target=target)
    names = auto_scheduler.feature.get_per_store_feature_names()
    fea = auto_scheduler.feature.get_per_store_features_from_states([s], task)[0]

    """
    lowered IR:

    Placeholder: A
    for i (0,64)
        for j (0,32)
            for ii (1)
                for jj (1)
                    B[...] = A[...]
            C[...] = B[...]
    """

    # check reuse distance and reuse type after fusion
    found = False
    for stage_fea in fea:
        for i, (name, value) in enumerate(zip(names, stage_fea)):
            if "reuse_type.kSerialMultipleReadWrite" in name and value > 0.5:
                # reuse distance in #iter
                assert fequal(stage_fea[i + 2], 1.0)
                # reuse distance in bytes
                assert fequal(stage_fea[i + 3], math.log2(16 + 1))
                found = True
    assert found


def test_gpu_feature():
    # Use records to build a complicated GPU program
    json_records = "\n".join(
        (
            """{"i": [["[\\"matmul_auto_scheduler_test\\", 512, 512, 512]", "cuda"], [[], [["CHW", 2, "local"], ["SP", 2, 0, 512, [1, 16, 32, 1], 1], ["SP", 2, 5, 512, [4, 1, 1, 16], 1], ["SP", 2, 10, 512, [1, 2], 1], ["RE", 2, [0, 5, 1, 6, 2, 7, 10, 11, 3, 8, 12, 4, 9]], ["FSP", 3, 0, 1, 3], ["FSP", 3, 4, 2, 3], ["RE", 3, [0, 4, 1, 5, 2, 6, 3, 7]], ["FU", 2, [0, 1]], ["FU", 3, [0, 1]], ["FU", 2, [1, 2]], ["FU", 3, [1, 2]], ["FU", 2, [2, 3]], ["FU", 3, [2, 3]], ["CA", 2, 3, 2], ["CHR", 1, "shared", [2]], ["CA", 2, 3, 3], ["FU", 2, [0, 1]], ["FFSP", 2, 0, [1, 2], 1, 1], ["AN", 2, 1, 6], ["CHR", 0, "shared", [3]], ["CA", 1, 4, 3], ["FU", 1, [0, 1]], ["FFSP", 1, 0, [1, 2], 1, 1], ["AN", 1, 1, 6], ["AN", 5, 0, 5], ["AN", 5, 1, 4], ["AN", 5, 2, 6], ["PR", 4, 0, "auto_unroll_max_step$1024"]]]], "r": [[0.00536798], 0, 2.49277, 1585564852], "v": "v0.1"}""",
        )
    )

    # load states
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(json_records)
        f.flush()
        inputs, _ = auto_scheduler.RecordReader(f.name).read_lines()

        inp = inputs[0]
        task = auto_scheduler.SearchTask(
            workload_key=inp.task.workload_key,
            target=inp.task.target,
            hardware_params=auto_scheduler.HardwareParams(
                100000, 16, 64, 1 << 30, 1 << 30, 1 << 30, 1 << 30, 1 << 30
            ),
        )

        state = task.compute_dag.infer_bound_from_state(inputs[0].state)
        fea = auto_scheduler.feature.get_per_store_features_from_states([state], task)[0]
        names = auto_scheduler.feature.get_per_store_feature_names()

        # build feature dict
        fea_dicts = []
        for i in range(len(fea)):
            tmp_dict = {}
            for j in range(len(names)):
                tmp_dict[names[j]] = fea[i][j]
            fea_dicts.append(tmp_dict)

        """
        lowered IR:

        Placeholder: A, B
        blockIdx.x i.0@j.0@ (0,8)
          vthread i.1@j.1@ (0,4)
            threadIdx.x i.2@j.2@ (0,16)
              C.local auto_unroll: 1024
              for k.0 (0,256)
                for ax0@ax1@.0 (0,8)
                  threadIdx.x ax0@ax1@.1 (0,16)
                    B.shared = ...
                for ax0@ax1@.0 (0,64)
                  threadIdx.x ax0@ax1@.1 (0,16)
                    A.shared = ...
                for i_c.3 (0,32)
                  for k.2 (0,2)
                    for j_c.4 (0,16)
                      C.local = ...
              for i.3 (0,32)
                for j.3 (0,16)
                  C = ...
        """

        # check gpu-related features
        assert fequal(fea_dicts[0]["blockIdx_x_len"], math.log2(8 + 1))
        assert fequal(fea_dicts[0]["vthread_len"], math.log2(4 + 1))
        assert fequal(fea_dicts[1]["threadIdx_x_len"], math.log2(16 + 1))
        assert fequal(fea_dicts[0]["threadIdx_y_len"], math.log2(1 + 1))
        assert fequal(fea_dicts[2]["blockIdx_z_len"], math.log2(1 + 1))
        assert fequal(fea_dicts[0]["is_gpu"], 1.0)


@T.prim_func
def tir_matmul(
    A: T.Buffer((256, 256), "float32"),
    B: T.Buffer((256, 256), "float32"),
    C: T.Buffer((256, 256), "float32"),
) -> None:
    # function attr dict
    T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
    A_flat = T.Buffer([16384], dtype="float32", data=A.data)
    B_flat = T.Buffer([16384], dtype="float32", data=B.data)
    C_flat = T.Buffer([16384], dtype="float32", data=C.data)
    # body
    for x, y in T.grid(128, 128):
        C_flat[x * 128 + y] = T.float32(0)
        for k in T.serial(128):
            C_flat[x * 128 + y] = C_flat[x * 128 + y] + A_flat[x * 128 + k] * B_flat[y * 128 + k]


def test_primfunc_without_lowering():
    features = auto_scheduler.feature.named_features_from_primfunc(tir_matmul)
    assert features["float_mad"].shape == (1,)
    # featurization does not handle multiple-add right now, so they are split out
    assert abs(features["float_addsub"][0] - 128 * 128 * 128) < 10
    assert abs(features["float_mul"][0] - 128 * 128 * 128) < 10
    for i in range(0, 3):
        assert abs(features[f"B{i}.unique_bytes"][0] - 128 * 128 * 4) < 10  # 4 bytes per float32


def test_primfunc_lowered():
    # Lower tir function so all passes get applied
    f = tvm.lower(tir_matmul)
    features = auto_scheduler.feature.named_features_from_primfunc(f["main"])
    assert features["float_mad"].shape == (1,)
    # featurization does not handle multiple-add right now, so they are split out
    assert abs(features["float_addsub"][0] - 128 * 128 * 128) < 10
    assert abs(features["float_mul"][0] - 128 * 128 * 128) < 10
    for i in range(0, 3):
        assert abs(features[f"B{i}.unique_bytes"][0] - 128 * 128 * 4) < 10  # 4 bytes per float32


def test_dense_lowered():
    a = relay.var("a", relay.TensorType((128, 128), "float32"))
    b = relay.var("b", relay.TensorType((128, 128), "float32"))
    c = relay.nn.dense(a, b)
    mod = tvm.IRModule.from_expr(relay.Function([a, b], c))
    target = "llvm"
    comp = relay.vm.VMCompiler()
    mod, params = comp.optimize(mod, params={}, target=target)
    for name, func in mod.functions.items():
        if name.name_hint != "main":
            break
    features = auto_scheduler.feature.named_features_from_primfunc(func)
    # featurization does not handle multiple-add right now, so they are split out
    assert features["float_addsub"].sum() >= 128 * 128 * 128
    assert features["float_mul"].sum() >= 128 * 128 * 128
    total_bytes_loaded = 0
    for i in range(0, 4):
        total_bytes_loaded += features[f"B{i}.unique_bytes"].sum()
    assert total_bytes_loaded > 2 * 128 * 128 * 4  # 4 bytes per float32


@T.prim_func
def negative_extent(A: T.Buffer((1,), "float32")):
    for j in range(0, -1):
        A[j] = A[j] + 1.0


def test_negative_extent():
    features = auto_scheduler.feature.named_features_from_primfunc(negative_extent)
    assert features["B0.unique_bytes"] == 0


@T.prim_func
def zero_dim(
    p2: T.Buffer((), "float32"),
    T_cast: T.Buffer((T.int64(1), T.int64(768)), "int8"),
):
    # function attr dict
    T.func_attr(
        {
            "tir.noalias": True,
            "Primitive": 1,
        }
    )
    # buffer definition
    T_cast_1 = T.buffer_decl([T.int64(768)], dtype="int8", data=T_cast.data)
    p2_1 = T.buffer_decl([1], dtype="float32", data=p2.data)
    # body
    for i0_i1_fused in T.serial(768):
        T_cast_1[i0_i1_fused] = p2_1[0]


def test_zero_dim():
    features = auto_scheduler.feature.named_features_from_primfunc(zero_dim)
    assert features["B1.stride"] == 1
    assert features["B0.stride"] == 1


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_fusion()
    test_gpu_feature()
