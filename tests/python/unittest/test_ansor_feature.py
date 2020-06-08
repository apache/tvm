"""Test feature extraction"""

import math
import tempfile

import tvm
from tvm import te, ansor

from test_ansor_common import matmul_nkkm


def fequal(a, b):
    return math.fabs(a - b) < 1e-6


def test_cpu_matmul():
    dag = ansor.ComputeDAG(matmul_nkkm(512, 512, 512))
    s = dag.get_init_state()
    C = 2

    i, j, k = s.stages[C].iters
    io, ii = s.split(C, i, [16])
    jo, ji = s.split(C, j, [8])
    s.reorder(C, [io, jo, k, ji, ii])
    s.vectorize(C, ji)
    s.parallel(C, io)
    s.parallel(C, jo)
    s.unroll(2, k)

    target = tvm.target.create('llvm')
    task = ansor.SearchTask(dag, "test", target)
    names = ansor.feature.get_per_stmt_feature_names()
    fea = ansor.feature.get_per_stmt_features_from_states([s.state_object], task)[0]

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

    assert fequal(fea_dict[c_name + ".bytes"], math.log2(512 ** 3 * 4 + 1))
    assert fequal(fea_dict[b_name + ".unique_bytes"], math.log2(512 ** 2 * 4 + 1))
    assert fequal(fea_dict[c_name + ".reuse_dis_iter"], math.log2(8 * 16 + 1))
    assert fequal(fea_dict[c_name + ".reuse_dis_bytes"], math.log2((8 * 16 + 8 + 16) * 4 + 1))
    assert fequal(fea_dict[c_name + ".reuse_ct"], math.log2(512 + 1))

    assert fequal(fea_dict["unroll_num"], math.log2(1 + 1))
    # assert fequal(fea_dict["unroll_type.kPosInnerReduce"], 1.0)
    assert fequal(fea_dict["vec_num"], math.log2(1 + 1))
    assert fequal(fea_dict["parallel_num"], math.log2(2 + 1))
    assert fequal(fea_dict["parallel_prod"], math.log2((512 * 512 / 16 / 8) + 1))


def test_cpu_fusion():
    def fusion_test(N, M):
        A = te.placeholder((N, M), name='A')
        B = te.compute((N, M), lambda i, j: A[i][j], name='B')
        C = te.compute((N, M), lambda i, j: B[i][j], name='C')
        return [A, B, C]

    dag = ansor.ComputeDAG(fusion_test(64, 32))
    s = dag.get_init_state()
    s.compute_at(1, 2, s.stages[2].iters[1])

    target = tvm.target.create('llvm')
    task = ansor.SearchTask(dag, "test", target)
    names = ansor.feature.get_per_stmt_feature_names()
    fea = ansor.feature.get_per_stmt_features_from_states([s.state_object], task)[0]

    found = False
    for stage_fea in fea:
        for i, (name, value) in enumerate(zip(names, stage_fea)):
            if 'reuse_type.kSerialMultipleReadWrite' in name and value > 0.5:
                assert fequal(stage_fea[i + 2], 1.0)
                assert fequal(stage_fea[i + 3], math.log2(16 + 1))
                found = True
    assert found


def test_gpu_feature():
    # todo(lmzheng)
    pass


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_fusion()
    test_gpu_feature()
