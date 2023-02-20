import numpy as np
import pytest
import tvm
from tvm import te
import scipy
from tvm import relay
import pytest
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
from tvm.contrib.nvcc import have_fp16
import tvm.testing
from tvm.topi.utils import get_const_tuple
from tvm.script import tir

executor_kind = tvm.testing.parameter("graph", "vm")


@tvm.testing.uses_gpu
def test_floor_div_op(target, dev):
    N = 100
    divisor = 5

    @tir.prim_func
    def func_64(
        A: tir.Buffer[(N + 100, 2), "int64"],
        B: tir.Buffer[(N), "int64"],
        C: tir.Buffer[(N), "int64"],
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int64"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int64"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_32(
        A: tir.Buffer[(N + 100, 2), "int32"],
        B: tir.Buffer[(N), "int32"],
        C: tir.Buffer[(N), "int32"],
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int32"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int32"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_16(
        A: tir.Buffer[(N + 100, 2), "int16"],
        B: tir.Buffer[(N), "int16"],
        C: tir.Buffer[(N), "int16"],
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int16"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int16"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_8(
        A: tir.Buffer[(N + 100, 2), "int8"], B: tir.Buffer[(N), "int8"], C: tir.Buffer[(N), "int8"]
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int8"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int8"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    for opfunc, type in [
        (func_8, "int8"),
        (func_16, "int16"),
        (func_32, "int32"),
        (func_64, "int64"),
    ]:
        built = tvm.build(opfunc, target=target)
        x_data = np.random.randint(te.min_value(type), te.max_value(type), size=(100), dtype=type)
        data = []
        for i in range(N):
            data.append(i)

        y_data = np.asarray(data, dtype=type)

        a_dev = tvm.nd.empty([N + 100, 2], type, dev)
        b_dev = tvm.nd.array(x_data, dev)
        c_dev = tvm.nd.array(y_data, dev)

        built(a_dev, b_dev, c_dev)

        a = a_dev.numpy()
        b = b_dev.numpy()
        c = c_dev.numpy()

        #python modulo behaves a bit different to tvm floormod for negative numbers
        for i in range(N+100):
            if a[i, 1] < 0:
                a[i, 1] = divisor+a[i, 1]

        np.testing.assert_array_equal(a[:100, 0], (c-te.max_value(type)) // divisor)
        np.testing.assert_array_equal(a[:100, 1], (c-te.max_value(type)) % divisor)
        np.testing.assert_array_equal(a[100:N+100, 0], b // divisor)
        np.testing.assert_array_equal(a[100:N+100, 1], b % divisor)
    
if __name__ == "__main__":
    tvm.testing.main()
