import tvm
import numpy as np
from tvm import te, tir, topi


SQNN_FP32 = 0
SQNN_INT8 = 1
SQNN_UINT8 = 2

SQNN_DATATYPE_MAP = {
    SQNN_FP32: 'float32',
    SQNN_INT8: 'int8',
    SQNN_UINT8: 'uint8',
}


def simulated_qnn(data, out_dtype, output_scale=None, output_zero_point=None, axis=-1):

    def _compute_fp32(value, *indices):
        return value[indices]

    def _compute_int8(value, *indices):
        assert output_scale is not None and output_zero_point is not None
        const_min = tvm.tir.min_value(SQNN_DATATYPE_MAP[SQNN_INT8])
        const_max = tvm.tir.max_value(SQNN_DATATYPE_MAP[SQNN_INT8])
        # Use indexmod to handle both scalar and per-channel QNN parameters.
        scale_idx = tir.indexmod(indices[axis], topi.shape(output_scale)[0])
        zp_idx = tir.indexmod(indices[axis], topi.shape(output_zero_point)[0])
        return te.max(te.min(te.round(value[indices] / output_scale[scale_idx]) + output_zero_point[zp_idx], const_max), const_min)

    def _compute_uint8(value, *indices):
        assert output_scale is not None and output_zero_point is not None
        const_min = tvm.tir.min_value(SQNN_DATATYPE_MAP[SQNN_UINT8])
        const_max = tvm.tir.max_value(SQNN_DATATYPE_MAP[SQNN_UINT8])
        # Use indexmod to handle both scalar and per-channel QNN parameters.
        scale_idx = tir.indexmod(indices[axis], topi.shape(output_scale)[0])
        zp_idx = tir.indexmod(indices[axis], topi.shape(output_zero_point)[0])
        return te.max(te.min(te.round(value[indices] / output_scale[scale_idx]) + output_zero_point[zp_idx], const_max), const_min)

    def _dispatch_sim_qnn(value):
        fp32_value = te.compute(data.shape, lambda *indices: _compute_fp32(value, *indices))
        int8_value = te.compute(
            data.shape, 
            lambda *indices: tir.if_then_else(out_dtype[0] == SQNN_INT8, _compute_int8(value, *indices), fp32_value[indices]))
        uint8_value = te.compute(
            data.shape, 
            lambda *indices: tir.if_then_else(out_dtype[0] == SQNN_UINT8, _compute_uint8(value, *indices), int8_value[indices]))

        return uint8_value


    #return _dispatch_qnn()
    return te.compute(
        data.shape,
        lambda *indices: _dispatch_sim_qnn(data)[indices]
    )
    
def test_sim_qnn():
    V = te.placeholder([8], name="value")
    D = te.placeholder([1], name="dtype", dtype='int32')
    S = te.placeholder([te.size_var("dim")], name="scale", dtype='float32')
    ZP = te.placeholder([te.size_var("dim")], name="zero_point", dtype='int32')
    Q = simulated_qnn(V, D, output_scale=S, output_zero_point=ZP)
    s = te.create_schedule([Q.op])
    a = tvm.nd.array(np.asarray([200] * 8).astype('float32'), tvm.cpu())
    q = tvm.nd.array(np.zeros((8), dtype='float32'), tvm.cpu())
    f = tvm.build(s, [V, D, S, ZP, Q], "llvm", name="sim_qnn")
    for d_val in [[0], [1], [2]]:
        d = tvm.nd.array(np.asarray(d_val).astype('int32'), tvm.cpu())
        s = tvm.nd.array(np.asarray([1, 2]*4).astype('float32'), tvm.cpu())
        zp = tvm.nd.array(np.asarray([0]*8).astype('int32'), tvm.cpu())
        f(a, d, s, zp, q)
        print(q)
        s = tvm.nd.array(np.asarray([1]).astype('float32'), tvm.cpu())
        zp = tvm.nd.array(np.asarray([0]).astype('int32'), tvm.cpu())
        f(a, d, s, zp, q)
        print(q)


if __name__ == "__main__":
    test_sim_qnn()


