import tvm
from tvm import te, tir
from tvm.script import tir as T

import numpy as np
import torch
import torch.nn.functional as F


def te_conv1d(G, K, C, X, F_X, s, d, p):
    i = te.placeholder((1, C, X), name="I")
    w = te.placeholder((K, C // G, F_X), name="W")
    b = te.placeholder((K,), name="B")

    rc = te.reduce_axis((0, C // G), name="rc")
    rx = te.reduce_axis((0, F_X), name="rx")
    o = te.compute(
        (1, G, K // G, (X + 2 * p - F_X - (F_X - 1) * (d - 1)) // s + 1),
        lambda nn, gg, kk, xx: te.sum(
            te.if_then_else(
                te.any(s * xx + d * rx - p < 0, s * xx + d * rx - p >= X),
                0.0,
                i[nn, gg * (C // G) + rc, s * xx + d * rx - p] * w[gg * (K // G) + kk, rc, rx],
            ),
            axis=[rc, rx],
            init=b[gg * (K // G) + kk],
        ),
        name="O",
    )

    return i, w, b, o


@T.prim_func
def tir_conv1d(
    i: T.handle,
    w: T.handle,
    b: T.handle,
    o: T.handle,
    G: T.int32,
    K: T.int32,
    C: T.int32,
    X: T.int32,
    F_X: T.int32,
    s: T.int32,
    d: T.int32,
    p: T.int32,
) -> None:
    I = T.match_buffer(i, (1, C, X))
    W = T.match_buffer(w, (K, C // G, F_X))
    B = T.match_buffer(b, (K,))
    O = T.match_buffer(o, (1, K, (X + 2 * p - F_X - (F_X - 1) * (d - 1)) // s + 1))

    for g in range(G):
        for k in range(K // G):
            for c in range(C // G):
                for f_x in range(F_X):
                    for x in range(0, (X + 2 * p - F_X - (F_X - 1) * (d - 1)) // s + 1):
                        with T.block("mac"):
                            if c == 0 and f_x == 0:
                                O[0, g * (K // G) + k, x] = B[g * (K // G) + k]
                            if s * x + d * f_x - p >= 0 and s * x + d * f_x - p < X:
                                O[0, g * (K // G) + k, x] += (
                                    I[0, g * (C // G) + c, s * x + d * f_x - p]
                                    * W[g * (K // G) + k, c, f_x]
                                )


def main():
    params = {"G": 1, "K": 8, "C": 8, "X": 20, "F_X": 9, "s": 1, "d": 1, "p": 0}
    _, _, _, _, G, K, C, X, F_X, s, d, p = tir_conv1d.params
    func = tir_conv1d.specialize(
        {
            G: params["G"],
            K: params["K"],
            C: params["C"],
            X: params["X"],
            F_X: params["F_X"],
            s: params["s"],
            d: params["d"],
            p: params["p"],
        }
    )
    sch = tir.Schedule(func, debug_mask="all")
    g, k, c, f_x, x = sch.get_loops(sch.get_block("mac"))
    k_0, k_1 = sch.split(k, factors=[2, 4])
    sch.reorder(g, k_0, c, f_x, x, k_1)
    print(sch.mod)

    # input (i), weights (w)
    i = torch.rand((1, params["C"], params["X"]))
    w = torch.rand((params["K"], params["C"] // params["G"], params["F_X"]))
    b = torch.rand((params["K"]))

    conv = F.conv1d(
        i,
        w,
        bias=b,
        stride=params["s"],
        padding=params["p"],
        dilation=params["d"],
        groups=params["G"],
    )

    i = tvm.nd.array(i.numpy())
    w = tvm.nd.array(w.numpy())
    b = tvm.nd.array(b.numpy())
    o = tvm.nd.array(
        np.zeros(
            (
                1,
                params["K"],
                (
                    params["X"]
                    + 2 * params["p"]
                    - params["F_X"]
                    - (params["F_X"] - 1) * (params["d"] - 1)
                )
                // params["s"]
                + 1,
            ),
            dtype="float32",
        )
    )
    mod = tvm.lower(sch.mod["main"])
    print(mod)
    tvm_conv1d = tvm.build(mod)
    tvm_conv1d(i, w, b, o)

    assert np.allclose(o.numpy(), conv.numpy())

    # ii, ww, bb, oo = te_conv1d(
    #     params["G"],
    #     params["K"],
    #     params["C"],
    #     params["X"],
    #     params["F_X"],
    #     params["s"],
    #     params["d"],
    #     params["p"],
    # )
    # schedule = te.create_schedule(oo.op)
    # # nn, gg, kk, xx = o.op.axis
    # mod = tvm.lower(schedule, [ii, ww, bb, oo], simple_mode=False)
    # func = tvm.build(mod, [ii, ww, bb, oo], "llvm")
    # func(i, w, b, o)
    # o = o.asnumpy().reshape(
    #     1,
    #     params["K"],
    #     (params["X"] + 2 * params["p"] - params["F_X"] - (params["F_X"] - 1) * (params["d"] - 1))
    #     // params["s"]
    #     + 1,
    # )
    # print(o)


if __name__ == "__main__":
    main()
