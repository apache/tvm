import tvm
import tvm.testing
import tvm.topi.testing
import numpy as np
from tvm import topi


@tvm.testing.requires_llvm
def test_embed():
    M = 64
    N = 128
    K = 10
    table = np.reshape(np.arange(N * M), (M, N)).astype("float64")
    indices = np.random.randint(M, size=(K,))
    out = table[indices, :]
    tvm.testing.compare_numpy_tvm(
        [table, indices],
        out,
        "llvm",
        tvm.context("cpu"),
        topi.nn.embed,
        tvm.topi.testing.get_injective_schedule("llvm"),
    )


@tvm.testing.parametrize_targets
def test_embed_grad(ctx, target):
    M = 30
    N = 50
    K = 10
    table = np.reshape(np.arange(N * M), (M, N)).astype("float64")
    indices = np.random.randint(M, size=(K,))
    indices[0] = indices[-1]  # ensure we have duplicate indices
    grad = np.reshape(np.arange(K * N), (K, N)).astype(table.dtype)
    grad_out = np.zeros((M, N)).astype(table.dtype)

    for i, ind in enumerate(indices):
        grad_out[ind, :] += grad[i, :]

    implementations = {
        "cpu": (topi.nn.embed_grad, topi.x86.schedule_embed_grad),
        "gpu": (topi.nn.embed_grad, topi.cuda.schedule_embed_grad),
    }
    fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
    tvm.testing.compare_numpy_tvm(
        [table, indices, grad], grad_out, target, ctx, fcompute, fschedule
    )


if __name__ == "__main__":
    test_embed()
    test_embed_grad(tvm.context("cpu"), "llvm")
