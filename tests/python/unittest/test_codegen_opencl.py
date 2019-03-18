import tvm

target = 'opencl'

def test_opencl_ternary_expression():
    def check_if_then_else(ctx, n, dtype):
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        true_value = tvm.const(1, dtype=dtype)
        false_value = tvm.const(3, dtype=dtype)
        max_lhs = tvm.const(2, dtype=dtype)
        max_rhs = tvm.if_then_else(A[0] > 0, true_value, false_value)
        C = tvm.compute((n,), lambda i: tvm.max(max_lhs, max_rhs), name='C')
        s = tvm.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tvm.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)

        a = tvm.nd.empty((n,), A.dtype, ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        # Only need to test compiling here
        fun(a, c)

    def check_select(ctx, n, dtype):
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        true_value = tvm.const(1, dtype=dtype)
        false_value = tvm.const(3, dtype=dtype)
        max_lhs = tvm.const(2, dtype=dtype)
        max_rhs = tvm.expr.Select(A[0] > 0, true_value, false_value)
        C = tvm.compute((n,), lambda i: tvm.max(max_lhs, max_rhs), name='C')
        s = tvm.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tvm.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)

        a = tvm.nd.empty((n,), A.dtype, ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        # Only need to test compiling here
        fun(a, c)

    if not tvm.module.enabled(target):
        print("skip because opencl is not enabled..")
        return

    ctx = tvm.context(target, 0)

    check_if_then_else(ctx, 1, 'int8')
    check_if_then_else(ctx, 1, 'uint8')
    check_if_then_else(ctx, 1, 'int16')
    check_if_then_else(ctx, 1, 'uint16')
    check_select(ctx, 1, 'int8')
    check_select(ctx, 1, 'uint8')
    check_select(ctx, 1, 'int16')
    check_select(ctx, 1, 'uint16')


if __name__ == "__main__":
    test_opencl_ternary_expression()
