import tvm

def test_storage_share():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    num_stage = 5
    B = A
    for t in range(num_stage):
        B = tvm.compute((m, l), lambda i, j: B[i, j] + (t+1), name='A%d' % t)

    s = tvm.create_schedule(B.op)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 1

def test_alloc_seq():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope="local.L0A")
            A[j] = 1.2
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="B", scope="local.L0A")
            A[j] = 1.3

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 200
    tvm.ir_pass.PostOrderVisit(body, verify)
    assert num_alloc[0] == 1

def test_alloc_different_dtypes():
    def stmt_generater(dtype_list, length):
        ib = tvm.ir_builder.create()
        base_dtype = dtype_list[0]
        global_a = tvm.placeholder((length,), name = "global_a", dtype = base_dtype)
        for index, dtype in enumerate(dtype_list):
            with ib.for_range(0, length, name="j") as j:
                A = ib.allocate(dtype, length, name="A_" + str(index), scope="local.L0A")
                A[j] = tvm.const(1, dtype = dtype)
        return ib.get()
    
    def dtype_bit_len(dtype):
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])

    def offset_generater(dtype_list, length):
        dtype_len_list = [dtype_bit_len(i) for i in dtype_list]
        base_len = dtype_len_list[0]
        return sum([i * length / base_len for i in dtype_len_list])

    def dtype_test(dtype_list, length):
        def verify(n):
            if isinstance(n, tvm.stmt.Allocate):
                assert n.extents[0].value == offset

        body = stmt_generater(dtype_list, length)
        offset = offset_generater(dtype_list, length)
        body = tvm.ir_pass.StorageRewrite(body)
        tvm.ir_pass.PostOrderVisit(body, verify)

    length = 1024
    dtype_list = ["float16", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float32", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float64", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)
    
    dtype_list = ["int8", "int32", "uint16", "uint8"]
    dtype_test(dtype_list, length)


def test_inplace_rule():
    m = 10
    A = tvm.placeholder((m,), name='A')
    A0 = tvm.compute((m,), lambda i: A[i], name='A0')
    A1 = tvm.compute((m,), lambda i: A[i] + 1, name='A1')
    AA =  tvm.compute((m,), lambda i: A0[i] + A1[i] + A1[0], name='AA')
    B = tvm.compute((m,), lambda i: AA[i] + 1, name='B')
    s = tvm.create_schedule(B.op)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 2


def test_storage_combine():
    n = 8
    A = tvm.placeholder((4,), name='A')
    num_stage = 5
    B = A
    stages = []
    for t in range(num_stage):
        B = tvm.compute((n, ), lambda i: B[i] + B[0] + (t+1), name='A%d' % t)
        stages.append(B)

    s = tvm.create_schedule(B.op)
    for S in stages[:-1]:
        s[S].set_scope("global:tag")
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
            assert (n.extents[0].value == 16)
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 1


def test_storage_share_gpu():
    m = tvm.var('m')
    A = [tvm.placeholder((m), name='A')]
    num_stage = 5
    for t in range(num_stage):
        A.append(tvm.compute((m,), lambda i: A[-1][i] + (t+1), name='A%d_s' % t))
        A.append(tvm.compute((m,), lambda i: A[-1][i], name='A%d' % t))
    s = tvm.create_schedule(A[-1].op)
    for t in range(num_stage):
        x = A[2*t+2].op.axis[0]
        bx, tx = s[A[2*t+2]].split(x, factor=32)
        s[A[2*t+2]].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[A[2*t+2]].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[A[2*t+1]].compute_at(s[A[2*t+2]], tx)
        s[A[2*t+1]].set_scope("shared")

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A[0].shape, A[0].dtype, name='A')
    Bb = tvm.decl_buffer(A[0].shape, A[0].dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A[0]: Ab, A[-1]: Bb}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    alloc_stats = {"global": 0, "shared": 0}

    def verify(n):
        if isinstance(n, tvm.stmt.AttrStmt):
            if n.attr_key == "storage_scope":
                alloc_stats[n.value.value] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert alloc_stats["global"] == 2
    assert alloc_stats["shared"] == num_stage

def test_parallel_alloc():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="i", for_type="parallel") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", n, name="A", scope="global")
            A[j] = A[j] + 2

    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)
    assert (isinstance(body.body.body, tvm.stmt.Allocate))

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    with ib.for_range(0, n, name="t") as i:
        ib.scope_attr(
            tvm.const(1) , "pragma_scope", tvm.make.StringImm("parallel_launch_point"))
        with ib.for_range(0, n, name="i", for_type="parallel") as i:
            with ib.for_range(0, 10, name="j") as j:
                A = ib.allocate("float32", n, name="A", scope="global")
                A[j] = A[j] + 2
    body = ib.get()
    body = tvm.ir_pass.StorageRewrite(body)

    assert(isinstance(body.body.body.body.body, tvm.stmt.Allocate))

def test_inplace_rule2():
    #Test Buffer
    scope_tb = "local_TB"
    @tvm.register_func("tvm.info.mem.%s" % scope_tb)
    def mem_info_inp_buffer():
        return tvm.make.node("MemoryInfo",
                        unit_bits= 16,
                        max_simd_bits=32,
                        max_num_bits=1024*1024*1024,
                        head_address=None)
    m = 10
    A = tvm.placeholder((m,), name='A')
    C = tvm.placeholder((m,), name='C')
    D = tvm.placeholder((m,), name='D')
    A0 = tvm.compute((m,), lambda i: A[i] + C[i], name='A0')
    A1 = tvm.compute((m,), lambda i: D[i] * D[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A0[i] + A1[i], name='A2')
    B = tvm.compute((m,), lambda i: A2[i], name='B')
    s = tvm.create_schedule(B.op)
    A0L = s.cache_read(A0, scope_tb, [A2])
    A1L = s.cache_read(A1, scope_tb, [A2])
    A2L = s.cache_read(A2, scope_tb, [B])
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    Cc = tvm.decl_buffer(C.shape, B.dtype, name='C')
    Dd = tvm.decl_buffer(D.shape, B.dtype, name='D')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb, C: Cc, D:Dd}, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 2

if __name__ == "__main__":
    test_alloc_seq()
    test_alloc_different_dtypes()
    test_inplace_rule()
    test_storage_share()
    test_parallel_alloc()
    test_storage_combine()
    test_storage_share_gpu()
    test_inplace_rule2()
