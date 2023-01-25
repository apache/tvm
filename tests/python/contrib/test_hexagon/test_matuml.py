import sys
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir import IndexMap
from tvm.contrib.hexagon import allocate_hexagon_array


pytest_plugins = [
    "tvm.testing.plugin",
    "tvm.contrib.hexagon.pytest_plugin",
]


np.set_printoptions(threshold=sys.maxsize, suppress=True)


def layout_primfunc_impl(dim0, dim1, dtype, input_scope, output_scope):
    @T.prim_func
    def copy(input: T.handle, output: T.handle):
        I = T.match_buffer(input, (dim0, dim1), dtype=dtype, scope=input_scope)
        O = T.match_buffer(output, (dim0, dim1), dtype=dtype, scope=output_scope)

        for i, j in T.grid(dim0, dim1):
            with T.block("compute"):
                vi, vj = T.axis.remap("SS", [i, j])
                O[vi, vj] = I[vi, vj]

    return copy


@tvm.testing.fixture
def matmul_primfunc(M, N, K, dtype, layout_transform_scope):
    input_scope = layout_transform_scope["input"]
    weight_scope = layout_transform_scope["weight"]
    output_scope = layout_transform_scope["output"]

    @T.prim_func
    def matmul(a: T.handle, w: T.handle, b: T.handle):
        A = T.match_buffer(a, (M, K), dtype=dtype, scope=input_scope)
        W = T.match_buffer(w, (K, N), dtype=dtype, scope=weight_scope)
        B = T.match_buffer(b, (M, N), dtype=dtype, scope=output_scope)

        for m, n, k in T.grid(M, N, K):
            with T.block("compute"):
                vm, vn, vk = T.axis.remap("SSR", [m, n, k])
                with T.init():
                    B[vm, vn] = T.cast(0, dtype=dtype)
                B[vm, vn] = B[vm, vn] + A[vm, vk] * W[vk, vn]

    return matmul


class MatMulBase:
    M = tvm.testing.parameter(64)
    N = tvm.testing.parameter(64)
    K = tvm.testing.parameter(64)
    dtype = tvm.testing.parameter("int8")
    # Perform layout transformation as part of the compute, or externally
    # through separate primfuncs.
    layout_primfunc = tvm.testing.parameter(None)
    # Note that when the global or layout transform scopes are set to
    # global.vtcm, cache_reads and writes in the matmul schedule are ommitted
    arg_scopes = ["input", "weight", "output"]
    global_memory_scope, layout_transform_scope = tvm.testing.parameters(
        # Slices of activation and weights are fetched to fast memory in matmul loopnest
        ("global", {k: "global" for k in arg_scopes}),
        # Simulates weight prefetch when the layout transform occurs in its own primfunc (layout_primfunc != None)
        # (
        #     "global",
        #     {k: v for k, v in zip(arg_scopes, ["global", "global.vtcm", "global"])},
        # ),
        # All compute including layout transform performed in fast memory, no fetching
        # ("global.vtcm", {k: "global.vtcm" for k in arg_scopes}),
        ids=[
            "all_global_buffers_with_vtcm_slices",
            # "global_activations_weight_prefetched_to_vtcm",
            # "all_buffers_preloaded_to_vtcm",
        ],
    )


@tvm.testing.fixture
def fast_memory_scope(tvm_target):
    if str(tvm_target.kind) == "hexagon":
        return "global.vtcm"
    return "global"


@tvm.testing.fixture
def numpy_data_range(dtype):
    if dtype == "int8":
        return (-128, 127)
    elif dtype == "uint8":
        return (0, 255)
    elif dtype == "float16":
        return (-1.0, 1.0)
    raise "Unsupported data type"


@tvm.testing.fixture
def ndarrays(
    M,
    N,
    K,
    dtype,
    layout_primfunc,
    activation_index_map,
    weight_index_map,
    output_index_map,
    global_memory_scope,
    layout_transform_scope,
    numpy_data_range,
    tvm_target,
    hexagon_session,
):
    a_shape = (M, K)
    w_shape = (K, N)
    b_shape = (M, N)
    if dtype == "float16":
        a = np.random.uniform(
            low=numpy_data_range[0], high=numpy_data_range[1], size=a_shape
        ).astype(dtype)
        w = np.random.uniform(
            low=numpy_data_range[0], high=numpy_data_range[1], size=w_shape
        ).astype(dtype)
    else:
        a = np.random.randint(
            low=numpy_data_range[0], high=numpy_data_range[1], size=a_shape, dtype=dtype
        )
        w = np.random.randint(
            low=numpy_data_range[0], high=numpy_data_range[1], size=w_shape, dtype=dtype
        )
    b = np.zeros(b_shape, dtype=dtype)
    if layout_primfunc:
        at = np.zeros([int(i) for i in activation_index_map.map_shape(a_shape)], dtype=dtype)
        wt = np.zeros([int(i) for i in weight_index_map.map_shape(w_shape)], dtype=dtype)
        bt = np.zeros([int(i) for i in output_index_map.map_shape(b_shape)], dtype=dtype)
        numpy_tensors = [
            (a, global_memory_scope),
            (at, layout_transform_scope["input"]),
            (w, global_memory_scope),
            (wt, layout_transform_scope["weight"]),
            (bt, layout_transform_scope["output"]),
            (b, global_memory_scope),
        ]
    else:
        numpy_tensors = zip([a, w, b], [global_memory_scope] * 3)

    if str(tvm_target.kind) == "hexagon":
        arrays = [
            allocate_hexagon_array(hexagon_session.device, data=tensor, mem_scope=scope)
            for tensor, scope in numpy_tensors
        ]
    elif str(tvm_target.kind) == "llvm":
        arrays = [tvm.nd.array(x, device=tvm.cpu(0)) for x in numpy_tensors]
    else:
        raise "Could not allocate arrays for target"
    return arrays


def get_ref(a, w):
    return np.matmul(a, w)


@tvm.testing.fixture
def tvm_target(target_str):
    if target_str == "hexagon":
        target_hexagon = tvm.target.hexagon("v68")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
    else:
        target = tvm.target.Target(target_str)
    return target


def module_loader(mod, hexagon_session=None):
    if mod.type_key == "hexagon":
        assert hexagon_session != None
        mod = hexagon_session.load_module(mod)
    return mod


@tvm.testing.fixture
def scheduled_matmul_primfunc(
    matmul_primfunc,
    layout_primfunc,
    activation_index_map,
    weight_index_map,
    global_memory_scope,
    layout_transform_scope,
    fast_memory_scope,
    dtype,
):
    mod = tvm.IRModule.from_expr(matmul_primfunc)
    sch = tir.Schedule(mod)

    compute_block = sch.get_block("compute")

    if layout_primfunc == None:
        # Materialize layout transformations on inputs and outputs
        At_block = sch.cache_read(compute_block, 0, layout_transform_scope["input"])
        Wt_block = sch.cache_read(compute_block, 1, layout_transform_scope["weight"])
        Bt_block = sch.cache_write(compute_block, 0, layout_transform_scope["output"])

    sch.transform_layout(compute_block, ("read", 0), index_map=activation_index_map)
    sch.transform_layout(compute_block, ("read", 1), index_map=weight_index_map)
    sch.transform_layout(compute_block, ("write", 0), index_map=activation_index_map)

    m, n, k = sch.get_loops(compute_block)
    if dtype == "float16":
        mo, mi = sch.split(m, factors=[None, 32])
    else:
        mo, mi = sch.split(m, factors=[None, 64])
    no, ni = sch.split(n, factors=[None, 32])
    ko, ki = sch.split(k, factors=[None, 32])
    sch.reorder(mo, no, ko, ki, mi, ni)

    def schedule_cache(loops, vectorize=False):
        fused = sch.fuse(*loops)
        # o, _, io, ii = sch.split(fused, factors=[4, None, 2, 128])
        # sch.unroll(io)
        # # sch.parallel(o)
        # if vectorize:
        #     sch.vectorize(ii)
        return

    if layout_transform_scope["input"] == "global":
        # Use fast memory caches for input
        Avtcm_block = sch.cache_read(compute_block, 0, fast_memory_scope)
        # Read one block row (along K) of input at a time
        sch.compute_at(Avtcm_block, mo)
        schedule_cache(sch.get_loops(Avtcm_block)[1:])

    if layout_transform_scope["weight"] == "global":
        # Use fast memory caches for weight
        Wvtcm_block = sch.cache_read(compute_block, 1, fast_memory_scope)
        # Read one column (along K) of weight at a time
        sch.compute_at(Wvtcm_block, no)
        schedule_cache(sch.get_loops(Wvtcm_block)[2:])

    if layout_transform_scope["output"] == "global":
        # Use fast memory caches for output
        Bvtcm_block = sch.cache_write(compute_block, 0, fast_memory_scope)
        # Write back to global after accumulating accross all of ko
        sch.reverse_compute_at(Bvtcm_block, no)
        schedule_cache(sch.get_loops(Bvtcm_block)[2:])

    # Use local memory accumulator
    Blocal_block = sch.cache_write(compute_block, 0, "local")
    # sch.reverse_compute_at(Blocal_block, no)

    # Split out initialization and update of accumulator and
    # place initialization as early as possible.
    init_block = sch.decompose_reduction(compute_block, ko)
    update_block = sch.get_block("compute_update")

    # Nest the software pipeline for mo and no
    sch.annotate(mo, "software_pipeline_stage", [1, 2, 2, 2])
    sch.annotate(mo, "software_pipeline_order", [0, 1, 2, 3])
    sch.annotate(mo, "software_pipeline_async_stages", [1])

    sch.annotate(no, "software_pipeline_stage", [0, 1, 1, 1, 2])
    sch.annotate(no, "software_pipeline_order", [0, 1, 2, 3, 4])
    sch.annotate(no, "software_pipeline_async_stages", [0])

    return sch.mod["main"]


@tvm.testing.fixture
def activation_index_map(dtype):
    if dtype == "int8" or dtype == "uint8":
        return IndexMap.from_func(lambda m, k: [m // 64, k // 32, m % 64, k % 32])
    elif dtype == "float16":
        return IndexMap.from_func(lambda m, k: [m // 32, k // 32, (m % 32) // 2, k % 32, m % 2])
    else:
        raise "Unsupported data type"


@tvm.testing.fixture
def output_index_map(activation_index_map):
    return activation_index_map


@tvm.testing.fixture
def weight_index_map(dtype, activation_index_map):
    if dtype == "int8" or dtype == "uint8":
        return IndexMap.from_func(lambda k, n: [n // 32, k // 32, (k % 32) // 4, n % 32, k % 4])
    elif dtype == "float16":
        return IndexMap.from_func(lambda k, n: [n // 32, k // 32, (k % 32) // 2, n % 32, k % 2])


@tvm.testing.fixture
def scheduled_input_layout_transform(
    M,
    K,
    dtype,
    layout_primfunc,
    activation_index_map,
    global_memory_scope,
    layout_transform_scope,
):
    mod = schedule_layout_transforms(
        (M, K),
        dtype,
        layout_primfunc,
        global_memory_scope,
        layout_transform_scope["input"],
    )
    if mod:
        sch = tir.Schedule(mod)
        compute_block = sch.get_block("compute")
        sch.transform_layout(compute_block, ("write", 0), index_map=activation_index_map)
        return sch.mod["main"]
    return mod


@tvm.testing.fixture
def scheduled_weight_layout_transform(
    N,
    K,
    dtype,
    layout_primfunc,
    weight_index_map,
    global_memory_scope,
    layout_transform_scope,
):
    mod = schedule_layout_transforms(
        (K, N),
        dtype,
        layout_primfunc,
        global_memory_scope,
        layout_transform_scope["weight"],
    )
    if mod:
        sch = tir.Schedule(mod)
        compute_block = sch.get_block("compute")
        sch.transform_layout(compute_block, ("write", 0), index_map=weight_index_map)
        return sch.mod["main"]
    return mod


@tvm.testing.fixture
def scheduled_output_layout_transform(
    M,
    N,
    dtype,
    layout_primfunc,
    activation_index_map,
    global_memory_scope,
    layout_transform_scope,
):
    mod = schedule_layout_transforms(
        (M, N),
        dtype,
        layout_primfunc,
        global_memory_scope,
        layout_transform_scope["output"],
    )
    if mod:
        sch = tir.Schedule(mod)
        compute_block = sch.get_block("compute")
        sch.transform_layout(compute_block, ("read", 0), index_map=activation_index_map)
        return sch.mod["main"]
    return mod


def schedule_layout_transforms(
    shape,
    dtype,
    layout_primfunc,
    global_memory_scope,
    scope,
):
    if layout_primfunc:
        mod = tvm.IRModule.from_expr(
            layout_primfunc(
                shape[0],
                shape[1],
                dtype,
                input_scope=global_memory_scope,
                output_scope=scope,
            )
        )
        return mod
    return None


@tvm.testing.fixture
def scheduled_runtime_modules(
    scheduled_input_layout_transform,
    scheduled_weight_layout_transform,
    scheduled_matmul_primfunc,
    scheduled_output_layout_transform,
    tvm_target,
):

    functions = {}
    if scheduled_input_layout_transform:
        functions["activation_transform"] = scheduled_input_layout_transform
    if scheduled_weight_layout_transform:
        functions["weight_transform"] = scheduled_weight_layout_transform
    functions["matmul"] = scheduled_matmul_primfunc
    if scheduled_output_layout_transform:
        functions["output_transform"] = scheduled_output_layout_transform

    functions = {k: v.with_attr("global_symbol", k) for (k, v) in functions.items()}
    irmod = tvm.IRModule(functions)
    print("TVMScript:")
    irmod.show()
    print("Lowered:")
    with tvm.transform.PassContext(
        config={
            "tir.use_async_copy": 1,
            "tir.experimental_dma_bypass_cache": 1,
            "tir.merge_async_commit_queue_scope": 1,
        }
    ):
        tvm.lower(irmod).show()

    with tvm.transform.PassContext(
        config={
            "tir.use_async_copy": 1,
            "tir.experimental_dma_bypass_cache": 1,
            "tir.merge_async_commit_queue_scope": 1,
        }
    ):
        mod = tvm.build(irmod, target=tvm_target)
    return mod


class TestMatMul(MatMulBase):
    target_str = tvm.testing.parameter("hexagon")

    @tvm.testing.requires_hexagon
    def test_matmul(
        self,
        M,
        N,
        K,
        scheduled_runtime_modules,
        hexagon_session,
        ndarrays,
        tvm_target,
        dtype,
    ):
        # return
        mod = module_loader(scheduled_runtime_modules, hexagon_session)

        if len(ndarrays) == 3:
            a, w, b = ndarrays
            timer = mod.time_evaluator("matmul", hexagon_session.device, number=20, repeat=2)
            timing_result = timer(a, w, b)
            print(timing_result)
        else:
            a, at, w, wt, bt, b = ndarrays
            mod["activation_transform"](a, at)
            mod["weight_transform"](w, wt)
            timer = mod.time_evaluator("matmul", hexagon_session.device, number=20, repeat=2)
            timing_result = timer(at, wt, bt)
            print(timing_result)
            mod["output_transform"](bt, b)
        print("Throughput: ", 2 * M * N * K / timing_result.mean / 1e9, "GOPS")

        ref = get_ref(a.numpy(), w.numpy())
        tvm.testing.assert_allclose(ndarrays[-1].numpy(), ref, atol=1e-3, rtol=1e-3)
