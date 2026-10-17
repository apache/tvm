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
"""Wrapping existing transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from collections.abc import Callable

from . import _ffi_api
from .function_pass import function_pass


def DecorateDeviceScope():
    """Wrap each function body in a ``tirx.device_scope`` region.

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(x: T.int32):
                T.evaluate(x)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(x: T.int32):
                with T.device_scope():
                    T.evaluate(x)

        def run_pass():
            return tr.DecorateDeviceScope()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.DecorateDeviceScope()


def Apply(ftransform):
    """Apply ftransform to each function in the Module.

    This function is a thin wrapper around :py:func:`function_pass`

    Parameters
    ----------
    ftransform: tvm_tirx.tirx.Function -> tvm_tirx.tirx.Function
       The transformation pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                pass

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                T.func_attr({"example": 1})

        def run_pass():
            return tr.Apply(lambda f: f.with_attr("example", 1))(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """

    # pylint: disable=unused-argument
    def _transform(func, mod, ctx):
        return ftransform(func)

    return function_pass(_transform, opt_level=0, name="Apply")  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """Lower vectorization loops.

    Target-dependent vectorization uses the Function's ``target`` attribute,
    not an ambient target context or nested target attributes. Target-independent
    fixed-width loops do not require a target. Code needing different vectorization
    targets must be separated into functions before this pass.

    Parameters
    ----------
    enable_vectorize : bool
        Whether vectorization is enabled.
        Will lower to scalar loop when it is turned off.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
                for i in T.vectorized(0, 4):
                    B[i] = A[i]

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
                B[0:4] = A[T.Ramp(0, 1, 4)]

        def run_pass():
            return tr.VectorizeLoop()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def StorageRewrite():
    """Plan storage by moving allocations and reusing compatible storage.

    Hoists allocations where valid and shares storage when lifetimes and
    memory-scope allocation properties permit it.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(output: T.Tensor((2,), "float32")):
                A = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
                for i in T.serial(4):
                    A[i] = T.float32(i)
                output[0] = A[3]
                B = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
                for i in T.serial(4):
                    B[i] = T.float32(i + 1)
                output[1] = B[3]

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(output: T.Tensor((2,), T.float32)):
                A = T.alloc_tensor(ty_args=[T.Tensor((4,), T.float32)])
                for i in range(4):
                    A[i] = T.Cast("float32", i)
                output[0] = A[3]
                A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((4,), T.float32)])
                for i_1 in range(4):
                    A_1[i_1] = T.Cast("float32", i_1 + 1)
                output[1] = A_1[3]

        def run_pass():
            return tr.StorageRewrite()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def PointerValueTypeRewrite():
    """Rewrite pointer and tensor element types to match their accesses.

    Adjusts argument and allocation types, shapes and accesses where possible
    to reduce backend casts, including packing scalar storage into vectors.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((16,), T.float32, scope="local")])
                A[0:4] = T.Broadcast(T.float32(1.0), 4)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((4,), T.float32x4, scope="local")])
                A[0] = T.Broadcast(T.float32(1.0), 4)

        def run_pass():
            return tr.PointerValueTypeRewrite()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.PointerValueTypeRewrite()  # type: ignore


def UnrollLoop():
    """Unroll loops with constant extents marked ``T.unroll``.

    The ``tirx.UnrollLoop`` PassContext configuration also controls automatic
    unrolling and whether to expand loops or retain unroll annotations.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((2,), T.int32)):
                for i in T.unroll(0, 2):
                    A[i] = 1

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((2,), T.int32)):
                A[0] = 1
                A[1] = 1

        def run_pass():
            return tr.UnrollLoop()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.UnrollLoop()  # type: ignore


def RemoveNoOp():
    """Remove effect-free statements and provably empty control flow.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32)):
                T.evaluate(1)
                A[0] = 1

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32)):
                A[0] = 1

        def run_pass():
            return tr.RemoveNoOp()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.RemoveNoOp()  # type: ignore


def HoistIf():
    """Hoist loop-invariant If statements outside eligible loops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass.

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.int32), n: T.int32):
                for i in range(4):
                    if n > 0:
                        A[i] = 1

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.int32), n: T.int32):
                if 0 < n:
                    for i in range(4):
                        A[i] = 1

        def run_pass():
            return tr.HoistIf()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.HoistIf()


def BF16ComputeLegalize():
    """Promote unsupported BF16 computations and internal temporaries to float32.

    Inserts conversions at BF16 boundaries. Leaves functions unchanged when
    their CUDA target reports native BF16 support.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((1,), "bfloat16")])
                A[0] = T.bfloat16(1)
                T.evaluate(A[0] + T.bfloat16(2))

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float32)])
                A[0] = T.float32(1.0)
                T.evaluate(A[0] + T.float32(2.0))

        def run_pass():
            return tr.BF16ComputeLegalize()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.BF16ComputeLegalize()  # type: ignore


def FP8ComputeLegalize(promote_dtype: str = "float32"):
    """Promote unsupported FP8 computations and internal temporaries.

    Inserts conversions at FP8 boundaries. Leaves functions unchanged when
    their CUDA target reports native FP8 support.

    Parameters
    ----------
    promote_dtype : str
        The data type we promote fp8 to, options: float16/float32.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((1,), "float8_e5m2")])
                A[0] = T.float8_e5m2(1)
                T.evaluate(A[0] + T.float8_e5m2(2))

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float16)])
                A[0] = T.float16(1.0)
                T.evaluate(A[0] + T.float16(2.0))

        def run_pass():
            return tr.FP8ComputeLegalize("float16")(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.FP8ComputeLegalize(promote_dtype)  # type: ignore


def BF16StorageLegalize():
    """Represent BF16 storage as uint16 while preserving its bit patterns.

    Run after compute legalization. Functions whose CUDA target reports
    native BF16 support are unchanged.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    After packed-API lowering, pointer storage uses integer elements.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Ptr(T.bfloat16)):
                T.evaluate(0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Ptr(T.uint16)):
                pass

        def run_pass():
            return tr.BF16StorageLegalize()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.BF16StorageLegalize()  # type: ignore


def FP8StorageLegalize():
    """Represent FP8 storage as uint8 while preserving its bit patterns.

    Run after compute legalization. Functions whose CUDA target reports
    native FP8 support are unchanged.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    After packed-API lowering, pointer storage uses integer elements.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Ptr(T.float8_e5m2)):
                T.evaluate(0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Ptr(T.uint8)):
                pass

        def run_pass():
            return tr.FP8StorageLegalize()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.FP8StorageLegalize()  # type: ignore


def CommonSubexprElim():
    """Replace redundant computations by new variables.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(B: T.Tensor((2,), T.int32), x: T.int32, y: T.int32):
                B[0] = x + y
                B[1] = x + y

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(B: T.Tensor((2,), T.int32), x: T.int32, y: T.int32):
                cse_v1: T.let[T.int32] = x + y
                B[0] = cse_v1
                B[1] = cse_v1

        def run_pass():
            return tr.CommonSubexprElim()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.CommonSubexprElim()  # type: ignore


def RenormalizeSplitPattern():
    """Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), i: T.int32):
                A[0] = i % 128 // 32

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), i: T.int32):
                A[0] = i // 32 % 4

        def run_pass():
            return tr.RenormalizeSplitPattern()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.RenormalizeSplitPattern()  # type: ignore


def StmtSimplify():
    """Run statement-level arithmetic simplifications on the TIR Function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.int32)):
                for i in range(4):
                    A[i] = i // 4

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.int32)):
                for i in range(4):
                    A[i] = 0

        def run_pass():
            return tr.StmtSimplify()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.StmtSimplify()  # type: ignore


def LowerThreadAllreduce():
    """Lower ``gpu_thread_allreduce`` to target-specific communication.

    Requires a function ``target`` attribute and explicit thread extents.
    Selects warp shuffles or shared-memory reductions as supported.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass.

    Examples
    --------
    A single participating thread reduces to a direct store.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.float32)):
                T.func_attr({"target": T.target({"kind": "cuda", "arch": "sm_75"})})
                R = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float32, scope="local")])
                with T.launch_thread("threadIdx.x", 1) as tx:
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda arg0, arg1: (arg0 + arg1,)),
                        (T.float32(0.0),),
                        (A[tx],),
                        T.bool(True),
                        (R[0],),
                        (tx,),
                    )

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.float32)):
                T.func_attr({"target": T.target({"kind": "cuda", "arch": "sm_75"})})
                R = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float32, scope="local")])
                with T.launch_thread("threadIdx.x", 1) as tx:
                    R[0] = A[tx]

        def run_pass():
            return tr.LowerThreadAllreduce()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.LowerThreadAllreduce()


def ConvertSSA():
    """Give every variable definition in an IRModule a distinct identity.

    This pass handles cases where the same `tirx.Var` appears in
    multiple functions within the same module.  For example, after
    extracting a fragment from one function into another, where the
    same `tirx.Var` may be defined both as within the body of the
    original function, and as a parameter within the hoisted function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    The input uses explicit ``T.Bind`` statements to reuse a variable and disables
    the SSA check. Ordinary parsed assignments create distinct variable bindings.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        v = T.dynamic("v", "int32")

        @I.ir_module(check_well_formed=False)
        class Before:
            @T.function(private=True)
            def main():
                T.Bind(v, 16)
                T.evaluate(v)
                T.Bind(v, 32)
                T.evaluate(v)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                v = 16
                T.evaluate(v)
                v_1 = 32
                T.evaluate(v_1)

        def run_pass():
            return tr.ConvertSSA()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.ConvertSSA()  # type: ignore


def MakePackedAPI():
    """Expose exported functions through the TVM FFI packed-call ABI.

    Requires a ``target`` attribute with a host target on functions with a
    ``global_symbol`` and the default calling convention. Private functions,
    functions with other calling conventions, and functions whose target has
    no host are left unchanged.

    Tensor parameter types generate runtime checks for shape, dtype and
    layout. Symbolic dimensions not defined by other arguments are bound from
    the incoming DLTensor metadata. For example, a ``T.Tensor((16, 32),
    "float32")`` argument is checked against that shape, while symbolic
    dimensions are read from the argument's shape. The resulting function
    uses the host target and an exported ``__tvm_ffi_`` symbol.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    An exported no-argument function illustrates the packed-call ABI.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function
            def main():
                T.func_attr({"target": T.target("llvm", host="llvm")})
                T.evaluate(0)

        @I.ir_module
        class Expected:
            @T.function
            def main(
                self_handle: T.Ptr(T.void),
                args: T.Ptr(T.void),
                num_args: T.int32,
                result: T.Ptr(T.void),
            ) -> T.int32:
                T.func_attr({"global_symbol": "__tvm_ffi_main"})
                T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
                assert num_args == 0, (
                    "TypeError",
                    ["Expected ", "0", " arguments", " when calling:\\n  `", "main()", "`"],
                )
                with T.compute_scope("main_compute_"):
                    pass
                return 0

        def run_pass():
            return tr.MakePackedAPI()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def SplitHostDevice():
    """Annotate, split, and lower host/device functions.

    This pass first annotates device regions within host functions,
    then splits them into host and device-side Functions, and finally
    lowers host-to-device calls into the device kernel launch ABI.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm")})
                with T.region("tirx.device_scope", []):
                    T.evaluate(n)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(n: T.int32):
                T.func_attr(
                    {
                        "target": T.target(
                            {
                                "host": {
                                    "keys": ["cpu"],
                                    "kind": "llvm",
                                    "mtriple": "",
                                    "tag": "",
                                },
                                "keys": ["cuda", "gpu"],
                                "kind": "cuda",
                                "max_num_threads": 1024,
                                "tag": "",
                                "thread_warp_size": 32,
                            }
                        )
                    }
                )
                T.call_ffi_kernel("main_kernel", n)

            @T.function
            def main_kernel(n: T.int32):
                T.func_attr(
                    {
                        "calling_conv": 2,
                        "target": T.target("cuda"),
                        "tirx.is_global_func": True,
                        "tirx.kernel_launch_params": [],
                        "tirx.noalias": True,
                    }
                )
                T.evaluate(n)

        def run_pass():
            return tr.SplitHostDevice()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def SkipAssert():
    """Replace assertion statements with no-ops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), n: T.int32):
                assert n > 0, ("RuntimeError", ["positive extent required"])
                A[0] = 1

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), n: T.int32):
                A[0] = 1

        def run_pass():
            return tr.SkipAssert()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.SkipAssert()  # type: ignore


def LowerTVMBuiltin():
    """Lower runtime builtins to explicit stack storage and ABI operations.

    For example, packed calls gain argument/result slots. Requires the
    function target attribute for target-dependent lowering.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.func_attr({"target": T.target("llvm")})
                T.call_packed("callback", 1)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                T.func_attr({"target": T.target("llvm")})
                stack_ffi_any: T.let[T.Ptr(T.void)] = T.stack_alloca("tvm_ffi_any", 2)
                T.abi_field_set(stack_ffi_any, 0, 13, 1)
                T.abi_field_set(stack_ffi_any, 0, 14, 0)
                T.abi_field_set(stack_ffi_any, 0, 15, T.Cast("int64", 1))
                T.abi_field_set(stack_ffi_any, 1, 13, 0)
                T.abi_field_set(stack_ffi_any, 1, 14, 0)
                T.abi_field_set(stack_ffi_any, 1, 15, T.int64(0))
                T.call_packed_lowered("callback", stack_ffi_any, 0, 1)

        def run_pass():
            return tr.LowerTVMBuiltin()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.LowerTVMBuiltin()  # type: ignore


def LowerIntrin():
    """Lower intrinsic calls using rules registered for the function target.

    Requires a function ``target`` attribute. Also lowers logical tensor
    pointer operations to address operations.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), x: T.int32):
                T.func_attr({"target": T.target("llvm")})
                A[0] = x // 8

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((1,), T.int32), x: T.int32):
                T.func_attr({"target": T.target("llvm")})
                A[0] = x >> 3

        def run_pass():
            return tr.LowerIntrin()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.LowerIntrin()  # type: ignore


def NarrowDataType(target_bits: int):
    """Narrow integer indices and loop variables when bounds prove it safe.

    Keeps wider types when the analyzed range does not fit ``target_bits``.

    Parameters
    ----------
    target_bits : int
        The target bit configuration.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Note
    ----
    Run this pass after FlattenTensor.

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                for i in range(T.int64(0), T.int64(4)):
                    A[i] = T.float32(1.0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                for i in range(4):
                    A[i] = T.float32(1.0)

        def run_pass():
            return tr.NarrowDataType(32)(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.NarrowDataType(target_bits)  # type: ignore


def ForceNarrowIndexToInt32():
    """Force index expressions to int32 without range proofs.

    Use only when the caller can guarantee representability. Unlike
    :py:func:`NarrowDataType`, this pass may change overflowing computations.
    Rejects signed integer tensor parameters and ordinary allocations wider
    than int32. Explicitly typed local scalars retain their storage type and
    are cast at narrowed index uses.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Note
    ----
    This pass should not be used in default cases.

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                for i in range(T.int64(0), T.int64(4)):
                    A[i] = T.float32(1.0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                for i in range(4):
                    A[i] = T.float32(1.0)

        def run_pass():
            return tr.ForceNarrowIndexToInt32()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.ForceNarrowIndexToInt32()  # type: ignore


def VerifyMemory():
    """Reject illegal host-side access to device memory.

    Returns the module unchanged on success and raises an error on invalid
    access. Functions without a target attribute are skipped, as are
    functions with a non-default calling convention.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Valid device memory accesses are checked without changing the IR.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                T.func_attr({"target": T.target({"kind": "cuda", "arch": "sm_75"})})
                with T.launch_thread("threadIdx.x", 4) as tx:
                    A[tx] = T.float32(1.0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main(A: T.Tensor((4,), T.float32)):
                T.func_attr({"target": T.target({"kind": "cuda", "arch": "sm_75"})})
                with T.launch_thread("threadIdx.x", 4) as tx:
                    A[tx] = T.float32(1.0)

        def run_pass():
            return tr.VerifyMemory()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.VerifyMemory()  # type: ignore


def FlattenTensor():
    """Flatten tensor allocations and accesses to one-dimensional storage.

    Uses shapes, strides and offsets to compute linear indices. Tensor
    parameters retain their interface types and gain flat views as needed.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((2, 4), T.float32, scope="local")])
                A[1, 2] = T.float32(1.0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((8,), T.float32, scope="local")])
                A[6] = T.float32(1.0)

        def run_pass():
            return tr.FlattenTensor()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.FlattenTensor()  # type: ignore


def BindTarget(target):
    """Bind targets and classify host/device helper functions.

    Preserves existing device targets and updates host information on exported
    functions. Helpers used on both sides may be cloned for the host.

    Device helper calls must be inside explicit launch_thread or device_entry
    regions. Lower S-TIR thread bindings before classifying host/device calls.

    Parameters
    ----------
    target : tvm_tirx.target.Target
        target

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.evaluate(0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                T.func_attr({"target": T.target("cuda")})

        def run_pass():
            return tr.BindTarget(tvm.target.Target("cuda"))(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.BindTarget(target)  # type: ignore


def AnnotateEntryFunc():
    """Infer and mark an unambiguous TIRx entry function.

    Sets ``tirx.is_entry_func`` when the module contains only one function
    and it is TIRx, or when exactly one function is externally exposed and
    it is a TIRx function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                pass

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                T.func_attr({"tirx.is_entry_func": True})

        def run_pass():
            return tr.AnnotateEntryFunc()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.AnnotateEntryFunc()  # type: ignore


def Filter(fcond: Callable):
    """Remove TIRx functions for which ``fcond(function)`` is false.

    The predicate receives a Function and returns a boolean.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.evaluate(0)

        @I.ir_module
        class Expected:
            pass

        def run_pass():
            return tr.Filter(lambda f: False)(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.Filter(fcond)  # type: ignore


def TileDispatch():
    """Lower TIRx tile calls using the active backend dispatch table.

    Resolves tile instructions and device launch/execution scopes. Select
    the backend with a target context; follow with :py:func:`LowerTensorLayouts`
    or the backend's own layout-lowering pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Dispatch a tile-wide move into a loop of scalar PTX moves.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_local((4, 1), "float32", layout=T.TileLayout(T.S[(4, 1) : (1, 1)]))
                T.cuda.tile.apply(T.ptx["mov.b32"], A[:, :], T.float32(0), scope="thread")

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_local((4, 1), "float32", layout=T.TileLayout(T.S[(4, 1) : (1, 1)]))
                tile_d = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (4,),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout(T.S[(4,) : (1,)]),
                        )
                    ],
                )
                for repeat_0 in T.unroll(0, 4):
                    T.ptx.mov(tile_d[repeat_0], T.float32(0.0), "b32", "")

        def run_pass():
            with tvm.target.Target({"kind": "cuda", "arch": "sm_75"}):
                return tr.TileDispatch()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.TileDispatch()  # type: ignore


def LowerTensorLayouts():
    """Apply tensor layouts to physical storage and accesses.

    Run after :py:func:`TileDispatch` for generic tile compilation, or
    independently on already dispatched IR. Backends may pair TileDispatch
    with their own layout-lowering pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T
        from tvm_tirx.tirx import transform as tr

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (3, 4),
                            T.uint32,
                            scope="local",
                            layout=T.TileLayout(T.S[(3, 4) : (16, 2)]),
                        )
                    ]
                )
                A[2, 3] = T.uint32(7)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                A = T.alloc_tensor(ty_args=[T.Tensor((39,), T.uint32, scope="local")])
                A[38] = T.uint32(7)

        def run_pass():
            with tvm.target.Target({"kind": "cuda", "arch": "sm_75"}):
                return tr.LowerTensorLayouts()(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_api.LowerTensorLayouts()  # type: ignore
