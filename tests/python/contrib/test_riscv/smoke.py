import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm import te
from tvm.relay.backend import Executor, Runtime
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import byoc
from tvm.testing.aot import (
    AOTTestModel,
    compile_and_run,
    compile_models,
    create_relay_module_and_inputs_from_tflite_file,
    generate_ref_data,
)
import os


def test_sub(type):
    # TE + llvm + tvm.build -> asm code
    target = "llvm  -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"

    m = te.var("m")
    A = te.placeholder(m, dtype=type, name="A")
    B = te.placeholder(m, dtype=type, name="B")
    C = te.compute((m), lambda i: A[i] - B[i], name="C")
    s = te.create_schedule([C.op])

    f = tvm.build(s, [A, B, C], target)

    # Verify we see SVE load instructions and sub instructions using z registers
    assembly = f.get_source("asm")
    print(assembly)

def test_sub2(type):
    # TE + llvm + tvm.build -> asm code
    relay_mod = tvm.relay.fromtext(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 4, 4, 4), float32], %weight: Tensor[(4, 4, 3, 3), float32], src_layout="OIHW", dst_layout="OIHW4i4o") -> Tensor[(1, 4, 4, 4), float32] {
        %0 = fn (%p02: Tensor[(1, 4, 4, 4), float32], Primitive=1, hash="9332b3872fb5292c", src_layout="NCHW", dst_layout="NCHW4c") -> Tensor[(1, 1, 4, 4, 4), float32] {
            layout_transform(%p02, src_layout="NCHW", dst_layout="NCHW4c") /* ty=Tensor[(1, 1, 4, 4, 4), float32] */
        };
        %1 = fn (%p03: Tensor[(4, 4, 3, 3), float32], Primitive=1, hash="9f0b2b8a24a4dab3", src_layout="OIHW", dst_layout="OIHW4i4o") -> Tensor[(1, 1, 3, 3, 4, 4), float32] {
            layout_transform(%p03, src_layout="OIHW", dst_layout="OIHW4i4o") /* ty=Tensor[(1, 1, 3, 3, 4, 4), float32] */
        };
        %2 = %0(%data) /* ty=Tensor[(1, 1, 4, 4, 4), float32] */;
        %3 = %1(%weight) /* ty=Tensor[(1, 1, 3, 3, 4, 4), float32] */;
        %4 = fn (%p01: Tensor[(1, 1, 4, 4, 4), float32], %p1: Tensor[(1, 1, 3, 3, 4, 4), float32], out_layout="NCHW4c", kernel_layout="OIHW4i4o", Primitive=1, data_layout="NCHW4c") -> Tensor[(1, 1, 4, 4, 4), float32] {
                                                                                                                                                                                                                                                      nn.contrib_conv2d_NCHWc(%p01, %p1, padding=[1, 1, 1, 1], channels=4, kernel_size=[3, 3], data_layout="NCHW4c", kernel_layout="OIHW4i4o", out_layout="NCHW4c") /* ty=Tensor[(1, 1, 4, 4, 4), float32] */
        };
        %5 = %4(%2, %3) /* ty=Tensor[(1, 1, 4, 4, 4), float32] */;
        %6 = fn (%p0: Tensor[(1, 1, 4, 4, 4), float32], Primitive=1, src_layout="NCHW4c", dst_layout="NCHW") -> Tensor[(1, 4, 4, 4), float32] {
            layout_transform(%p0, src_layout="NCHW4c", dst_layout="NCHW") /* ty=Tensor[(1, 4, 4, 4), float32] */
        };
        %6(%5) /* ty=Tensor[(1, 4, 4, 4), float32] */
        }
        """
    )
    # pylint: enable=line-too-long

    compiled_test_mods = compile_models(
        models=AOTTestModel(module=relay_mod, inputs=None, outputs=None),
        interface_api="c",
        use_unpacked_api=True,
        pass_config={"tir.usmp.enable": False},
    )
    source = compiled_test_mods[0].executor_factory.lib.imported_modules[0].get_source()
    
    print(source)
    # Verify we see SVE load instructions and sub instructions using z registers


if __name__ == "__main__":
    test_sub("int8")
    test_sub2("int8")