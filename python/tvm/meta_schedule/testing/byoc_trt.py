import tvm
from tvm.runtime import Module
from tvm.target import Target
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder, BuilderResult
from typing import List


def relay_build_with_tensorrt(
    mod: Module,
    target: Target,
    params: dict,
) -> List[BuilderResult]:
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

    mod, config = partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        return tvm.relay.build_module._build_module_no_factory(mod, "cuda", "llvm", params)
