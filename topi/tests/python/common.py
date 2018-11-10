"""Common utility for topi test"""

from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity


def get_all_backend():
    """return all supported target

    Returns
    -------
    targets: list
        A list of all supported targets
    """
    return ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx',
            'llvm -device=arm_cpu', 'opencl -device=mali', 'aocl_sw_emu']


class NCHWcInt8Fallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'int8'
        self.memory[key] = cfg
        return cfg
