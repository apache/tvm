import tvm
from tvm.target import Target

def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


class CUDA(object):    

    def __init__(self, target: Target):
        self.target: Target = target
        self.sm_version: int = check_sm_version(self.target.arch)
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.device = device
        self.platform = "CUDA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap = 65536
        self.max_smem_usage = 2 * self.smem_cap
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.bandwidth = [750, 12080]
