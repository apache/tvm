"""Common utility for topi test"""

def get_all_backend():
    """return all supported target

    Returns
    -------
    targets: list
        A list of all supported targets
    """
    return ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx',
            'llvm -device=arm_cpu', 'opencl -device=mali', 'aocl_sw_emu']
