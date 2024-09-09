from setuptools import setup
import torch.utils.cpp_extension
import subprocess
from packaging.version import parse, Version
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
torch.utils.cpp_extension.CUDAExtension.debug = True


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

cc_flag = []
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("12.3"):
    raise RuntimeError("FA Hopper is only supported on CUDA 12.3 and above")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
        # "-keep"
    ]

# extra_compile_args = {
#     'cxx': ['-O3', '-std=c++17'],
#     'nvcc': [
#         # '-arch=sm_90a',
#         '-gencode arch=compute_90a,code=compute_90a',
#         '--use_fast_math',
#         '-std=c++17',
#         "-O3",
#         "-U__CUDA_NO_HALF_OPERATORS__",
#         "-U__CUDA_NO_HALF_CONVERSIONS__",
#         "-U__CUDA_NO_BFLOAT16_OPERATORS__",
#         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
#         "-U__CUDA_NO_BFLOAT162_OPERATORS__",
#         "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
#         "--expt-relaxed-constexpr",
#         "--expt-extended-lambda",
#         "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",  # printing out number of registers
#         '-I/usr/local/cuda/include',
#         '-I/home/msra/cy/tvm.tl/src/tl',
#         '-I/home/msra/cy/tvm.tl/cutlass/include',
#         '-lcuda',
#         '-lineinfo',
#         "-lnvToolsExt",
#         "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
#         "-DNDEBUG",  # Important, otherwise performance is severely impacted
#         # '-keep' # Uncomment this line to keep the generated .ptx file
#     ],
# }

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": append_nvcc_threads(
        nvcc_flags + ["-DEXECMODE=0"] + cc_flag
    ),
}

include_dirs = [
    '/home/msra/cy/tvm.tl/src/tl',
    '/home/msra/cy/tvm.tl/cutlass/include',
    '/usr/local/cuda/include'
]

setup(
    name='fa_test',
    ext_modules=[
        CUDAExtension(
            'fa_test',
            sources=['cuda_interface.cpp', 'fa_kernel.cu', 'fa_no_tma.cu'],
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            libraries=["cuda"]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH TMPDIR=~/cy/ncu_tmp ncu --set full -k regex:"main_kernel" --launch-count 1 --launch-skip 10 --target-processes application-only --cache-control none --clock-control none --apply-rules yes --import-source yes --check-exit-code yes -f -o reports/tl_8_2048_8_256_false /home/msra/miniconda3/envs/tl/bin/python main.py