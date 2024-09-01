from setuptools import setup
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
torch.utils.cpp_extension.CUDAExtension.debug = True

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-arch=sm_90a',
        '--use_fast_math',
        '-std=c++17',
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        '-I/usr/local/cuda/include',
        '-I/home/msra/cy/tvm.tl/src/tl',
        '-I/home/msra/cy/tvm.tl/cutlass/include',
        '-lcuda',
        # '-keep' # Uncomment this line to keep the generated .ptx file
    ],
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
