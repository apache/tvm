import os
import os.path as osp
# from tvm.contrib import nvcc
import subprocess

with open("gemmx1.cu", "r") as f:
    code = f.read()

tvm_root = osp.join(osp.dirname(__file__), "../..")
tl_template_path = osp.abspath(osp.join(tvm_root, "src/tl"))
if "TL_CUTLASS_PATH" in os.environ:
    cutlass_path = os.environ["TL_CUTLASS_PATH"]
else:
    cutlass_path = osp.abspath(osp.join(tvm_root, "3rdparty/cutlass/include"))


format = "ptx"
arch = f"sm_90a"

# print(tl_template_path)
# print(cutlass_path)

nvcc_command = [
    "nvcc",
    "-o", "gemmx1",
    "-arch=" + arch,
    "--use_fast_math",
    "-std=c++17",
    "-I" + tl_template_path,
    "-I" + cutlass_path,
    "-lcuda",
    "gemmx1.cu"
]

subprocess.run(nvcc_command, check=True)

# nvcc -ptx fa_kernel.cu -o fa_kernel.ptx -O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT162_CONVERSIONS__ --expt-relaxed-constexpr  --expt-extended-lambda -arch=sm_90a --use_fast_math -std=c++17 -I/home/msra/cy/tvm.tl/src/tl -I/home/msra/cy/tvm.tl/cutlass/include -lcuda 
"-O3",
"-U__CUDA_NO_HALF_OPERATORS__",
"-U__CUDA_NO_HALF_CONVERSIONS__",
"-U__CUDA_NO_BFLOAT16_OPERATORS__",
"-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
"-U__CUDA_NO_BFLOAT162_OPERATORS__",
"-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
"--expt-relaxed-constexpr",
"--expt-extended-lambda",