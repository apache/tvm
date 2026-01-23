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
# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""

from __future__ import absolute_import as _abs

import glob
import os
import platform
import subprocess
import warnings
from typing import Tuple

import tvm_ffi

import tvm
from tvm.target import Target

from ..base import py_str
from . import utils


def compile_cuda(
    code, target_format=None, arch=None, options=None, path_target=None, compiler="nvcc"
):
    """Compile cuda code with NVCC or NVRTC.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : str
        The target format of the compiler ("ptx", "cubin", or "fatbin").

    arch : str
        The cuda architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    compiler : str, optional
        Compiler backend: "nvcc" or "nvrtc".
        This can be set by the TVM_CUDA_COMPILE_MODE environment variable.

    Returns
    -------
    res_binary : bytearray
        The bytearray of the compiled binary (ptx/cubin/fatbin).

    Notes
    -----
    - NVRTC is a "runtime" compilation library and can be faster for JIT compilation.
    - NVRTC requires cuda-python: pip install cuda-python
    """
    use_nvshmem = "#include <nvshmem.h>" in code or "#include <nvshmemx.h>" in code

    if compiler == "nvcc":
        result = _compile_cuda_nvcc(code, target_format, arch, options, path_target, use_nvshmem)
    elif compiler == "nvrtc":
        result = _compile_cuda_nvrtc(code, target_format, arch, options, path_target, use_nvshmem)
    else:
        raise ValueError(f"cuda compiler must be 'nvcc' or 'nvrtc', got: {compiler}")

    return result


def _compile_cuda_nvcc(
    code,
    target_format=None,
    arch=None,
    options=None,
    path_target=None,
    use_nvshmem=False,
):
    """Compile CUDA code using nvcc.

    Parameters
    ----------
    code : str
        The CUDA source code.
    target_format : str, optional
        Output format: "ptx", "cubin", or "fatbin".
    arch : str, optional
        Target architecture. Auto-detected if None.
    options : str or list of str, optional
        Additional nvcc options.
    path_target : str, optional
        Output file path.

    Returns
    -------
    bytearray
        Compiled binary data.
    """
    # Check for NVSHMEM dependency
    nvshmem_include_path, nvshmem_lib_path = None, None
    if use_nvshmem:
        # NOTE: we cannot check whether nvshmem is used based on whether
        # the global function "runtime.nvshmem.cumodule_init" is defined.
        # The reason is because that if the input code does not use any NVSHMEM functions
        # while the global function is defined, using cubin to compile the
        # code may cause a compilation error.
        target_format = "cubin"
        nvshmem_include_path, nvshmem_lib_path = find_nvshmem_paths()

    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "sm_xx", or a list, such as
        # [
        #   "-gencode", "arch=compute_52,code=sm_52",
        #   "-gencode", "arch=compute_70,code=sm_70"
        # ]
        compute_version = "".join(
            get_target_compute_version(Target.current(allow_none=True)).split(".")
        )
        arch = ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]

    temp = utils.tempdir()
    file_name = "tvm_kernels"
    if target_format is None and not use_nvshmem:
        target_format = "ptx"
    if target_format not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target_format must be in cubin, ptx, fatbin")
    temp_code = temp.relpath(f"{file_name}.cu")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm_ffi.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = (
        pass_context.config["cuda.kernels_output_dir"]
        if "cuda.kernels_output_dir" in pass_context.config
        else None
    )
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.cu")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    if use_nvshmem:
        file_prefix = os.path.splitext(file_target)[0]
        file_target = f"{file_prefix}.o"  # in the first stage, compile to object file

    cmd = ["nvcc"]
    cmd += [f"--{target_format}", "-O3"]
    if kernels_output_dir is not None:
        cmd += ["-lineinfo"]
    if isinstance(arch, list):
        cmd += arch
    elif isinstance(arch, str):
        cmd += ["-arch", arch]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    if not use_nvshmem:
        cmd += [temp_code]
    else:
        cmd += ["-c", temp_code]
        cmd += ["-rdc=true"]
        cmd += ["-I", nvshmem_include_path]

    # NOTE: ccbin option can be used to tell nvcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require nvcc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    # Second stage for NVSHMEM
    if use_nvshmem:
        cmd = ["nvlink"]
        cmd += [f"-arch=sm_{compute_version}"]
        cmd += ["-L", nvshmem_lib_path]
        cmd += ["-L", os.path.join(find_cuda_path(), "lib64")]
        cmd += ["-l", "nvshmem_device"]
        cmd += ["-l", "cudadevrt"]
        cmd += ["-o", f"{file_prefix}.cubin"]
        cmd += [file_target]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        (out, _) = proc.communicate()

        if proc.returncode != 0:
            msg = code
            msg += "\nCompilation error:\n"
            msg += py_str(out)
            raise RuntimeError(msg)

        file_target = f"{file_prefix}.cubin"

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def _compile_cuda_nvrtc(
    code, target_format=None, arch=None, options=None, path_target=None, use_nvshmem=False
):
    """Compile CUDA code using NVRTC (NVIDIA Runtime Compilation).

    Parameters
    ----------
    code : str
        The CUDA source code.
    target_format : str, optional
        Output format: "cubin" or "ptx". Default: "cubin"
    arch : str, optional
        Target architecture (e.g., "sm_80"). Auto-detected if None.
    options : str or list of str, optional
        Additional NVRTC options.
    path_target : str, optional
        Output file path. If provided, the compiled binary is written to this path.
    use_nvshmem : bool, optional
        Whether NVSHMEM is used. Default: False

    Returns
    -------
    bytearray
        Compiled binary data.
    """
    try:
        from cuda.bindings import nvrtc  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise RuntimeError(
            "Failed to compile CUDA with NVRTC because the `cuda-python` package "
            "is not available.\n"
            "Please install it with: pip install cuda-python\n"
            "See: https://nvidia.github.io/cuda-python/"
        ) from e

    # For NVSHMEM, we also need the CUDA driver API to initialize the context for linking
    if use_nvshmem:
        import importlib.util  # pylint: disable=import-outside-toplevel

        if importlib.util.find_spec("cuda.bindings.driver") is None:
            raise RuntimeError(
                "Failed to compile CUDA with NVRTC+NVSHMEM because the `cuda-python` package "
                "is not available.\n"
                "Please install it with: pip install cuda-python\n"
                "See: https://nvidia.github.io/cuda-python/"
            )

    # NVSHMEM requires linking with device library, which always produces cubin
    if use_nvshmem or target_format is None:
        target_format = "cubin"

    # Validate target_format (NVRTC doesn't support fatbin)
    if target_format == "fatbin":
        raise ValueError(
            "NVRTC does not support fatbin generation yet. "
            "Use target_format='cubin' or 'ptx' with NVRTC, "
            "or set compiler='nvcc' for fatbin compilation."
        )
    if target_format not in ["cubin", "ptx"]:
        raise ValueError(f"target_format must be 'cubin' or 'ptx', got: {target_format}")

    # Validate options
    if options is not None and not isinstance(options, (str, list)):
        raise ValueError("options must be str or list of str")

    # Auto-detect architecture
    if arch is None:
        compute_version = get_target_compute_version(Target.current(allow_none=True))
        arch = f"sm_{''.join(compute_version.split('.'))}"

    # Get NVSHMEM paths if needed
    nvshmem_include_path, nvshmem_lib_path = None, None
    if use_nvshmem:
        nvshmem_include_path, nvshmem_lib_path = find_nvshmem_paths()

    # Strip host-only headers for NVRTC. NVRTC compiles device code and does not
    # require the CUDA driver header or host C++ headers.
    headers_to_strip = {"#include <cuda.h>"}
    code_filtered = "\n".join(
        line for line in code.splitlines() if line.strip() not in headers_to_strip
    )

    # NVRTC compiles device code and does not include the host-side cuda.h.
    # CUtensorMap is a host-side structure, to reference and use it in device code,
    # we must forward-declare it for NVRTC.
    if "CUtensorMap" in code_filtered:
        code_filtered = (
            "struct __align__(128) CUtensorMap {\n"
            "  unsigned long long opaque[16];\n"
            "};\n\n" + code_filtered
        )

    # Add standard type definitions and compatibility macros that NVRTC doesn't provide.
    nvrtc_preamble = """#include <cuda/std/cstdint>
using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;

// NVRTC uses asm/volatile instead of __asm__/__volatile__
#ifndef __asm__
#define __asm__ asm
#endif
#ifndef __volatile__
#define __volatile__ volatile
#endif

"""
    code_filtered = nvrtc_preamble + code_filtered

    # For NVSHMEM, add preamble to map cuda::std type traits to std namespace.
    # NVSHMEM headers require std:: type traits but NVRTC uses cuda::std::.
    if use_nvshmem:
        nvshmem_preamble = """#include <cuda/std/type_traits>

// Map cuda::std type traits to std namespace for NVSHMEM headers
namespace std {
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::is_same;
    using cuda::std::enable_if;
    using cuda::std::conditional;
}

"""
        code_filtered = nvshmem_preamble + code_filtered

    # Create NVRTC program
    # Use "tvm_kernels.cu" for consistency with nvcc path
    result, prog = nvrtc.nvrtcCreateProgram(
        str.encode(code_filtered), b"tvm_kernels.cu", 0, None, None
    )
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Failed to create NVRTC program: {nvrtc.nvrtcGetErrorString(result)}")

    # Prepare compilation options
    cuda_path = find_cuda_path()
    compile_opts = [
        f"--gpu-architecture={arch}".encode(),
        b"-default-device",
    ]

    if use_nvshmem:
        compile_opts.extend([b"-rdc", b"true"])

    # Add CUDA include paths. NVRTC needs explicit include paths for CUDA headers.
    # Standard installations: cuda_path/include
    # Conda/architecture-specific installations: cuda_path/targets/<arch>/include
    include_paths = []

    # Check standard include directory
    standard_include = os.path.join(cuda_path, "include")
    if os.path.isdir(standard_include):
        include_paths.append(standard_include)

    # Check architecture-specific include directory
    arch_include = os.path.join(
        cuda_path,
        "targets",
        f"{platform.machine()}-{platform.system().lower()}",
        "include",
    )
    if os.path.isdir(arch_include):
        include_paths.append(arch_include)

    # Check for CCCL include directory (required for cuda/std/cstdint and type_traits)
    # CCCL provides standard library functionality for device code
    cccl_include = os.path.join(arch_include, "cccl") if os.path.isdir(arch_include) else None
    if cccl_include and os.path.isdir(cccl_include):
        include_paths.append(cccl_include)

    # Verify we can find essential CUDA headers
    if not any(os.path.isfile(os.path.join(p, "cuda_runtime.h")) for p in include_paths):
        raise RuntimeError(
            f"Cannot find CUDA headers in {cuda_path}. "
            f"Searched in: {include_paths}. "
            "Please ensure CUDA is properly installed."
        )

    # Add all valid include paths
    for include_path in include_paths:
        compile_opts.append(f"-I{include_path}".encode())

    # Add NVSHMEM include path
    if use_nvshmem and nvshmem_include_path:
        compile_opts.append(f"-I{nvshmem_include_path}".encode())

    # For NVSHMEM, add deprecation and type conversion macros
    if use_nvshmem:
        compile_opts.extend(
            [
                # Define deprecation macros as empty (not properly defined in NVRTC context)
                b"-D__NV_SILENCE_DEPRECATION_BEGIN=",
                b"-D__NV_SILENCE_DEPRECATION_END=",
                b"-D__NV_SILENCE_HOST_DEPRECATION_BEGIN=",
                b"-D__NV_SILENCE_HOST_DEPRECATION_END=",
                # Disable FP8/FP6/FP4 extended types that cause issues with NVRTC
                b"-D__CUDA_NO_FP8_CONVERSIONS__",
                b"-D__CUDA_NO_FP6_CONVERSIONS__",
                b"-D__CUDA_NO_FP4_CONVERSIONS__",
            ]
        )

    compile_opts.extend(
        [
            b"-U__CUDA_NO_HALF_OPERATORS__",
            b"-U__CUDA_NO_HALF_CONVERSIONS__",
            b"-U__CUDA_NO_BFLOAT16_OPERATORS__",
            b"-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            b"-U__CUDA_NO_BFLOAT162_OPERATORS__",
            b"-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            b"--use_fast_math",
        ]
    )

    # Add user-provided options, filtering out nvcc-specific flags that nvrtc doesn't support
    if options:
        nvcc_only_prefixes = (
            "-c",
            "-O",
            "-std",
            "--std",
            "-Xcompiler",
            "-Xlinker",
            "-Xarchive",
            "-Xcudafe",
            "-Xptxas",
            "--compile",
            "--compiler-options",
            "--linker-options",
            "-fPIC",
            "-shared",
            "-o",
        )
        if isinstance(options, str):
            options = [options]
        for opt in options:
            if isinstance(opt, str):
                opt_str = opt
            elif isinstance(opt, bytes):
                opt_str = opt.decode()
            else:
                opt_str = str(opt)
            skip = any(
                opt_str.startswith(prefix) or opt_str == prefix for prefix in nvcc_only_prefixes
            )
            if skip:
                continue
            compile_opts.append(opt.encode() if isinstance(opt, str) else opt)

    # Compile
    (result,) = nvrtc.nvrtcCompileProgram(prog, len(compile_opts), compile_opts)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        # Get compilation log
        result_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if result_log == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 0:
            log_buf = bytearray(log_size)
            (result_log,) = nvrtc.nvrtcGetProgramLog(prog, log_buf)
            if result_log == nvrtc.nvrtcResult.NVRTC_SUCCESS:
                error_msg = f"NVRTC compilation failed:\n{log_buf.decode('utf-8')}"
            else:
                error_msg = f"NVRTC compilation failed (couldn't get log): {result}"
        else:
            error_msg = f"NVRTC compilation failed: {result}"

        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(error_msg)

    # Get compiled binary
    if target_format == "cubin":
        result, binary_size = nvrtc.nvrtcGetCUBINSize(prog)
        if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"Failed to get CUBIN size: {nvrtc.nvrtcGetErrorString(result)}")
        binary_buf = bytearray(binary_size)
        (result,) = nvrtc.nvrtcGetCUBIN(prog, binary_buf)
        if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"Failed to get CUBIN: {nvrtc.nvrtcGetErrorString(result)}")
    else:  # ptx
        result, binary_size = nvrtc.nvrtcGetPTXSize(prog)
        if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"Failed to get PTX size: {nvrtc.nvrtcGetErrorString(result)}")
        binary_buf = bytearray(binary_size)
        (result,) = nvrtc.nvrtcGetPTX(prog, binary_buf)
        if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"Failed to get PTX: {nvrtc.nvrtcGetErrorString(result)}")

    # Clean up NVRTC program
    nvrtc.nvrtcDestroyProgram(prog)

    # Link stage for NVSHMEM
    if use_nvshmem:
        binary_buf = _link_nvshmem_nvrtc(binary_buf, nvshmem_lib_path)

    if path_target:
        with open(path_target, "wb") as f:
            f.write(binary_buf)
    return binary_buf


def _link_nvshmem_nvrtc(binary_buf, nvshmem_lib_path):
    """Link compiled CUBIN with NVSHMEM device library using CUDA driver API."""
    import ctypes  # pylint: disable=import-outside-toplevel

    from cuda.bindings import driver as cu  # pylint: disable=import-outside-toplevel

    # cuLinkCreate requires a valid CUDA context.
    (result,) = cu.cuInit(0)
    if result != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to initialize CUDA: {result}")

    # Check if there's already a CUDA context; create one if not
    context_created = False
    result, context = cu.cuCtxGetCurrent()
    if result != cu.CUresult.CUDA_SUCCESS or not context:
        result, device = cu.cuDeviceGet(0)
        if result != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to get CUDA device: {result}")
        result, context = cu.cuCtxCreate(None, 0, device)
        if result != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create CUDA context: {result}")
        context_created = True

    try:
        # Create linker
        result, link_state = cu.cuLinkCreate(0, [], [])
        if result != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create CUDA linker: {result}")

        try:
            # Add our compiled CUBIN
            (result,) = cu.cuLinkAddData(
                link_state,
                cu.CUjitInputType.CU_JIT_INPUT_CUBIN,
                binary_buf,
                len(binary_buf),
                b"tvm_kernels.cubin",
                0,
                [],
                [],
            )
            if result != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to add CUBIN to linker: {result}")

            # Add NVSHMEM device library
            nvshmem_device_lib = os.path.join(nvshmem_lib_path, "libnvshmem_device.a")
            if not os.path.exists(nvshmem_device_lib):
                raise RuntimeError(f"NVSHMEM device library not found: {nvshmem_device_lib}")

            (result,) = cu.cuLinkAddFile(
                link_state,
                cu.CUjitInputType.CU_JIT_INPUT_LIBRARY,
                nvshmem_device_lib.encode(),
                0,
                [],
                [],
            )
            if result != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to add NVSHMEM device library: {result}")

            # Complete linking
            result, linked_cubin, linked_size = cu.cuLinkComplete(link_state)
            if result != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to complete NVSHMEM linking: {result}")

            # Copy linked binary before destroying linker
            binary_buf = bytearray(ctypes.string_at(linked_cubin, linked_size))
            if not binary_buf:
                raise RuntimeError("Compilation error: empty result is generated")
        finally:
            # Clean up linker
            cu.cuLinkDestroy(link_state)
    finally:
        # Clean up context if we created it
        if context_created and context:
            cu.cuCtxDestroy(context)

    return binary_buf


def find_cuda_path():
    """Utility function to find cuda path

    Returns
    -------
    path : str
        Path to cuda root.
    """
    if "CUDA_PATH" in os.environ:
        return os.environ["CUDA_PATH"]
    cmd = ["which", "nvcc"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(str(out).strip(), "../.."))
    cuda_path = "/usr/local/cuda"
    if os.path.exists(os.path.join(cuda_path, "bin/nvcc")):
        return cuda_path
    raise RuntimeError("Cannot find cuda path")


def get_cuda_version(cuda_path=None):
    """Utility function to get cuda version

    Parameters
    ----------
    cuda_path : Optional[str]

        Path to cuda root.  If None is passed, will use
        `find_cuda_path()` as default.

    Returns
    -------
    version : float
        The cuda version

    """
    if cuda_path is None:
        cuda_path = find_cuda_path()

    version_file_path = os.path.join(cuda_path, "version.txt")
    if not os.path.exists(version_file_path):
        # Debian/Ubuntu repackaged CUDA path
        version_file_path = os.path.join(cuda_path, "lib", "cuda", "version.txt")
    try:
        with open(version_file_path) as f:
            version_str = f.read().strip().split()[-1]
            return tuple(int(field) for field in version_str.split("."))
    except FileNotFoundError:
        pass

    cmd = [os.path.join(cuda_path, "bin", "nvcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [line for line in out.split("\n") if "release" in line][0]
        release_fields = [s.strip() for s in release_line.split(",")]
        version_str = [f[1:] for f in release_fields if f.startswith("V")][0]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read cuda version file")


def find_nvshmem_paths() -> Tuple[str, str]:
    """
    Searches for the NVSHMEM include and library directories.

    Returns
    -------
    A tuple containing the path to the include directory and the library directory.
    """
    candidate_roots = []

    # 1. NVSHMEM_HOME env variable
    if "NVSHMEM_HOME" in os.environ:
        candidate_roots.append(os.environ["NVSHMEM_HOME"])

    # 2. CUDA Toolkit
    try:
        cuda_home = find_cuda_path()
        candidate_roots.append(cuda_home)
    except RuntimeError:
        pass

    # 3. Other common system installation paths
    candidate_roots.extend(["/usr/local", "/usr"])

    seen = set()
    unique_candidates = []
    for path in candidate_roots:
        if path and path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    for root in unique_candidates:
        # Check both standard include path and versioned subdirectories (e.g., nvshmem_12)
        include_paths_to_check = [os.path.join(root, "include")]

        # Add versioned subdirectories like include/nvshmem_*
        versioned_includes = glob.glob(os.path.join(root, "include", "nvshmem_*"))
        include_paths_to_check.extend(versioned_includes)

        # Check standard and architecture-specific lib directories
        lib_paths_to_check = [
            os.path.join(root, "lib64"),
            os.path.join(root, "lib"),
        ]

        # Add architecture-specific lib paths (e.g., lib/x86_64-linux-gnu)
        machine = platform.machine()
        system = platform.system().lower()
        lib_paths_to_check.extend(
            [
                os.path.join(root, "lib", f"{machine}-{system}-gnu"),
                os.path.join(root, "lib64", f"{machine}-{system}-gnu"),
            ]
        )

        for include_path in include_paths_to_check:
            if os.path.isfile(os.path.join(include_path, "nvshmem.h")):
                for lib_path in lib_paths_to_check:
                    # Check for both static (.a) and shared (.so) libraries
                    if os.path.isfile(os.path.join(lib_path, "libnvshmem.a")) or os.path.isfile(
                        os.path.join(lib_path, "libnvshmem.so")
                    ):
                        return include_path, lib_path

    error_message = [
        "Error: Could not find NVSHMEM installation.",
        "Searched in the following locations:",
    ]
    error_message.extend([f"  - {path}" for path in unique_candidates])
    error_message.extend(
        [
            "",
            "Please ensure NVSHMEM is installed and try one of the following:",
            (
                "  1. Set the 'NVSHMEM_HOME' environment variable "
                "to your NVSHMEM installation directory."
            ),
            (
                "  2. Ensure your CUDA Toolkit installation includes NVSHMEM and "
                "'nvcc' is on your PATH."
            ),
        ]
    )
    raise RuntimeError("\n".join(error_message))


@tvm_ffi.register_global_func
def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
    """
    Compile CUDA code using the configured backend (nvcc or nvrtc).

    This callback is invoked by TVM's C++ backend during CUDA module compilation.
    By default, uses nvcc to generate fatbin.

    Environment Variables
    ---------------------
    TVM_CUDA_COMPILE_MODE : str
        Compiler backend: "nvcc" (default) or "nvrtc"
        - "nvcc": Use nvcc subprocess, generates fatbin
        - "nvrtc": Use NVRTC via cuda-python for faster JIT, generates cubin

    Parameters
    ----------
    code : str
        CUDA source code to compile
    target : Target
        TVM target architecture

    Returns
    -------
    bytes
        Compiled binary (fatbin for nvcc, cubin for nvrtc)
    """
    compiler = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvcc").lower()

    if compiler == "nvrtc":
        return compile_cuda(code, target_format="cubin", compiler="nvrtc")
    if compiler == "nvcc":
        return compile_cuda(code, target_format="fatbin", compiler="nvcc")

    raise ValueError(f"Invalid TVM_CUDA_COMPILE_MODE: {compiler}. Expected 'nvcc' or 'nvrtc'.")


@tvm_ffi.register_global_func("tvm_callback_libdevice_path")
def find_libdevice_path(arch):
    """Utility function to find libdevice

    Parameters
    ----------
    arch : int
        The compute architecture in int

    Returns
    -------
    path : str
        Path to libdevice.
    """
    cuda_path = find_cuda_path()
    lib_path = os.path.join(cuda_path, "nvvm/libdevice")
    if not os.path.exists(lib_path):
        # Debian/Ubuntu repackaged CUDA path
        lib_path = os.path.join(cuda_path, "lib/nvidia-cuda-toolkit/libdevice")
    selected_ver = 0
    selected_path = None
    cuda_ver = get_cuda_version(cuda_path)
    major_minor = (cuda_ver[0], cuda_ver[1])
    if major_minor in (
        (9, 0),
        (9, 1),
        (10, 0),
        (10, 1),
        (10, 2),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
    ):
        path = os.path.join(lib_path, "libdevice.10.bc")
    else:
        for fn in os.listdir(lib_path):
            if not fn.startswith("libdevice"):
                continue

            try:
                # expected pattern: libdevice.${ARCH}.10.bc
                #             e.g., libdevice.compute_20.10.bc
                ver = int(fn.split(".")[-3].split("_")[-1])
                if selected_ver < ver <= arch:
                    selected_ver = ver
                    selected_path = fn
            except ValueError:
                # it can just be `libdevice.10.bc` in CUDA 10
                selected_path = fn

        if selected_path is None:
            raise RuntimeError(f"Cannot find libdevice for arch {arch}")
        path = os.path.join(lib_path, selected_path)
    return path


def callback_libdevice_path(arch):
    try:
        return find_libdevice_path(arch)
    except RuntimeError:
        warnings.warn("Cannot find libdevice path")
        return ""


@tvm_ffi.register_global_func("tvm.contrib.nvcc.get_compute_version")
def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.6")
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and target.arch:
        arch = target.arch.split("_")[1]
        if len(arch) < 2:
            raise ValueError(f"The arch is not expected {target.arch}")
        if arch[-1].isalpha():
            # This is for arch like "sm_90a"
            suffix = arch[-1]
            major = arch[:-2]
            minor = arch[-2]
            return major + "." + minor + "." + suffix
        return arch[:-1] + "." + arch[-1]

    # 3. GPU compute version
    if tvm.cuda(0).exist:
        return tvm.cuda(0).compute_version

    raise ValueError(
        "No CUDA architecture was specified or GPU detected."
        "Try specifying it by adding '-arch=sm_xx' to your target."
    )


def parse_compute_version(compute_version):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error: " + str(err))


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, minor = parse_compute_version(compute_version)
    # fp 16 support in reference to:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions
    if major == 5 and minor == 3:
        return True
    if major >= 6:
        return True

    return False


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 6:
        return True

    return False


def have_tensorcore(compute_version=None, target=None):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    target : tvm.target.Target, optional
        The compilation target, will be used to determine arch if compute_version
        isn't specified.
    """
    if compute_version is None:
        if tvm.cuda(0).exist:
            compute_version = tvm.cuda(0).compute_version
        else:
            if target is None or "arch" not in target.attrs:
                warnings.warn(
                    "Tensorcore will be disabled due to no CUDA architecture specified."
                    "Try specifying it by adding '-arch=sm_xx' to your target."
                )
                return False
            compute_version = target.attrs["arch"]
            # Compute version will be in the form "sm_{major}{minor}"
            major, minor = compute_version.split("_")[1]
            compute_version = major + "." + minor
    major, _ = parse_compute_version(compute_version)
    if major >= 7:
        return True

    return False


def have_cudagraph():
    """Either CUDA Graph support is provided"""
    try:
        cuda_ver = get_cuda_version()
        if cuda_ver < (10, 0):
            return False
        return True
    except RuntimeError:
        return False


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_bf16")
def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 8:
        return True

    return False


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_fp8")
def have_fp8(compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    major, minor = parse_compute_version(compute_version)
    # fp8 is suppored in Ada Lovelace (8.9) or later architectures.
    if major == 8 and minor == 9:
        return True
    if major >= 9:
        return True
    return False


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_fp4")
def have_fp4(compute_version):
    """Whether fp4 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    major, minor = parse_compute_version(compute_version)
    # fp4 is suppored in Blackwell (10.0) or later architectures.
    if major == 10 and minor == 0:
        return True
    return False
