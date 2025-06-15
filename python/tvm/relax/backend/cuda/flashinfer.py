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

"""FlashInfer JIT compilation module for CUDA backend"""
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import tvm
from tvm.target import Target


def _compile_flashinfer_kernels(
    name: str, source_paths: List[Path], target: Target, num_threads: int
) -> List[Path]:
    from flashinfer.jit.env import (  # pylint: disable=import-outside-toplevel
        CUTLASS_INCLUDE_DIRS,
        FLASHINFER_CSRC_DIR,
        FLASHINFER_INCLUDE_DIR,
        FLASHINFER_JIT_DIR,
        FLASHINFER_TVM_BINDING_DIR,
    )

    # ------------------------------------------------------------------------
    # Caching Flow: create build_directory and compute cache hash.
    # ------------------------------------------------------------------------
    build_directory = FLASHINFER_JIT_DIR / name
    build_directory.mkdir(parents=True, exist_ok=True)

    def get_object_file_path(src: Path) -> Path:
        obj_name = src.stem + ".o"
        obj_path = build_directory / obj_name
        return obj_path

    # Compute latest modification time among all source files
    latest_src_mtime = max(src.stat().st_mtime for src in source_paths)

    # Get modification time for the current file (the one that contains this function)
    current_file_mtime = Path(__file__).stat().st_mtime

    # Build the hash key from metadata
    hash_key = {
        "name": name,
        "target": str(target),
        "latest_src_mtime": latest_src_mtime,
        "current_file_mtime": current_file_mtime,
    }

    hash_value = hashlib.md5(
        json.dumps(hash_key, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()

    # Check if a valid hash exists in the build directory
    hash_file = build_directory / "hash.md5"
    if hash_file.exists():
        with open(hash_file, "r") as f:
            cached_hash = f.read().strip()
        if cached_hash == hash_value:
            # Check that all object files exist
            object_files = []
            all_exist = True
            for src in source_paths:
                obj_path = get_object_file_path(src)
                if not obj_path.exists():
                    all_exist = False
                    break
                object_files.append(obj_path)
            if all_exist:
                return object_files

    # If we are here, cache is missing or outdated. Write the new hash and compile the paths
    with open(hash_file, "w") as f:
        f.write(hash_value)

    # ------------------------------------------------------------------------
    # 1) Common CUDA compile flags
    # ------------------------------------------------------------------------
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads",
        str(num_threads),
        "-g",
        "-use_fast_math",
        "--expt-relaxed-constexpr",
        # DMLC default
        "-DDMLC_USE_FOPEN64=0",
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
        # Enable `-fPIC` for the host compiler
        "-Xcompiler=-fPIC",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]

    # Determine compute version
    compute_version = "".join(tvm.contrib.nvcc.get_target_compute_version(target).split("."))
    if compute_version in ["90"]:
        compute_version += "a"
    cuda_cflags += [
        "-gencode",
        f"arch=compute_{compute_version},code=sm_{compute_version}",
    ]

    # ------------------------------------------------------------------------
    # 2) Include paths
    # ------------------------------------------------------------------------
    tvm_home = os.environ["TVM_SOURCE_DIR"]
    include_paths = [
        FLASHINFER_INCLUDE_DIR,
        FLASHINFER_CSRC_DIR,
        FLASHINFER_TVM_BINDING_DIR,
        Path(tvm_home).resolve() / "include",
        Path(tvm_home).resolve() / "ffi" / "include",
        Path(tvm_home).resolve() / "3rdparty" / "dlpack" / "include",
        Path(tvm_home).resolve() / "3rdparty" / "dmlc-core" / "include",
    ] + CUTLASS_INCLUDE_DIRS

    # ------------------------------------------------------------------------
    # 3) Function to compile a single source file
    # ------------------------------------------------------------------------
    def compile_single_source(src: Path) -> Path:
        # Derive the .o filename from the source filename
        obj_path = get_object_file_path(src)

        # Construct the command
        cmd = (
            ["nvcc"]
            + cuda_cflags
            + [f"-I{inc_path}" for inc_path in include_paths]
            + ["-c", "-o", str(obj_path), str(src)]
        )

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"FlashInfer JIT compilation failed for {src}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{out.decode('utf-8')}\n"
                f"stderr:\n{err.decode('utf-8')}"
            )
        return obj_path

    # ------------------------------------------------------------------------
    # 4) Compile each source in parallel using ThreadPoolExecutor
    # ------------------------------------------------------------------------
    object_files = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compile_single_source, src) for src in source_paths]
        for f in futures:
            object_files.append(f.result())  # Will raise if there's a compilation error

    # Return list of generated object files for any further linking steps
    return object_files


def _load_flashinfer_modules(object_files: List[Path]) -> List[tvm.runtime.Module]:
    return [
        tvm.runtime.load_static_library(str(obj_path.absolute()), func_names=[])
        for obj_path in object_files
    ]


def gen_flashinfer_prefill_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    qk_head_dim: int,
    v_head_dim: int,
    target: Target,
    enable_inline_rope: bool = True,
    num_threads: int = 8,
) -> List[tvm.runtime.Module]:
    """Generate a FlashInfer module for prefill.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    qk_head_dim : int
        The head dimension of the query and key tensors.
    v_head_dim : int
        The head dimension of the value tensor.
    target : Target
        The target device to compile for.
    enable_inline_rope : bool
        Whether to enable inline rotary positional embedding.
    num_threads : int
        The number of threads to use for compilation.

    Returns
    -------
    A list of compiled static library modules for FlashInfer prefill kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_customize_batch_prefill_tvm_binding,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    if enable_inline_rope and qk_head_dim != v_head_dim:
        raise ValueError("Inline rope mode is not supported when qk_head_dim == v_head_dim")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    # Todo(tvm-team): decide which backend ("fa2/fa3") to use
    backend = "fa2"
    variant_name = (
        "DefaultAttention<false, false, false, false>"
        if backend == "fa2"
        else "DefaultAttention<false>"
    )
    variant_decl = (
        "#include <flashinfer/attention/variants.cuh>"
        if backend == "fa2"
        else "#include <flashinfer/attention/hopper/variants.cuh>"
    )
    jit_args = {
        "backend": backend,
        "uri": f"batch_prefill_tvm_dtype_q_{dtype_q}_"
        + f"dtype_kv_{dtype_kv}_"
        + f"dtype_o_{dtype_o}_"
        + f"qk_head_dim_{qk_head_dim}_"
        + f"v_head_dim_{v_head_dim}_"
        + f"enable_inline_rope_{enable_inline_rope}",
        "dtype_q": torch_dtype_q,
        "dtype_kv": torch_dtype_kv,
        "dtype_o": torch_dtype_o,
        "idtype": torch.int32,
        "head_dim_qk": qk_head_dim,
        "head_dim_vo": v_head_dim,
        "additional_tensor_names": [],
        "additional_tensor_dtypes": [],
        "additional_scalar_names": ["sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        "additional_scalar_dtypes": ["double", "double", "double"],
        "variant_name": variant_name,
        "variant_decl": variant_decl,
        "enable_inline_rope": enable_inline_rope,
    }
    uri, source_paths = gen_customize_batch_prefill_tvm_binding(**jit_args)
    object_files = _compile_flashinfer_kernels(uri, source_paths, target, num_threads)
    modules = _load_flashinfer_modules(object_files)
    return modules


def gen_flashinfer_decode_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    qk_head_dim: int,
    v_head_dim: int,
    target: Target,
    num_threads: int = 8,
) -> List[tvm.runtime.Module]:
    """Generate a FlashInfer module for decode.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    qk_head_dim : int
        The head dimension of the query and key tensors.
    v_head_dim : int
        The head dimension of the value tensor.
    target : Target
        The target device to compile for.
    num_threads : int
        The number of threads to use for compilation.

    Returns
    -------
    A list of compiled static library modules for FlashInfer decode kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_customize_batch_decode_tvm_binding,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    jit_args = {
        "uri": f"batch_decode_tvm_dtype_q_{dtype_q}_"
        + f"dtype_kv_{dtype_kv}_"
        + f"dtype_o_{dtype_o}_"
        + f"qk_head_dim_{qk_head_dim}_"
        + f"v_head_dim_{v_head_dim}",
        "dtype_q": torch_dtype_q,
        "dtype_kv": torch_dtype_kv,
        "dtype_o": torch_dtype_o,
        "idtype": torch.int32,
        "head_dim_qk": qk_head_dim,
        "head_dim_vo": v_head_dim,
        "additional_tensor_names": [],
        "additional_tensor_dtypes": [],
        "additional_scalar_names": ["sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        "additional_scalar_dtypes": ["double", "double", "double"],
        "variant_name": "DefaultAttention<false, false, false, false>",
        "variant_decl": "#include <flashinfer/attention/variants.cuh>",
    }
    uri, source_paths = gen_customize_batch_decode_tvm_binding(**jit_args)
    object_files = _compile_flashinfer_kernels(uri, source_paths, target, num_threads)
    modules = _load_flashinfer_modules(object_files)
    return modules


def gen_flashinfer_mla_module(
    dtype_q: str,
    dtype_kv: str,
    dtype_o: str,
    head_dim_ckv: int,
    head_dim_kpe: int,
    target: Target,
    num_threads: int = 8,
) -> List[tvm.runtime.Module]:
    """Generate a FlashInfer module for MLA.

    Parameters
    ----------
    dtype_q : str
        The data type of the query tensor.
    dtype_kv : str
        The data type of the key/value tensors.
    dtype_o : str
        The data type of the output tensor.
    head_dim_ckv : int
        The head dimension of the compressed key/value tensors.
    head_dim_kpe : int
        The head dimension of the query/key positional embedding.
    target : Target
        The target device to compile for.
    num_threads : int
        The number of threads to use for compilation.

    Returns
    -------
    A list of compiled static library modules for FlashInfer MLA kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_batch_mla_tvm_binding,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install PyTorch to use FlashInfer.")

    torch_dtype_q = getattr(torch, dtype_q)
    torch_dtype_kv = getattr(torch, dtype_kv)
    torch_dtype_o = getattr(torch, dtype_o)
    jit_args = {
        "uri": f"batch_mla_tvm_dtype_q_{dtype_q}_"
        + f"dtype_kv_{dtype_kv}_"
        + f"dtype_o_{dtype_o}_"
        + f"head_dim_ckv_{head_dim_ckv}_"
        + f"head_dim_kpe_{head_dim_kpe}",
        "dtype_q": torch_dtype_q,
        "dtype_kv": torch_dtype_kv,
        "dtype_o": torch_dtype_o,
        "dtype_idx": torch.int32,
        "head_dim_ckv": head_dim_ckv,
        "head_dim_kpe": head_dim_kpe,
    }
    uri, source_paths = gen_batch_mla_tvm_binding(**jit_args)
    object_files = _compile_flashinfer_kernels(uri, source_paths, target, num_threads)
    modules = _load_flashinfer_modules(object_files)
    return modules


def gen_sampling_module(target: Target, num_threads: int = 8):
    """
    Generate a FlashInfer module for sampling kernels.

    Parameters
    ----------
    target : Target
        The target device for which the module will be compiled.
    num_threads : int, optional
        The number of threads to use during compilation (default is 8).

    Returns
    -------
    List[tvm.runtime.Module]
        A list of compiled static library modules for the FlashInfer sampling kernels.
    """
    try:
        from flashinfer.jit import (  # pylint: disable=import-outside-toplevel
            gen_sampling_tvm_binding,
        )
    except ImportError:
        raise ImportError(
            "FlashInfer is not installed. Please follow instructions "
            "in https://docs.flashinfer.ai to install FlashInfer."
        )
    uri, source_paths = gen_sampling_tvm_binding(uri="sampling")
    object_files = _compile_flashinfer_kernels(uri, source_paths, target, num_threads)
    modules = _load_flashinfer_modules(object_files)
    return modules
