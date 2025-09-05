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

from typing import Sequence, Optional, Mapping
import os
import sys
import glob
import hashlib
import shutil
import subprocess
import functools

from tvm_ffi.module import Module, load_module
from tvm_ffi.utils import FileLock
from tvm_ffi.libinfo import find_include_path, find_dlpack_include_path

IS_WINDOWS = sys.platform == "win32"


def _hash_sources(
    cpp_source: str,
    cuda_source: str,
    cpp_functions: Mapping[str, str],
    cuda_functions: Mapping[str, str],
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> str:
    """Generate a unique hash for the given sources and functions."""
    m = hashlib.sha256()
    m.update(cpp_source.encode("utf-8"))
    m.update(cuda_source.encode("utf-8"))
    for name, doc in sorted(cpp_functions.items()):
        m.update(name.encode("utf-8"))
        m.update(doc.encode("utf-8"))
    for name, doc in sorted(cuda_functions.items()):
        m.update(name.encode("utf-8"))
        m.update(doc.encode("utf-8"))
    for flag in extra_cflags:
        m.update(flag.encode("utf-8"))
    for flag in extra_cuda_cflags:
        m.update(flag.encode("utf-8"))
    for flag in extra_ldflags:
        m.update(flag.encode("utf-8"))
    for path in extra_include_paths:
        m.update(path.encode("utf-8"))
    return m.hexdigest()[:16]


def _maybe_write(path: str, content: str) -> None:
    """Write content to path if it does not already exist with the same content."""
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_content = f.read()
        if existing_content == content:
            return
    with open(path, "w") as f:
        f.write(content)


@functools.lru_cache
def _find_cuda_home() -> Optional[str]:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                raise RuntimeError(
                    "Could not find CUDA installation. "
                    "Please set CUDA_HOME environment variable."
                )
    return cuda_home


def _get_cuda_target() -> str:
    """Get the CUDA target architecture flag."""
    if "TVM_FFI_CUDA_ARCH_LIST" in os.environ:
        arch_list = os.environ["TVM_FFI_CUDA_ARCH_LIST"].split()  # e.g., "8.9 9.0a"
        flags = []
        for arch in arch_list:
            if len(arch.split(".")) != 2:
                raise ValueError(f"Invalid CUDA architecture: {arch}")
            major, minor = arch.split(".")
            flags.append(f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}")
        return " ".join(flags)
    else:
        #
        try:
            status = subprocess.run(
                args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                check=True,
            )
            compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
            major, minor = compute_cap.split(".")
            return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        except Exception:
            # fallback to a reasonable default
            return "-gencode=arch=compute_70,code=sm_70"


def _generate_ninja_build(
    name: str,
    build_dir: str,
    with_cuda: bool,
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> str:
    """Generate the content of build.ninja for building the module."""
    default_include_paths = [find_include_path(), find_dlpack_include_path()]

    if IS_WINDOWS:
        default_cflags = ["/std:c++17"]
        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]
        default_ldflags = ["/DLL"]
    else:
        default_cflags = ["-std=c++17", "-fPIC", "-O2"]
        default_cuda_cflags = ["-Xcompiler", "-fPIC", "-std=c++17", "-O2"]
        default_ldflags = ["-shared"]

        if with_cuda:
            # determine the compute capability of the current GPU
            default_cuda_cflags += [_get_cuda_target()]
            default_ldflags += ["-L{}".format(os.path.join(_find_cuda_home(), "lib64")), "-lcudart"]

    cflags = default_cflags + [flag.strip() for flag in extra_cflags]
    cuda_cflags = default_cuda_cflags + [flag.strip() for flag in extra_cuda_cflags]
    ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
    include_paths = default_include_paths + [os.path.abspath(path) for path in extra_include_paths]

    # append include paths
    for path in include_paths:
        cflags.append("-I{}".format(path))
        cuda_cflags.append("-I{}".format(path))

    # flags
    ninja = []
    ninja.append("ninja_required_version = 1.3")
    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))
    ninja.append("cflags = {}".format(" ".join(cflags)))
    if with_cuda:
        ninja.append("nvcc = {}".format(os.path.join(_find_cuda_home(), "bin", "nvcc")))
        ninja.append("cuda_cflags = {}".format(" ".join(cuda_cflags)))
    ninja.append("ldflags = {}".format(" ".join(ldflags)))

    # rules
    ninja.append("")
    ninja.append("rule compile")
    ninja.append("  depfile = $out.d")
    ninja.append("  deps = gcc")
    ninja.append("  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out")
    ninja.append("")

    if with_cuda:
        ninja.append("rule compile_cuda")
        ninja.append("  depfile = $out.d")
        ninja.append("  deps = gcc")
        ninja.append(
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out"
        )
        ninja.append("")

    ninja.append("rule link")
    ninja.append("  command = $cxx $in $ldflags -o $out")
    ninja.append("")

    # build targets
    ninja.append(
        "build main.o: compile {}".format(os.path.abspath(os.path.join(build_dir, "main.cpp")))
    )
    if with_cuda:
        ninja.append(
            "build cuda.o: compile_cuda {}".format(
                os.path.abspath(os.path.join(build_dir, "cuda.cu"))
            )
        )
    ninja.append("build {}.so: link main.o{}".format(name, " cuda.o" if with_cuda else ""))
    ninja.append("")

    # default target
    ninja.append("default {}.so".format(name))
    ninja.append("")
    return "\n".join(ninja)


def _build_ninja(build_dir: str) -> None:
    """Build the module in the given build directory using ninja."""
    command = ["ninja", "-v"]
    num_workers = os.environ.get("MAX_JOBS", None)
    if num_workers is not None:
        command += ["-j", num_workers]
    status = subprocess.run(args=command, cwd=build_dir, capture_output=True)
    if status.returncode != 0:
        msg = ["ninja exited with status {}".format(status.returncode)]
        if status.stdout:
            msg.append("stdout:\n{}".format(status.stdout.decode("utf-8")))
        if status.stderr:
            msg.append("stderr:\n{}".format(status.stderr.decode("utf-8")))

        raise RuntimeError("\n".join(msg))


def _decorate_with_tvm_ffi(source: str, functions: Mapping[str, str]) -> str:
    """Decorate the given source code with TVM FFI export macros."""
    sources = [
        "#include <tvm/ffi/dtype.h>",
        "#include <tvm/ffi/error.h>",
        "#include <tvm/ffi/extra/c_env_api.h>",
        "#include <tvm/ffi/function.h>",
        "",
        source,
    ]

    for exported_name, func_name_in_source in functions.items():
        sources.append(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({exported_name}, {func_name_in_source});")
    sources.append("")

    return "\n".join(sources)


def load_inline(
    name: str,
    *,
    cpp_source: str | None = None,
    cuda_source: str | None = None,
    cpp_functions: Mapping[str, str] | None = None,
    cuda_functions: Mapping[str, str] | None = None,
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    extra_ldflags: Sequence[str] | None = None,
    extra_include_paths: Sequence[str] | None = None,
) -> Module:
    """Compile and load a C++/CUDA tvm ffi module from inline source code.

    This function compiles the given C++ and/or CUDA source code into a shared library. Both cpp_source and cuda_source
    are compiled to an object file, and then linked together into a shared library. It's possible to only provide
    cpp_source or cuda_source.

    The `cpp_functions` and `cuda_functions` parameters are used to specify which functions in the source code
    should be exported to the tvm ffi module. The keys of the mapping are the names of the exported functions, and the
    values are the names of the functions in the source code. The exported name and the function name in the source code
    must be different. The exported name must be a valid C identifier while the function name in the source code can
    contain namespace qualifiers.

    Extra compiler and linker flags can be provided via the `extra_cflags`, `extra_cuda_cflags`, and `extra_ldflags`
    parameters. The default flags are generally sufficient for most use cases, but you may need to provide additional
    flags for your specific use case.

    The include dir of tvm ffi and dlpack are used by default for linker to find the headers. Thus, you can include
    any header from tvm ffi and dlpack in your source code. You can also provide additional include paths via the
    `extra_include_paths` parameter and include custom headers in your source code.

    The compiled shared library is cached in a cache directory to avoid recompilation. The cache directory can be
    specified via the `TVM_FFI_CACHE_DIR` environment variable. If not specified, the default cache directory is
    `~/.cache/tvm-ffi`.

    Parameters
    ----------
    name: str
        The name of the tvm ffi module.
    cpp_source: str, optional
        The C++ source code.
    cuda_source: str, optional
        The CUDA source code.
    cpp_functions: Mapping[str, str], optional
        The mapping from the exported function name to the function name in the C++ source code.
    cuda_functions: Mapping[str, str], optional
        The mapping from the exported function name to the function name in the CUDA source code.
    extra_cflags: Sequence[str], optional
        The extra compiler flags for C++ compilation.
        The default flags are:
        - On Linux/macOS: ['-std=c++17', '-fPIC', '-O2']
        - On Windows: ['/std:c++17']
    extra_cuda_cflags:
        The extra compiler flags for CUDA compilation.
        The default flags are:
        - On Linux/macOS: ['-Xcompiler', '-fPIC', '-std=c++17', '-O2']
        - On Windows: ['-Xcompiler', '/std:c++17', '/O2']
    extra_ldflags: Sequence[str], optional
        The extra linker flags.
        The default flags are:
        - On Linux/macOS: ['-shared']
        - On Windows: ['/DLL']
    extra_include_paths: Sequence[str], optional
        The extra include paths.
        The default include paths are:
        - The include path of tvm ffi
    Returns
    -------
    mod: Module
        The loaded tvm ffi module.
    """
    if cpp_source is None:
        cpp_source = ""
    if cuda_source is None:
        cuda_source = ""
    if cpp_functions is None:
        cpp_functions = {}
    if cuda_functions is None:
        cuda_functions = {}
    extra_ldflags = extra_ldflags or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_include_paths = extra_include_paths or []

    # whether we have cuda source in this module
    with_cuda = len(cuda_source.strip()) > 0

    # add function registration code to sources
    cpp_source = _decorate_with_tvm_ffi(cpp_source, cpp_functions)
    cuda_source = _decorate_with_tvm_ffi(cuda_source, cuda_functions)

    # determine the cache dir for the built module
    cache_dir = os.path.join(
        os.environ.get("TVM_FFI_CACHE_DIR", os.path.expanduser("~/.cache/tvm-ffi"))
    )
    source_hash: str = _hash_sources(
        cpp_source,
        cuda_source,
        cpp_functions,
        cuda_functions,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
    )
    build_dir: str = os.path.join(cache_dir, "{}_{}".format(name, source_hash))
    os.makedirs(build_dir, exist_ok=True)

    # generate build.ninja
    ninja_source = _generate_ninja_build(
        name=name,
        build_dir=build_dir,
        with_cuda=with_cuda,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
    )

    with FileLock(os.path.join(build_dir, "lock")):
        # write source files and build.ninja if they do not already exist
        _maybe_write(os.path.join(build_dir, "main.cpp"), cpp_source)
        if with_cuda:
            _maybe_write(os.path.join(build_dir, "cuda.cu"), cuda_source)
        _maybe_write(os.path.join(build_dir, "build.ninja"), ninja_source)

        # build the module
        _build_ninja(build_dir)

        return load_module(os.path.join(build_dir, "{}.so".format(name)))
