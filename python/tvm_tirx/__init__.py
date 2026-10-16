# isort: skip_file
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
# pylint: disable=redefined-builtin, wildcard-import
"""Tensor-level intermediate representations, compilation, and runtime execution."""

import sys
import os
from typing import TYPE_CHECKING


def _is_config_mode():
    """Keep the configuration CLI independent of native library loading."""
    argv = getattr(sys, "orig_argv", ())
    return os.path.basename(sys.argv[0]) in ("tvm-tirx-config", "tvm-tirx-config.exe") or any(
        arg == "-m" and argv[index + 1 : index + 2] == ["tvm_tirx.config"]
        for index, arg in enumerate(argv)
    )


if TYPE_CHECKING or not _is_config_mode():
    import multiprocessing
    import os

    # ffi module must load first
    import tvm_ffi as _tvm_ffi

    # top-level alias
    from .libinfo import __version__
    from .base import _RUNTIME_ONLY

    # tvm.runtime
    from .runtime._tensor import cpu, cuda, opencl, vulkan, metal
    from .runtime._tensor import vpi, rocm, ext_dev
    from .runtime import DataTypeCode

    # tvm.ir
    from .ir import IRModule, transform
    from . import ir

    # tvm.script — must be imported before any dialect package so that
    # tvm.script.register_dialect is reachable when dialect __init__.py files run.
    from . import script

    # tvm.tirx — registers itself via tvm.script.register_dialect in its __init__
    from . import tirx
    from .ir import prim

    # tvm.backend — owns backend Python load hooks
    from . import backend

    # tvm.target
    from . import target
    from .target import device_from_target

    # Shared code generation helpers
    from . import codegen

    # others
    from . import sym

    # support infra
    from . import support

    # Side-effect imports: register CUDA/ROCm FFI callbacks at TVM startup
    from .support import rocm as _rocm, nvcc as _nvcc

    def tvm_wrap_excepthook(exception_hook):
        """Wrap given excepthook with TVM additional work."""

        def wrapper(exctype, value, trbk):
            """Clean subprocesses when TVM is interrupted."""
            exception_hook(exctype, value, trbk)

            if hasattr(multiprocessing, "active_children"):
                # pylint: disable=not-callable
                for p in multiprocessing.active_children():
                    p.terminate()

        return wrapper

    sys.excepthook = tvm_wrap_excepthook(sys.excepthook)

    # Autoload loads built-in and out-of-tree backends. Out-of-tree extensions opt
    # into being loaded automatically at ``import tvm`` time by declaring an entry
    # point in the ``tvm.backends`` group:
    # [project.entry-points."tvm.backends"] tvm_foo = "tvm_foo:_autoload".
    # Autoload can be disabled via ``TVM_DEVICE_BACKEND_AUTOLOAD=0``.
    from .backend._autoload_backends import _autoload_backends

    _autoload_backends()

    # Load the optional ORC JIT extension for its global FFI registrations.  LLVM
    # function lookup reports a targeted installation error if this import fails;
    # importing TVM itself remains silent for AOT-only workflows.
    try:
        import tvm_ffi_orcjit as _tvm_ffi_orcjit
    except Exception:  # pylint: disable=broad-exception-caught
        pass
