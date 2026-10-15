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
"""Helpers for tests that use exclusive machine resources."""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from tvm_ffi.utils import FileLock

_LOCK_DIR_ENV_VAR = "TVM_TEST_LOCK_DIR"
_LOCK_DIR_NAME = "tvm-testing-locks"
_R = TypeVar("_R")


def _ensure_test_lock_path(filename: str) -> Path:
    lock_dir_override = os.environ.get(_LOCK_DIR_ENV_VAR)
    if lock_dir_override:
        lock_dir = Path(lock_dir_override).expanduser()
    else:
        lock_dir = Path(tempfile.gettempdir()) / _LOCK_DIR_NAME

    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / filename


def run_with_gpu_lock(func: Callable[..., _R], /, *args: Any, **kwargs: Any) -> _R:
    """Run a callable while holding the machine-local GPU lock.

    The lock avoids contentious GPU access that may break GPU-related tests.

    Parameters
    ----------
    func : Callable
        Callable containing the complete live local-GPU lifetime.
    *args : Any
        Positional arguments forwarded to ``func``.
    **kwargs : Any
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    result : Any
        The return value of ``func``.
    """

    lock_path = _ensure_test_lock_path("gpu.lock")
    with FileLock(str(lock_path)):
        return func(*args, **kwargs)
