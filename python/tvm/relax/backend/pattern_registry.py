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

"""Pattern registry for BYOC backends"""

import atexit
from typing import Callable, List, Mapping, Optional, Set, Tuple, Union

from tvm.relax.dpl import DFPattern
from tvm.relax.transform import FusionPattern

from ..expr import Expr
from . import _ffi_api

_REGISTERED_PATTERN_NAMES: Set[str] = set()


def _cleanup_registered_patterns():
    _ffi_api.RemovePatterns(list(_REGISTERED_PATTERN_NAMES))  # type: ignore # pylint: disable=no-member


_CLEANUP_REGISTERED = False


def _ensure_cleanup_function_registered():
    """
    Add a cleanup function to be called on interpreter termination, to remove all
    patterns registered on the Python side. Without cleaning up those patterns,
    program will segfault on termination. It's because the check functions of pattern
    entries are referenced from the static memory of libtvm, thus they will be cleaned
    up at the very end, making calls to Py_DecRef after Python interpreter terminates.
    """
    global _CLEANUP_REGISTERED  # pylint: disable=global-statement

    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_registered_patterns)
        _CLEANUP_REGISTERED = True


CheckFunc = Callable[[Mapping[DFPattern, Expr], Expr], bool]
Pattern = Union[
    FusionPattern,
    Tuple[str, DFPattern],
    Tuple[str, DFPattern, Mapping[str, DFPattern]],
    Tuple[str, DFPattern, Mapping[str, DFPattern], CheckFunc],
]


def register_patterns(patterns: List[Pattern]):
    """
    Register patterns which will be used to partition the DataflowBlock into
    subgraphs that are supported by external backends.

    Parameters
    ----------
    patterns: List[Pattern]
        Patterns to be registered. Patterns that appear later in the list have
        higher priority when partitioning DataflowBlock.
    """
    _ensure_cleanup_function_registered()

    entries = []
    for item in patterns:
        if isinstance(item, FusionPattern):
            entries.append(item)
        elif isinstance(item, tuple):
            entries.append(FusionPattern(*item))
            _REGISTERED_PATTERN_NAMES.add(item[0])
        else:
            raise TypeError(f"Cannot register type {type(item)} as pattern")
    _ffi_api.RegisterPatterns(entries)


def get_patterns_with_prefix(prefix: str) -> List[FusionPattern]:
    """
    Get a list of patterns whose names startwith `prefix`.

    Parameters
    ----------
    prefix: str
        The prefix of pattern name.

    Returns
    -------
    patterns: FusionPattern
        Matched patterns, ordered by priority from high to low.
    """
    return _ffi_api.GetPatternsWithPrefix(prefix)


def get_pattern(name: str) -> Optional[FusionPattern]:
    """
    Find the pattern with a particular name.

    Parameters
    ----------
    name: str
        The pattern name.

    Returns
    -------
    pattern: Optional[FusionPattern]
        The matched pattern. Returns None if such pattern is not found.
    """
    return _ffi_api.GetPattern(name)
