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

from typing import List, Mapping, Optional, Tuple, Union

import tvm
from tvm.relax.dpl import DFPattern
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("relax.backend.PatternRegistryEntry")
class PatternRegistryEntry(Object):
    """
    An entry in the pattern registry. This represents a single pattern that
    can be used to identify expressions that can be handled by external
    backends, like CUTLASS and TensorRT.

    Parameters
    ----------
    name: str
        The name of pattern. Usually it starts with the name of backend, like 'cutlass.matmul'.

    pattern: DFPattern
        The dataflow pattern that will be used to match expressions that can be handled
        by external backends.

    arg_patterns: Mapping[str, DFPattern]
        The mapping from arg name to its pattern. It can be used to extract arg expression
        from match result. All DFPattern in this map should be part of the `pattern`.
    """

    name: str
    pattern: DFPattern
    arg_patterns: Mapping[str, DFPattern]

    def __init__(self, name: str, pattern: DFPattern, arg_patterns: Mapping[str, DFPattern]):
        self.__init_handle_by_constructor__(
            _ffi_api.PatternRegistryEntry, name, pattern, arg_patterns  # type: ignore
        )


Pattern = Union[
    PatternRegistryEntry,
    Tuple[str, DFPattern],
    Tuple[str, Tuple[DFPattern, Mapping[str, DFPattern]]],
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
    entries = []
    for item in patterns:
        if isinstance(item, PatternRegistryEntry):
            entries.append(item)
        elif isinstance(item, tuple):
            name, pattern_or_tuple = item
            if isinstance(pattern_or_tuple, tuple):
                pattern, arg_patterns = pattern_or_tuple
            else:
                pattern, arg_patterns = pattern_or_tuple, {}
            entries.append(PatternRegistryEntry(name, pattern, arg_patterns))
        else:
            raise TypeError(f"Cannot register type {type(pattern)} as pattern")
    _ffi_api.RegisterPatterns(entries)


def get_patterns_with_prefix(prefix: str) -> List[PatternRegistryEntry]:
    """
    Get a list of patterns whose names startwith `prefix`.

    Parameters
    ----------
    prefix: str
        The prefix of pattern name.

    Returns
    -------
    patterns: PatternRegistryEntry
        Matched patterns, ordered by priority from high to low.
    """
    return _ffi_api.GetPatternsWithPrefix(prefix)


def get_pattern(name: str) -> Optional[PatternRegistryEntry]:
    """
    Find the pattern with a particular name.

    Parameters
    ----------
    name: str
        The pattern name.

    Returns
    -------
    pattern: Optional[PatternRegistryEntry]
        The matched pattern. Returns None if such pattern is not found.
    """
    return _ffi_api.GetPattern(name)
