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
"""
This file contains the entry point of TVMScript Unified Printer.
"""

from typing import Dict, Optional

from tvm.runtime import Object, ObjectPath

from . import _ffi_api


def script(  # pylint: disable=too-many-arguments
    root_node: Object,
    ir_name: str,
    ir_prefix: Dict[str, str],
    indent_spaces: int = 4,
    print_line_numbers: bool = False,
    num_context_lines: int = -1,
    path_to_underline: Optional[ObjectPath] = None,
) -> str:
    """
    Print IR graph as TVMScript code

    Parameters
    ----------
    root_node : Object
        The root node to print.
    ir_name : str
        The dispatch token of the target IR, e.g., "tir", "relax".
    ir_prefix : Dict[str, str]
        The symbol name for TVMScript IR namespaces. For example,
        {"tir": "T"}.
    indent_spaces : int
        The number of indent spaces to use in the output
    print_line_numbers: bool
        Whether to print line numbers
    num_context_lines : Optional[int]
        Number of context lines to print around the underlined text
    path_to_underline : Optional[ObjectPath]
        Object path to be underlined

    Returns
    -------
    script : str
        The TVMScript code of the root_node
    """
    return _ffi_api.Script(  # type: ignore # pylint: disable=no-member
        root_node,
        ir_name,
        ir_prefix,
        indent_spaces,
        print_line_numbers,
        num_context_lines,
        path_to_underline,
    )
