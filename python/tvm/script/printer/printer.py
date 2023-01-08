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
"""The printer interface"""

from typing import Mapping, Optional

from tvm.runtime.object_path import ObjectPath

from . import _ffi_api


def script(
    obj,
    ir_prefix: Optional[Mapping[str, str]] = None,
    indent_space: int = 4,
    print_line_number: bool = False,
    num_context_lines: int = -1,
    path_to_underline: Optional[ObjectPath] = None,
):
    """Print a TVM IR as a TVMScript text format.

    Parameters
    ----------
    obj : object
        An TVM object representing TVM IR
    ir_prefix : Optional[Mapping[str, str]]
        A mapping from IR type to the prefix of the script.
        Default to {"ir": "I", "tir": T}
    indent_space : int = 4
        The number of spaces to indent
    print_line_number : bool = False
        Whether to print line number
    num_context_lines : int = -1
        The number of context lines to print. -1 means all lines.
    path_to_underline : Optional[ObjectPath]
        The path to underline in the script.

    Returns
    -------
    script : str
        The TVMScript text format
    """
    if ir_prefix is None:
        ir_prefix = {
            "ir": "I",
            "tir": "T",
        }
    return _ffi_api.Script(  # type: ignore # pylint: disable=no-member
        obj, ir_prefix, indent_space, print_line_number, num_context_lines, path_to_underline
    )
