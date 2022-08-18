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
"""Functions to print doc into text format"""

from typing import Optional
from tvm.runtime.object_path import ObjectPath
from . import _ffi_api
from .doc import Doc


def to_python_script(
    doc: Doc,
    indent_spaces: int = 4,
    print_line_numbers: bool = False,
    num_context_lines: Optional[int] = None,
    path_to_underline: Optional[ObjectPath] = None,
) -> str:
    """Convert Doc into Python script.

    Parameters
    ----------
    doc : Doc
        The doc to convert into Python script
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
        The text representation of Doc in Python syntax
    """
    if num_context_lines is None:
        num_context_lines = -1
    return _ffi_api.DocToPythonScript(  # type: ignore
        doc, indent_spaces, print_line_numbers, num_context_lines, path_to_underline
    )
