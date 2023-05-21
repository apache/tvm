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
"""Util to invoke clang in the system."""
# pylint: disable=invalid-name
import subprocess

from tvm._ffi.base import py_str
import tvm.target
from . import utils


def find_clang(required=True):
    """Find clang in system.

    Parameters
    ----------
    required : bool
        Whether it is required,
        runtime error will be raised if the compiler is required.

    Returns
    -------
    valid_list : list of str
        List of possible paths.

    Note
    ----
    This function will first search clang that
    matches the major llvm version that built with tvm
    """
    cc_list = []
    major = tvm.target.codegen.llvm_version_major(allow_none=True)
    if major is not None:
        cc_list += [f"clang-{major}.0"]
        cc_list += [f"clang-{major}"]
    cc_list += ["clang"]
    cc_list += ["clang.exe"]
    valid_list = [utils.which(x) for x in cc_list]
    valid_list = [x for x in valid_list if x]
    if not valid_list and required:
        raise RuntimeError("cannot find clang, candidates are: " + str(cc_list))
    return valid_list


def create_llvm(inputs, output=None, options=None, cc=None):
    """Create llvm text ir.

    Parameters
    ----------
    inputs : list of str
        List of input files name or code source.

    output : str, optional
        Output file, if it is none
        a temporary file is created

    options : list
        The list of additional options string.

    cc : str, optional
        The clang compiler, if not specified,
        we will try to guess the matched clang version.

    Returns
    -------
    code : str
        The generated llvm text IR.
    """
    cc = cc if cc else find_clang()[0]
    cmd = [cc]
    cmd += ["-S", "-emit-llvm"]
    temp = utils.tempdir()
    output = output if output else temp.relpath("output.ll")
    inputs = [inputs] if isinstance(inputs, str) else inputs
    input_files = []
    for i, code in enumerate(inputs):
        if utils.is_source_path(code):
            input_files.append(code)
        else:
            temp_path = temp.relpath(f"input{i}.cc")
            with open(temp_path, "w") as output_file:
                output_file.write(code)
            input_files.append(temp_path)
    if options:
        cmd += options
    cmd += ["-o", output]
    cmd += input_files
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    return open(output).read()
