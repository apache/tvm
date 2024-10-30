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

"""Runtime support infra of TVM."""

import re

import tvm._ffi


@tvm._ffi.register_func("tvm.runtime.regex_match")
def _regex_match(regex_pattern: str, match_against: str) -> bool:
    """Check if a pattern matches a regular expression

    This function should be used instead of `std::regex` within C++
    call sites, to avoid ABI incompatibilities with pytorch.

    Currently, the pytorch wheels available through pip install use
    the pre-C++11 ABI by setting `-DUSE_CXX11_ABI=0` [0]. If TVM were to
    user the pre-C++11 ABI, this would cause breakages with
    dynamically-linked LLVM environments.

    Use of the `<regex>` header in TVM should be avoided, as its
    implementation is not supported by gcc's dual ABI. This ABI
    incompatibility results in runtime errors either when `std::regex`
    is called from TVM, or when `std::regex` is called from pytorch,
    depending on which library was loaded first.  This restriction can
    be removed when a version of pytorch compiled using
    `-DUSE_CXX11_ABI=1` is available from PyPI.

    This is exposed as part of `libtvm_runtime.so` as it is used by
    the DNNL runtime.

    [0] https://github.com/pytorch/pytorch/issues/51039

    Parameters
    ----------
    regex_pattern: str

         The regular expression

    match_against: str

        The string against which to match the regular expression

    Returns
    -------
    match_result: bool

        True if `match_against` matches the pattern defined by
        `regex_pattern`, and False otherwise.

    """
    match = re.match(regex_pattern, match_against)
    return match is not None
