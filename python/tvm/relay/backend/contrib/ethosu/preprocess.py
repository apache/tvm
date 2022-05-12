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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Set of passes to pre-process the IRModule to support Arm(R)-Ethos(TM)-U
NPU code generation. These set of passes will mutate both the main and the
external functions.
"""
import tvm  # type: ignore
from . import _ffi_api  # type: ignore


def preprocess_ext_io() -> tvm.transform.Pass:
    """This pass mutates the number of inputs going to / outputs coming out to/from
    external functions to one. This is achieved via concatenation
    of inputs and splitting of outputs in around the call to the external function.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to mutate the IO of the external functions and their calls.
    """
    return _ffi_api.PreprocessExternalFuncIO()  # type: ignore # pylint: disable=no-member
