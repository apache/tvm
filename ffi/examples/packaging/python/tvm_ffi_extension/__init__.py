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
# specific language governing permissions and limitations.
from .base import _LIB
from . import _ffi_api


def add_one(x, y):
    """
    Adds one to the input tensor.

    Parameters
    ----------
    x : Tensor
      The input tensor.
    y : Tensor
      The output tensor.
    """
    return _LIB.add_one(x, y)


def raise_error(msg):
    """
    Raises an error with the given message.

    Parameters
    ----------
    msg : str
        The message to raise the error with.

    Raises
    ------
    RuntimeError
        The error raised by the function.
    """
    return _ffi_api.raise_error(msg)
