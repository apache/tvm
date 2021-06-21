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
"""Utilities for PlatformSimulator."""
import functools
import re
import tvm


def _get_type_size(tipe):
    if isinstance(tipe, tvm.ir.type.TupleType):
        return sum([_get_type_size(f) for f in tipe.fields])

    dtype = str(tipe.dtype)
    shape = list([int(i) for i in tipe.shape])
    nbits = (lambda s: int(s) if s != "" else 8)(re.sub("[a-z]", "", dtype))
    assert nbits % 8 == 0
    return functools.reduce(
        lambda x, y: x * y,
        shape,
        nbits / 8,  # use byte as basic unit
    )


def get_node_size(node):
    """Get node size in bytes.

    Parameters
    ----------
    node: tvm.relay.Expr
        The Relay expression whose size is to be calculated.
    """
    return _get_type_size(node.checked_type)
