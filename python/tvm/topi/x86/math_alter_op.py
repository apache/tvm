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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Legalization transforms for math operations on x86"""

import logging

from tvm import relay
from ..math import erf_legalize

logger = logging.getLogger("topi")


@erf_legalize.register("cpu")
def _erf_legalize(attrs, inputs, arg_types):
    """Legalizes ERF op if needed.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # Extract types and expressions.
    data = inputs[0]
    data_tensor = arg_types[0]
    # Check if the input type is supported.
    data_dtype = data_tensor.dtype
    # If input is not fp32, we must cast to it.
    if data_dtype != "float32":
        data = relay.cast(data, "float32")
        output = relay.erf(data)
        return relay.cast(output, data_dtype)

    # Otherwise do nothing.
    return None
