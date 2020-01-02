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
# pylint: disable=wildcard-import, redefined-builtin, invalid-name
""" Utility functions that are used across many directories. """
from __future__ import absolute_import
import numpy as np
from . import expr as _expr

def get_scalar_from_constant(expr):
    """ Returns scalar value from Relay constant scalar. """
    assert isinstance(expr, _expr.Constant) and not expr.data.shape, \
            "Expr is not a constant scalar."
    value = expr.data.asnumpy()
    if value.dtype == np.dtype(np.int32):
        return int(value)
    if value.dtype == np.dtype(np.float32):
        return float(value)
    assert False, "Constant expr must be float32/int32"
    return None  # To suppress pylint
