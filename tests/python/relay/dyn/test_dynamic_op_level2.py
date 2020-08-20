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
""" Support level2 dynamic operator test cases.
"""

import numpy as np
import tvm
from tvm import relay
from tvm import te
from tvm.relay.testing import ctx_list
import random
from test_dynamic_op_level3 import verify_func
import tvm.topi.testing
from tvm.relay.testing import run_infer_type

def test_dyn_pad():
    def verify_pad(dshape, pad_width, pad_val, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), 'int64'))
        pad_val_var = relay.var("pad_val_var", relay.TensorType((), dtype))
        y = relay.nn.pad(x, pad_width_var, pad_val_var)
        yy = run_infer_type(y)

        assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * ndim, dtype)
        func = relay.Function([x, pad_width_var, pad_val_var], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, pad_width, 'constant', constant_values=(((pad_val,)*2),) * ndim)
        pad_width = np.array(pad_width).astype('int64')

        verify_func(func, [data, pad_width, np.array(pad_val).astype(dtype)], ref_res)

    def verify_pad_default_fill(dshape, pad_width, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), 'int64'))
        y = relay.nn.pad(x, pad_width_var)
        yy = run_infer_type(y)

        assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * ndim, dtype)
        func = relay.Function([x, pad_width_var], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, pad_width)
        pad_width = np.array(pad_width).astype('int64')

        verify_func(func, [data, pad_width], ref_res)

    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), 2.0, "int32")
    verify_pad((2, 7), ((1, 4), (2, 2)), 4.0, "float64")
    verify_pad_default_fill((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), "float64")
    verify_pad_default_fill((2, 7), ((1, 4), (2, 2)), "int32")

if __name__ == "__main__":
    test_dyn_pad()
