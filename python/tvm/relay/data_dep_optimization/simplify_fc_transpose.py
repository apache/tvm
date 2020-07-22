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
#pylint: disable=unused-argument, not-context-manager
"""Automatic optimize fc tranpose"""
import numpy as np

import tvm
from tvm import relay
from tvm.relay.analysis import search_fc_transpose

from .utils import _run_opt_pass


def convert(func, params):
    """convert all ```y = nn.dense(x, transpose(w, [1, 0]))``` to
        ```y = nn.dense(x, wt)```

    Parameters
    ----------
    func : relay.Expr
        Expr will be optimized
    params : Dict[String, tvm.nd.array]
        Parameters of Expr

    Returns
    -------
    new_func : relay.Expr
        Mutated Expr from ```y = nn.dense(x, transpose(w, [1, 0]))``` to
        ```y = nn.dense(x, wt)```
    params: Dict[String, tvm.nd.array]
        Parameters of mutated Expr, with weights pre-transposed
    """
    weight_info = search_fc_transpose(func)
    for item in weight_info:
        name = str(item)
        w_np = params[name].asnumpy()
        new_w = np.transpose(w_np, axes=[1, 0])
        params[name + ".T"] = tvm.nd.array(new_w)
        del params[name]
    new_func = _run_opt_pass(
        func,
        relay.transform.SimplifyFCTranspose(
            weight_info,
        )
    )
    return new_func, params
