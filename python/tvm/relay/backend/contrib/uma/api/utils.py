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
"""Utility methods for the Universal Modular Accelerator Interface (UMA)"""

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator

from enum import Enum


##############################
# Extract constants workaround
##############################
class ExtractConstants(ExprMutator):
    """The actual mutator pass to extract the constants from a function and replace them with
    Vars so the function can be lowered to a TE graph. Additionally returns all the values of
    the constants extracted."""

    def __init__(self):
        super().__init__()
        self.constants = {}
        self.const_vars = []

    def visit_constant(self, const):
        if isinstance(const.checked_type, relay.ty.TensorType):
            name = "p" + str(len(self.constants))
            self.constants[name] = const.data
            var = relay.var(type_annotation=const.checked_type, name_hint=name)
            self.const_vars.append(var)
            return var

        return const

    def visit_function(self, fn):
        new_body = self.visit(fn.body)
        new_params = list(fn.params) + self.const_vars
        return relay.Function(new_params, new_body, attrs=fn.attrs)

    def extract_constants(self, func):
        new_func = self.visit(func)
        return new_func, self.constants


def extract_constants(func):
    """Extract the constants from a function and replace them with
    Vars so the function can be lowered to a TE graph. Additionally
    returns all the values of the constants extracted.
    Parameters
    ----------
    func : tvm.relay.Function
        The Relay function from which to extract constants.
    Returns
    -------
    new_func : tvm.relay.Function
        The Relay function with constants replaced by vars.
    const_dict : dict of int to numpy.ndarray
        A dict of the extracted constants keyed by their param index.
    """
    new_func, consts = ExtractConstants().extract_constants(func)
    new_func = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(new_func))[
        func.attrs["global_symbol"]
    ]
    return new_func, consts
