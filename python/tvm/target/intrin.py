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
"""Target dependent intrinsic registration."""
import tvm._ffi
from tvm.tir import call_pure_extern


# Intrinsic rule related code
def register_intrin_rule(target, intrin, f=None, override=False):
    """Register an intrinsic function generation rule.

    Intrinsic generation rules are callback functions for
    code generator to get device specific calls.
    This function simply translates to.

    :code:`register_func("tvm.intrin.rule.%s.%s" % (target, intrin), f, override)`

    TVM may already pre-register intrinsic rules in the backend.
    However, user can use this function to change the intrinsic translation
    behavior or add new intrinsic rules during runtime.

    Parameters
    ----------
    target : str
        The name of codegen target.

    intrin : str
        The name of the intrinsic.

    f : function, optional
        The function to be registered.

    override: boolean optional
        Whether override existing entry.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    The following code registers exp expansion rule for opencl.

    .. code-block:: python

        register_intrin_rule("opencl", "exp", my_exp_rule, override=True)
    """
    return tvm._ffi.register_func("tvm.intrin.rule.%s.%s" % (target, intrin), f, override)


def _rule_float_suffix(op):
    """Intrinsic rule: Add float suffix if it is float32.

    This is an example intrinsic generation rule.

    Parameters
    ----------
    op : PrimExpr
        The call expression of original intrinsic.

    Returns
    -------
    ret : PrimExpr
        The translated intrinsic rule.
        Return same op if no translation is possible.

    See Also
    --------
    register_intrin_rule : The registeration function for intrin rule.
    """
    name = op.op.name
    assert name.startswith("tir.")
    prefix = name[4:]

    if op.dtype == "float32":
        return call_pure_extern(op.dtype, "%sf" % prefix, *op.args)
    if op.dtype == "float64":
        return call_pure_extern(op.dtype, prefix, *op.args)
    return op


def _rule_float_direct(op):
    """Intrinsic rule: Directly call pure extern function for floats.

    This is an example intrinsic generation rule.

    Parameters
    ----------
    op : PrimExpr
        The call expression of original intrinsic.

    Returns
    -------
    ret : PrimExpr
        The translated intrinsic rule.
        Return same op if no translation is possible.

    See Also
    --------
    register_intrin_rule : The registeration function for intrin rule.
    """
    if str(op.dtype).startswith("float"):
        return call_pure_extern(op.dtype, op.op.name[4:], *op.args)
    return None

# opencl pattern for exp
register_intrin_rule("opencl", "exp", _rule_float_direct, override=True)
# default pattern for exp
register_intrin_rule("default", "exp", _rule_float_suffix, override=True)
