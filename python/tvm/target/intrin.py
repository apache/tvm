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
from tvm.ir import register_intrin_lowering
from tvm.tir import call_pure_extern


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
    register_intrin_lowering : The registration function for intrinsic lowering rule.
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
    register_intrin_lowering : The registration function for intrinsic lowering rule.
    """
    if str(op.dtype).startswith("float"):
        return call_pure_extern(op.dtype, op.op.name[4:], *op.args)
    return None


# opencl pattern for exp
register_intrin_lowering("tir.exp", target="opencl", f=_rule_float_direct, level=99)
# default pattern for exp
register_intrin_lowering("tir.exp", target="default", f=_rule_float_suffix, level=99)
