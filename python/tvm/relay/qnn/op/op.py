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
# pylint: disable=unused-argument
"""The register functions for the QNN dialect."""
import tvm.ir


def register_qnn_legalize(op_name, legal_op=None, level=10):
    """Register legal transformation function for a QNN op.

    This helps QNN match hardware intrinsics better and is run before
    canonicalization.

    Parameters
    ----------
    op_name : str
        The name of the operator

    legal_op: function (attrs: Attrs, inputs: List[Expr]) -> new_expr: Expr
        The function for transforming an expr to another expr.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMQnnLegalize", legal_op, level)


def register_qnn_canonicalize(op_name, legal_op=None, level=10):
    """Register canonicalization function for a QNN op.

    This transforms QNN ops to mainline Relay components.

    Parameters
    ----------
    op_name : str
        The name of the operator

    legal_op: function (Attrs, List[Expr], List[relay.Type]) -> Expr
        The function for transforming an expr to another expr.

    level : int
        The priority level
    """

    return tvm.ir.register_op_attr(op_name, "FTVMQnnCanonicalize", legal_op, level)
