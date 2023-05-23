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
"""Parser dispatching infrastructure"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type

from .doc import AST

if TYPE_CHECKING:
    from .parser import Parser


ParseMethod = Callable[["Parser", AST], None]
ParseVTable: Dict[Tuple[str, str], ParseMethod] = {}

OpMethod = Callable[..., Any]
OpVTable: Dict[Tuple[Type, AST, int], OpMethod] = {}


def register(token: str, type_name: str):
    """Register a method for a dispatch token and type name.

    Parameters
    ----------
    token : str
        The token for IR, e.g., T for TIR and R for Relax.

    type_name : str
        The type name of AST node, e.g., FunctionDef, With, For.

    Returns
    -------
    func : callable
        The function to register dispatched method of parsing
        corresponding token and AST node type.
    """

    def func(method: ParseMethod):
        """Register a method in parser virtual table.

        Parameters
        ----------
        method : ParseMethod
            The dispatched method to be registered in parser virtual table.
        """
        ParseVTable[(token, type_name)] = method

    return func


def get(
    token: str,
    type_name: str,
    default: Optional[ParseMethod] = None,
) -> Optional[ParseMethod]:
    """Get a registered method for a dispatch token and type name,
    or return a default method if no registered methods with this dispatch token and type name.

    Parameters
    ----------
    token : str
        The token for IR, e.g., T for TIR and R for Relax.

    type_name : str
        The type name of AST node, e.g., FunctionDef, With, For.

    default : Optional[ParseMethod]
        The default method when no registered methods with this dispatch token and type name.

    Returns
    -------
    func : Optional[ParseMethod]
        The dispatched method of parsing corresponding token and AST node type.
    """
    return ParseVTable.get((token, type_name), default)


def register_op(operand_type: Type, op_node_type: AST, operand_index: int):
    """Register a method for a operand type, AST operator node and operand index.

    Parameters
    ----------
    operand_type : Type
        The type of operands, e.g., tir.PrimExpr, tir.IterVar.

    op_node_type : AST
        The doc AST operator node type, e.g., doc.Add, doc.Eq.

    operand_index : int
        The operand index, i.e., 0 for left operand and 1 for right operand.

    Returns
    -------
    func : callable
        The function to register dispatched method of parsing
        corresponding a operand type, AST operator node and operand index.
    """

    def func(method: OpMethod):
        """Register a method in parser operator virtual table.

        Parameters
        ----------
        method : ParseMethod
            The dispatched method to be registered in parser operator virtual table.
        """
        OpVTable[(operand_type, op_node_type, operand_index)] = method

    return func


def get_op(
    operand_type: Type,
    op_node_type: Type,
    operand_index: int,
    default: Optional[OpMethod] = None,
) -> Optional[OpMethod]:
    """Register a method for a operand type, AST operator node and operand index.

    Parameters
    ----------
    operand_type : Type
        The type of operands, e.g., tir.PrimExpr, tir.IterVar.

    op_node_type : AST
        The doc AST operator node type, e.g., doc.Add, doc.Eq.

    operand_index : int
        The operand index, i.e., 0 for left operand and 1 for right operand.


    default : Optional[OpMethod]
        The default method when no registered methods with this operand type,
        AST operator node and operand index.

    Returns
    -------
    func : Optional[OpMethod]
        The function to register dispatched method of parsing
        corresponding a operand type, AST operator node and operand index.
    """
    return OpVTable.get((operand_type, op_node_type, operand_index), default)
