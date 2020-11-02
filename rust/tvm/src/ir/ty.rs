/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use super::span::Span;
use crate::runtime::{IsObject, Object, ObjectPtr};
use tvm_macros::Object;
use tvm_rt::{array::Array, DataType};

use super::PrimExpr;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Type"]
#[type_key = "Type"]
pub struct TypeNode {
    pub base: Object,
    pub span: Span,
}

impl TypeNode {
    fn base<T: IsObject>(span: Span) -> Self {
        TypeNode {
            base: Object::base::<T>(),
            span,
        }
    }
}

/*
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
#[repr(C)]
#[derive(Object)]
#[ref_name = "PrimType"]
#[type_key = "PrimType"]
pub struct PrimTypeNode {
    pub base: TypeNode,
    /// The corresponding dtype field.
    pub dtype: DataType,
}

/*
 *!
 * \brief Low-level raw pointer type.
 *
 *  PointerType represents type hints in the TIR to be
 *  passed to the final code generator.
 *
 *  PointerType should not occur in the high-level analysis.
 *
 * \sa PointerType
 */

#[repr(C)]
#[derive(Object)]
#[ref_name = "PointerType"]
#[type_key = "PointerType"]
pub struct PointerTypeNode {
    pub base: TypeNode,
    /// The type of the element which the pointer points to.
    pub element_type: Type,
}

/// Possible kinds of type variables.
pub enum TypeKind {
    Type = 0,
    /// Template variable in shape expression.
    ShapeVar = 1,
    Constraint = 4,
    AdtHandle = 5,
    TypeData = 6,
}

/*
 * \brief Type parameter in functions.
 *
 * A type variable can be viewed as template parameter in c++ template function.
 *
 * For example, in the following pesudo code,
 * the TypeVar of f is TypeVar("n", kind=kShapeVar).
 * This function can take in a Tensor with shape=(3, 3) and
 * returns a Tensor with shape=(9,)
 *
 * \code
 *
 *  template<i32 n>
 *  f(x : Tensor[i32, (n, n)]) -> Tensor[i32, (n * n)]
 *
 * \endcode
 * \sa TypeVar, TypeKind
 */
#[repr(C)]
#[derive(Object)]
#[ref_name = "TypeVar"]
#[type_key = "TypeVar"]
pub struct TypeVarNode {
    pub base: TypeNode,
    pub name_hint: String,
    pub kind: TypeKind,
}

/// A global type variable that is used for defining new types or type aliases.
#[repr(C)]
#[derive(Object)]
#[ref_name = "GlobalTypeVar"]
#[type_key = "GlobalTypeVar"]
pub struct GlobalTypeVarNode {
    pub base: TypeNode,
    pub name_hint: String,
    pub kind: TypeKind,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "TupleType"]
#[type_key = "TupleType"]
pub struct TupleTypeNode {
    pub base: TypeNode,
    pub fields: Array<Type>,
}

impl TupleType {
    pub fn empty() -> TupleType {
        todo!()
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "TypeConstraint"]
#[type_key = "TypeConstraint"]
pub struct TypeConstraintNode {
    pub base: TypeNode,
}

/// The representation of a polymorphic function type.
#[repr(C)]
#[derive(Object)]
#[ref_name = "FuncType"]
#[type_key = "FuncType"]
pub struct FuncTypeNode {
    pub base: TypeNode,
    /// The type of arguments.
    pub arg_types: Array<Type>,
    /// The return type of the function.
    pub ret_type: Type,
    /// ...
    pub type_params: Array<TypeVar>,
    /// Type constraints that must hold when
    /// calling this function.
    pub type_constraints: Array<TypeConstraint>,
}

/*
 * \brief Intermediate values that is used to indicate incomplete type
 *         during type inference.
 *
 * If we view the type relations as "computational graph of types",
 * then IncompleteType represents intermediate values of the graph,
 * TypeVar represents the input to the graph.
 */
#[repr(C)]
#[derive(Object)]
#[ref_name = "IncompleteType"]
#[type_key = "IncompleteType"]
pub struct IncompleteTypeNode {
    pub base: TypeNode,
    pub kind: TypeKind,
}

/*
 * \brief Reference Type High-level Relay IR.
 *
 * \sa RelayRefType.
 */
#[repr(C)]
#[derive(Object)]
#[ref_name = "RefType"]
#[type_key = "relay.RefType"]
pub struct RelayRefTypeNode {
    pub base: TypeNode,
    pub value: Type,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseTensorType"]
#[type_key = "relay.BaseTensorType"]
pub struct BaseTensorTypeNode {
    pub base: TypeNode,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "TensorType"]
#[type_key = "relay.TensorType"]
pub struct TensorTypeNode {
    pub base: TypeNode,
    pub shape: Array<PrimExpr>,
    pub dtype: DataType,
}

impl TensorType {
    pub fn new(shape: Array<PrimExpr>, dtype: DataType, span: Span) -> TensorType {
        let node = TensorTypeNode {
            base: TypeNode::base::<TensorTypeNode>(span),
            shape,
            dtype,
        };
        ObjectPtr::new(node).into()
    }
}
// TODO(@jroesch): implement these in future.
//
// using TypeCall = tvm::TypeCall;
// using TypeCallNode = tvm::TypeCallNode;
// using TypeRelation = tvm::TypeRelation;
// using TypeRelationNode = tvm::TypeRelationNode;
// using TypeRelationFn = tvm::TypeRelationFn;
// using TypeReporter = tvm::TypeReporter;
// using TypeReporterNode = tvm::TypeReporterNode;
