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

use tvm_macros::Object;
use tvm_rt::{array::Array, DataType};

use crate::ir::relay::Constructor;
use crate::ir::span::Span;
use crate::ir::PrimExpr;
use crate::runtime::{string::String as TString, IsObject, IsObjectRef, Object, ObjectPtr};

#[repr(C)]
#[derive(Object, Debug)]
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
#[derive(Object, Debug)]
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
#[derive(Object, Debug)]
#[ref_name = "PointerType"]
#[type_key = "PointerType"]
pub struct PointerTypeNode {
    pub base: TypeNode,
    /// The type of the element which the pointer points to.
    pub element_type: Type,
}

/// Possible kinds of type variables.
#[derive(PartialEq, Eq, Debug)]
pub enum TypeKind {
    Type = 0,
    /// Template variable in shape expression.
    ShapeVar = 1,
    Constraint = 4,
    AdtHandle = 5,
    TypeData = 6,
}

/// Type parameter in functions.
///
/// A type variable can be viewed as template parameter in c++ template function.
///
/// For example, in the following pesudo code,
/// the TypeVar of f is TypeVar("n", kind=kShapeVar).
/// This function can take in a Tensor with shape=(3, 3) and
/// returns a Tensor with shape=(9,)
#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TypeVar"]
#[type_key = "TypeVar"]
pub struct TypeVarNode {
    pub base: TypeNode,
    pub name_hint: TString,
    pub kind: TypeKind,
}

/// A global type variable that is used for defining new types or type aliases.
#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "GlobalTypeVar"]
#[type_key = "GlobalTypeVar"]
pub struct GlobalTypeVarNode {
    pub base: TypeNode,
    pub name_hint: TString,
    pub kind: TypeKind,
}

impl GlobalTypeVar {
    pub fn new<S>(name_hint: S, kind: TypeKind, span: Span) -> GlobalTypeVar
    where
        S: Into<TString>,
    {
        let node = GlobalTypeVarNode {
            base: TypeNode::base::<GlobalTypeVarNode>(span),
            name_hint: name_hint.into(),
            kind: kind,
        };
        ObjectPtr::new(node).into()
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TupleType"]
#[type_key = "TupleType"]
pub struct TupleTypeNode {
    pub base: TypeNode,
    pub fields: Array<Type>,
}

impl TupleType {
    // todo add coercion
    pub fn new(fields: Vec<Type>, span: Span) -> Self {
        let node = TupleTypeNode {
            base: TypeNode::base::<TupleTypeNode>(span),
            fields: Array::from_vec(fields).unwrap(),
        };
        ObjectPtr::new(node).into()
    }

    pub fn empty() -> TupleType {
        TupleType::new(vec![], Span::null())
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TypeConstraint"]
#[type_key = "TypeConstraint"]
pub struct TypeConstraintNode {
    pub base: TypeNode,
}

/// The representation of a polymorphic function type.
#[repr(C)]
#[derive(Object, Debug)]
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
#[derive(Object, Debug)]
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
#[derive(Object, Debug)]
#[ref_name = "RefType"]
#[type_key = "relay.RefType"]
pub struct RelayRefTypeNode {
    pub base: TypeNode,
    pub value: Type,
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BaseTensorType"]
#[type_key = "relay.BaseTensorType"]
pub struct BaseTensorTypeNode {
    pub base: TypeNode,
}

#[repr(C)]
#[derive(Object, Debug)]
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

    pub fn static_sh(shape: Vec<i32>, dtype: DataType, span: Span) -> TensorType {
        let sh = Array::from_vec(shape.into_iter().map(Into::into).collect()).unwrap();
        Self::new(sh, dtype, span)
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

/* TypeData container node.
\brief Stores all data for an Algebraic Data Type (ADT).

In particular, it stores the handle (global type var) for an ADT
and the constructors used to build it and is kept in the module. Note
that type parameters are also indicated in the type data: this means that
for any instance of an ADT, the type parameters must be indicated. That is,
an ADT definition is treated as a type-level function, so an ADT handle
must be wrapped in a TypeCall node that instantiates the type-level arguments.
The kind checker enforces this. */
#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TypeData"]
#[type_key = "relay.TypeData"]
pub struct TypeDataNode {
    /// The header is simply the name of the ADT.
    /// We adopt nominal typing for ADT definitions;
    /// that is, differently-named ADT definitions with same constructors
    /// have different types.
    pub base: TypeNode,
    pub type_name: GlobalTypeVar,
    /// The type variables (to allow for polymorphism).
    pub type_vars: Array<TypeVar>,
    /// The constructors.
    pub constructors: Array<Constructor>,
}

impl TypeData {
    pub fn new<TypeVars, Ctors>(
        type_name: GlobalTypeVar,
        type_vars: TypeVars,
        constructors: Ctors,
        span: Span,
    ) -> TypeData
    where
        TypeVars: IntoIterator<Item = TypeVar>,
        Ctors: IntoIterator<Item = Constructor>,
    {
        use std::iter::FromIterator;
        let type_data = TypeDataNode {
            base: TypeNode::base::<TypeDataNode>(span),
            type_name,
            type_vars: Array::from_iter(type_vars),
            constructors: Array::from_iter(constructors),
        };
        TypeData(Some(ObjectPtr::new(type_data)))
    }
}
