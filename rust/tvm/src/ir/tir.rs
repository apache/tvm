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

use super::{PrimExpr, PrimExprNode};

use crate::ir::span::Span;
use crate::runtime::{IsObjectRef, String as TVMString};
use crate::DataType;

use tvm_macros::Object;

macro_rules! define_node {
    ($name:ident, $ref:expr, $typekey:expr; $node:ident { $($id:ident : $t:ty),*}) => {
        #[repr(C)]
        #[derive(Object, Debug)]
        #[ref_name = $ref]
        #[type_key = $typekey]
        pub struct $node {
            base: PrimExprNode,
            $(pub $id : $t),*
        }

        impl $name {
            pub fn new(datatype: DataType, $($id : $t,)*) -> $name {
                let base = PrimExprNode::base::<$node>(datatype, Span::null());
                let node = $node { base, $($id),* };
                node.into()
            }
        }
    }
}

// TODO(@jroesch): should move up to expr.rs to mirror TVM.
define_node!(IntImm, "IntImm", "IntImm";
             IntImmNode { value: i64 });

impl From<i32> for IntImm {
    fn from(i: i32) -> IntImm {
        IntImm::new(DataType::int(32, 1), i as i64)
    }
}

impl From<i32> for PrimExpr {
    fn from(i: i32) -> PrimExpr {
        IntImm::from(i).upcast()
    }
}

define_node!(Var, "Var", "tir.Var";
             VarNode { name_hint: TVMString });

define_node!(Add, "Add", "tir.Add"; AddNode { a: PrimExpr, b: PrimExpr });
define_node!(Sub, "Sub", "tir.Sub"; SubNode { a: PrimExpr, b: PrimExpr });
define_node!(Mul, "Mul", "tir.Mul"; MulNode { a: PrimExpr, b: PrimExpr });

define_node!(Div, "Div", "tir.Div"; DivNode { a: PrimExpr, b: PrimExpr });
define_node!(Mod, "Mod", "tir.Mod"; ModNode { a: PrimExpr, b: PrimExpr });
define_node!(FloorDiv, "FloorDiv", "tir.FloorDiv"; FloorDivNode { a: PrimExpr, b: PrimExpr });
define_node!(FloorMod, "FloorMod", "tir.FloorMod"; FloorModNode { a: PrimExpr, b: PrimExpr });

define_node!(Min, "Min", "tir.Min"; MinNode { a: PrimExpr, b: PrimExpr });
define_node!(Max, "Max", "tir.Max"; MaxNode { a: PrimExpr, b: PrimExpr });

// the new datatype is in the base expr
define_node!(Cast, "Cast", "tir.Cast"; CastNode { value: PrimExpr });

// renamed base to start to avoid name clash
define_node!(Ramp, "Ramp", "tir.Ramp"; RampNode { start: PrimExpr, stride: PrimExpr, lanes: i32 });

define_node!(Select, "Select", "tir.Select";
             SelectNode { condition: PrimExpr, true_value: PrimExpr, false_value: PrimExpr });

define_node!(Eq, "Eq", "tir.EQ"; EqNode { a: PrimExpr, b: PrimExpr });
define_node!(Ne, "Ne", "tir.NE"; NeNode { a: PrimExpr, b: PrimExpr });
define_node!(Lt, "Lt", "tir.LT"; LtNode { a: PrimExpr, b: PrimExpr });
define_node!(Le, "Le", "tir.LE"; LeNode { a: PrimExpr, b: PrimExpr });
define_node!(Gt, "Gt", "tir.GT"; GtNode { a: PrimExpr, b: PrimExpr });
define_node!(Ge, "Ge", "tir.GE"; GeNode { a: PrimExpr, b: PrimExpr });

define_node!(And, "And", "tir.And"; AndNode { a: PrimExpr, b: PrimExpr });
define_node!(Or,  "Or",  "tir.Or";  OrNode  { a: PrimExpr, b: PrimExpr });
define_node!(Not, "Not", "tir.Not"; NotNode { value: PrimExpr });

define_node!(Let, "Let", "tir.Let"; LetNode { var: Var, value: PrimExpr, body: PrimExpr });
