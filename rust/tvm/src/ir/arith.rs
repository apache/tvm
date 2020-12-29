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

use crate::runtime::{Object, ObjectPtr};

use tvm_macros::Object;

macro_rules! define_node {
    ($name:ident, $ref:expr, $typekey:expr; $node:ident { $($id:ident : $t:ty),*}) => {
        #[repr(C)]
        #[derive(Object, Debug)]
        #[ref_name = $ref]
        #[type_key = $typekey]
        pub struct $node {
            base: Object,
            $(pub $id : $t),*
        }

        impl $name {
            pub fn new($($id : $t,)*) -> $name {
                let base = Object::base::<$node>();
                let node = $node { base, $($id),* };
                $name(Some(ObjectPtr::new(node)))
            }
        }
    }
}

define_node!(ConstIntBound, "ConstIntBound", "arith.ConstIntBound";
             ConstIntBoundNode { min_value: i64, max_value: i64 });
