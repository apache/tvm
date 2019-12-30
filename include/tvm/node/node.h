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
/*!
 * \file tvm/node/node.h
 * \brief Definitions and helper macros for IR/AST nodes.
 *
 *  The node folder contains base utilities for IR/AST nodes,
 *  invariant of which specific language dialect.
 *
 *  We implement AST/IR nodes as sub-classes of runtime::Object.
 *  The base class Node is just an alias of runtime::Object.
 *
 *  Besides the runtime type checking provided by Object,
 *  node folder contains additional functionalities such as
 *  reflection and serialization, which are important features
 *  for building a compiler infra.
 */
#ifndef TVM_NODE_NODE_H_
#define TVM_NODE_NODE_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/node/reflection.h>

#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace tvm {

using runtime::TypeIndex;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::GetRef;
using runtime::Downcast;
using runtime::ObjectHash;
using runtime::ObjectEqual;
using runtime::make_object;

}  // namespace tvm
#endif  // TVM_NODE_NODE_H_
