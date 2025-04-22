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
 * \file tvm/runtime/container/shape_tuple.h
 * \brief Runtime ShapeTuple container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
#define TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_

#include <ostream>
#include <utility>
#include <vector>
#include <tvm/ffi/container/shape.h>
#include "./base.h"

namespace tvm {
namespace runtime {

using ShapeTuple = tvm::ffi::Shape;
using ShapeTupleObj = tvm::ffi::ShapeObj;
using IntTuple = ShapeTuple;
using IntTupleObj = ShapeTupleObj;

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::IntTuple;
using runtime::IntTupleObj;
using runtime::ShapeTuple;
using runtime::ShapeTupleObj;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
