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

#ifndef TVM_TIRX_IR_SPECIALIZE_H_
#define TVM_TIRX_IR_SPECIALIZE_H_

#include <tvm/ir/object_functor.h>

namespace tvm {
namespace tirx {

using SpecializeVisitorVTable =
    ObjectFunctor<ffi::Optional<VisitInterrupt>(const ffi::Object*, ObjectVisitor*)>;

// Register dialect-specific buffer planning before any specialization is run.
// Initializers extend the inherited native traversal table without exposing the
// specializer's private buffer remapping and declaration state.
void RegisterSpecializeBufferPlannerExtension(void (*init)(SpecializeVisitorVTable*));

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_SPECIALIZE_H_
