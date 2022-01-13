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
 * \file src/relay/op/vm/vm.h
 * \brief Dialect operators for Relay VM.
 */
#ifndef TVM_RELAY_OP_VM_VM_H_
#define TVM_RELAY_OP_VM_VM_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

Expr InvokeTVMOp(Expr func, Expr inputs, Expr outputs, DictAttrs attrs);
Expr ShapeOf(Expr expr);
Expr ReshapeTensor(Expr data, Expr shape, Array<PrimExpr> newshape);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_VM_VM_H_
