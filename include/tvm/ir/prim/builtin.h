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
#ifndef TVM_IR_PRIM_BUILTIN_H_
#define TVM_IR_PRIM_BUILTIN_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace prim {
namespace builtin {

/*!
 * \brief Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
 * (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
 */
TVM_DLL const Op& vscale();

/*! \brief Left shift. */
TVM_DLL const Op& shift_left();

/*! \brief Right shift. */
TVM_DLL const Op& shift_right();

/*! \brief Bitwise and operator. */
TVM_DLL const Op& bitwise_and();

/*! \brief Bitwise or operator. */
TVM_DLL const Op& bitwise_or();

/*! \brief Bitwise xor operator. */
TVM_DLL const Op& bitwise_xor();

/*! \brief Bitwise not operator. */
TVM_DLL const Op& bitwise_not();

/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
TVM_DLL const Op& if_then_else();

/*! \brief Marks a condition is likely going to happen. */
TVM_DLL const Op& likely();

}  // namespace builtin
}  // namespace prim
}  // namespace tvm
#endif  // TVM_IR_PRIM_BUILTIN_H_
