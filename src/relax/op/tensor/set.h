/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file set.h
 * \brief The functions to make Relax set operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_SET_H_
#define TVM_RELAX_OP_TENSOR_SET_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

Expr unique(Expr x, PrimValue sorted, PrimValue return_index, PrimValue return_inverse,
            PrimValue return_counts, Optional<PrimValue> axis);
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_SET_H_
