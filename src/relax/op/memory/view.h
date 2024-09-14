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
 * \file view.h
 * \brief The functions to make Relax tensor view calls.
 */
#ifndef TVM_RELAX_OP_MEMORY_VIEW_H_
#define TVM_RELAX_OP_MEMORY_VIEW_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief View a tensor with different properties. */
Expr view(Expr x, Optional<Expr> shape, Optional<Expr> dtype, Optional<Expr> relative_byte_offset);

/*! \brief Ensure the tensor has elem_offset == 0. A copy will be made if necessary. */
Expr ensure_aligned(const Expr& x);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_MEMORY_VIEW_H_
