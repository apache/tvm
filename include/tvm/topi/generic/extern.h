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
 * \file generic/extern.h
 * \brief Schedule for extern followed by injective ops
 */
#ifndef TVM_TOPI_GENERIC_EXTERN_H_
#define TVM_TOPI_GENERIC_EXTERN_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/detail/fuse.h>
#include <tvm/topi/generic/injective.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace generic {
/*!
 * \brief Schedule an extern op followed by injective operations
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the op.
 */
inline Schedule schedule_extern(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  tvm::te::AutoInlineInjective(s);
  for (auto out : outs) {
    if (out->op->IsInstance<ExternOpNode>()) {
      continue;
    }
    tvm::GenericFunc::Get("schedule_injective_from_existing")(s, out);
  }

  return s;
}

}  // namespace generic
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_GENERIC_EXTERN_H_
