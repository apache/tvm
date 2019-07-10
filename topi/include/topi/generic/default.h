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
 * \file generic/default.h
 * \brief Generic default schedule
 */
#ifndef TOPI_GENERIC_DEFAULT_H_
#define TOPI_GENERIC_DEFAULT_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/operation.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace generic {
/*!
* \brief Create a generic default schedule for the given output tensors.
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

/*!
* \brief Create a generic default schedule for the given output tensors, and apply
* auto inline
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule_auto_inline(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto x = outs[0];
  tvm::schedule::AutoInlineInjective(s);
  auto axis = s[x]->op.as<ComputeOpNode>()->axis;
  if (axis.size() > 0) {
    detail::Fuse(s[x], axis);
  }
  return s;
}

}  // namespace generic
}  // namespace topi
#endif  // TOPI_GENERIC_DEFAULT_H_
