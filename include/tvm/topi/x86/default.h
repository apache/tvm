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
 * \file x86/default.h
 * \brief default x86 schedule
 */
#ifndef TVM_TOPI_X86_DEFAULT_H_
#define TVM_TOPI_X86_DEFAULT_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/detail/fuse.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace x86 {
/*!
 * \brief Helper to create a default x86 schedule for the given ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 * \param auto_inline Whether to apply the auto inline step.
 *
 * \return A schedule for the given ops.
 */
inline Schedule MakeDefaultSchedule(const Target& target, const Array<Tensor>& outs,
                                    bool auto_inline) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto x = outs[0];
  auto axis = s[x]->op.as<ComputeOpNode>()->axis;

  if (auto_inline) {
    tvm::te::AutoInlineInjective(s);
    if (axis.size() > 0) {
      detail::Fuse(s[x], axis);
    }
    return s;
  }

  if (axis.size() == 4) {
    auto n = axis[0];
    auto c = axis[1];
    auto fused = detail::Fuse(s[x], {n, c});  // for nhwc layout, fuse n and h
    s[x].parallel(fused);
  } else {
    s[x].parallel(axis[0]);
  }

  return s;
}

/*!
 * \brief Create a default x86 schedule for the given ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule default_schedule(const Target& target, const Array<Tensor>& outs) {
  return MakeDefaultSchedule(target, outs, false);
}

/*!
 * \brief Create a default x86 schedule for the given ops, with auto inline
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule default_schedule_auto_inline(const Target& target, const Array<Tensor>& outs) {
  return MakeDefaultSchedule(target, outs, true);
}

}  // namespace x86
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_X86_DEFAULT_H_
