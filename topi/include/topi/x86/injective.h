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
 * \file x86/injective.h
 * \brief x86 schedule for injective ops
 */
#ifndef TOPI_X86_INJECTIVE_H_
#define TOPI_X86_INJECTIVE_H_

#include <topi/detail/fuse.h>
#include <topi/tags.h>
#include <tvm/runtime/threading_backend.h>
#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>

#include <vector>

namespace topi {
using namespace tvm;
using namespace tvm::te;

namespace x86 {

/*!
 * \brief Updates an existing schedule for the given injective ops.
 *
 * \param sch The schedule to update.
 * \param out The tensor representing the injective op.
 *
 * \return The updated schedule.
 */
inline Schedule schedule_injective_from_existing(Schedule sch, const Tensor& out) {
  const auto& axes = sch[out]->op.as<ComputeOpNode>()->axis;
  std::vector<IterVar> to_fuse;
  int64_t prod_len = 1;

  for (const auto& axis : axes) {
    const auto* pint = axis->dom->extent.as<IntImmNode>();
    if (pint == nullptr) {
      break;
    }
    prod_len *= pint->value;
    to_fuse.push_back(axis);
    if (prod_len >= tvm::runtime::threading::MaxConcurrency()) {
      break;
    }
  }

  if (to_fuse.empty()) {
    return sch;
  } else if (to_fuse.size() == 1) {
    sch[out].parallel(to_fuse[0]);
  } else {
    auto fused = detail::Fuse(sch[out], to_fuse);
    sch[out].parallel(fused);
  }
  return sch;
}

/*!
 * \brief Create an x86 schedule for the given injective ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_injective(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  tvm::te::AutoInlineInjective(s);

  auto x = outs[0];
  schedule_injective_from_existing(s, x);

  return s;
}

}  // namespace x86
}  // namespace topi
#endif  // TOPI_X86_INJECTIVE_H_
