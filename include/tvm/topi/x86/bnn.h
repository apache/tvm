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
 * \file x86/bnn.h
 * \brief x86 schedule for binary operations
 */
#ifndef TVM_TOPI_X86_BNN_H_
#define TVM_TOPI_X86_BNN_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/topi/detail/fuse.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace x86 {
/*!
 * \brief Create a generic schedule for binarize_pack
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_binarize_pack(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule = [&](const Tensor& out) {
    s[out].parallel(out->op.as<ComputeOpNode>()->axis[0]);
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    if (op->tag == "binarize_pack") {
      _schedule(op.output(0));
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  return s;
}

/*!
 * \brief Create a generic schedule for binary_dense
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_binary_dense(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule = [&](const Tensor& A, const Tensor& B, const Tensor& C) {
    IterVar co, ci;
    s[C].split(s[C]->op.as<ComputeOpNode>()->reduce_axis[0], 8, &co, &ci);
    s[C].parallel(s[C]->op.as<ComputeOpNode>()->axis[0]);

    Tensor out;
    if (detail::contains(s->outputs, C->op)) {
      out = C;
    } else {
      out = outs[0]->op.output(0);
    }

    IterVar xo, xi;
    s[out].split(out->op.as<ComputeOpNode>()->axis[1], 8, &xo, &xi);
    s[out].vectorize(xi);
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_broadcast(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag == "binary_dense") {
      auto output = op.output(0);
      auto data = op->InputTensors()[0];
      auto weight = op->InputTensors()[1];
      _schedule(data, weight, output);
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  return s;
}

}  // namespace x86
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_X86_BNN_H_
