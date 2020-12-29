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
 * \file adjoint.cc
 * \brief Perform reverse-mode autodiff.
 *        Suppose we have f(x) = g(h1(x), h2(x), ..., hn(x)),
 *        df/dx = \sum_i df/dhi * dhi/dx
 *        We call df/dx as adjoint(x), df/dhi as adjoint(hi), dhi/dx is the Jacobian
 *        The idea is to first construct the reverse-dependency {input->outputs} between tensors,
 *        start from one input,
 *        (1) collect adjoints from all its dependencies (outputs),
 *        (2) multiply the Jacobian (PartialAdjoint),
 *        (3) and sum them together to get the adjoint of the input itself.
 *        The three steps are computed recursively.
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/autodiff.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <memory>
#include <vector>

#include "ad_utils.h"

namespace tvm {
namespace te {

Tensor Identity(const Tensor& output) {
  Array<PrimExpr> shape = output->shape;
  for (auto e : output->shape) {
    // add extra dimension for Jacobian
    shape.push_back(e);
  }
  auto func = [&output](const Array<Var>& input_indices) {
    PrimExpr res = const_true();
    for (size_t i = 0; i < output->shape.size(); ++i) {
      res =
          res && (PrimExpr(input_indices[i]) == PrimExpr(input_indices[output->shape.size() + i]));
    }
    return Cast(output->dtype, res);
  };
  return te::compute(shape, func, "identity");
}

Tensor VectorJacobianProduct(const Tensor& output, const Tensor& input, const Tensor& head) {
  Tensor jac = Jacobian(output, input);
  Tensor result = topi::tensordot(head, jac, /*axes=*/output->shape.size(),
                                  output->op->name + "." + input->op->name + ".grad");
  result = InlineTensorAccess(result, {jac}, false);
  result = RemoveJacobianAndLiftNonzeroCond(result);
  // inline tail call
  result = InlineTailTensorAccess(result);
  return result;
}

Array<Tensor> Gradient(const Tensor& output, const Array<Tensor>& inputs,
                       const Tensor& head_or_null) {
  // Diagonal identity tensor
  Tensor head = head_or_null.get() ? head_or_null : Identity(output);

  // This Map{input -> outputs} maps a tensor to the list of tensors
  // immediately depending on it (using it in their bodies)
  std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies;
  std::vector<Tensor> stack({output});
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    stack.pop_back();
    for (const Tensor& input : tensor->op->InputTensors()) {
      if (!reverse_dependencies.count(input)) {
        stack.push_back(input);
      }
      reverse_dependencies[input].push_back(tensor);
    }
  }

  // This map maps tensors to the corresponding adjoints (dLoss/dTensor)
  std::unordered_map<Tensor, Tensor> adjoints;
  // head is the adjoint of output by definition
  adjoints[output] = head;

  // This is a recursive function that does all the work. It computes the adjoint for a given
  // tensor, adds it to the map, and returns it
  std::function<Tensor(const Tensor&)> compute_adjoint;
  compute_adjoint = [&compute_adjoint, &adjoints, &reverse_dependencies, &head,
                     &output](const Tensor& tensor) {
    if (!adjoints.count(tensor)) {
      // Here the adjoint hasn't been computed yet
      Tensor res_adjoint;
      std::vector<Tensor> direct_consumers = reverse_dependencies[tensor];
      if (direct_consumers.empty()) {
        // No reverse dependencies means that the output does not depend on this tensor,
        // return a zero tensor of the appropriate shape
        // (i.e., output shape + tensor shape, aka shape of Jacobian)
        Array<PrimExpr> result_shape(head->shape.begin(), head->shape.end() - output->shape.size());
        for (auto e : tensor->shape) {
          result_shape.push_back(e);
        }
        res_adjoint = topi::full(result_shape, output->dtype, make_zero(output->dtype));
      } else {
        // The new adjoint is computed as a sum of the reverse dependencies' adjoints multiplied
        // by the corresponding "local" jacobians (dDep/dTensor). The computation of the jacobian
        // and the multiplication is done in the function VectorJacobianProduct
        for (const Tensor& direct_consumer : direct_consumers) {
          // part = (adjoint of direct_consumer) * Jacobian(direct_consumer, tensor)
          Tensor part =
              VectorJacobianProduct(direct_consumer, tensor, compute_adjoint(direct_consumer));
          res_adjoint = res_adjoint.get() ? topi::add(res_adjoint, part) : part;
        }
      }

      adjoints[tensor] = res_adjoint;
      return res_adjoint;
    } else {
      return adjoints[tensor];
    }
  };

  // Adjoints corresponding to inputs
  Array<Tensor> result;
  // Compute an adjoint for each input
  for (const Tensor& input : inputs) {
    result.push_back(compute_adjoint(input));
  }

  return result;
}

TVM_REGISTER_GLOBAL("te.Gradient").set_body([](TVMArgs args, TVMRetValue* ret) {
  LOG(WARNING) << "te.Gradient is an experimental feature.";
  if (args.size() == 2) {
    *ret = Gradient(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = Gradient(args[0], args[1], args[2]);
  }
});

}  // namespace te
}  // namespace tvm
