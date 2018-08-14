/*!
 *  Copyright (c) 2018 by Contributors
 * \file dump_tensor.cc
 * \brief Print out tensors recursively.
 */
#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

namespace tvm {
namespace ir {

std::string PrintTensorName(const Tensor& tensor) {
  if (!tensor.get()) {
    return "NULL_TENSOR";
  }

  std::ostringstream oss;
  oss << tensor->op->name << "[" << tensor->value_index << "]";
  return oss.str();
}

std::string PrintIterVars(const Array<IterVar>& itervars) {
  std::ostringstream oss;
  oss << "(";
  bool first = true;
  for (const IterVar& iv : itervars) {
    if (!first) oss << ", ";
    first = false;
    oss << iv->var << " : " << "[" << iv->dom->min
        << ", " << (iv->dom->min + iv->dom->extent - 1) << "]";
  }
  oss << ")";
  return oss.str();
}

std::string PrintTensorsRecursively(const Array<Tensor>& tensors) {
  std::vector<Tensor> unprocessed;
  std::unordered_set<Tensor> processed;
  std::ostringstream oss;

  for (const Tensor& t : tensors) {
    unprocessed.push_back(t);
  }

  while (!unprocessed.empty()) {
    Tensor cur = unprocessed.back();
    unprocessed.pop_back();
    processed.insert(cur);

    oss << "tensor " << PrintTensorName(cur) << " : " << cur->dtype << " " << cur->shape << "\n";
    if (const ComputeOpNode* comp = cur->op.as<ComputeOpNode>()) {
      oss << "axes " << PrintIterVars(comp->axis) << "\n";
      Expr body = comp->body[cur->value_index];

      for (const Tensor& t : comp->InputTensors()) {
        if (processed.count(t) == 0) {
          unprocessed.push_back(t);
        }
      }

      if (const Reduce* red = body.as<Reduce>()) {
        oss << "Reduction\n";
        oss << "    identity " << red->combiner->identity_element << "\n";
        oss << "    lhs " << red->combiner->lhs << "  rhs " << red->combiner->rhs << "\n";
        oss << "    combiner " << red->combiner->result << "\n";
        oss << "    axis " << PrintIterVars(red->axis) << "\n";
        oss << "    condition " << red->condition << "\n";
        for (size_t i = 0; i < red->source.size(); ++i) {
          oss << "    source[" << i << "] = " << red->source[i] << "\n";
        }
      } else {
        oss << "    " << body << "\n";
      }
    } else {
      oss << "    " << cur->op << "\n";
    }
    oss << "\n";
  }

  return oss.str();
}

std::string PrintTensorRecursively(const Tensor& tensor) {
  return PrintTensorsRecursively({tensor});
}

TVM_REGISTER_API("PrintTensorRecursively")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = PrintTensorRecursively(args[0]);
  });

TVM_REGISTER_API("PrintTensorsRecursively")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = PrintTensorsRecursively(args[0]);
  });

}  // namespace ir
}  // namespace tvm
