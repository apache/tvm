/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {
namespace {

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  explicit InjectRealize(std::vector<Tensor> tensors)
      : tensors_(tensors) {}
  std::vector<Tensor> tensors_;
};


}  // namespace
}  // namespace ir
}  // namespace tvm
