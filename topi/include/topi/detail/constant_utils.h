/*!
*  Copyright (c) 2017 by Contributors
* \file constant_utils.h
* \brief Utility functions for handling constants in TVM expressions
*/
#ifndef TOPI_DETAIL_CONSTANT_UTILS_H_
#define TOPI_DETAIL_CONSTANT_UTILS_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

bool IsConstInt(Expr expr) {
  return
    expr->derived_from<tvm::ir::IntImm>() ||
    expr->derived_from<tvm::ir::UIntImm>();
}

int64_t GetConstInt(Expr expr) {
  if (expr->derived_from<tvm::ir::IntImm>()) {
    return expr.as<tvm::ir::IntImm>()->value;
  }
  if (expr->derived_from<tvm::ir::UIntImm>()) {
    return expr.as<tvm::ir::UIntImm>()->value;
  }
  LOG(ERROR) << "expr must be a constant integer";
}

}  // namespace topi
#endif  // TOPI_DETAIL_CONSTANT_UTILS_H_
