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

/*! \brief Return true iff the given expr is a constant int or uint */
bool IsConstInt(Expr expr) {
  return
    expr->derived_from<tvm::ir::IntImm>() ||
    expr->derived_from<tvm::ir::UIntImm>();
}

/*! \brief Get the value of the given constant integer expression */
int64_t GetConstInt(Expr expr) {
  if (expr->derived_from<tvm::ir::IntImm>()) {
    return expr.as<tvm::ir::IntImm>()->value;
  }
  if (expr->derived_from<tvm::ir::UIntImm>()) {
    return expr.as<tvm::ir::UIntImm>()->value;
  }
  LOG(ERROR) << "expr must be a constant integer";
  return -1;
}

/*! \brief Get the value of all the constant integer expressions in the given array */
std::vector<int> GetConstIntValues(Array<Expr> exprs, const std::string& var_name) {
  std::vector<int> result;
  for (auto expr : exprs) {
    CHECK(IsConstInt(expr)) << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

}  // namespace topi
#endif  // TOPI_DETAIL_CONSTANT_UTILS_H_
