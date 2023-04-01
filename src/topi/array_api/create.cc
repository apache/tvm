#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace topi {
namespace array_api {

using te::Tensor;
using tir::Var;

Tensor full(Array<PrimExpr> shape, PrimExpr fill_value, DataType dtype) {
  return te::compute(
      shape,
      [&](const Array<Var>& i) {
        return dtype.is_void() ? fill_value : tvm::cast(dtype, fill_value);
      },
      "full");
}

Tensor full_like(Tensor x, PrimExpr fill_value, DataType dtype) {
  return te::compute(
      x->shape,
      [&](const Array<Var>& i) {
        DataType result_dtype = dtype;
        if (result_dtype.is_void()) {
          result_dtype = x->dtype;
        }
        if (result_dtype.is_void()) {
          result_dtype = fill_value->dtype;
        }
        return tvm::cast(result_dtype, fill_value);
      },
      "full_like");
}

Tensor ones(Array<PrimExpr> shape, DataType dtype) {
  return te::compute(
      shape, [&](const Array<Var>& i) { return tir::make_const(dtype, 1); }, "ones");
}

Tensor ones_like(Tensor x, DataType dtype) {
  if (dtype.is_void()) {
    dtype = x->dtype;
  }
  return te::compute(
      x->shape, [&](const Array<Var>& i) { return tir::make_const(dtype, 1); }, "ones_like");
}

Tensor zeros(Array<PrimExpr> shape, DataType dtype) {
  return te::compute(
      shape, [&](const Array<Var>& i) { return tir::make_const(dtype, 0); }, "zeros");
}

Tensor zeros_like(Tensor x, DataType dtype) {
  if (dtype.is_void()) {
    dtype = x->dtype;
  }
  return te::compute(
      x->shape, [&](const Array<Var>& i) { return tir::make_const(dtype, 0); }, "zeros_like");
}

Tensor triu(Tensor x, PrimExpr k) {
  using tir::Select;
  int ndim = x->shape.size();
  CHECK_GE(ndim, 2) << "ValueError: triu requires input tensor to be at least 2D";
  return te::compute(
      x->shape,
      [&](const Array<Var>& indices) {
        PrimExpr i = indices[ndim - 2];
        PrimExpr j = indices[ndim - 1];
        return Select(i <= j - k, x(indices), tir::make_zero(x->dtype));
      },
      "triu");
}

Tensor tril(Tensor x, PrimExpr k) {
  using tir::Select;
  int ndim = x->shape.size();
  CHECK_GE(ndim, 2) << "ValueError: tril requires input tensor to be at least 2D";
  return te::compute(
      x->shape,
      [&](const Array<Var>& indices) {
        PrimExpr i = indices[ndim - 2];
        PrimExpr j = indices[ndim - 1];
        return Select(j - k <= i, x(indices), tir::make_zero(x->dtype));
      },
      "tril");
}

TVM_REGISTER_GLOBAL("topi.array_api.full").set_body_typed(full);
TVM_REGISTER_GLOBAL("topi.array_api.full_like").set_body_typed(full_like);
TVM_REGISTER_GLOBAL("topi.array_api.ones").set_body_typed(ones);
TVM_REGISTER_GLOBAL("topi.array_api.ones_like").set_body_typed(ones_like);
TVM_REGISTER_GLOBAL("topi.array_api.zeros").set_body_typed(zeros);
TVM_REGISTER_GLOBAL("topi.array_api.zeros_like").set_body_typed(zeros_like);
TVM_REGISTER_GLOBAL("topi.array_api.triu").set_body_typed(triu);
TVM_REGISTER_GLOBAL("topi.array_api.tril").set_body_typed(tril);

}  // namespace array_api
}  // namespace topi
}  // namespace tvm
