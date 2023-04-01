#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace topi {
namespace array_api {

using te::Tensor;
using tir::Var;

Array<PrimExpr> BroadcastShape(const Array<PrimExpr>& a, const Array<PrimExpr>& b) {
  int64_t ndim_a = a.size();
  int64_t ndim_b = b.size();
  int64_t ndim = std::max(ndim_a, ndim_b);
  Array<PrimExpr> out_shape;
  out_shape.reserve(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    PrimExpr dim_a = i < ndim_a ? a[ndim_a - 1 - i] : IntImm(DataType::Int(64), 1);
    PrimExpr dim_b = i < ndim_b ? b[ndim_b - 1 - i] : IntImm(DataType::Int(64), 1);
    if (tir::is_one(dim_a)) {
      out_shape.push_back(dim_b);
    } else if (tir::is_one(dim_b)) {
      out_shape.push_back(dim_a);
    } else if (tir::ExprDeepEqual()(dim_a, dim_b)) {
      out_shape.push_back(dim_a);
    } else {
      LOG(FATAL) << "ValueError: Unable to deduce output shape from inputs: " << a << " and " << b;
    }
  }
  return out_shape;
}

#define TVM_TOPI_DECLARE_UNARY(OpName, OpFunc)                                              \
  TVM_DLL Tensor OpName(const Tensor& x) {                                                  \
    return te::compute(/*shape=*/x->shape,                                                  \
                       /*f=*/[&](const Array<Var>& indices) { return OpFunc(x(indices)); }, \
                       /*name=*/#OpName);                                                   \
  }                                                                                         \
  TVM_REGISTER_GLOBAL("topi.array_api." #OpName).set_body_typed(OpName);

#define TVM_TOPI_DECLARE_BINARY(OpName, OpFunc)                                          \
  TVM_DLL Tensor OpName(const Tensor& x, const Tensor& y) {                              \
    return te::compute(                                                                  \
        /*shape=*/BroadcastShape(x->shape, y->shape),                                    \
        /*f=*/[&](const Array<Var>& indices) { return OpFunc(x(indices), y(indices)); }, \
        /*name=*/#OpName);                                                               \
  }                                                                                      \
  TVM_REGISTER_GLOBAL("topi.array_api." #OpName).set_body_typed(OpName);

// Category 1. Basic arithmetic operators
TVM_TOPI_DECLARE_BINARY(add, ::tvm::add);
TVM_TOPI_DECLARE_BINARY(subtract, ::tvm::sub);
TVM_TOPI_DECLARE_BINARY(multiply, ::tvm::mul);
TVM_TOPI_DECLARE_BINARY(divide, ::tvm::div);  // TODO(@junrushao): revisit its semantics
TVM_TOPI_DECLARE_BINARY(floor_divide, ::tvm::floordiv);
TVM_TOPI_DECLARE_BINARY(remainder, ::tvm::floormod);  // TODO(@junrushao): revisit its semantics
TVM_TOPI_DECLARE_BINARY(pow, ::tvm::pow);

// Category 2. Trigonometric functions
TVM_TOPI_DECLARE_UNARY(acos, ::tvm::acos);
TVM_TOPI_DECLARE_UNARY(acosh, ::tvm::acosh);
TVM_TOPI_DECLARE_UNARY(asin, ::tvm::asin);
TVM_TOPI_DECLARE_UNARY(asinh, ::tvm::asinh);
TVM_TOPI_DECLARE_UNARY(atan, ::tvm::atan);
TVM_TOPI_DECLARE_UNARY(atanh, ::tvm::atanh);
TVM_TOPI_DECLARE_UNARY(cos, ::tvm::cos);
TVM_TOPI_DECLARE_UNARY(cosh, ::tvm::cosh);
TVM_TOPI_DECLARE_UNARY(sin, ::tvm::sin);
TVM_TOPI_DECLARE_UNARY(sinh, ::tvm::sinh);
TVM_TOPI_DECLARE_UNARY(tan, ::tvm::tan);
TVM_TOPI_DECLARE_UNARY(tanh, ::tvm::tanh);
TVM_TOPI_DECLARE_BINARY(atan2, ::tvm::atan2);

// Category 3. Exp/log/square operators
TVM_TOPI_DECLARE_UNARY(exp, ::tvm::exp)
// TVM_TOPI_DECLARE_UNARY(expm1, ::tvm::expm1);  // TODO(@junrushao): add
TVM_TOPI_DECLARE_UNARY(log, ::tvm::log);
TVM_TOPI_DECLARE_UNARY(log1p, ::tvm::log1p);
TVM_TOPI_DECLARE_UNARY(log2, ::tvm::log2);
TVM_TOPI_DECLARE_UNARY(log10, ::tvm::log10);
TVM_TOPI_DECLARE_UNARY(square, [](const PrimExpr& e) { return e * e; });  // TODO(@junrushao): add
TVM_TOPI_DECLARE_UNARY(sqrt, ::tvm::sqrt);
// TVM_TOPI_DECLARE_BINARY(logaddexp, ::tvm::logaddexp); // TODO(@junrushao): add

// Category 4. Rounding/Sign operators
TVM_TOPI_DECLARE_UNARY(abs, ::tvm::abs);
TVM_TOPI_DECLARE_UNARY(ceil, ::tvm::ceil);
TVM_TOPI_DECLARE_UNARY(floor, ::tvm::floor);
TVM_TOPI_DECLARE_UNARY(trunc, ::tvm::trunc);
TVM_TOPI_DECLARE_UNARY(round, ::tvm::round);
// TVM_TOPI_DECLARE_UNARY(sign, ::tvm::sign); TODO(@junrushao): add intrin: sign
TVM_TOPI_DECLARE_UNARY(positive, [](const PrimExpr& e) { return e; });
TVM_TOPI_DECLARE_UNARY(negative, ::tvm::neg);

// Category 5. Comparison operators
TVM_TOPI_DECLARE_BINARY(equal, ::tvm::equal);
TVM_TOPI_DECLARE_BINARY(not_equal, ::tvm::not_equal);
TVM_TOPI_DECLARE_BINARY(greater, ::tvm::greater);
TVM_TOPI_DECLARE_BINARY(greater_equal, ::tvm::greater_equal);
TVM_TOPI_DECLARE_BINARY(less, ::tvm::less);
TVM_TOPI_DECLARE_BINARY(less_equal, ::tvm::less_equal);

// Category 6. Bitwise operators
TVM_TOPI_DECLARE_BINARY(bitwise_and, ::tvm::bitwise_and);
TVM_TOPI_DECLARE_BINARY(bitwise_or, ::tvm::bitwise_or);
TVM_TOPI_DECLARE_BINARY(bitwise_xor, ::tvm::bitwise_xor);
TVM_TOPI_DECLARE_UNARY(bitwise_invert, ::tvm::bitwise_neg);
TVM_TOPI_DECLARE_BINARY(bitwise_left_shift, ::tvm::left_shift);
TVM_TOPI_DECLARE_BINARY(bitwise_right_shift, ::tvm::right_shift);

// Category 7. Logical operators
TVM_TOPI_DECLARE_BINARY(logical_and, ::tvm::logical_and);
TVM_TOPI_DECLARE_BINARY(logical_or, ::tvm::logical_or);
TVM_TOPI_DECLARE_UNARY(logical_not, ::tvm::logical_not);
// TVM_TOPI_DECLARE_BINARY(logical_xor, ::tvm::logical_xor); // TODO(@junrushao): add

// Category 8. Special value checking for floating point numbers
TVM_TOPI_DECLARE_UNARY(isfinite, ::tvm::isfinite);
TVM_TOPI_DECLARE_UNARY(isinf, ::tvm::isinf);
TVM_TOPI_DECLARE_UNARY(isnan, ::tvm::isnan);

}  // namespace array_api
}  // namespace topi
}  // namespace tvm
