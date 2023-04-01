#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/array_api/base.h>

namespace tvm {
namespace topi {
namespace array_api {

using te::Tensor;
using tir::Var;

TVM_DLL Tensor astype(const Tensor& x, const runtime::DataType& dtype) {
  if (x->dtype.is_void() || dtype.is_void()) {
    throw NotDerivable("NotDerivable: unknown dtype");
  }
  return te ::compute(
      x->shape,
      [&](const Array<Var>& indices) {  //
        return tvm::cast(dtype, x(indices));
      },
      "astype");
}

TVM_REGISTER_GLOBAL("topi.array_api.astype").set_body_typed(astype);

}  // namespace array_api
}  // namespace topi
}  // namespace tvm
