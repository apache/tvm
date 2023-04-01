#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/array_api/base.h>
#include <tvm/topi/reduction.h>

namespace tvm {
namespace topi {
namespace array_api {

using runtime::DataType;
using te::Tensor;

using Indexer = std::function<PrimExpr(Tensor, Array<PrimExpr>)>;

inline void GetSpatialReductionDims(Array<PrimExpr> shape, bool reduction_last,
                                    PrimExpr* spatial_dim, PrimExpr* reduce_dim,
                                    Array<PrimExpr>* shape_prefix, bool* fake_spatial) {
  int ndim = shape.size();
  ICHECK_GE(ndim, 1);
  if (ndim == 1) {
    *spatial_dim = IntImm(DataType::Int(64), 1);
    *reduce_dim = shape[0];
    *shape_prefix = {};
    *fake_spatial = true;
  } else if (reduction_last) {
    *spatial_dim = shape[ndim - 2];
    *reduce_dim = shape[ndim - 1];
    *shape_prefix = Array<PrimExpr>{shape.begin(), shape.end() - 2};
    *fake_spatial = false;
  } else {
    *spatial_dim = shape[ndim - 1];
    *reduce_dim = shape[ndim - 2];
    *shape_prefix = Array<PrimExpr>{shape.begin(), shape.end() - 2};
    *fake_spatial = false;
  }
}

inline Array<PrimExpr> IndexTensor(Array<PrimExpr> indices, const Optional<PrimExpr>& spatial_index,
                                   const PrimExpr& reduction_axis, bool reduction_last) {
  if (reduction_last) {
    if (spatial_index.defined()) {
      indices.push_back(spatial_index.value());
    }
    indices.push_back(reduction_axis);
  } else {
    indices.push_back(reduction_axis);
    if (spatial_index.defined()) {
      indices.push_back(spatial_index.value());
    }
  }
  return indices;
}

inline PrimExpr IndexWithCast(const Tensor& x, const Array<PrimExpr>& indices, DataType dtype) {
  PrimExpr val = x(indices);
  return dtype.is_void() ? val : tvm::cast(dtype, val);
}

Tensor Matmul(Tensor a, Tensor b, DataType out_dtype, arith::Analyzer* analyzer) {
  static IntImm zero = IntImm(DataType::Int(64), 0);
  constexpr const bool transpose_a = false;
  constexpr const bool transpose_b = false;
  int a_ndim = a->shape.size();
  int b_ndim = b->shape.size();
  if (a_ndim == 0 || b_ndim == 0) {
    LOG(FATAL) << "ValueError: matmul does not support 0-dimensional tensors, but got " << a->shape
               << " and " << b->shape;
  }
  PrimExpr a_reduce_dim, a_spatial_dim;
  Array<PrimExpr> a_shape_prefix;
  bool a_fake_spatial;
  GetSpatialReductionDims(a->shape, !transpose_a, &a_spatial_dim, &a_reduce_dim, &a_shape_prefix,
                          &a_fake_spatial);
  PrimExpr b_reduce_dim, b_spatial_dim;
  Array<PrimExpr> b_shape_prefix;
  bool b_fake_spatial;
  GetSpatialReductionDims(b->shape, transpose_b, &b_spatial_dim, &b_reduce_dim, &b_shape_prefix,
                          &b_fake_spatial);
  if (!ProveEqual(analyzer, a_reduce_dim, b_reduce_dim)) {
    LOG(FATAL) << "ValueError: shapes " << a->shape << " and " << b->shape
               << " are not aligned for matmul";
  }
  Array<PrimExpr> shape_prefix = BroadcastShape(a_shape_prefix, b_shape_prefix, analyzer);
  Array<PrimExpr> shape{shape_prefix.begin(), shape_prefix.end()};
  if (!a_fake_spatial) {
    shape.push_back(a_spatial_dim);
  }
  if (!b_fake_spatial) {
    shape.push_back(b_spatial_dim);
  }
  tir::IterVar k = te::reduce_axis(Range(zero, a_reduce_dim), "k");
  try {
    return te::compute(
        shape,
        [&](const Array<Var>& indices) -> PrimExpr {
          Optional<PrimExpr> a_spatial_idx, b_spacial_idx;
          if (a_fake_spatial && b_fake_spatial) {
            a_spatial_idx = NullOpt;
            b_spacial_idx = NullOpt;
          } else if (a_fake_spatial) {
            a_spatial_idx = NullOpt;
            b_spacial_idx = indices[indices.size() - 1];
          } else if (b_fake_spatial) {
            a_spatial_idx = indices[indices.size() - 1];
            b_spacial_idx = NullOpt;
          } else {
            a_spatial_idx = indices[indices.size() - 2];
            b_spacial_idx = indices[indices.size() - 1];
          }
          Array<PrimExpr> indices_prefix{indices.begin(), indices.begin() + shape_prefix.size()};
          PrimExpr a_val =
              a(IndexTensor(BroadcastIndices(indices_prefix, a_shape_prefix, shape_prefix),
                            a_spatial_idx, k, !transpose_a));
          PrimExpr b_val =
              b(IndexTensor(BroadcastIndices(indices_prefix, b_shape_prefix, shape_prefix),
                            b_spacial_idx, k, transpose_b));
          DataType a_dtype = a->dtype;
          DataType b_dtype = b->dtype;
          if (!a_dtype.is_void() && !b_dtype.is_void()) {
            if (!out_dtype.is_void()) {
              a_val = tvm::cast(out_dtype, a_val);
              b_val = tvm::cast(out_dtype, b_val);
            }
            return tvm::sum(a_val * b_val, {k});
          }
          throw NotDerivable("NotDerivable: dtype unknown");
        },
        "matmul");
  } catch (const NotDerivable& e) {
    return te::placeholder(shape, out_dtype, "matmul");
  }
}

TVM_REGISTER_GLOBAL("topi.array_api.matmul")
    .set_body_typed([](Tensor a, Tensor b, DataType out_dtype) {
      return Matmul(a, b, out_dtype, nullptr);
    });

}  // namespace array_api
}  // namespace topi
}  // namespace tvm
