#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/array_api/base.h>
#include <tvm/topi/reduction.h>

#include <numeric>

namespace tvm {
namespace topi {
namespace array_api {

Array<PrimExpr> PrefixSum(const Array<PrimExpr>& a) {
  if (a.empty()) {
    return Array<PrimExpr>();
  }
  int n = a.size();
  Array<PrimExpr> ret;
  ret.reserve(n);
  ret.push_back(a[0]);
  for (int i = 1; i < n; ++i) {
    ret.push_back(ret[i - 1] + a[i]);
  }
  return ret;
}

PrimExpr SearchInOrderedArray(const Array<PrimExpr>& separators, const PrimExpr& i,
                              const std::function<PrimExpr(int)>& f_pos) {
  int n = separators.size();
  ICHECK_GT(n, 0);
  if (n == 1) {
    return f_pos(0);
  }
  PrimExpr ret = tvm::if_then_else(/*cond=*/i < separators[n - 2],
                                   /*true_value=*/f_pos(n - 2),
                                   /*false_value=*/f_pos(n - 1));
  for (int j = n - 3; j >= 0; --j) {
    ret = tvm::if_then_else(
        /*cond=*/i < separators[j],
        /*true_value=*/f_pos(j),
        /*false_value=*/ret);
  }
  return ret;
}

te::Tensor broadcast_to(te::Tensor data, Array<PrimExpr> shape) {
  CHECK_LE(data->shape.size(), shape.size())
      << "ValueError: Cannot broadcast shape " << data->shape << " to " << shape;
  return te::compute(shape, [&](const Array<Var>& indices) -> PrimExpr {
    Array<PrimExpr> new_indices =
        BroadcastIndices({indices.begin(), indices.end()}, data->shape, shape);
    return data(new_indices);
  });
}

te::Tensor repeat(te::Tensor a, Array<PrimExpr> repeats, Optional<IntImm> _axis) {
  int ndim = a->shape.size();
  CHECK(!repeats.empty()) << "ValueError: repeats must be non-empty";
  // Case 1. axis == None and |repeats| == 1
  if (!_axis.defined()) {
    CHECK_EQ(repeats.size(), 1) << "ValueError: repeats must be a single integer when axis is None";
    Array<PrimExpr> new_shape = {ProdShape(a->shape) * repeats[0]};

    return te::compute(
        new_shape,
        [&](const Array<Var>& indices) -> PrimExpr {
          ICHECK_EQ(indices.size(), 1);
          PrimExpr i = tvm::floordiv(indices[0], repeats[0]);
          return a(UnravelIndex(i, a->shape));
        },
        "repeat");
  }
  // Case 2. axis != None and |repeats| == 1
  int axis = NormalizeAxis(ndim, _axis.value()->value);
  if (repeats.size() == 1) {
    Array<PrimExpr> new_shape = a->shape;
    new_shape.Set(axis, a->shape[axis] * repeats[0]);
    return te::compute(
        new_shape,
        [&](const Array<Var>& _indices) -> PrimExpr {
          Array<PrimExpr> indices(_indices.begin(), _indices.end());
          indices.Set(axis, tvm::floordiv(indices[axis], repeats[0]));
          return a(indices);
        },
        "repeat");
  }
  // Case 3. axis != None and |repeats| >= 2
  int n = repeats.size();
  ICHECK_GE(n, 2);
  Array<PrimExpr> sum_repeat = PrefixSum(repeats);
  {
    // Check shape[axis] == |repeats|
    const auto* dim = a->shape[axis].as<IntImmNode>();
    CHECK(dim && dim->value == static_cast<int64_t>(repeats.size()))
        << "ValueError: repeats must have the same length as the dimension on axis " << axis;
  }
  Array<PrimExpr> new_shape = a->shape;
  new_shape.Set(axis, sum_repeat[n - 1]);
  return te::compute(
      new_shape,
      [&](const Array<Var>& _indices) -> PrimExpr {
        Array<PrimExpr> indices(_indices.begin(), _indices.end());
        indices.Set(axis, SearchInOrderedArray(sum_repeat, indices[axis],
                                               [](int i) { return IntImm(DataType::Int(64), i); }));
        return a(indices);
      },
      "repeat");
}

te::Tensor tile(te::Tensor a, Array<PrimExpr> repeats) {
  DataType i64 = DataType::Int(64);
  if (repeats.size() < a->shape.size()) {
    Array<PrimExpr> new_repeats;
    new_repeats.reserve(a->shape.size());
    int diff = static_cast<int>(a->shape.size()) - static_cast<int>(repeats.size());
    for (int i = 0; i < diff; ++i) {
      new_repeats.push_back(IntImm(i64, 1));
    }
    for (uint32_t i = 0; i < repeats.size(); ++i) {
      new_repeats.push_back(repeats[i]);
    }
    repeats = new_repeats;
  }
  int diff = static_cast<int>(repeats.size()) - static_cast<int>(a->shape.size());
  int a_ndim = a->shape.size();
  Array<PrimExpr> new_shape;
  new_shape.reserve(repeats.size());
  for (int i = 0; i < diff; ++i) {
    new_shape.push_back(repeats[i]);
  }
  for (int i = 0; i < a_ndim; ++i) {
    new_shape.push_back(a->shape[i] * repeats[i + diff]);
  }
  return te::compute(
      new_shape,
      [&](const Array<Var>& indices) -> PrimExpr {
        Array<PrimExpr> new_indices;
        new_indices.reserve(a_ndim);
        for (int i = 0; i < a_ndim; ++i) {
          new_indices.push_back(tvm::floormod(indices[i + diff], a->shape[i]));
        }
        return a(new_indices);
      },
      "tile");
}

te::Tensor expand_dims(te::Tensor a, Array<IntImm> _axis) {
  int ndim = a->shape.size();
  int new_ndim = ndim + _axis.size();
  std::vector<int64_t> axis = NormalizeAxes(new_ndim, _axis);
  Array<PrimExpr> new_shape;
  new_shape.reserve(new_ndim);
  for (int i = 0, j = 0; i < new_ndim; ++i) {
    if (std::binary_search(axis.begin(), axis.end(), i)) {
      new_shape.push_back(IntImm(DataType::Int(64), 1));
    } else {
      new_shape.push_back(a->shape[j++]);
    }
  }
  ICHECK_EQ(new_shape.size(), new_ndim);
  return te::compute(
      new_shape,
      [&](const Array<Var>& indices) -> PrimExpr {
        Array<PrimExpr> new_indices;
        new_indices.reserve(ndim);
        for (int i = 0; i < new_ndim; ++i) {
          if (!std::binary_search(axis.begin(), axis.end(), i)) {
            new_indices.push_back(indices[i]);
          }
        }
        return a(new_indices);
      },
      "expand_dims");
}

te::Tensor permute_dims(te::Tensor a, Optional<Array<IntImm>> _axes) {
  int ndim = a->shape.size();
  std::vector<int64_t> axes;
  if (_axes.defined()) {
    axes = ArrayToVector(_axes.value());
    for (int i = 0; i < ndim; ++i) {
      int64_t axis = axes[i];
      CHECK(-ndim <= axis && axis < ndim) << "ValueError: axis must be in range [-ndim, ndim), "
                                             "where ndim is "
                                          << ndim << ", but got: " << _axes;
      if (axis < 0) {
        axes[i] += ndim;
      }
    }
    std::set<int64_t> axis_set(axes.begin(), axes.end());
    for (int i = 0; i < ndim; ++i) {
      CHECK(axis_set.count(i)) << "ValueError: axis must be permutation of [0, ..., ndim), "
                                  "where ndim is "
                               << ndim << ", but got: " << _axes;
    }
  } else {
    axes.resize(ndim);
    std::iota(axes.rbegin(), axes.rend(), 0);
  }
  std::vector<int64_t> inv_axis(axes.size());
  Array<PrimExpr> new_shape;
  new_shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    new_shape.push_back(a->shape[axes[i]]);
    inv_axis[axes[i]] = i;
  }
  return te::compute(
      new_shape,
      [&](const Array<Var>& indices) -> PrimExpr {
        Array<PrimExpr> new_indices;
        new_indices.reserve(ndim);
        for (int i = 0; i < ndim; ++i) {
          new_indices.push_back(indices[inv_axis[i]]);
        }
        return a(new_indices);
      },
      "permute_dims");
}

te::Tensor squeeze(te::Tensor a, Optional<Array<IntImm>> _axes) {
  int ndim = a->shape.size();
  std::vector<int64_t> axes;
  if (_axes.defined()) {
    axes = NormalizeAxes(ndim, _axes.value());
    for (int i : axes) {
      const auto* dim = a->shape[i].as<IntImmNode>();
      if (dim == nullptr) {
        // TODO(@junrushao): handle symbolic shape better
        // assume it is 1;
        continue;
      }
      CHECK_EQ(dim->value, 1) << "ValueError: cannot select an axis to squeeze out "
                                 "which has size not equal to one. "
                                 "Squeeze axis "
                              << i << " has size " << dim->value << ", the shape is " << a->shape;
    }
  } else {
    for (int i = 0; i < ndim; ++i) {
      const auto* dim = a->shape[i].as<IntImmNode>();
      if (!dim) {
        // TODO(@junrushao): handle symbolic shape better
        // assume it is 1;
        throw NotDerivable("NotDerivable: symbolic shape");
      }
      if (dim->value == 1) {
        axes.push_back(i);
      }
    }
  }
  Array<PrimExpr> new_shape;
  new_shape.reserve(ndim - axes.size());
  for (int i = 0; i < ndim; ++i) {
    if (!std::binary_search(axes.begin(), axes.end(), i)) {
      new_shape.push_back(a->shape[i]);
    }
  }
  return te::compute(
      new_shape,
      [&](const Array<Var>& indices) -> PrimExpr {
        Array<PrimExpr> new_indices;
        new_indices.reserve(ndim);
        for (int i = 0, j = 0; i < ndim; ++i) {
          if (!std::binary_search(axes.begin(), axes.end(), i)) {
            new_indices.push_back(indices[j++]);
          } else {
            new_indices.push_back(IntImm(DataType::Int(64), 0));
          }
        }
        return a(new_indices);
      },
      "squeeze");
}

te::Tensor reshape(te::Tensor x, Array<PrimExpr> shape) {
  Array<PrimExpr> x_shape = x->shape;
  int unknown_dim = -1;
  PrimExpr prod_dim = IntImm(DataType::Int(64), 1);
  PrimExpr prod_known_dim = IntImm(DataType::Int(64), 1);
  for (PrimExpr dim : x->shape) {
    prod_dim *= dim;
  }
  for (uint32_t i = 0; i < shape.size(); ++i) {
    PrimExpr dim = shape[i];
    if (const auto* int_imm = dim.as<IntImmNode>()) {
      if (int_imm->value != -1 && int_imm->value <= 0) {
        LOG(FATAL) << "ValueError: cannot reshape to a shape with negative dimension besides -1, "
                   << "but got " << shape;
      }
      if (int_imm->value == -1) {
        if (unknown_dim != -1) {
          LOG(FATAL) << "ValueError: Can only specify one unknown dimension, "
                     << "but got " << shape;
        }
        unknown_dim = i;
        continue;
      }
    }
    prod_known_dim *= dim;
  }
  if (const auto* int_prod_dim = prod_dim.as<IntImmNode>()) {
    if (const auto* int_prod_known_dim = prod_known_dim.as<IntImmNode>()) {
      if (unknown_dim == -1) {
        CHECK_EQ(int_prod_dim->value, int_prod_known_dim->value)
            << "ValueError: cannot reshape array of shape " << x_shape << " into shape " << shape;
      } else {
        CHECK_EQ(int_prod_dim->value % int_prod_known_dim->value, 0)
            << "ValueError: cannot reshape array of shape " << x_shape << " into shape " << shape;
      }
    }
  }
  if (unknown_dim != -1) {
    // TODO(@junrushao): handle symbolic shape divisibility
    shape.Set(unknown_dim, floordiv(prod_dim, prod_known_dim));
  }
  return te::compute(
      shape,
      [&](const Array<Var>& _indices) -> PrimExpr {
        Array<PrimExpr> indices{_indices.begin(), _indices.end()};
        PrimExpr i = RavelIndex(indices, shape);
        Array<PrimExpr> x_indices = UnravelIndex(i, x_shape);
        return x(x_indices);
      },
      "reshape");
}

te::Tensor flatten(te::Tensor x, Integer _start_axis, Integer _end_axis) {
  int start_axis = _start_axis->value;
  int end_axis = _end_axis->value;
  Array<PrimExpr> shape = x->shape;
  int ndim = shape.size();
  if (ndim == 0) {
    CHECK(start_axis == 0 && end_axis == -1)
        << "ValueError: Only allow start_axis = 0 and end_axis = -1 for 0-dim input, but got "
           "start_axis = "
        << start_axis << ", end_axis = " << end_axis << ", x.shape = ()";
    Array<PrimExpr> result_shape;
    result_shape.push_back(IntImm(DataType::Int(64), 1));
    return te::compute(
        result_shape,
        [&](const Array<Var>& _indices) -> PrimExpr {
          Array<PrimExpr> idx;
          return x(idx);
        },
        "flatten");
  }
  start_axis = NormalizeAxis(ndim, start_axis);
  end_axis = NormalizeAxis(ndim, end_axis);
  CHECK_LE(start_axis, end_axis)
      << "ValueError: start_axis must be less than or equal to end_axis, but got start_axis = "
      << start_axis << ", end_axis = " << end_axis << ", x.shape = " << shape;
  Array<PrimExpr> new_shape;
  {
    new_shape.reserve(ndim);
    new_shape.insert(new_shape.end(), shape.begin(), shape.begin() + start_axis);
    PrimExpr prod = shape[start_axis];
    for (int i = start_axis + 1; i <= end_axis; ++i) {
      prod *= shape[i];
    }
    new_shape.push_back(prod);
    new_shape.insert(new_shape.end(), shape.begin() + end_axis + 1, shape.end());
  }
  return te::compute(new_shape, [&](const Array<Var>& indices) -> PrimExpr {
    Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    new_indices.insert(new_indices.end(), indices.begin(), indices.begin() + start_axis);
    Array<PrimExpr> unraveled = UnravelIndex(indices[start_axis],          //
                                             {shape.begin() + start_axis,  //
                                              shape.begin() + end_axis + 1});
    new_indices.insert(new_indices.end(), unraveled.begin(), unraveled.end());
    new_indices.insert(new_indices.end(), indices.begin() + start_axis + 1, indices.end());
    return x(new_indices);
  });
}

Array<te::Tensor> SplitByRange(te::Tensor a, Array<Range> ranges, int axis,
                               arith::Analyzer* analyzer) {
  axis = NormalizeAxis(a->shape.size(), axis);
  Array<te::Tensor> result;
  result.reserve(ranges.size());
  for (Range range : ranges) {
    Array<PrimExpr> result_shape = a->shape;
    result_shape.Set(axis, analyzer->Simplify(range->extent));
    result.push_back(te::compute(
        result_shape,
        [&](const Array<Var>& indices) -> PrimExpr {
          Array<PrimExpr> idx;
          idx.reserve(a->shape.size());
          idx.insert(idx.end(), indices.begin(), indices.begin() + axis);
          idx.push_back(indices[axis] + analyzer->Simplify(range->min));
          idx.insert(idx.end(), indices.begin() + axis + 1, indices.end());
          return a(idx);
        },
        "split"));
  }
  return result;
}

Array<te::Tensor> split(te::Tensor a, Array<PrimExpr> sections, int axis,
                        arith::Analyzer* analyzer) {
  axis = NormalizeAxis(a->shape.size(), axis);
  PrimExpr n = a->shape[axis];
  std::vector<Range> ranges;
  ranges.reserve(sections.size() + 1);
  PrimExpr cur = IntImm(DataType::Int(64), 0);
  for (PrimExpr section : sections) {
    if (analyzer->CanProve(section <= 0)) {
      section = make_zero(section.dtype());
    } else if (analyzer->CanProve(n < section)) {
      section = n;
    }
    Range range(cur, section);
    ranges.push_back(range);
    cur = section;
  }
  ranges.push_back(Range(cur, n));
  for (Range& range : ranges) {
    if (analyzer->CanProve(range->extent <= 0)) {
      range = Range::FromMinExtent(range->min, make_zero(range->extent.dtype()));
    }
  }
  return SplitByRange(a, ranges, axis, analyzer);
}

Array<te::Tensor> split(te::Tensor a, int num_parts, int axis, arith::Analyzer* analyzer) {
  CHECK_GT(num_parts, 0) << "ValueError: num_parts must be positive, but got " << num_parts;
  axis = NormalizeAxis(a->shape.size(), axis);
  PrimExpr n = a->shape[axis];
  PrimExpr len = ceildiv(n, IntImm(DataType::Int(64), num_parts));
  Array<Range> ranges;
  ranges.reserve(num_parts);
  for (int i = 0; i < num_parts; ++i) {
    if (i == num_parts - 1) {
      ranges.push_back(Range(len * i, n));
    } else {
      ranges.push_back(Range::FromMinExtent(len * i, len));
    }
  }
  return SplitByRange(a, ranges, axis, analyzer);
}

te::Tensor concat(Array<te::Tensor> arrays, int axis, arith::Analyzer* analyzer) {
  CHECK(!arrays.empty()) << "ValueError: There must be at least one array to be concatenated";
  int n = arrays.size();
  // Check `ndim`s are the same for all inputs
  for (int i = 1; i < n; ++i) {
    if (arrays[i]->shape.size() != arrays[0]->shape.size()) {
      std::ostringstream os;
      os << "ValueError: all the input array dimensions except for the "
            "concatenation axis must match exactly, but got:";
      for (int j = 0; j < n; ++j) {
        os << " " << arrays[j]->shape;
      }
      LOG(FATAL) << os.str();
    }
  }
  int ndim = arrays[0]->shape.size();
  axis = NormalizeAxis(ndim, axis);
  // Check the other dimensions are equal, and summarize the `axis` dimension
  Array<PrimExpr> sum_dim;
  sum_dim.reserve(n);
  sum_dim.push_back(arrays[0]->shape[axis]);
  for (int i = 1; i < n; ++i) {
    for (int dim = 0; dim < ndim; ++dim) {
      if (dim == axis) {
        sum_dim.push_back(sum_dim[i - 1] + arrays[i]->shape[dim]);
      } else if (analyzer->CanProve(arrays[0]->shape[dim] != arrays[i]->shape[dim])) {
        std::ostringstream os;
        os << "ValueError: all the input array dimensions except for the "
              "concatenation axis must match exactly, but got:";
        for (int j = 0; j < n; ++j) {
          os << " " << arrays[j]->shape;
        }
        LOG(FATAL) << os.str();
      } else if (!analyzer->CanProveEqual(arrays[0]->shape[dim], arrays[i]->shape[dim])) {
        std::ostringstream os;
        os << "NotDerivable: all the input array dimensions except for the "
              "concatenation axis must match exactly, but got:";
        for (int j = 0; j < n; ++j) {
          os << " " << arrays[j]->shape;
        }
        throw NotDerivable(os.str());
      }
    }
  }
  Array<PrimExpr> shape = arrays[0]->shape;
  shape.Set(axis, sum_dim[n - 1]);
  return te::compute(
      shape,
      [&](const Array<Var>& _indices) -> PrimExpr {
        return SearchInOrderedArray(sum_dim, _indices[axis], [&](int i) {
          te::Tensor a = arrays[i];
          Array<PrimExpr> indices{_indices.begin(), _indices.end()};
          if (i == 0) {
            return a(indices);
          }
          indices.Set(axis, indices[axis] - sum_dim[i - 1]);
          return a(indices);
        });
      },
      "concat");
}

// Array<te::Tensor> Split(te::Tensor a, int in

TVM_REGISTER_GLOBAL("topi.array_api.broadcast_to").set_body_typed(broadcast_to);
TVM_REGISTER_GLOBAL("topi.array_api.repeat").set_body_typed(repeat);
TVM_REGISTER_GLOBAL("topi.array_api.tile").set_body_typed(tile);
TVM_REGISTER_GLOBAL("topi.array_api.expand_dims").set_body_typed(expand_dims);
TVM_REGISTER_GLOBAL("topi.array_api.permute_dims").set_body_typed(permute_dims);
TVM_REGISTER_GLOBAL("topi.array_api.squeeze").set_body_typed(squeeze);
TVM_REGISTER_GLOBAL("topi.array_api.reshape").set_body_typed(reshape);
TVM_REGISTER_GLOBAL("topi.array_api.flatten").set_body_typed(flatten);
TVM_REGISTER_GLOBAL("topi.array_api.split")
    .set_body_typed([](te::Tensor a, ObjectRef indices_or_sections,
                       IntImm axis) -> Array<te::Tensor> {
      arith::Analyzer analyzer;
      if (const auto* sections = indices_or_sections.as<IntImmNode>()) {
        return split(a, sections->value, axis->value, &analyzer);
      } else {
        return split(a, Downcast<Array<PrimExpr>>(indices_or_sections), axis->value, &analyzer);
      }
    });
TVM_REGISTER_GLOBAL("topi.array_api.concat")
    .set_body_typed([](Array<te::Tensor> arrays, IntImm axis) -> te::Tensor {
      arith::Analyzer analyzer;
      return concat(arrays, axis->value, &analyzer);
    });

}  // namespace array_api
}  // namespace topi
}  // namespace tvm
