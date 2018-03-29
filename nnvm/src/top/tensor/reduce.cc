/*!
 *  Copyright (c) 2017 by Contributors
 * \file reduce.cc
 * \brief reduce operator.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/reduction.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// reduce
DMLC_REGISTER_PARAMETER(ReduceParam);

inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  if (axis.ndim() == 0) {
    if (keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }
  CHECK_LT(axis[axis.ndim() - 1], ishape.ndim())
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " Exceeds input dimensions " << ishape;

  TShape in_axis = axis;
  for (auto& i : in_axis) {
    i = i < 0 ? i + ishape.ndim(): i;
    CHECK_GE(i, 0) << "axis out of bounds in reduce operator";
    CHECK_LT(i, ishape.ndim()) << "axis out of bounds in reduce operator";
  }
  std::sort(in_axis.begin(), in_axis.end());

  if (keepdims) {
    TShape oshape(ishape);
    if (exclude) {
      for (dim_t i = 0, j = 0; i < ishape.ndim(); ++i) {
        if (j < in_axis.ndim() && i == in_axis[j]) {
          ++j;
          continue;
        }
        oshape[i] = 1;
      }
      return oshape;
    }

    for (dim_t i = 0; i < in_axis.ndim(); ++i) {
      oshape[in_axis[i]] = 1;
    }
    return oshape;
  }

  if (exclude) {
    TShape oshape = TShape(in_axis.ndim());
    for (dim_t i = 0; i < in_axis.ndim(); ++i) {
      oshape[i] = ishape[in_axis[i]];
    }
    return oshape;
  }
  TShape oshape = TShape(std::max<dim_t>(1, ishape.ndim() - in_axis.ndim()));
  for (dim_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
    if (j < in_axis.ndim() && i == in_axis[j]) {
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
  NNVM_ASSIGN_INPUT_SHAPE(
      attrs, *out_attrs, 0,
      ReduceShapeImpl((*in_attrs)[0], param.axis,
                      param.keepdims, param.exclude));
  return true;
}

template<typename PType>
inline void AxesParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

#define NNVM_REGISTER_REDUCE_OP(op)                                     \
  NNVM_REGISTER_OP(op)                                                  \
  .add_argument("data", "Tensor", "The input")                          \
  .add_arguments(ReduceParam::__FIELDS__())                             \
  .set_attr_parser(AxesParamParser<ReduceParam>)                        \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
  .set_attr<FInferShape>("FInferShape", ReduceShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)               \
  .set_num_inputs(1)                                                    \
  .set_num_outputs(1)

NNVM_REGISTER_REDUCE_OP(sum)
.describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    Array<Expr> axis;
    if (param.exclude) {
      std::set<dim_t> exclude_axis;
      for (dim_t i = 0; i < param.axis.ndim(); ++i) {
        exclude_axis.insert(param.axis[i]);
      }
      for (dim_t i = 0; i < inputs[0].ndim(); ++i) {
        if (exclude_axis.count(i) == 0) {
          axis.push_back(make_const(Int(32), i));
        }
      }
    } else {
      axis = ShapeToArray(param.axis);
    }
    return Array<Tensor>{
      topi::sum(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    return std::vector<NodeEntry>{
      MakeNode("expand_like", n->attrs.name + "_grad",
               {ograds[0], n->inputs[0]},
               {{"axis", axis.str()},
                {"exclude", std::to_string(param.exclude)}})
  };
});

NNVM_REGISTER_REDUCE_OP(max)
.describe(R"code(Computes the max of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    auto axis = ShapeToArray(param.axis);
    return Array<Tensor>{
      topi::max(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                             {ograds[0], n->inputs[0]},
                             {{"axis", axis.str()},
                              {"keepdims", std::to_string(param.keepdims)},
                              {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_max_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});

NNVM_REGISTER_REDUCE_OP(min)
.describe(R"code(Computes the min of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    auto axis = ShapeToArray(param.axis);
    return Array<Tensor>{
      topi::min(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                              {ograds[0], n->inputs[0]},
                              {{"axis", axis.str()},
                               {"keepdims", std::to_string(param.keepdims)},
                               {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_min_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});


}  // namespace top
}  // namespace nnvm
