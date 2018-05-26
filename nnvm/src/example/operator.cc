// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNVM

#include <nnvm/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <utility>

namespace myproject {

using nnvm::FListInputNames;
using nnvm::FMutateInput;
using nnvm::FInferShape;
using nnvm::NodeAttrs;
using nnvm::TShape;
using nnvm::array_view;

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      array_view<TShape*> ishape,
                      array_view<TShape*> oshape) {
  if (ishape.size() == 0 || ishape[0]->ndim() == 0) return false;
  for (TShape* pshape : oshape) {
    *pshape = *ishape[0];
  }
  for (TShape* pshape : ishape) {
    *pshape = *ishape[0];
  }
  return true;
}

NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(__add_symbol__)
.describe("Alias of add")
.set_num_inputs(2);

NNVM_REGISTER_OP(exp)
.describe("take exponmential")
.set_num_inputs(1)
.attr("inplace_pair", std::make_pair(0, 0))
.attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(conv2d)
.describe("take conv of input")
.set_num_inputs(2)
.attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  });

NNVM_REGISTER_OP(add)
.attr<std::string>("nick_name", "plus");

NNVM_REGISTER_OP(assign)
.set_num_inputs(2)
.set_num_outputs(1)
.attr<FMutateInput>("FMutateInput", [](const NodeAttrs& attrs, uint32_t index) {
    return index == 0;
  });

}  // namespace myproject
