// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNVM

#include <nnvm/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/graph_attr_types.h>
#include <utility>

namespace myproject {

using nnvm::FListInputNames;
using nnvm::FMutateInputs;
using nnvm::FInferShape;
using nnvm::FInferType;
using nnvm::FInplaceOption;
using nnvm::NodeAttrs;
using nnvm::TShape;
using nnvm::array_view;

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

inline std::vector<std::pair<int, int> > InplaceIn0Out0(const NodeAttrs& attrs) {
  return {{0, 0}};
}

// simple demonstration of reshape.
NNVM_REGISTER_OP(reshape)
.describe("reshape source to target shape")
.set_num_inputs(1)
.set_attr_parser(
    [](NodeAttrs* attrs) {
      // parse attr parser to get target attribute
      TShape target;
      std::istringstream is(attrs->dict.at("target"));
      CHECK(is >> target);
      attrs->parsed = std::move(target);
    })
.attr<FInferShape>(
    "FInferShape", [] (const NodeAttrs& attrs,
                       std::vector<TShape> *ishape,
                       std::vector<TShape> *oshape) {
      // get parsed attribute
      const TShape& target = nnvm::get<TShape>(attrs.parsed);
      (*oshape)[0] = target;
      if ((*ishape)[0].ndim() == 0) return false;
      CHECK_EQ((*ishape)[0].Size(), target.Size())
          << "Reshape op: source target shape mismatch";
      return true;
    })
.attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);


NNVM_REGISTER_OP(cast)
.describe("cast source type to target")
.set_num_inputs(1)
.set_attr_parser(
    [](NodeAttrs* attrs) {
      // parse attr parser to get target attribute
      int dtype;
      std::istringstream is(attrs->dict.at("dtype"));
      CHECK(is >> dtype);
      attrs->parsed = std::move(dtype);
    })
.attr<FInferShape>("FInferShape", SameShape)
.attr<FInferType>(
    "FInferType", [](const NodeAttrs& attrs,
                     std::vector<int> *itype,
                     std::vector<int> *otype) {
      (*otype)[0] = nnvm::get<int>(attrs.parsed);
      return true;
    });


NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr<FInferShape>("FInferShape", SameShape)
.attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);

NNVM_REGISTER_OP(__add_symbol__)
.describe("Alias of add")
.set_num_inputs(2);

NNVM_REGISTER_OP(exp)
.describe("take exponential")
.set_num_inputs(1)
.attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(cross_device_copy)
.describe("Copy data across device.")
.set_num_inputs(1)
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
.attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  });

}  // namespace myproject
