// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNVM

#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <utility>

using nnvm::FListInputNames;
using nnvm::FMutateInput;
using nnvm::NodeAttrs;

NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr("inplace_pair", std::make_pair(0, 0));


NNVM_REGISTER_OP(exp)
.describe("take exponmential")
.set_num_inputs(1)
.attr("inplace_pair", std::make_pair(0, 0));


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
