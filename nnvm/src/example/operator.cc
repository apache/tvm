// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNVM

#include <nnvm/op.h>
#include <utility>

NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr("inplace_pair", std::make_pair(0, 0));


NNVM_REGISTER_OP(add)
.attr<std::string>("nick_name", "plus");
