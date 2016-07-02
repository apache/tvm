// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNGRAPH

#include <nngraph/op.h>
#include <utility>

NNGRAPH_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr("inplace_pair", std::make_pair(0, 0));


NNGRAPH_REGISTER_OP(add)
.attr<std::string>("nick_name", "plus");
