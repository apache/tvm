 /*!
 *  Copyright (c) 2018 by Contributors
 *  Code mainly used for test purposes.
 * \file api_test.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/attrs.h>
#include <tvm/api_registry.h>

namespace tvm {
// Attrs used to python API
struct TestAttrs : public AttrsNode<TestAttrs> {
  int axis;
  std::string name;
  Array<Expr> padding;
  TypedEnvFunc<int(int)> func;

  TVM_DECLARE_ATTRS(TestAttrs, "attrs.TestAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(10)
        .set_lower_bound(1)
        .set_upper_bound(10)
        .describe("axis field");
    TVM_ATTR_FIELD(name)
        .describe("name");
    TVM_ATTR_FIELD(padding)
        .describe("padding of input")
        .set_default(Array<Expr>({0, 0}));
    TVM_ATTR_FIELD(func)
        .describe("some random env function")
        .set_default(TypedEnvFunc<int(int)>(nullptr));
  }
};

TVM_REGISTER_NODE_TYPE(TestAttrs);

TVM_REGISTER_API("_nop")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
  });

TVM_REGISTER_API("_context_test")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    DLContext ctx = args[0];
    int dtype = args[1];
    int did = args[2];
    CHECK_EQ(static_cast<int>(ctx.device_type), dtype);
    CHECK_EQ(static_cast<int>(ctx.device_id), did);
    *ret = ctx;
  });

// internal fucntion used for debug and testing purposes
TVM_REGISTER_API("_ndarray_use_count")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    runtime::NDArray nd = args[0];
    // substract the current one
    *ret = (nd.use_count() - 1);
  });

}  // namespace tvm
