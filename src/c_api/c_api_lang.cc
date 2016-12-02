/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Higher DSL build.
 * \file c_api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/split.h>
#include <tvm/schedule.h>
#include "./c_api_registry.h"

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_const)
.set_body([](const ArgStack& args,  RetValue *ret) {
    using Halide::Internal::make_const;
    if (args.at(0).type_id == kLong) {
      *ret = make_const(args.at(1), args.at(0).operator int64_t());
    } else if (args.at(0).type_id == kDouble) {
      *ret = make_const(args.at(1), args.at(0).operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  })
.add_argument("src", "Number", "source number")
.add_argument("dtype", "str", "data type");

TVM_REGISTER_API(_Array)
.set_body([](const ArgStack& args,  RetValue *ret) {
    std::vector<std::shared_ptr<Node> > data;
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args.at(i).type_id == kNodeHandle)
          << "need content of array to be NodeBase";
      data.push_back(args.at(i).sptr);
    }
    auto node = std::make_shared<ArrayNode>();
    node->data = std::move(data);
    ret->type_id = kNodeHandle;
    ret->sptr = node;
  });

TVM_REGISTER_API(_ArrayGetItem)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    int64_t i = args.at(1);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    ret->sptr = n->data[i];
    ret->type_id = kNodeHandle;
  });

TVM_REGISTER_API(_ArraySize)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(sptr.get())->data.size());
  });

TVM_REGISTER_API(Range)
.set_body([](const ArgStack& args,  RetValue *ret) {
    if (args.size() == 1) {
      *ret = Range(0, args.at(0));
    } else {
      *ret = Range(args.at(0), args.at(1));
    }
  })
.describe("create a domain range")
.add_argument("begin", "Expr", "beginning of the range.")
.add_argument("end", "Expr", "extent of the range");

TVM_REGISTER_API(_Tensor)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = TensorNode::make(args.at(0),
                            args.at(1),
                            args.at(2),
                            args.at(3),
                            args.at(4));
  });

TVM_REGISTER_API(_ComputeOp)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = ComputeOpNode::make(args.at(0),
                               args.at(1),
                               args.at(2),
                               args.at(3));
  });


TVM_REGISTER_API(_IterVar)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = IterVar(args.at(0), args.at(1), args.at(2));
  });


TVM_REGISTER_API(_DimSplit)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = DimSplitNode::make(args.at(0), args.at(1));
  });

TVM_REGISTER_API(_Schedule)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Schedule(args.at(0), args.at(1));
  });

}  // namespace tvm
