/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Higher DSL build.
 * \file c_api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/buffer.h>
#include <tvm/schedule.h>
#include "./c_api_registry.h"

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_const)
.set_body([](const ArgStack& args,  RetValue *ret) {
    if (args.at(0).type_code == kInt) {
      *ret = make_const(args.at(1), args.at(0).operator int64_t());
    } else if (args.at(0).type_code == kFloat) {
      *ret = make_const(args.at(1), args.at(0).operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  })
.add_argument("src", "Number", "source number")
.add_argument("dtype", "str", "data type");


TVM_REGISTER_API(_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
  *ret = ir::StringImm::make(args.at(0));
});


TVM_REGISTER_API(_Array)
.set_body([](const ArgStack& args,  RetValue *ret) {
    std::vector<std::shared_ptr<Node> > data;
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args.at(i).type_code == kNodeHandle)
          << "need content of array to be NodeBase";
      data.push_back(args.at(i).sptr);
    }
    auto node = std::make_shared<ArrayNode>();
    node->data = std::move(data);
    ret->type_code = kNodeHandle;
    ret->sptr = node;
  });

TVM_REGISTER_API(_ArrayGetItem)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    int64_t i = args.at(1);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    ret->sptr = n->data[i];
    ret->type_code = kNodeHandle;
  });

TVM_REGISTER_API(_ArraySize)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(sptr.get())->data.size());
  });

TVM_REGISTER_API(_Map)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK_EQ(args.size() % 2, 0U);
    MapNode::ContainerType data;
    for (size_t i = 0; i < args.size(); i += 2) {
      CHECK(args.at(i).type_code == kNodeHandle)
          << "need content of array to be NodeBase";
      CHECK(args.at(i + 1).type_code == kNodeHandle)
          << "need content of array to be NodeBase";
      data.emplace(std::make_pair(args.at(i).sptr, args.at(i + 1).sptr));
    }
    auto node = std::make_shared<MapNode>();
    node->data = std::move(data);
    ret->type_code = kNodeHandle;
    ret->sptr = node;
  });

TVM_REGISTER_API(_MapSize)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(n->data.size());
  });

TVM_REGISTER_API(_MapGetItem)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    CHECK(args.at(1).type_code == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    auto it = n->data.find(args.at(1).sptr);
    CHECK(it != n->data.end())
        << "cannot find the corresponding key in the Map";
    ret->sptr = (*it).second;
    ret->type_code = kNodeHandle;
  });

TVM_REGISTER_API(_MapCount)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    CHECK(args.at(1).type_code == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(n->data.count(args.at(1).sptr));
  });

TVM_REGISTER_API(_MapItems)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_code == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    auto rkvs = std::make_shared<ArrayNode>();
    for (const auto& kv : n->data) {
      rkvs->data.push_back(kv.first);
      rkvs->data.push_back(kv.second);
    }
    ret->sptr = rkvs;
    ret->type_code = kNodeHandle;
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

TVM_REGISTER_API(_Buffer)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = BufferNode::make(args.at(0),
                            args.at(1),
                            args.at(2),
                            args.at(3),
                            args.at(4));
  });

TVM_REGISTER_API(_Tensor)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = TensorNode::make(args.at(0),
                            args.at(1),
                            args.at(2),
                            args.at(3));
  });

TVM_REGISTER_API(_TensorEqual)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = args.at(0).operator Tensor() == args.at(1).operator Tensor();
  });

TVM_REGISTER_API(_TensorHash)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = static_cast<int64_t>(
        std::hash<Tensor>()(args.at(0).operator Tensor()));
  });

TVM_REGISTER_API(_Placeholder)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Placeholder(args.at(0),
                       args.at(1),
                       args.at(2));
  });

TVM_REGISTER_API(_ComputeOp)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = ComputeOpNode::make(args.at(0),
                               args.at(1),
                               args.at(2));
  });

TVM_REGISTER_API(_OpGetOutput)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = args.at(0).operator Operation().output(
        args.at(1).operator int64_t());
  });


TVM_REGISTER_API(_IterVar)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = IterVar(args.at(0), args.at(1), args.at(2));
  });

TVM_REGISTER_API(_Schedule)
.set_body([](const ArgStack& args, RetValue *ret) {
    *ret = Schedule(args.at(0).operator Array<Operation>());
  });

TVM_REGISTER_API(_StageSetScope)
.set_body([](const ArgStack& args, RetValue *ret) {
    args.at(0).operator Stage()
        .set_scope(args.at(1));
  });

TVM_REGISTER_API(_StageSplitByFactor)
.set_body([](const ArgStack& args, RetValue *ret) {
    IterVar outer, inner;
    args.at(0).operator Stage()
        .split(args.at(1), &outer, &inner, args.at(2));
    *ret = Array<IterVar>({outer, inner});
  });

TVM_REGISTER_API(_StageSplitByOuter)
.set_body([](const ArgStack& args, RetValue *ret) {
    IterVar inner;
    args.at(0).operator Stage()
        .split(args.at(1), args.at(2), &inner, args.at(3));
    *ret = inner;
  });

TVM_REGISTER_API(_StageFuse)
.set_body([](const ArgStack& args, RetValue *ret) {
    IterVar fused;
    args.at(0).operator Stage()
        .split(args.at(1), args.at(2), &fused);
    *ret = fused;
  });

TVM_REGISTER_API(_StageComputeAt)
.set_body([](const ArgStack& args, RetValue *ret) {
    args.at(0).operator Stage()
        .compute_at(args.at(1), args.at(2));
  });

TVM_REGISTER_API(_StageComputeInline)
.set_body([](const ArgStack& args, RetValue *ret) {
    args.at(0).operator Stage()
        .compute_inline();
  });

TVM_REGISTER_API(_StageComputeRoot)
.set_body([](const ArgStack& args, RetValue *ret) {
    args.at(0).operator Stage()
        .compute_root();
  });

TVM_REGISTER_API(_StageReorder)
.set_body([](const ArgStack& args, RetValue *ret) {
    args.at(0).operator Stage()
        .reorder(args.at(1));
  });

TVM_REGISTER_API(_StageTile)
  .set_body([](const ArgStack& args, RetValue *ret) {
    IterVar x_outer, y_outer, x_inner, y_inner;
    args.at(0).operator Stage()
        .tile(args.at(1), args.at(2), &x_outer, &y_outer,
              &x_inner, &y_inner, args.at(3), args.at(4));
    *ret = Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
  });

}  // namespace tvm
