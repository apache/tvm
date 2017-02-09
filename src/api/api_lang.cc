/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Higher DSL build.
 * \file api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/buffer.h>
#include <tvm/schedule.h>
#include <tvm/api_registry.h>

namespace tvm {

TVM_REGISTER_API(_const)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args[0].type_code() == kInt) {
      *ret = make_const(args[1], args[0].operator int64_t());
    } else if (args[0].type_code() == kFloat) {
      *ret = make_const(args[1], args[0].operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  });


TVM_REGISTER_API(_str)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = ir::StringImm::make(args[0]);
});


TVM_REGISTER_API(_Array)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    std::vector<std::shared_ptr<Node> > data;
    for (int i = 0; i < args.size(); ++i) {
      data.push_back(args[i].node_sptr());
    }
    auto node = std::make_shared<ArrayNode>();
    node->data = std::move(data);
    *ret = node;
  });

TVM_REGISTER_API(_ArrayGetItem)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    int64_t i = args[1];
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    *ret = n->data[i];
  });

TVM_REGISTER_API(_ArraySize)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(sptr.get())->data.size());
  });

TVM_REGISTER_API(_Map)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args.size() % 2, 0);
    MapNode::ContainerType data;
    for (int i = 0; i < args.num_args; i += 2) {
      CHECK(args[i].type_code() == kNodeHandle)
          << "need content of array to be NodeBase";
      CHECK(args[i + 1].type_code() == kNodeHandle)
          << "need content of array to be NodeBase";
      data.emplace(std::make_pair(args[i].node_sptr(),
                                  args[i + 1].node_sptr()));
    }
    auto node = std::make_shared<MapNode>();
    node->data = std::move(data);
    *ret = node;
  });

TVM_REGISTER_API(_MapSize)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(n->data.size());
  });

TVM_REGISTER_API(_MapGetItem)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    CHECK(args[1].type_code() == kNodeHandle);
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    auto it = n->data.find(args[1].node_sptr());
    CHECK(it != n->data.end())
        << "cannot find the corresponding key in the Map";
    *ret = (*it).second;
  });

TVM_REGISTER_API(_MapCount)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    CHECK(args[1].type_code() == kNodeHandle);
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(
        n->data.count(args[1].node_sptr()));
  });

TVM_REGISTER_API(_MapItems)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    auto rkvs = std::make_shared<ArrayNode>();
    for (const auto& kv : n->data) {
      rkvs->data.push_back(kv.first);
      rkvs->data.push_back(kv.second);
    }
    *ret = rkvs;
  });

TVM_REGISTER_API(Range)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args.size() == 1) {
      *ret = Range(0, args[0]);
    } else {
      *ret = Range(args[0], args[1]);
    }
  });

TVM_REGISTER_API(_Buffer)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = BufferNode::make(args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4]);
  });

TVM_REGISTER_API(_Tensor)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = TensorNode::make(args[0],
                            args[1],
                            args[2],
                            args[3]);
  });

TVM_REGISTER_API(_TensorEqual)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Tensor() == args[1].operator Tensor();
  });

TVM_REGISTER_API(_TensorHash)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = static_cast<int64_t>(
        std::hash<Tensor>()(args[0].operator Tensor()));
  });

TVM_REGISTER_API(_Placeholder)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = Placeholder(args[0],
                       args[1],
                       args[2]);
  });

TVM_REGISTER_API(_ComputeOp)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = ComputeOpNode::make(args[0],
                               args[1],
                               args[2]);
  });

TVM_REGISTER_API(_OpGetOutput)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Operation().output(
        args[1].operator int64_t());
  });


TVM_REGISTER_API(_IterVar)
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = IterVar(args[0], args[1], args[2]);
  });

TVM_REGISTER_API(_Schedule)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = Schedule(args[0].operator Array<Operation>());
  });

TVM_REGISTER_API(_StageSetScope)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .set_scope(args[1]);
  });

TVM_REGISTER_API(_StageSplitByFactor)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar outer, inner;
    args[0].operator Stage()
        .split(args[1], &outer, &inner, args[2]);
    *ret = Array<IterVar>({outer, inner});
  });

TVM_REGISTER_API(_StageSplitByOuter)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar inner;
    args[0].operator Stage()
        .split(args[1], args[2], &inner, args[3]);
    *ret = inner;
  });

TVM_REGISTER_API(_StageFuse)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar fused;
    args[0].operator Stage()
        .fuse(args[1], args[2], &fused);
    *ret = fused;
  });

TVM_REGISTER_API(_StageComputeAt)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_at(args[1], args[2]);
  });

TVM_REGISTER_API(_StageComputeInline)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_inline();
  });

TVM_REGISTER_API(_StageComputeRoot)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_root();
  });

TVM_REGISTER_API(_StageReorder)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .reorder(args[1]);
  });

TVM_REGISTER_API(_StageTile)
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar x_outer, y_outer, x_inner, y_inner;
    args[0].operator Stage()
        .tile(args[1], args[2], &x_outer, &y_outer,
              &x_inner, &y_inner, args[3], args[4]);
    *ret = Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
  });

TVM_REGISTER_API(_StageUnroll)
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .unroll(args[1]);
  });

TVM_REGISTER_API(_StageVectorize)
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .vectorize(args[1]);
  });

TVM_REGISTER_API(_ScheduleNormalize)
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Schedule()
        .normalize();
  });

}  // namespace tvm
