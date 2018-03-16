/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Higher DSL build.
 * \file api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/operation.h>
#include <tvm/buffer.h>
#include <tvm/schedule.h>
#include <tvm/api_registry.h>
#include <tvm/build_module.h>

namespace tvm {

TVM_REGISTER_API("_min_value")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    Type t = args[0].operator Type();
    *ret = t.min();
  });

TVM_REGISTER_API("_max_value")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    Type t = args[0].operator Type();
    *ret = t.max();
  });

TVM_REGISTER_API("_const")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args[0].type_code() == kDLInt) {
      *ret = make_const(args[1], args[0].operator int64_t());
    } else if (args[0].type_code() == kDLFloat) {
      *ret = make_const(args[1], args[0].operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  });

TVM_REGISTER_API("_str")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = ir::StringImm::make(args[0]);
});


TVM_REGISTER_API("_Array")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    std::vector<std::shared_ptr<Node> > data;
    for (int i = 0; i < args.size(); ++i) {
      data.push_back(args[i].node_sptr());
    }
    auto node = std::make_shared<ArrayNode>();
    node->data = std::move(data);
    *ret = node;
  });

TVM_REGISTER_API("_ArrayGetItem")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    int64_t i = args[1];
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    *ret = n->data[static_cast<size_t>(i)];
  });

TVM_REGISTER_API("_ArraySize")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(sptr.get())->data.size());
  });

TVM_REGISTER_API("_Map")
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

TVM_REGISTER_API("_MapSize")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(n->data.size());
  });

TVM_REGISTER_API("_MapGetItem")
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

TVM_REGISTER_API("_MapCount")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    CHECK(args[1].type_code() == kNodeHandle);
    auto& sptr = args[0].node_sptr();
    CHECK(sptr->is_type<MapNode>());
    auto* n = static_cast<const MapNode*>(sptr.get());
    *ret = static_cast<int64_t>(
        n->data.count(args[1].node_sptr()));
  });

TVM_REGISTER_API("_MapItems")
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

TVM_REGISTER_API("Range")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args.size() == 1) {
      *ret = Range(0, args[0]);
    } else {
      *ret = Range(args[0], args[1]);
    }
  });

TVM_REGISTER_API("_Buffer")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = BufferNode::make(args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[5],
                            args[6],
                            args[7],
                            args[8]);
  });

TVM_REGISTER_API("_BufferAccessPtr")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Buffer()
        .access_ptr(args[1], args[2], args[3], args[4]);
  });

TVM_REGISTER_API("_BufferVLoad")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Buffer()
        .vload(args[1], args[2]);
  });

TVM_REGISTER_API("_BufferVStore")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Buffer()
        .vstore(args[1], args[2]);
  });

TVM_REGISTER_API("_Tensor")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = TensorNode::make(args[0],
                            args[1],
                            args[2],
                            args[3]);
  });

TVM_REGISTER_API("_TensorIntrin")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = TensorIntrinNode::make(args[0],
                                  args[1],
                                  args[2],
                                  args[3],
                                  args[4],
                                  args[5],
                                  args[6]);
  });

TVM_REGISTER_API("_TensorEqual")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Tensor() == args[1].operator Tensor();
  });

TVM_REGISTER_API("_TensorHash")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = static_cast<int64_t>(
        std::hash<Tensor>()(args[0].operator Tensor()));
  });

TVM_REGISTER_API("_Placeholder")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = placeholder(args[0],
                       args[1],
                       args[2]);
  });

TVM_REGISTER_API("_ComputeOp")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = ComputeOpNode::make(args[0],
                               args[1],
                               args[2],
                               args[3]);
  });

TVM_REGISTER_API("_ScanOp")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = ScanOpNode::make(args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[5],
                            args[6]);
  });

TVM_REGISTER_API("_ExternOp")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = ExternOpNode::make(args[0],
                              args[1],
                              args[2],
                              args[3],
                              args[4],
                              args[5]);
  });

TVM_REGISTER_API("_OpGetOutput")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Operation().output(
        static_cast<size_t>(args[1].operator int64_t()));
  });

TVM_REGISTER_API("_OpNumOutputs")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Operation()->num_outputs();
  });

TVM_REGISTER_API("_OpInputTensors")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Operation()->InputTensors();
  });

TVM_REGISTER_API("_IterVar")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = IterVarNode::make(
        args[0], args[1],
        static_cast<IterVarType>(args[2].operator int()),
        args[3]);
  });

TVM_REGISTER_API("_CreateSchedule")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = create_schedule(args[0].operator Array<Operation>());
  });

TVM_REGISTER_API("_StageSetScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .set_scope(args[1]);
  });

TVM_REGISTER_API("_StageBind")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .bind(args[1], args[2]);
  });

TVM_REGISTER_API("_StageSplitByFactor")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar outer, inner;
    args[0].operator Stage()
        .split(args[1], args[2], &outer, &inner);
    *ret = Array<IterVar>({outer, inner});
  });

TVM_REGISTER_API("_StageSplitByNParts")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar outer, inner;
    args[0].operator Stage()
        .split_by_nparts(args[1], args[2], &outer, &inner);
    *ret = Array<IterVar>({outer, inner});
  });

TVM_REGISTER_API("_StageFuse")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar fused;
    args[0].operator Stage()
        .fuse(args[1], args[2], &fused);
    *ret = fused;
  });

TVM_REGISTER_API("_StageComputeAt")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_at(args[1], args[2]);
  });

TVM_REGISTER_API("_StageComputeInline")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_inline();
  });

TVM_REGISTER_API("_StageComputeRoot")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .compute_root();
  });

TVM_REGISTER_API("_StageReorder")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .reorder(args[1]);
  });

TVM_REGISTER_API("_StageTile")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    IterVar x_outer, y_outer, x_inner, y_inner;
    args[0].operator Stage()
        .tile(args[1], args[2],
              args[3], args[4],
              &x_outer, &y_outer,
              &x_inner, &y_inner);
    *ret = Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
  });

TVM_REGISTER_API("_StageEnvThreads")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .env_threads(args[1]);
  });

TVM_REGISTER_API("_StageSetStorePredicate")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .set_store_predicate(args[1]);
  });

TVM_REGISTER_API("_StageUnroll")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .unroll(args[1]);
  });

TVM_REGISTER_API("_StageVectorize")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .vectorize(args[1]);
  });

TVM_REGISTER_API("_StageTensorize")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .tensorize(args[1], args[2]);
  });

TVM_REGISTER_API("_StageParallel")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .parallel(args[1]);
  });

TVM_REGISTER_API("_StagePragma")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
    args[0].operator Stage()
        .pragma(args[1], args[2]);
  });

TVM_REGISTER_API("_StagePrefetch")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator Stage()
        .prefetch(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("_StageStorageAlign")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator Stage()
        .storage_align(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("_StageDoubleBuffer")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator Stage().double_buffer();
  });

TVM_REGISTER_API("_StageOpenGL")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator Stage().opengl();
  });

TVM_REGISTER_API("_ScheduleNormalize")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = args[0].operator Schedule()
        .normalize();
  });

TVM_REGISTER_API("_ScheduleCreateGroup")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = args[0].operator Schedule()
        .create_group(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("_ScheduleCacheRead")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = args[0].operator Schedule()
        .cache_read(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("_ScheduleCacheWrite")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = args[0].operator Schedule()
        .cache_write(args[1], args[2]);
  });

TVM_REGISTER_API("_ScheduleRFactor")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = args[0].operator Schedule()
        .rfactor(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("_CommReducerCombine")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    const ir::CommReducerNode* combiner =
      args[0].operator ir::CommReducer().as<ir::CommReducerNode>();
    *ret = (*combiner)(args[1], args[2]);
  });

}  // namespace tvm
