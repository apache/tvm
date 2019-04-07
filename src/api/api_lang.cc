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
#include <tvm/data_layout.h>


namespace tvm {

TVM_REGISTER_API("_min_value")
.set_body_method(&Type::min);

TVM_REGISTER_API("_max_value")
.set_body_method(&Type::max);

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
.set_body_simple(ir::StringImm::make);


TVM_REGISTER_API("_Array")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    std::vector<NodePtr<Node> > data;
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].type_code() != kNull) {
        data.push_back(args[i].node_sptr());
      } else {
        data.push_back(NodePtr<Node>(nullptr));
      }
    }
    auto node = make_node<ArrayNode>();
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
    if (args.size() != 0 && args[0].type_code() == kStr) {
      // StrMap
      StrMapNode::ContainerType data;
      for (int i = 0; i < args.num_args; i += 2) {
        CHECK(args[i].type_code() == kStr)
            << "key of str map need to be str";
        CHECK(args[i + 1].type_code() == kNodeHandle)
            << "value of the map to be NodeRef";
        data.emplace(std::make_pair(args[i].operator std::string(),
                                    args[i + 1].node_sptr()));
      }
      auto node = make_node<StrMapNode>();
      node->data = std::move(data);
      *ret = node;
    } else {
      // Container node.
      MapNode::ContainerType data;
      for (int i = 0; i < args.num_args; i += 2) {
        CHECK(args[i].type_code() == kNodeHandle)
            << "key of str map need to be str";
        CHECK(args[i + 1].type_code() == kNodeHandle)
            << "value of map to be NodeRef";
        data.emplace(std::make_pair(args[i].node_sptr(),
                                    args[i + 1].node_sptr()));
      }
      auto node = make_node<MapNode>();
      node->data = std::move(data);
      *ret = node;
    }
  });

TVM_REGISTER_API("_MapSize")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    if (sptr->is_type<MapNode>()) {
      auto* n = static_cast<const MapNode*>(sptr.get());
      *ret = static_cast<int64_t>(n->data.size());
    } else {
      CHECK(sptr->is_type<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(sptr.get());
      *ret = static_cast<int64_t>(n->data.size());
    }
  });

TVM_REGISTER_API("_MapGetItem")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    auto& sptr = args[0].node_sptr();
    if (sptr->is_type<MapNode>()) {
      CHECK(args[1].type_code() == kNodeHandle);
      auto* n = static_cast<const MapNode*>(sptr.get());
      auto it = n->data.find(args[1].node_sptr());
      CHECK(it != n->data.end())
          << "cannot find the corresponding key in the Map";
      *ret = (*it).second;
    } else {
      CHECK(sptr->is_type<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(sptr.get());
      auto it = n->data.find(args[1].operator std::string());
      CHECK(it != n->data.end())
          << "cannot find the corresponding key in the Map";
      *ret = (*it).second;
    }
  });

TVM_REGISTER_API("_MapCount")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    auto& sptr = args[0].node_sptr();
    if (sptr->is_type<MapNode>()) {
      auto* n = static_cast<const MapNode*>(sptr.get());
      CHECK(args[1].type_code() == kNodeHandle);
      *ret = static_cast<int64_t>(
          n->data.count(args[1].node_sptr()));
    } else {
      CHECK(sptr->is_type<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(sptr.get());
      *ret = static_cast<int64_t>(
          n->data.count(args[1].operator std::string()));
    }
  });

TVM_REGISTER_API("_MapItems")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    auto& sptr = args[0].node_sptr();
    if (sptr->is_type<MapNode>()) {
      auto* n = static_cast<const MapNode*>(sptr.get());
      auto rkvs = make_node<ArrayNode>();
      for (const auto& kv : n->data) {
        rkvs->data.push_back(kv.first);
        rkvs->data.push_back(kv.second);
      }
      *ret = rkvs;
    } else {
      auto* n = static_cast<const StrMapNode*>(sptr.get());
      auto rkvs = make_node<ArrayNode>();
      for (const auto& kv : n->data) {
        rkvs->data.push_back(ir::StringImm::make(kv.first).node_);
        rkvs->data.push_back(kv.second);
      }
      *ret = rkvs;
    }
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
.set_body_simple(BufferNode::make);

TVM_REGISTER_API("_BufferAccessPtr")
.set_body_method(&Buffer::access_ptr);

TVM_REGISTER_API("_BufferVLoad")
.set_body_method(&Buffer::vload);

TVM_REGISTER_API("_BufferVStore")
.set_body_method(&Buffer::vstore);

TVM_REGISTER_API("_Layout")
.set_body_simple(LayoutNode::make);

TVM_REGISTER_API("_LayoutIndexOf")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = args[0].operator Layout()
      .IndexOf(LayoutAxis::make(args[1]));
});

TVM_REGISTER_API("_LayoutFactorOf")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = args[0].operator Layout()
      .FactorOf(LayoutAxis::make(args[1]));
});

TVM_REGISTER_API("_LayoutNdim")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = static_cast<int64_t>(args[0].operator Layout().ndim());
});

TVM_REGISTER_API("_LayoutGetItem")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  const LayoutAxis& axis = args[0].operator Layout()[args[1]];
  *ret = axis.name();
});

TVM_REGISTER_API("_BijectiveLayout")
.set_body_simple(BijectiveLayoutNode::make);

TVM_REGISTER_API("_BijectiveLayoutForwardIndex")
.set_body_method(&BijectiveLayout::ForwardIndex);

TVM_REGISTER_API("_BijectiveLayoutBackwardIndex")
.set_body_method(&BijectiveLayout::BackwardIndex);

TVM_REGISTER_API("_BijectiveLayoutForwardShape")
.set_body_method(&BijectiveLayout::ForwardShape);

TVM_REGISTER_API("_BijectiveLayoutBackwardShape")
.set_body_method(&BijectiveLayout::BackwardShape);

TVM_REGISTER_API("_Tensor")
.set_body_simple(TensorNode::make);

TVM_REGISTER_API("_TensorIntrin")
.set_body_simple(TensorIntrinNode::make);

TVM_REGISTER_API("_TensorIntrinCall")
.set_body_simple(TensorIntrinCallNode::make);

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
.set_body_simple(ComputeOpNode::make);

TVM_REGISTER_API("_ScanOp")
.set_body_simple(ScanOpNode::make);

TVM_REGISTER_API("_TensorComputeOp")
.set_body_simple(TensorComputeOpNode::make);

TVM_REGISTER_API("_ExternOp")
.set_body_simple(ExternOpNode::make);

TVM_REGISTER_API("_HybridOp")
.set_body_simple(HybridOpNode::make);

TVM_REGISTER_API("_OpGetOutput")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = args[0].operator Operation().output(
        static_cast<size_t>(args[1].operator int64_t()));
  });


TVM_REGISTER_API("_OpNumOutputs")
.set_body_node_method<Operation>(&OperationNode::num_outputs);


TVM_REGISTER_API("_OpInputTensors")
.set_body_node_method<Operation>(&OperationNode::InputTensors);

TVM_REGISTER_API("_IterVar")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = IterVarNode::make(
        args[0], args[1],
        static_cast<IterVarType>(args[2].operator int()),
        args[3]);
  });

TVM_REGISTER_API("_CreateSchedule")
.set_body_simple(create_schedule);

TVM_REGISTER_API("_StageSetScope")
.set_body_method(&Stage::set_scope);

TVM_REGISTER_API("_StageBind")
.set_body_method(&Stage::bind);

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
        .fuse(args[1], &fused);
    *ret = fused;
  });

TVM_REGISTER_API("_StageComputeAt")
.set_body_method(&Stage::compute_at);

TVM_REGISTER_API("_StageComputeInline")
.set_body_method(&Stage::compute_inline);

TVM_REGISTER_API("_StageComputeRoot")
.set_body_method(&Stage::compute_root);

TVM_REGISTER_API("_StageReorder")
.set_body_method(&Stage::reorder);

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
.set_body_method(&Stage::env_threads);

TVM_REGISTER_API("_StageSetStorePredicate")
.set_body_method(&Stage::set_store_predicate);

TVM_REGISTER_API("_StageUnroll")
.set_body_method(&Stage::unroll);

TVM_REGISTER_API("_StageVectorize")
.set_body_method(&Stage::vectorize);

TVM_REGISTER_API("_StageTensorize")
.set_body_method(&Stage::tensorize);

TVM_REGISTER_API("_StageParallel")
.set_body_method(&Stage::parallel);

TVM_REGISTER_API("_StagePragma")
.set_body_method(&Stage::pragma);

TVM_REGISTER_API("_StagePrefetch")
.set_body_method(&Stage::prefetch);

TVM_REGISTER_API("_StageStorageAlign")
.set_body_method(&Stage::storage_align);

TVM_REGISTER_API("_StageDoubleBuffer")
.set_body_method(&Stage::double_buffer);

TVM_REGISTER_API("_StageOpenGL")
.set_body_method(&Stage::opengl);

TVM_REGISTER_API("_ScheduleNormalize")
.set_body_method(&Schedule::normalize);

TVM_REGISTER_API("_ScheduleCreateGroup")
.set_body_method(&Schedule::create_group);

TVM_REGISTER_API("_ScheduleCacheRead")
.set_body_method(&Schedule::cache_read);

TVM_REGISTER_API("_ScheduleCacheWrite")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    if (args[1].IsNodeType<Tensor>()) {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Tensor(), args[2]);
    } else {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Array<Tensor>(), args[2]);
    }
  });

TVM_REGISTER_API("_ScheduleRFactor")
.set_body_method(&Schedule::rfactor);

TVM_REGISTER_API("_CommReducerCombine")
.set_body_node_method<ir::CommReducer>(&ir::CommReducerNode::operator());

}  // namespace tvm
