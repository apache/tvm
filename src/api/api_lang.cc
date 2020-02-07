/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Implementation of API functions related to Higher DSL build.
 * \file api_lang.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/te/tensor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/te/schedule.h>
#include <tvm/runtime/registry.h>

#include <tvm/driver/driver_api.h>
#include <tvm/tir/data_layout.h>

namespace tvm {

TVM_REGISTER_GLOBAL("_min_value")
.set_body_typed(min_value);

TVM_REGISTER_GLOBAL("_max_value")
.set_body_typed(max_value);

TVM_REGISTER_GLOBAL("Range")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args.size() == 1) {
      *ret = Range(0, args[0]);
    } else {
      *ret = Range(args[0], args[1]);
    }
  });

namespace tir {

TVM_REGISTER_GLOBAL("_Buffer")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    CHECK_EQ(args.size(), 10);
    auto buffer_type = args[9].operator std::string();
    BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
    *ret = BufferNode::make(args[0], args[1], args[2], args[3], args[4],
                            args[5], args[6], args[7], args[8], type);
  });

TVM_REGISTER_GLOBAL("_BufferAccessPtr")
.set_body_method(&Buffer::access_ptr);

TVM_REGISTER_GLOBAL("_BufferVLoad")
.set_body_method(&Buffer::vload);

TVM_REGISTER_GLOBAL("_BufferVStore")
.set_body_method(&Buffer::vstore);

TVM_REGISTER_GLOBAL("_Layout")
.set_body_typed(LayoutNode::make);

TVM_REGISTER_GLOBAL("_LayoutIndexOf")
.set_body_typed([](Layout layout, std::string axis) -> int {
  return layout.IndexOf(LayoutAxis::make(axis));
});

TVM_REGISTER_GLOBAL("_LayoutFactorOf")
.set_body_typed([](Layout layout, std::string axis) -> int {
  return layout.FactorOf(LayoutAxis::make(axis));
});

TVM_REGISTER_GLOBAL("_LayoutNdim")
.set_body_typed([](Layout layout) -> int {
  return layout.ndim();
});

TVM_REGISTER_GLOBAL("_LayoutGetItem")
.set_body_typed([](Layout layout, int idx) -> std::string {
  const LayoutAxis& axis = layout[idx];
  return axis.name();
});

TVM_REGISTER_GLOBAL("_BijectiveLayout")
.set_body_typed(BijectiveLayoutNode::make);

TVM_REGISTER_GLOBAL("_BijectiveLayoutForwardIndex")
.set_body_method(&BijectiveLayout::ForwardIndex);

TVM_REGISTER_GLOBAL("_BijectiveLayoutBackwardIndex")
.set_body_method(&BijectiveLayout::BackwardIndex);

TVM_REGISTER_GLOBAL("_BijectiveLayoutForwardShape")
.set_body_method(&BijectiveLayout::ForwardShape);

TVM_REGISTER_GLOBAL("_BijectiveLayoutBackwardShape")
.set_body_method(&BijectiveLayout::BackwardShape);
}  // namespace tir

namespace te {
TVM_REGISTER_GLOBAL("_Tensor")
.set_body_typed(TensorNode::make);

TVM_REGISTER_GLOBAL("_TensorIntrin")
.set_body_typed(TensorIntrinNode::make);

TVM_REGISTER_GLOBAL("_TensorIntrinCall")
.set_body_typed(TensorIntrinCallNode::make);

TVM_REGISTER_GLOBAL("_TensorEqual")
.set_body_method(&Tensor::operator==);

TVM_REGISTER_GLOBAL("_TensorHash")
.set_body_typed([](Tensor tensor) -> int64_t {
    return static_cast<int64_t>(std::hash<Tensor>()(tensor));
  });

TVM_REGISTER_GLOBAL("_Placeholder")
.set_body_typed([](Array<PrimExpr> shape, DataType dtype, std::string name) {
  return placeholder(shape, dtype, name);
});

TVM_REGISTER_GLOBAL("_ComputeOp")
.set_body_typed(ComputeOpNode::make);

TVM_REGISTER_GLOBAL("_ScanOp")
.set_body_typed(ScanOpNode::make);

TVM_REGISTER_GLOBAL("_TensorComputeOp")
.set_body_typed(TensorComputeOpNode::make);

TVM_REGISTER_GLOBAL("_ExternOp")
.set_body_typed(ExternOpNode::make);

TVM_REGISTER_GLOBAL("_HybridOp")
.set_body_typed(HybridOpNode::make);

TVM_REGISTER_GLOBAL("_OpGetOutput")
.set_body_typed([](Operation op, int64_t output) {
  return op.output(static_cast<size_t>(output));
});

TVM_REGISTER_GLOBAL("_OpNumOutputs")
.set_body_method<Operation>(&OperationNode::num_outputs);

TVM_REGISTER_GLOBAL("_OpInputTensors")
.set_body_method<Operation>(&OperationNode::InputTensors);

TVM_REGISTER_GLOBAL("_IterVar")
.set_body_typed([](Range dom, Var var, int iter_type, std::string thread_tag) {
  return IterVarNode::make(
      dom, var,
      static_cast<IterVarType>(iter_type),
      thread_tag);
});

TVM_REGISTER_GLOBAL("_CreateSchedule")
.set_body_typed(create_schedule);

TVM_REGISTER_GLOBAL("_StageSetScope")
.set_body_method(&Stage::set_scope);

TVM_REGISTER_GLOBAL("_StageBind")
.set_body_method(&Stage::bind);

TVM_REGISTER_GLOBAL("_StageSplitByFactor")
.set_body_typed([](Stage stage, IterVar parent, PrimExpr factor) {
  IterVar outer, inner;
  stage.split(parent, factor, &outer, &inner);
  return Array<IterVar>({outer, inner});
});

TVM_REGISTER_GLOBAL("_StageSplitByNParts")
.set_body_typed([](Stage stage, IterVar parent, PrimExpr nparts) {
  IterVar outer, inner;
  stage.split_by_nparts(parent, nparts, &outer, &inner);
  return Array<IterVar>({outer, inner});
});

TVM_REGISTER_GLOBAL("_StageFuse")
.set_body_typed([](Stage stage, Array<IterVar> axes) {
    IterVar fused;
    stage.fuse(axes, &fused);
    return fused;
  });

TVM_REGISTER_GLOBAL("_StageComputeAt")
.set_body_method(&Stage::compute_at);

TVM_REGISTER_GLOBAL("_StageComputeInline")
.set_body_method(&Stage::compute_inline);

TVM_REGISTER_GLOBAL("_StageComputeRoot")
.set_body_method(&Stage::compute_root);

TVM_REGISTER_GLOBAL("_StageReorder")
.set_body_method(&Stage::reorder);

TVM_REGISTER_GLOBAL("_StageTile")
.set_body_typed([](
  Stage stage,
  IterVar x_parent, IterVar y_parent,
  PrimExpr x_factor, PrimExpr y_factor
) {
    IterVar x_outer, y_outer, x_inner, y_inner;
    stage.tile(x_parent, y_parent,
               x_factor, y_factor,
               &x_outer, &y_outer,
               &x_inner, &y_inner);
    return Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
  });

TVM_REGISTER_GLOBAL("_StageEnvThreads")
.set_body_method(&Stage::env_threads);

TVM_REGISTER_GLOBAL("_StageSetStorePredicate")
.set_body_method(&Stage::set_store_predicate);

TVM_REGISTER_GLOBAL("_StageUnroll")
.set_body_method(&Stage::unroll);

TVM_REGISTER_GLOBAL("_StageVectorize")
.set_body_method(&Stage::vectorize);

TVM_REGISTER_GLOBAL("_StageTensorize")
.set_body_method(&Stage::tensorize);

TVM_REGISTER_GLOBAL("_StageParallel")
.set_body_method(&Stage::parallel);

TVM_REGISTER_GLOBAL("_StagePragma")
.set_body_method(&Stage::pragma);

TVM_REGISTER_GLOBAL("_StagePrefetch")
.set_body_method(&Stage::prefetch);

TVM_REGISTER_GLOBAL("_StageStorageAlign")
.set_body_method(&Stage::storage_align);

TVM_REGISTER_GLOBAL("_StageDoubleBuffer")
.set_body_method(&Stage::double_buffer);

TVM_REGISTER_GLOBAL("_StageOpenGL")
.set_body_method(&Stage::opengl);

TVM_REGISTER_GLOBAL("_ScheduleNormalize")
.set_body_method(&Schedule::normalize);

TVM_REGISTER_GLOBAL("_ScheduleCreateGroup")
.set_body_method(&Schedule::create_group);

TVM_REGISTER_GLOBAL("_ScheduleCacheRead")
.set_body_method(&Schedule::cache_read);

TVM_REGISTER_GLOBAL("_ScheduleCacheWrite")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    if (args[1].IsObjectRef<Tensor>()) {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Tensor(), args[2]);
    } else {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Array<Tensor>(), args[2]);
    }
  });

TVM_REGISTER_GLOBAL("_ScheduleRFactor")
.set_body_method(&Schedule::rfactor);
}  // namespace te

TVM_REGISTER_GLOBAL("_CommReducerCombine")
.set_body_method<tir::CommReducer>(&tir::CommReducerNode::operator());

}  // namespace tvm
