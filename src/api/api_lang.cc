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

TVM_REGISTER_GLOBAL("tir.min_value")
.set_body_typed(min_value);

TVM_REGISTER_GLOBAL("tir.max_value")
.set_body_typed(max_value);

TVM_REGISTER_GLOBAL("ir.Range")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = Range(args[0], args[1]);
  });

namespace tir {
TVM_REGISTER_GLOBAL("tir.IterVar")
.set_body_typed([](Range dom, Var var, int iter_type, std::string thread_tag) {
  return IterVarNode::make(
      dom, var,
      static_cast<IterVarType>(iter_type),
      thread_tag);
});
}

namespace te {
TVM_REGISTER_GLOBAL("te.Tensor")
.set_body_typed(TensorNode::make);

TVM_REGISTER_GLOBAL("te.TensorIntrin")
.set_body_typed(TensorIntrinNode::make);

TVM_REGISTER_GLOBAL("te.TensorIntrinCall")
.set_body_typed(TensorIntrinCallNode::make);

TVM_REGISTER_GLOBAL("te.TensorEqual")
.set_body_method(&Tensor::operator==);

TVM_REGISTER_GLOBAL("te.TensorHash")
.set_body_typed([](Tensor tensor) -> int64_t {
    return static_cast<int64_t>(std::hash<Tensor>()(tensor));
  });

TVM_REGISTER_GLOBAL("te.Placeholder")
.set_body_typed([](Array<PrimExpr> shape, DataType dtype, std::string name) {
  return placeholder(shape, dtype, name);
});

TVM_REGISTER_GLOBAL("te.ComputeOp")
.set_body_typed(ComputeOpNode::make);

TVM_REGISTER_GLOBAL("te.ScanOp")
.set_body_typed(ScanOpNode::make);

TVM_REGISTER_GLOBAL("te.TensorComputeOp")
.set_body_typed(TensorComputeOpNode::make);

TVM_REGISTER_GLOBAL("te.ExternOp")
.set_body_typed(ExternOpNode::make);

TVM_REGISTER_GLOBAL("te.HybridOp")
.set_body_typed(HybridOpNode::make);

TVM_REGISTER_GLOBAL("te.OpGetOutput")
.set_body_typed([](Operation op, int64_t output) {
  return op.output(static_cast<size_t>(output));
});

TVM_REGISTER_GLOBAL("te.OpNumOutputs")
.set_body_method<Operation>(&OperationNode::num_outputs);

TVM_REGISTER_GLOBAL("te.OpInputTensors")
.set_body_method<Operation>(&OperationNode::InputTensors);

TVM_REGISTER_GLOBAL("te.CreateSchedule")
.set_body_typed(create_schedule);

TVM_REGISTER_GLOBAL("te.StageSetScope")
.set_body_method(&Stage::set_scope);

TVM_REGISTER_GLOBAL("te.StageBind")
.set_body_method(&Stage::bind);

TVM_REGISTER_GLOBAL("te.StageSplitByFactor")
.set_body_typed([](Stage stage, IterVar parent, PrimExpr factor) {
  IterVar outer, inner;
  stage.split(parent, factor, &outer, &inner);
  return Array<IterVar>({outer, inner});
});

TVM_REGISTER_GLOBAL("te.StageSplitByNParts")
.set_body_typed([](Stage stage, IterVar parent, PrimExpr nparts) {
  IterVar outer, inner;
  stage.split_by_nparts(parent, nparts, &outer, &inner);
  return Array<IterVar>({outer, inner});
});

TVM_REGISTER_GLOBAL("te.StageFuse")
.set_body_typed([](Stage stage, Array<IterVar> axes) {
    IterVar fused;
    stage.fuse(axes, &fused);
    return fused;
  });

TVM_REGISTER_GLOBAL("te.StageComputeAt")
.set_body_method(&Stage::compute_at);

TVM_REGISTER_GLOBAL("te.StageComputeInline")
.set_body_method(&Stage::compute_inline);

TVM_REGISTER_GLOBAL("te.StageComputeRoot")
.set_body_method(&Stage::compute_root);

TVM_REGISTER_GLOBAL("te.StageReorder")
.set_body_method(&Stage::reorder);

TVM_REGISTER_GLOBAL("te.StageTile")
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

TVM_REGISTER_GLOBAL("te.StageEnvThreads")
.set_body_method(&Stage::env_threads);

TVM_REGISTER_GLOBAL("te.StageSetStorePredicate")
.set_body_method(&Stage::set_store_predicate);

TVM_REGISTER_GLOBAL("te.StageUnroll")
.set_body_method(&Stage::unroll);

TVM_REGISTER_GLOBAL("te.StageVectorize")
.set_body_method(&Stage::vectorize);

TVM_REGISTER_GLOBAL("te.StageTensorize")
.set_body_method(&Stage::tensorize);

TVM_REGISTER_GLOBAL("te.StageParallel")
.set_body_method(&Stage::parallel);

TVM_REGISTER_GLOBAL("te.StagePragma")
.set_body_method(&Stage::pragma);

TVM_REGISTER_GLOBAL("te.StagePrefetch")
.set_body_method(&Stage::prefetch);

TVM_REGISTER_GLOBAL("te.StageStorageAlign")
.set_body_method(&Stage::storage_align);

TVM_REGISTER_GLOBAL("te.StageDoubleBuffer")
.set_body_method(&Stage::double_buffer);

TVM_REGISTER_GLOBAL("te.StageOpenGL")
.set_body_method(&Stage::opengl);

TVM_REGISTER_GLOBAL("te.ScheduleNormalize")
.set_body_method(&Schedule::normalize);

TVM_REGISTER_GLOBAL("te.ScheduleCreateGroup")
.set_body_method(&Schedule::create_group);

TVM_REGISTER_GLOBAL("te.ScheduleCacheRead")
.set_body_method(&Schedule::cache_read);

TVM_REGISTER_GLOBAL("te.ScheduleCacheWrite")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    if (args[1].IsObjectRef<Tensor>()) {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Tensor(), args[2]);
    } else {
      *ret = args[0].operator Schedule()
          .cache_write(args[1].operator Array<Tensor>(), args[2]);
    }
  });

TVM_REGISTER_GLOBAL("te.ScheduleRFactor")
.set_body_method(&Schedule::rfactor);
}  // namespace te

TVM_REGISTER_GLOBAL("te.CommReducerCombine")
.set_body_method<tir::CommReducer>(&tir::CommReducerNode::operator());

}  // namespace tvm
