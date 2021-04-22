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
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace tir {

/**************** Constructor ****************/

BlockRV::BlockRV() { this->data_ = make_object<BlockRVNode>(); }

LoopRV::LoopRV() { this->data_ = make_object<LoopRVNode>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleNode::GetSRef(const StmtNode* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_OBJECT_TYPE(ScheduleNode);

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleModule")  //
    .set_body_method<Schedule>(&ScheduleNode::mod);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetState")  //
    .set_body_method<Schedule>(&ScheduleNode::state);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSeed")  //
    .set_body_method<Schedule>(&ScheduleNode::Seed);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCopy")  //
    .set_body_method<Schedule>(&ScheduleNode::Copy);

/**************** (FFI) Constructor ****************/

TVM_REGISTER_GLOBAL("tir.schedule.ConcreteSchedule")
    .set_body_typed([](ObjectRef obj, int debug_mode) -> Schedule {
      IRModule mod{nullptr};
      if (const auto* func = obj.as<PrimFuncNode>()) {
        mod = IRModule({{GlobalVar("main"), GetRef<BaseFunc>(func)}});
      } else if (const auto* p_mod = obj.as<IRModuleNode>()) {
        mod = GetRef<IRModule>(p_mod);
      } else {
        LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: "
                   << obj->GetTypeKey();
      }
      return Schedule::Concrete(mod, debug_mode);
    });

/******** (FFI) Lookup random variables ********/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGet")
    .set_body_typed([](Schedule self, ObjectRef obj) -> ObjectRef {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->Get(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->Get(GetRef<BlockRV>(block_rv));
      }
      if (const auto* int_rv = obj.as<IntRVNode>()) {
        int64_t result = self->Get(GetRef<IntRV>(int_rv));
        return IntImm(DataType::Int(32), result);
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetSRef")
    .set_body_typed([](Schedule self, ObjectRef obj) -> Optional<ObjectRef> {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->GetSRef(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->GetSRef(GetRef<BlockRV>(block_rv));
      }
      if (const auto* stmt = obj.as<StmtNode>()) {
        return self->GetSRef(GetRef<Stmt>(stmt));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRemoveRV")
    .set_body_typed([](Schedule self, ObjectRef obj) -> void {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->RemoveRV(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->RemoveRV(GetRef<BlockRV>(block_rv));
      }
      if (const auto* int_rv = obj.as<IntRVNode>()) {
        return self->RemoveRV(GetRef<IntRV>(int_rv));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });

/***** (FFI) Block/Loop relation *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetBlock")
    .set_body_method<Schedule>(&ScheduleNode::GetBlock);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetLoops")
    .set_body_method<Schedule>(&ScheduleNode::GetLoops);

}  // namespace tir
}  // namespace tvm
