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
 * \file src/relax/transform/lower_alloc_tensor.cc
 * \brief Lower any relax.builtin.alloc_tensor remaining after static planning
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
class Mutator : public ExprMutator {
 public:
  explicit Mutator(IRModule mod) : ctx_mod_(mod) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallNode* op) override {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    if (op->op.same_as(alloc_tensor_op)) {
      CHECK_EQ(op->args.size(), 4) << "Op " << op->op << " should have three arguments, "
                                   << "[shape, dtype, runtime_device_index, storage_scope].  "
                                   << "However, received " << ffi::GetRef<Call>(op);

      auto shape_arg = op->args[0];
      auto dtype = Downcast<DataTypeImm>(op->args[1]);
      PrimValue runtime_device_index = Downcast<PrimValue>(op->args[2]);
      StringImm storage_scope = Downcast<StringImm>(op->args[3]);

      auto shape = [&]() -> ffi::Array<PrimExpr> {
        if (auto ptr = shape_arg.as<ShapeExprNode>()) {
          return ptr->values;
        }

        auto sinfo = GetStructInfo(shape_arg);
        if (auto ptr = sinfo.as<ShapeStructInfoNode>()) {
          if (ptr->values) {
            return ptr->values.value();
          }
        }

        LOG(FATAL) << "Shape argument for " << alloc_tensor_op << " should be a ShapeExpr, "
                   << "or a variable that holds a ShapeExpr.  "
                   << "However, received argument " << shape_arg << " with struct info " << sinfo;
        TVM_FFI_UNREACHABLE();
      }();

      PrimExpr nbytes = [&]() -> PrimExpr {
        PrimExpr nbytes = tir::make_const(DataType::Int(64), dtype->value.bytes());
        for (const auto& dim : shape) {
          nbytes *= dim;
        }
        return nbytes;
      }();

      ShapeExpr size({nbytes});

      int64_t vdevice_index = -1;
      if (auto* prim_value_node = op->args[2].as<PrimValueNode>()) {
        vdevice_index = prim_value_node->value.as<IntImmNode>()->value;
      }
      ffi::Optional<VDevice> vdevice = GetGlobalVDevice(ctx_mod_, vdevice_index);

      if (vdevice.defined()) {
        std::string dev_kind = vdevice.value()->target->kind->name;
        PrimExpr dev_size = tir::make_const(DataType::Int(64), 1);
        if (vdevice.value()->memory_scope != "global") {
          auto device_size_handler =
              tvm::ffi::Function::GetGlobal(std::string("DeviceGetMemSize.") + dev_kind);
          if (device_size_handler.has_value()) {
            dev_size *=
                (*device_size_handler)(shape, dtype->value, vdevice.value()).cast<PrimExpr>();
            size = ShapeExpr({dev_size});
          }
          auto device_scope_handler =
              tvm::ffi::Function::GetGlobal(std::string("DeviceScopeCompatibility.") + dev_kind);
          if (device_scope_handler.has_value()) {
            ffi::String dev_scope =
                (*device_scope_handler)(vdevice.value()->target, vdevice.value()->memory_scope)
                    .cast<ffi::String>();
            storage_scope = StringImm(dev_scope);
          }
        }
      }

      auto offset = PrimValue::Int64(0);

      Expr storage = relax::Call(mem_alloc_storage_op, {size, runtime_device_index, storage_scope,
                                                        DataTypeImm(DataType::UInt(8))});
      storage = builder_->Emit(storage, "storage");
      Expr tensor =
          relax::Call(mem_alloc_tensor_op, {storage, offset, shape_arg, dtype, op->args[2]});
      return tensor;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  IRModule ctx_mod_;
};
}  // namespace

Expr LowerAllocTensor(IRModule m, Expr expr) {
  Mutator mutator(m);
  return mutator(expr);
}

namespace transform {

Pass LowerAllocTensor() {
  auto pass_func = [=](Function func, IRModule m, PassContext pc) {
    return Downcast<Function>(relax::LowerAllocTensor(m, std::move(func)));
  };
  return CreateFunctionPass(pass_func, /*opt_level=*/0, "LowerAllocTensor", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LowerAllocTensor", LowerAllocTensor);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
