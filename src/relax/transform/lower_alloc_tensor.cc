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
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

namespace {
class Mutator : public ExprMutator {
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallNode* op) override {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    if (op->op.same_as(alloc_tensor_op)) {
      CHECK_EQ(op->args.size(), 3) << "Op " << op->op << " should have three arguments, "
                                   << "[shape, dtype, runtime_device_index].  "
                                   << "However, received " << GetRef<Call>(op);

      auto shape_arg = op->args[0];
      auto dtype = Downcast<DataTypeImm>(op->args[1]);
      PrimValue runtime_device_index = Downcast<PrimValue>(op->args[2]);
      std::string storage_scope = "global";

      auto shape = [&]() -> Array<PrimExpr> {
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
      }();

      PrimExpr nbytes = [&]() -> PrimExpr {
        PrimExpr nbytes = tir::make_const(DataType::Int(64), dtype->value.bytes());
        for (const auto& dim : shape) {
          nbytes *= dim;
        }
        return nbytes;
      }();

      auto offset = PrimValue::Int64(0);

      Expr storage = relax::Call(mem_alloc_storage_op,
                                 {ShapeExpr({nbytes}), runtime_device_index,
                                  StringImm(storage_scope), DataTypeImm(DataType::UInt(8))});
      storage = builder_->Emit(storage, "storage");
      Expr tensor = relax::Call(mem_alloc_tensor_op, {storage, offset, shape_arg, dtype});
      return tensor;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }
};
}  // namespace

Expr LowerAllocTensor(Expr expr) {
  Mutator mutator;
  return mutator(expr);
}

namespace transform {

Pass LowerAllocTensor() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function func, IRModule m, PassContext pc) {
        return Downcast<Function>(relax::LowerAllocTensor(std::move(func)));
      };
  return CreateFunctionPass(pass_func, /*opt_level=*/0, "LowerAllocTensor", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LowerAllocTensor").set_body_typed(LowerAllocTensor);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
