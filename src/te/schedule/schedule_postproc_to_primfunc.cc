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
 * \file schedule_postproc_to_primfunc.cc
 *
 * \brief Translate the function body generated by ScheduleOps
 *  with te related dialects that incorporates Tensor
 *  into the Stmts to a PrimFunc.
 *
 *  Perform this translation before running any TIR optimizations.
 *
 *  Rationale: The body generated by ScheduleOps is not
 *  a formal PrimFunc and cannot be used for further optimization.
 *  This function canonicalize that body and creates a formal PrimFunc.
 *
 *  List of actions taken by the function:
 *  - Remove occurences of te::Tensor, te::Operation in the IR
 *    and replace them by corresponding IR nodes via tir::Buffer.
 *  - Add annotation of extern buffers using the buffer_map field
 *    in the PrimFunc type.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include <utility>
#include <unordered_map>

namespace tvm {
namespace te {

// create a buffer for tensor.
Buffer CreateBufferFor(const Tensor& tensor) {
  std::string name = tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    name += ".v" + std::to_string(tensor->value_index);
  }
  Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, name);
  return buffer;
}

// A remapper that maps tensor to buffer
class TensorToBufferMapper : public StmtExprMutator {
 public:
  explicit TensorToBufferMapper(std::unordered_map<Tensor, Buffer> buffer_map)
      : buffer_map_(buffer_map) {
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // TODO(tvm-team): remove realize_scope, turn the info into
    // Buffer's scope field in this pass.
    if (op->attr_key == tir::attr::realize_scope ||
        op->attr_key == tir::attr::double_buffer_scope) {
      Stmt body = op->body;
      Operation operation = Downcast<Operation>(op->node);
      for (int i = operation->num_outputs(); i != 0; --i) {
        Buffer buffer = GetOrAllocBuffer(operation.output(i - 1));
        body = AttrStmtNode::make(
            buffer, op->attr_key, op->value, body);
      }
      return body;
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef> >(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      return AttrStmtNode::make(
          Array<ObjectRef>{tuple[0], GetOrAllocBuffer(tensor)},
          op->attr_key, op->value, op->body);
    } else if (op->attr_key == tir::attr::buffer_dim_align||
               op->attr_key == tir::attr::prefetch_scope) {
      Tensor tensor = Downcast<Tensor>(op->node);
      Buffer buffer = GetOrAllocBuffer(tensor);
      return AttrStmtNode::make(
          buffer, op->attr_key, op->value, op->body);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const RealizeNode* op) final {
    Tensor tensor = Downcast<Operation>(op->func).output(op->value_index);
    Buffer buffer = GetOrAllocBuffer(tensor);

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<RealizeNode>();

    return BufferRealize(buffer, op->bounds, op->condition, op->body);
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    Tensor tensor = Downcast<Operation>(op->func).output(op->value_index);
    Buffer buffer = GetBuffer(tensor);

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProvideNode>();

    return BufferStore(buffer, op->value, op->args);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    auto ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<CallNode>();

    if (op->call_type == CallNode::Halide) {
      Tensor tensor = Downcast<Operation>(op->func).output(op->value_index);
      Buffer buffer = GetBuffer(tensor);
      return tir::BufferLoad(buffer, op->args);
    } else {
      return ret;
    }
  }

 private:
  Buffer GetOrAllocBuffer(const Tensor& tensor) {
    return GetBuffer(tensor, true);
  }

  Buffer GetBuffer(const Tensor& tensor, bool allow_alloc = false) {
    auto it = buffer_map_.find(tensor);
    if (it != buffer_map_.end()) return it->second;
    CHECK(allow_alloc) << "Cannot find the Realization point of tensor " << tensor;

    auto buffer = CreateBufferFor(tensor);
    buffer_map_[tensor] = buffer;
    return buffer;
  }

  // maps tensor to buffer.
  std::unordered_map<Tensor, Buffer> buffer_map_;
};


PrimFunc SchedulePostProcToPrimFunc(Array<ObjectRef> arg_list,
                                    Stmt body,
                                    Optional<Map<Tensor, Buffer>> extern_buffer_opt) {
  std::unordered_map<Tensor, Buffer> extern_buffer;

  if (extern_buffer_opt.defined()) {
    auto v = extern_buffer_opt.value();
    extern_buffer = std::unordered_map<Tensor, Buffer>(v.begin(), v.end());
  }

  Array<tir::Var> params;
  Map<tir::Var, tir::Buffer> buffer_map;

  for (auto var : arg_list) {
    if (auto* n = var.as<tir::VarNode>()) {
      params.push_back(GetRef<tir::Var>(n));
    } else if (auto* n = var.as<te::TensorNode>()) {
      te::Tensor tensor = GetRef<te::Tensor>(n);
      CHECK(!extern_buffer.count(tensor));

      tir::Buffer buffer = CreateBufferFor(tensor);
      tir::Var bptr(buffer->name, DataType::Handle());
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
      extern_buffer[tensor] = buffer;
    } else {
      tir::Buffer buffer = Downcast<tir::Buffer>(var);
      tir::Var bptr(buffer->name, DataType::Handle());
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
    }
  }

  body = TensorToBufferMapper(std::move(extern_buffer))(std::move(body));
  return tir::PrimFunc(params, body, VoidType(), buffer_map);
}

TVM_REGISTER_GLOBAL("schedule.SchedulePostProcToPrimFunc")
.set_body_typed(SchedulePostProcToPrimFunc);

}  // namespace te
}  // namespace tvm
