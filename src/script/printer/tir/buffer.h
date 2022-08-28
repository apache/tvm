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
#ifndef TVM_SCRIPT_PRINTER_TIR_BUFFER_H_
#define TVM_SCRIPT_PRINTER_TIR_BUFFER_H_

#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/tir/buffer.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

class BufferAssociatedVariables {
 public:
  void Disassociate(const tir::VarNode* var) { var2buffer_.erase(var); }

  void AssociateIfNotAlready(const tir::VarNode* var, const tir::Buffer& buffer) {
    var2buffer_.insert({var, buffer});
  }

  bool IsAssociated(const tir::VarNode* var) const { return var2buffer_.count(var) != 0; }

  bool IsAssociatedWith(const PrimExpr& e, const tir::Buffer& buffer) const {
    if (const auto* v = e.as<tir::VarNode>()) {
      auto it = var2buffer_.find(v);
      return it != var2buffer_.end() && it->second == buffer;
    }
    return false;
  }

  void Define(VarTableNode* vars, const Frame& frame) const {
    for (const auto& kv : var2buffer_) {
      const tir::VarNode* var = kv.first;
      const tir::Buffer& buffer = kv.second;

      ExprDoc buffer_name = vars->GetVarDoc(MakeTraced(buffer)).value();
      buffer_name->source_paths.clear();

      if (buffer->data.get() == var) {
        vars->DefineByDoc(
            buffer->data, [buffer_name]() { return buffer_name->Attr("data"); }, frame);
      } else if (buffer->elem_offset.get() == var) {
        vars->DefineByDoc(
            buffer->elem_offset, [buffer_name]() { return buffer_name->Attr("elem_offset"); },
            frame);
      } else {
        ICHECK(false) << "Unexpected association. Buffer: " << buffer
                      << "; Var: " << GetRef<tir::Var>(var);
      }
    }
  }

 private:
  std::unordered_map<const tir::VarNode*, tir::Buffer> var2buffer_;
};

struct BufferPrintInfo {
  TracedObject<tir::Buffer> buffer;
  TracedArray<PrimExpr> shape;
  Optional<ExprDoc> dtype;
  TracedOptional<tir::Var> data;
  TracedOptional<Array<PrimExpr>> strides;
  TracedOptional<PrimExpr> elem_offset;
  Optional<ExprDoc> scope;
  Optional<ExprDoc> align;
  Optional<ExprDoc> offset_factor;
  Optional<ExprDoc> buffer_type;

  ExprDoc AsCall(const ExprDoc& prefix,
                 std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const;
  ExprDoc AsCall(const ExprDoc& prefix, const Array<ExprDoc>& extra_args,
                 std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const;
};

std::vector<BufferPrintInfo> GetBufferPrintInfo(
    const std::vector<TracedObject<tir::Buffer>>& buffers,  //
    std::function<bool(const tir::VarNode*)> f_var_defined,
    std::unordered_map<const tir::VarNode*, ObjectPath>* var_explicit_def,
    BufferAssociatedVariables* associated_vars);

std::vector<IdDoc> DefineBuffers(const std::vector<TracedObject<tir::Buffer>>& buffers,
                                 const Frame& frame, const IRDocsifier& p,
                                 const ExprDoc& definition_prefix,
                                 std::function<void(IdDoc, ExprDoc)> add_definiton);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_BUFFER_H_
