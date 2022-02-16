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
 * \file extract_constants.cc
 * \brief Collects constants from PrimFunc
    TODO:
    For more information, see the RFC:
    TODO
    https://discuss.tvm.apache.org/t/rfc-introducing-a-rolling-buffer-scheduling-primitive/9836
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

// Replaces constant data to index into mod's "Constants" attrs array.
// Only processes tir::PrimFunc and ignores everything else
using ConstArrayType = Array<runtime::NDArray>;
class Applicator : public tir::StmtExprVisitor {
 protected:
  // returns index of the a in constant_array_, if not found - appends
  size_t deDup(const runtime::NDArray& a) {
    tvm::SEqualReducer eql;
    auto it = std::find_if(
        constant_array_.begin(), constant_array_.end(), [&eql, a](const runtime::NDArray& v) {
          return NDArrayContainerTrait::SEqualReduce(a.as<runtime::NDArray::Container>(),
                                                     v.as<runtime::NDArray::Container>(), eql);
        });
    if (it != constant_array_.end()) {
      return it - constant_array_.begin();
    }
    constant_array_.push_back(std::move(a));
    return constant_array_.size() - 1;
  }

 public:
  ConstArrayType Apply(tir::Stmt body, const ConstArrayType& constant_array) {
    constant_array_ = constant_array;
    this->VisitStmt(body);
    return constant_array_;
  }

  void VisitStmt_(const tir::AllocateConstNode* acn) override {
    tir::AllocateConstNode* node = const_cast<tir::AllocateConstNode*>(acn);
    // Check whether the data already defined within the module's attrs
    // and replace it with array index;
    ICHECK(node->data) << "data field should be defined";
    if (node->data) {
      node->irmod_storage_idx = Optional<Integer>(Integer(deDup(node->data.value())));
    }
    tir::StmtExprVisitor::VisitStmt_(acn);
  }

 private:
  ConstArrayType constant_array_;
};

namespace transform {

Pass ExtractPrimFuncConstants() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* func = f.CopyOnWrite();
    if (!m->attrs.defined()) {
      m->attrs = DictAttrs(Map<String, ObjectRef>());
    }
    auto* attrs = m->attrs.CopyOnWrite();
    ConstArrayType constant_array_ =
        (attrs->dict.count(tvm::attr::kConstantsArray))
            ? Downcast<ConstArrayType>(attrs->dict[tvm::attr::kConstantsArray])
            : ConstArrayType();

    const ConstArrayType constant_list = Applicator().Apply(func->body, constant_array_);
    if (constant_list.size()) {
      attrs->dict.Set(tvm::attr::kConstantsArray, constant_list);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ExtractPrimFuncConstants", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractPrimFuncConstants")
    .set_body_typed(ExtractPrimFuncConstants);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
