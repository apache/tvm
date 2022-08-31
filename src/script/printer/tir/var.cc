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
#include "./var.h"

#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include "../utils.h"
#include "./tir.h"
#include "tvm/script/printer/doc.h"

namespace tvm {
namespace script {
namespace printer {

TracedObject<String> GetVarNameHint(const TracedObject<tir::Var>& var) {
  TracedObject<String> name_hint = var.GetAttr(&tir::VarNode::name_hint);
  if (name_hint.Get().empty()) {
    return MakeTraced(String("v"), var.GetPath());
  } else {
    return name_hint;
  }
}

ExprDoc GetTypeAnnotationForVar(const TracedObject<tir::Var>& var, const IRDocsifier& p) {
  TracedObject<Type> type_annotation = var.GetAttr(&tir::VarNode::type_annotation);
  if (type_annotation.Get().defined()) {
    return p->AsExprDoc(type_annotation);
  } else {
    TracedBasicValue<DataType> dtype = var.GetAttr(&tir::VarNode::dtype);
    Type raw_type = GetTypeFromRuntimeDataType(dtype.Get());
    return p->AsExprDoc(MakeTraced(raw_type, dtype.GetPath()));
  }
}

// a = T.var("int32")
IdDoc DefineVar(const TracedObject<tir::Var>& var, const Frame& frame, const IRDocsifier& p,
                std::function<void(AssignDoc)> add_def) {
  IdDoc id = p->vars->Define(var.Get(), GetVarNameHint(var), frame);
  if (add_def != nullptr) {
    add_def(AssignDoc(id,
                      TIR(p)->Attr("var")->Call({DType2Literal(var.GetAttr(&tir::VarNode::dtype))}),
                      NullOpt));
  }
  return id;
}
// a: T.int32
IdDoc DeclareVar(const TracedObject<tir::Var>& var, const Frame& frame, const IRDocsifier& p,
                 std::function<void(AssignDoc)> add_decl) {
  IdDoc id = p->vars->Define(var.Get(), GetVarNameHint(var), frame);
  if (add_decl != nullptr) {
    add_decl(AssignDoc(id, NullOpt, GetTypeAnnotationForVar(var, p)));
  }
  return id;
}

static String GetIterTypePyStr(tir::IterVarType iter_type) {
  switch (iter_type) {
    case tir::kDataPar:
      return "DataPar";
    case tir::kThreadIndex:
      return "ThreadIndex";
    case tir::kCommReduce:
      return "CommReduce";
    case tir::kOrdered:
      return "Ordered";
    case tir::kOpaque:
      return "DimInfo";
    case tir::kUnrolled:
      return "Unrolled";
    case tir::kVectorized:
      return "Vectorized";
    case tir::kParallelized:
      return "Parallelized";
    case tir::kTensorized:
      return "Tensorized";
    default:
      LOG(FATAL) << "Unknown iter type: " << iter_type;
      throw;
  }
}

// T.iter_var(...)
ExprDoc IterVarDef(const TracedObject<tir::IterVar>& iter_var, const IRDocsifier& p) {
  Array<ExprDoc> args;

  args.push_back(p->AsExprDoc(iter_var.GetAttr(&tir::IterVarNode::var)));

  if (iter_var.Get()->dom.defined()) {
    auto dom = iter_var.GetAttr(&tir::IterVarNode::dom);
    auto min = dom.GetAttr(&RangeNode::min);
    auto extent = dom.GetAttr(&RangeNode::extent);
    if (tir::is_zero(min.Get())) {
      auto extent_doc = p->AsExprDoc(extent);
      extent_doc->source_paths.push_back(min.GetPath());
      args.push_back(extent_doc);
    } else {
      auto max = MakeTraced(min.Get() + extent.Get(), extent.GetPath());
      args.push_back(TupleDoc({p->AsExprDoc(min), p->AsExprDoc(max)}));
    }
  } else {
    args.push_back(LiteralDoc::None());
  }

  auto iter_type = iter_var.GetAttr(&tir::IterVarNode::iter_type);
  args.push_back(
      LiteralDoc::Str(MakeTraced(GetIterTypePyStr(iter_type.Get()), iter_type.GetPath())));
  args.push_back(LiteralDoc::Str(iter_var.GetAttr(&tir::IterVarNode::thread_tag)));

  ExprDoc result = TIR(p)->Attr("iter_var")->Call(args);
  result->source_paths.push_back(iter_var.GetPath());
  return result;
}

// T.axis.S/R(...)
ExprDoc IterVarBlockVar(const TracedObject<tir::IterVar>& iter_var, const IRDocsifier& p) {
  // TODO(yelite): implement this in the PR for ForStmt
  throw;
}

// T.launch_thread(...)
ExprDoc IterVarLaunchThread(const TracedObject<tir::IterVar>& iter_var,
                            const TracedObject<PrimExpr>& value, const Frame& frame,
                            const IRDocsifier& p,
                            std::function<void(tir::Var, AssignDoc)> add_thread_binding) {
  TracedObject<tir::Var> var = iter_var.GetAttr(&tir::IterVarNode::var);

  if (!p->vars->IsVarDefined(var.Get())) {
    p->vars->Define(var.Get(), GetVarNameHint(var), frame);
    auto thread_tag = LiteralDoc::Str(iter_var.GetAttr(&tir::IterVarNode::thread_tag));
    add_thread_binding(
        var.Get(),
        AssignDoc(p->AsExprDoc(var), TIR(p)->Attr("env_thread")->Call({thread_tag}), NullOpt));
  }

  return TIR(p)->Attr("launch_thread")->Call({p->AsExprDoc(var), p->AsExprDoc(value)});
}

ExprDoc PrintVariable(TracedObject<tir::Var> var, IRDocsifier p) {
  Optional<ExprDoc> doc = p->vars->GetVarDoc(var);
  if (doc.defined()) {
    return doc.value();
  } else {
    // TODO(yelite): When implementing the PrimFunc printing, the logic here
    // needs to change, putting variable def into PrimFuncFrame if it exists.
    TIRTopLevelFrame top_level_frame = p->GetFrame<TIRTopLevelFrame>().value();
    IdDoc doc = DeclareVar(var, top_level_frame, p,
                           [&decls = top_level_frame->free_var_definitions](AssignDoc decl) {
                             decls.push_back(decl);
                           });
    return doc;
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Var>(PrintVariable);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SizeVar>([](TracedObject<tir::SizeVar> var, IRDocsifier p) {
      return PrintVariable(MakeTraced(var.Get(), var.GetPath()), p);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IterVar>([](TracedObject<tir::IterVar> v, IRDocsifier p) -> Doc {
      LOG(FATAL) << "Cannot print IterVar directly. Please use the helper functions in var.h for "
                    "specific usage of IterVar.";
      throw;
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
