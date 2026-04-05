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
 * \file src/tirx/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/struct_info.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>

#include <unordered_map>
#include <unordered_set>

#include "../../ir/printer_utils.h"
#include "script_print_utils.h"

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  PrimFuncNode::RegisterReflection();
  TensorIntrinNode::RegisterReflection();
}

namespace {
relax::StructInfo InferStructInfo(const PrimFunc& prim_func) {
  ffi::Array<relax::StructInfo> params;
  for (const auto& param : prim_func->params) {
    relax::StructInfo param_sinfo = [&]() -> relax::StructInfo {
      if (auto opt_buf = prim_func->buffer_map.Get(param)) {
        auto buf = opt_buf.value();
        relax::ShapeExpr shape(
            buf->shape.Map([](PrimExpr dim) { return cast(DataType::Int(64), dim); }));
        return relax::TensorStructInfo(shape, buf->dtype);
      }

      if (auto prim_type = param->type_annotation.as<PrimTypeNode>();
          prim_type && prim_type->dtype.is_handle()) {
        return relax::ObjectStructInfo();
      }

      return relax::PrimStructInfo(param->dtype);
    }();
    params.push_back(param_sinfo);
  }

  relax::StructInfo ret = [&]() -> relax::StructInfo {
    if (const auto* prim = prim_func->ret_type.as<PrimTypeNode>()) {
      return relax::PrimStructInfo(prim->dtype);
    } else if (IsVoidType(prim_func->ret_type)) {
      return relax::TupleStructInfo(ffi::Array<relax::StructInfo>{});
    } else {
      return relax::ObjectStructInfo();
    }
  }();

  bool purity = prim_func->body.defined() ? s_tir::IsPureFunction(prim_func) : false;

  return relax::FuncStructInfo(params, ret, purity);
}
}  // namespace

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(ffi::Array<tirx::Var> params, Stmt body, Type ret_type,
                   ffi::Map<tirx::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
  if (!attrs.defined()) {
    attrs = DictAttrs();
  }

  if (!ret_type.defined()) {
    ret_type = VoidType();
  }

  auto n = ffi::make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->struct_info_ = relax::FuncStructInfo::OpaqueFunc();
  n->span = std::move(span);
  data_ = std::move(n);

  (*this)->struct_info_ = InferStructInfo(*this);
}

FuncType PrimFuncNode::func_type_annotation() const {
  ffi::Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type);
}

class TensorIntrinManager {
 public:
  ffi::Map<ffi::String, tirx::TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin::TensorIntrin(PrimFunc desc, PrimFunc impl) {
  // Check the number of func var is equal
  TVM_FFI_CHECK_EQ(desc->params.size(), impl->params.size(), ValueError)
      << "The number of parameters of the description and the implementation of the "
         "tensor intrinsic doesn't match.";
  for (size_t i = 0; i < desc->params.size(); i++) {
    TVM_FFI_CHECK(desc->params[i]->dtype.is_handle(), ValueError)
        << "Parameters of the description of the "
           "tensor intrinsic should be handle only.";
    TVM_FFI_CHECK(impl->params[i]->dtype.is_handle(), ValueError)
        << "Parameters of the implementation of "
           "the tensor intrinsic should be handle only.";
  }
  TVM_FFI_ICHECK_EQ(desc->buffer_map.size(), impl->buffer_map.size());

  ObjectPtr<TensorIntrinNode> n = ffi::make_object<TensorIntrinNode>();
  n->desc = std::move(desc);
  n->impl = std::move(impl);
  data_ = std::move(n);
}

void TensorIntrin::Register(ffi::String name, TensorIntrin intrin, bool override) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  if (!override) {
    TVM_FFI_CHECK_EQ(manager->reg.count(name), 0, ValueError)
        << "TensorIntrin '" << name << "' has already been registered";
  }
  manager->reg.Set(name, intrin);
}

ffi::Optional<TensorIntrin> TensorIntrin::Get(ffi::String name, bool allow_missing) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  auto it = manager->reg.find(name);
  if (it == manager->reg.end()) {
    if (allow_missing) {
      return std::nullopt;
    } else {
      TVM_FFI_THROW(ValueError) << "TensorIntrin '" << name << "' is not registered";
    }
  }
  return (*it).second;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PrimFunc",
           [](ffi::Array<tirx::Var> params, Stmt body, Type ret_type,
              ffi::Map<tirx::Var, Buffer> buffer_map, DictAttrs attrs,
              Span span) { return PrimFunc(params, body, ret_type, buffer_map, attrs, span); })
      .def("tirx.TensorIntrin",
           [](PrimFunc desc_func, PrimFunc intrin_func) {
             return TensorIntrin(desc_func, intrin_func);
           })
      .def("tirx.TensorIntrinRegister", TensorIntrin::Register)
      .def("tirx.TensorIntrinGet", TensorIntrin::Get);
}

// ---------------------------------------------------------------------------
// __ffi_text_print__ override
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;

  // PrimFunc -> @T.prim_func \n def name(params...): body
  refl::TypeAttrDef<PrimFuncNode>().def(
      "__ffi_text_print__",
      [](PrimFunc func, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        // Determine function name
        ffi::String func_name = "main";
        bool in_module = printer->VarIsDefined(func);
        if (in_module) {
          if (auto binding_expr = printer->VarGet(func)) {
            if (auto* id_node = binding_expr.value().as<::tvm::ffi::ir::text::IdASTObj>()) {
              func_name = id_node->name;
            }
          }
        } else if (func->attrs.defined()) {
          auto it = func->attrs->dict.find("global_symbol");
          if (it != func->attrs->dict.end()) {
            func_name = (*it).second.cast<ffi::String>();
          }
        }
        text::IdAST name = text::IdAST(func_name);

        // Build decorator
        ffi::List<text::ExprAST> decorators;
        bool has_global_symbol = func->attrs.defined() &&
                                 func->attrs->dict.count("global_symbol");
        if (!has_global_symbol) {
          decorators.push_back(
              text::ExprCallKw(TIR("prim_func"), {}, {ffi::String("private")}, {text::LiteralAST::Bool(true)}));
        } else {
          decorators.push_back(TIR("prim_func"));
        }

        // Push frame
        text::DefaultFrame frame;
        printer->FramePush(frame);

        // Pre-compute buffer_data_counter
        int n_args = func->params.size();
        std::unordered_map<const tirx::VarNode*, int> buffer_data_counter;
        for (const auto& pair : func->buffer_map) {
          const tirx::VarNode* data_var = pair.second->data.get();
          if (!buffer_data_counter.count(data_var)) {
            buffer_data_counter.insert({data_var, 0});
          }
          ++buffer_data_counter.at(data_var);
        }

        // Step 1. Handle params with buffer inlining
        ffi::List<text::AssignAST> params;
        std::unordered_set<const BufferNode*> buffer_inlined;

        for (int i = 0; i < n_args; ++i) {
          Var var = func->params[i];
          text::AccessPath var_p = path->Attr("params")->ArrayItem(i);

          if (CountVarOccurrence(func, var) == 2 && func->buffer_map.count(var)) {
            Buffer buffer = func->buffer_map[var];
            if (IsSimpleBuffer(buffer) && buffer_data_counter.at(buffer->data.get()) == 1) {
              text::AccessPath buffer_p = path->Attr("buffer_map")->MapItem(var);
              printer->VarDef(buffer->name, buffer, frame);
              DefineBufferDataVar(buffer, printer);
              text::ExprAST buf_id = printer->VarGet(buffer).value();
              text::ExprAST annotation = PrintBufferAnnotation(buffer, printer, buffer_p);
              params.push_back(text::AssignAST(buf_id, ffi::Optional<text::ExprAST>(),
                                          ffi::Optional<text::ExprAST>(annotation)));
              buffer_inlined.insert(buffer.get());
              continue;
            }
          }

          text::ExprAST var_id = DefineVar(var, printer, var_p);
          ffi::Optional<text::ExprAST> annotation;
          if (var->type_annotation.defined()) {
            annotation = Print(printer, var->type_annotation, var_p->Attr("type_annotation"));
          }
          params.push_back(
              text::AssignAST(var_id, ffi::Optional<text::ExprAST>(), annotation));
        }

        // Step 2. Handle func->attrs
        auto PrintAttrDict = [&](const ffi::Map<ffi::String, ffi::Any>& dict,
                                  const text::AccessPath& dict_p) -> text::ExprAST {
          ffi::List<text::ExprAST> keys;
          ffi::List<text::ExprAST> vals;
          for (const auto& kv : dict) {
            keys.push_back(text::LiteralAST::Str(kv.first));
            vals.push_back(Print(printer, kv.second, dict_p));
          }
          return text::DictAST(std::move(keys), std::move(vals));
        };

        if (func->attrs.defined() && !func->attrs->dict.empty()) {
          if (func->attrs->dict.count("global_symbol") &&
              func->attrs->dict.at("global_symbol").cast<ffi::String>() == func_name) {
            ffi::Map<ffi::String, ffi::Any> new_attrs;
            for (const auto& kv : func->attrs->dict) {
              if (kv.first != "global_symbol") {
                new_attrs.Set(kv.first, kv.second);
              }
            }
            if (!new_attrs.empty()) {
              text::ExprAST attr_dict = PrintAttrDict(new_attrs, path->Attr("attrs"));
              frame->stmts.push_back(
                  text::ExprStmtAST(text::ExprCall(TIR("func_attr"), {attr_dict})));
            }
          } else {
            text::ExprAST attr_dict = PrintAttrDict(func->attrs->dict, path->Attr("attrs"));
            frame->stmts.push_back(
                text::ExprStmtAST(text::ExprCall(TIR("func_attr"), {attr_dict})));
          }
        }

        // Step 3. Handle buffer_map: non-inlined entries
        for (int i = 0; i < n_args; ++i) {
          Var param = func->params[i];
          if (func->buffer_map.count(param)) {
            Buffer buffer = func->buffer_map[param];
            if (buffer_inlined.count(buffer.get())) continue;
            DefineBufferVars(buffer, printer, frame);
            text::AccessPath buffer_p = path->Attr("buffer_map")->MapItem(param);
            printer->VarDef(buffer->name, buffer, frame);
            text::ExprAST buf_id = printer->VarGet(buffer).value();
            text::ExprAST param_doc = params[i]->lhs;
            ffi::List<text::ExprAST> extra_args;
            extra_args.push_back(param_doc);
            text::ExprAST rhs = PrintBufferDecl(buffer, "match_buffer", std::move(extra_args),
                                           printer, buffer_p);
            DefineBufferDataVar(buffer, printer);
            frame->stmts.push_back(text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>()));
          }
        }

        // Step 3b. Emit declarations for undefined Vars
        {
          struct ThreadVarInfo {
            std::vector<IterVar> iter_vars;
          };
          std::unordered_map<std::string, ThreadVarInfo> thread_var_info;
          std::unordered_set<const tirx::VarNode*> thread_vars;
          {
            class ThreadVarCollector : public tirx::StmtVisitor {
             public:
              std::unordered_map<std::string, ThreadVarInfo>* info;
              void VisitStmt_(const tirx::AttrStmtNode* op) final {
                if ((op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") &&
                    (op->node.type_index() >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) &&
                    op->node.cast<ffi::ObjectRef>()->IsInstance<IterVarNode>()) {
                  IterVar iv = op->node.cast<IterVar>();
                  std::string key = iv->thread_tag;
                  (*info)[key].iter_vars.push_back(iv);
                }
                tirx::StmtVisitor::VisitStmt_(op);
              }
            };
            ThreadVarCollector collector;
            collector.info = &thread_var_info;
            Stmt body_to_walk = func->body;
            {
              const SBlockRealizeNode* root_br = func->body.as<SBlockRealizeNode>();
              if (root_br && !root_br->iter_values.size() && is_one(root_br->predicate)) {
                SBlock rb = root_br->block;
                if (!rb->annotations.size() && !rb->match_buffers.size() &&
                    !rb->reads.size() && !rb->writes.size() && !rb->init.defined()) {
                  const SBlockRealizeNode* inner_br = rb->body.as<SBlockRealizeNode>();
                  if (rb->alloc_buffers.size() ||
                      (inner_br && inner_br->block->iter_vars.size()) ||
                      (!inner_br && ContainsNode<SBlockRealizeNode>(rb->body))) {
                    body_to_walk = rb->body;
                  }
                }
              }
            }
            collector(body_to_walk);

            for (const auto& kv : thread_var_info) {
              const std::vector<IterVar>& ivs = kv.second.iter_vars;
              std::unordered_map<const tirx::VarNode*, int> var_ptr_count;
              for (const IterVar& iv : ivs) {
                thread_vars.insert(iv->var.get());
                var_ptr_count[iv->var.get()]++;
              }
              for (const auto& vpc : var_ptr_count) {
                if (vpc.second > 1) {
                  for (const IterVar& iv : ivs) {
                    if (iv->var.get() == vpc.first) {
                      DefineVar(iv->var, printer, text::AccessPath::Root());
                      text::ExprAST var_id = printer->VarGet(iv->var).value();
                      text::ExprAST rhs = text::ExprCall(TIR("env_thread"),
                                             {text::LiteralAST::Str(iv->thread_tag)});
                      frame->stmts.push_back(text::AssignAST(var_id, rhs, ffi::Optional<text::ExprAST>()));
                      break;
                    }
                  }
                }
              }
            }
          }

          ffi::Array<Var> defined_vars;
          for (const auto& param : func->params) {
            defined_vars.push_back(param);
          }
          for (const auto& pair : func->buffer_map) {
            Buffer buf = pair.second;
            defined_vars.push_back(buf->data);
            for (const PrimExpr& s : buf->shape) {
              if (const auto* v = s.as<tirx::VarNode>()) {
                defined_vars.push_back(ffi::GetRef<Var>(v));
              }
            }
            for (const PrimExpr& s : buf->strides) {
              if (const auto* v = s.as<tirx::VarNode>()) {
                defined_vars.push_back(ffi::GetRef<Var>(v));
              }
            }
            if (const auto* v = buf->elem_offset.as<tirx::VarNode>()) {
              defined_vars.push_back(ffi::GetRef<Var>(v));
            }
          }
          {
            class SBlockVarCollector : public tirx::StmtVisitor {
             public:
              ffi::Array<Var>* vars;
              void VisitStmt_(const tirx::SBlockNode* op) final {
                for (const IterVar& iv : op->iter_vars) {
                  vars->push_back(iv->var);
                }
                for (const tirx::Buffer& buf : op->alloc_buffers) {
                  vars->push_back(buf->data);
                }
                for (const tirx::MatchBufferRegion& mb : op->match_buffers) {
                  tirx::Buffer buf = mb->buffer;
                  vars->push_back(buf->data);
                  for (const PrimExpr& s : buf->shape) {
                    if (const auto* v = s.as<tirx::VarNode>()) {
                      vars->push_back(ffi::GetRef<Var>(v));
                    }
                  }
                  for (const PrimExpr& s : buf->strides) {
                    if (const auto* v = s.as<tirx::VarNode>()) {
                      vars->push_back(ffi::GetRef<Var>(v));
                    }
                  }
                  if (const auto* v = buf->elem_offset.as<tirx::VarNode>()) {
                    vars->push_back(ffi::GetRef<Var>(v));
                  }
                }
                tirx::StmtVisitor::VisitStmt_(op);
              }
            };
            SBlockVarCollector collector;
            collector.vars = &defined_vars;
            collector(func->body);
          }
          Stmt body_to_scan = func->body;
          ffi::Array<Var> undef = tirx::UndefinedVars(body_to_scan, defined_vars);
          std::unordered_set<const tirx::VarNode*> seen;
          for (const Var& v : undef) {
            if (seen.count(v.get())) continue;
            seen.insert(v.get());
            if (thread_vars.count(v.get())) continue;
            if (!printer->VarGet(v).has_value()) {
              DefineNewTIRVar(v, printer, frame);
            }
          }
        }

        // Step 4. Handle func->body with implicit root block detection
        ffi::Optional<SBlock> implicit_root_block;
        {
          const SBlockRealizeNode* root_block_realize =
              func->body.as<SBlockRealizeNode>();
          if (root_block_realize && !root_block_realize->iter_values.size() &&
              is_one(root_block_realize->predicate)) {
            SBlock root_block = root_block_realize->block;
            if (!root_block->annotations.size() && !root_block->match_buffers.size() &&
                !root_block->reads.size() && !root_block->writes.size() &&
                !root_block->init.defined()) {
              const SBlockRealizeNode* block_realize =
                  root_block->body.as<SBlockRealizeNode>();
              if (root_block->alloc_buffers.size() ||
                  (block_realize && block_realize->block->iter_vars.size()) ||
                  (!block_realize &&
                   ContainsNode<SBlockRealizeNode>(root_block->body))) {
                implicit_root_block = root_block;
              }
            }
          }
        }

        ffi::List<text::StmtAST> body_stmts;
        if (implicit_root_block.defined()) {
          SBlock root_block = implicit_root_block.value();
          text::AccessPath root_block_p = path->Attr("body")->Attr("block");
          frame->stmts.push_back(text::CommentAST(ffi::Optional<ffi::String>(ffi::String("with T.sblock(\"root\"):"))));
          for (int i = 0, n = root_block->alloc_buffers.size(); i < n; ++i) {
            Buffer buffer = root_block->alloc_buffers[i];
            text::AccessPath buffer_p = root_block_p->Attr("alloc_buffers")->ArrayItem(i);
            std::string buf_name = buffer->name;
            if (buf_name.empty()) buf_name = "buffer";
            printer->VarDef(buf_name, buffer, frame);
            text::ExprAST buf_id = printer->VarGet(buffer).value();
            ffi::List<text::ExprAST> no_extra;
            text::ExprAST rhs = PrintBufferDecl(buffer, "sblock_alloc_buffer", std::move(no_extra),
                                           printer, buffer_p);
            DefineBufferDataVar(buffer, printer);
            frame->stmts.push_back(text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>()));
          }
          body_stmts = PrintBodyStmts(root_block->body, printer, root_block_p->Attr("body"));
        } else {
          body_stmts = PrintBodyStmts(func->body, printer, path->Attr("body"));
        }

        // Merge frame stmts + body
        ffi::List<text::StmtAST> all_body;
        for (const auto& s : frame->stmts) all_body.push_back(s);
        for (const auto& s : body_stmts) all_body.push_back(s);

        printer->FramePop();

        // Return type annotation
        ffi::Optional<text::ExprAST> ret_type;
        if (func->ret_type.defined()) {
          const auto* as_tuple = func->ret_type.as<TupleTypeNode>();
          if (!as_tuple || as_tuple->fields.size()) {
            ret_type = Print(printer, func->ret_type, path->Attr("ret_type"));
          }
        }

        text::FunctionAST func_ast(name, params, decorators, ret_type, all_body);

        if (!in_module) {
          ffi::List<text::StmtAST> result;
          result.push_back(text::CommentAST(
              ffi::Optional<ffi::String>(ffi::String("from tvm.script import tirx as T"))));
          result.push_back(text::CommentAST(ffi::Optional<ffi::String>()));
          result.push_back(func_ast);
          return text::StmtBlockAST(result);
        }
        return func_ast;
      });
}

}  // namespace tirx
}  // namespace tvm
