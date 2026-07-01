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
#include "../../../tirx/transform/ir_utils.h"  // For `GetPtrStorageScope`
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc DoConciseScoping(const ffi::Optional<ExprDoc>& lhs, const ExprDoc& rhs,
                     ffi::Array<StmtDoc>* stmts, bool concise_scoping) {
  if (concise_scoping) {
    if (lhs.defined()) {
      stmts->insert(stmts->begin(), AssignDoc(lhs.value(), rhs, std::nullopt));
    } else {
      stmts->insert(stmts->begin(), ExprStmtDoc(rhs));
    }
    return StmtBlockDoc(*stmts);
  } else {
    return ScopeDoc(lhs, rhs, *stmts);
  }
}

bool AllowConciseScoping(const IRDocsifier& d, const ffi::ObjectRef& obj) {
  if (d->cfg.defined()) {
    if (d->cfg->obj_to_annotate.count(obj)) {
      // if the object requires annotation, do not fold this frame
      return false;
    }
  }
  TVM_FFI_ICHECK(!d->frames.empty());
  if (const auto* f = d->frames.back().as<TIRFrameNode>()) {
    return f->allow_concise_scoping;
  }
  TVM_FFI_THROW(NotImplementedError) << "fragment printing";
  TVM_FFI_UNREACHABLE();
}

bool IsAncestorOfAllVarUse(const tirx::Stmt& node, const ffi::ObjectRef& var,
                           const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return false;
  }
  const std::vector<const ffi::Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (*it == node.get()) {
      return true;
    }
  }
  return false;
}

ffi::Optional<PrimExpr> FindReturnValue(const tirx::Stmt& node) {
  auto eval = node.as<tirx::EvaluateNode>();
  if (!eval) return std::nullopt;

  auto call = eval->value.as<CallNode>();
  if (!call) return std::nullopt;

  if (!call->op.same_as(tirx::builtin::ret())) return std::nullopt;

  if (call->args.size() != 1) return std::nullopt;

  return call->args[0].as_or_throw<PrimExpr>();
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::TilePrimitiveCall>(
        "", [](tirx::TilePrimitiveCall op_call, AccessPath p, IRDocsifier d) -> Doc {
          static const OpAttrMap<tirx::TScriptPrinterName>& op_names =
              Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
          auto op = op_call->op;
          if (op_names.count(op) == 0) {
            LOG(WARNING) << "No TScriptPrinterName attribute for " << op->name;
          }

          static const auto& category_map = Op::GetAttrMap<tirx::TIRxOpCategory>("TIRxOpCategory");
          bool is_tile_primitive = category_map.get(op, ffi::String("")) == "tile_primitive";
          TVM_FFI_ICHECK(is_tile_primitive)
              << "Only tile primitive ops can be used in tirx::TilePrimitiveCall";
          ffi::String name = op_names.get(op, op->name);
          // Per-call execution scope is printed as a namespace prefix on the op,
          // e.g. ``T.warp.copy(...)``. ``warpgroup`` prints as ``wg``. The
          // default ``thread`` scope prints through the explicit tile namespace,
          // e.g. ``T.tile.copy(...)``, so canonical script only needs the full
          // TIRx dialect import. ``Tx`` remains a handwritten shorthand for
          // ``T.tile`` and ``T.<scope>`` tile calls.
          auto scope_ns = [](tirx::ScopeKind k) -> ffi::Optional<ffi::String> {
            switch (k) {
              case tirx::ScopeKind::kWarp:
                return ffi::String("warp");
              case tirx::ScopeKind::kWarpgroup:
                return ffi::String("wg");
              case tirx::ScopeKind::kCta:
                return ffi::String("cta");
              case tirx::ScopeKind::kCluster:
                return ffi::String("cluster");
              default:  // kThread -> no prefix
                return std::nullopt;
            }
          };
          auto scoped_callee = [&](const ffi::String& op_name) -> ExprDoc {
            ffi::Optional<ffi::String> ns = scope_ns(op_call->scope->kind);
            if (ns.has_value()) {
              return TIRx(d, ns.value())->Attr(op_name);
            }
            return TIRx(d, "tile")->Attr(op_name);
          };
          if (!op.same_as(tirx::compose_op())) {
            // Trim trailing None args (e.g. optional bias=None, scale=None)
            size_t n_args = op_call->args.size();
            while (n_args > 0 &&
                   op_call->args[n_args - 1].type_index() == ffi::TypeIndex::kTVMFFINone) {
              --n_args;
            }
            // Detect in-place unary ops: after trimming Nones, if exactly 2 args
            // and args[0]/args[1] refer to the same buffer region, collapse to 1 arg
            bool inplace_unary = false;
            if (n_args == 2) {
              auto dst_opt = op_call->args[0].as<tirx::BufferRegion>();
              auto src_opt = op_call->args[1].as<tirx::BufferRegion>();
              if (dst_opt.has_value() && src_opt.has_value() &&
                  dst_opt.value()->buffer.same_as(src_opt.value()->buffer) &&
                  StructuralEqual()(dst_opt.value()->region, src_opt.value()->region)) {
                inplace_unary = true;
              }
            }
            ffi::Array<Doc> args;
            for (size_t i = 0; i < n_args; ++i) {
              if (inplace_unary && i == 1) continue;  // skip duplicate src
              args.push_back(d->AsDoc<Doc>(op_call->args[i], p->Attr("args")->ArrayItem(i)));
            }
            ffi::Optional<ExprDoc> disp = std::nullopt;
            if (op_call->dispatch.has_value()) {
              disp = LiteralDoc::Str(op_call->dispatch.value(), p->Attr("dispatch"));
            }
            return OpCallDoc(scoped_callee(name), args,
                             d->AsDoc<DictDoc>(op_call->workspace, p->Attr("workspace")),
                             d->AsDoc<DictDoc>(op_call->config, p->Attr("config")), disp);
          } else {
            With<TIRFrame> f(d, op_call);
            ffi::Array<tirx::Stmt> stmts;
            for (size_t i = 0, n = op_call->args.size(); i < n; ++i) {
              stmts.push_back(op_call->args[i].as_or_throw<tirx::Stmt>());
            }
            tirx::SeqStmt seq_stmt(stmts);
            AsDocBody(seq_stmt, p->Attr("args"), f->get(), d);
            // Build kwargs: workspace, dispatch, then flatten config
            ffi::Array<ffi::String> kw_keys;
            ffi::Array<ExprDoc> kw_values;
            if (!op_call->workspace.empty()) {
              kw_keys.push_back("workspace");
              kw_values.push_back(d->AsDoc<DictDoc>(op_call->workspace, p->Attr("workspace")));
            }
            if (op_call->dispatch.has_value()) {
              kw_keys.push_back("dispatch");
              kw_values.push_back(LiteralDoc::Str(op_call->dispatch.value(), p->Attr("dispatch")));
            }
            using POO = std::pair<ffi::String, ffi::Any>;
            std::vector<POO> items{op_call->config.begin(), op_call->config.end()};
            std::sort(items.begin(), items.end(),
                      [](const POO& a, const POO& b) { return a.first < b.first; });
            for (const auto& kv : items) {
              kw_keys.push_back(kv.first);
              kw_values.push_back(
                  d->AsDoc<ExprDoc>(kv.second, p->Attr("config")->MapItem(kv.first)));
            }
            return ScopeDoc(std::nullopt, scoped_callee("compose_op")->Call({}, kw_keys, kw_values),
                            (*f)->stmts);
          }
        });
TVM_SCRIPT_REPR(tirx::TilePrimitiveCallNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Evaluate>("", [](tirx::Evaluate eval, AccessPath p, IRDocsifier d) -> Doc {
      if (d->cfg->syntax_sugar) {
        if (auto return_value = FindReturnValue(eval)) {
          ExprDoc value =
              d->AsDoc<ExprDoc>(return_value.value(), p->Attr("value")->Attr("args")->ArrayItem(0));
          return ReturnDoc(value);
        }
      }

      ExprDoc value = d->AsDoc<ExprDoc>(eval->value, p->Attr("value"));
      if (eval->value->IsInstance<CallNode>()) {
        return ExprStmtDoc(value);
      }
      return ExprStmtDoc(TIR(d, "evaluate")->Call({value}));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Bind>("", [](tirx::Bind stmt, AccessPath p, IRDocsifier d) -> Doc {
      // Step 1. Type annotation
      TVM_FFI_ICHECK(!stmt->var->ty.IsMissing())
          << "Type annotation is required for variable: " << stmt->var->name_hint;
      ffi::Optional<ExprDoc> type_doc = d->AsDoc<ExprDoc>(stmt->var->ty,  //
                                                          p->Attr("var")->Attr("type_annotation"));
      if (const auto* tuple_type = stmt->var->ty.as<TupleTypeNode>()) {
        if (tuple_type->fields.empty()) {
          type_doc = std::nullopt;
        }
      }
      // Step 2. RHS
      ExprDoc rhs = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
      // Step 3. LHS - Bind is flat, define var if new, otherwise just assign
      if (!d->IsVarDefined(stmt->var)) {
        TVM_FFI_ICHECK(!d->frames.empty());
        ExprDoc lhs = DefineVar(stmt->var, d->frames.back(), d);
        ExprDoc let_ann = type_doc.defined() ? ExprDoc(IndexDoc(TIR(d, "let"), {type_doc.value()}))
                                             : TIR(d, "let");
        return AssignDoc(lhs, rhs, let_ann);
      } else {
        ExprDoc lhs = d->AsDoc<ExprDoc>(stmt->var, p->Attr("var"));
        return AssignDoc(lhs, rhs, std::nullopt);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AssertStmt>(
        "", [](tirx::AssertStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          // Always emit the canonical tuple form: assert cond, ("Kind", ["part0", "part1", ...])
          ffi::Array<ExprDoc> parts;
          auto parts_path = p->Attr("message_parts");
          for (size_t i = 0; i < stmt->message_parts.size(); ++i) {
            parts.push_back(d->AsDoc<ExprDoc>(stmt->message_parts[i], parts_path->ArrayItem(i)));
          }
          ExprDoc kind_doc = d->AsDoc<ExprDoc>(stmt->error_kind, p->Attr("error_kind"));
          return AssertDoc(cond, TupleDoc({kind_doc, ListDoc(parts)}));
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::While>("", [](tirx::While stmt, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      return WhileDoc(cond, (*f)->stmts);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Break>("", [](tirx::Break stmt, AccessPath p, IRDocsifier d) -> Doc {
      return BreakDoc();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Continue>("", [](tirx::Continue stmt, AccessPath p, IRDocsifier d) -> Doc {
      return ContinueDoc();
    });

namespace {

/*!
 * \brief Find all parent buffers that share the same data pointer with the given child buffer.
 * \param child The child buffer.
 * \param d The IRDocsifier.
 * \return A list of candidate parent buffers.
 */
std::vector<tirx::Buffer> FindParentBuffers(const tirx::Buffer& child, const IRDocsifier& d) {
  std::vector<tirx::Buffer> results;
  for (const auto& [obj, info] : d->obj2info) {
    if (const auto* buf = obj.as<tirx::BufferNode>()) {
      tirx::Buffer parent = ffi::GetRef<tirx::Buffer>(buf);
      if (parent.same_as(child)) continue;
      if (parent->data.same_as(child->data)) {
        results.push_back(parent);
      }
    }
  }
  return results;
}

/*!
 * \brief Check if a layout is the default layout for a given shape.
 */
bool IsDefaultLayout(const ffi::Optional<tirx::Layout>& layout, const ffi::Array<PrimExpr>& shape) {
  if (!layout.defined()) return false;
  return StructuralEqual()(layout.value(), tirx::TileLayoutNode::DefaultLayout(shape));
}

/*!
 * \brief Try to produce a DeclBuffer sugar expression for the given child buffer
 *        with respect to a specific parent buffer.
 *
 * Returns std::nullopt if no sugar pattern matches.
 */
ffi::Optional<ExprDoc> TryDeclBufferSugarWithParent(const tirx::Buffer& child, const AccessPath& p,
                                                    const IRDocsifier& d,
                                                    const tirx::Buffer& parent) {
  ffi::Optional<ExprDoc> parent_doc = d->GetVarDoc(parent);
  if (!parent_doc.defined()) return std::nullopt;
  ExprDoc pdoc = parent_doc.value();

  tirx::ExprDeepEqual expr_equal;

  // Check elem_offset equality
  bool same_elem_offset = expr_equal(child->elem_offset, parent->elem_offset);
  // Check dtype equality
  bool same_dtype = (child->dtype == parent->dtype);
  // Check shape equality
  bool same_shape = (child->shape.size() == parent->shape.size());
  if (same_shape) {
    for (size_t i = 0; i < child->shape.size(); ++i) {
      if (!expr_equal(child->shape[i], parent->shape[i])) {
        same_shape = false;
        break;
      }
    }
  }

  bool child_is_default = IsDefaultLayout(child->layout, child->shape);
  bool parent_is_default = IsDefaultLayout(parent->layout, parent->shape);

  // --- (a) Slice (default layout, different elem_offset) ---
  if (!same_elem_offset && same_dtype && !parent->shape.empty()) {
    // Reconstruct start indices from elem_offset difference and parent strides (row-major)
    // offset_diff = child->elem_offset - parent->elem_offset
    // For row-major: strides[i] = prod(shape[i+1:])
    // start[i] = offset_diff / strides[i]; offset_diff %= strides[i]
    // Build slice doc: parent[start:start+extent, ...]
    // We only support this for IntImm offsets
    auto* child_off = child->elem_offset.as<IntImmNode>();
    auto* parent_off = parent->elem_offset.as<IntImmNode>();
    if (child_off && parent_off) {
      int64_t offset_diff = child_off->value - parent_off->value;
      // Compute row-major strides
      std::vector<int64_t> strides(parent->shape.size());
      int64_t stride = 1;
      for (int i = static_cast<int>(parent->shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        if (auto* s = parent->shape[i].as<IntImmNode>()) {
          stride *= s->value;
        } else {
          return std::nullopt;  // Non-constant shape, can't decompose
        }
      }
      // Check child shape is also all IntImm
      for (size_t i = 0; i < child->shape.size(); ++i) {
        if (!child->shape[i].as<IntImmNode>()) return std::nullopt;
      }
      if (child->shape.size() != parent->shape.size()) return std::nullopt;

      ffi::Array<Doc> slices;
      int64_t remaining = offset_diff;
      bool in_bounds = true;
      for (size_t i = 0; i < parent->shape.size(); ++i) {
        int64_t start_val = remaining / strides[i];
        remaining %= strides[i];
        int64_t extent_val = child->shape[i].as<IntImmNode>()->value;
        int64_t parent_dim = parent->shape[i].as<IntImmNode>()->value;
        int64_t stop_val = start_val + extent_val;
        // Bounds check: start + extent must be within parent dim
        if (stop_val > parent_dim) {
          in_bounds = false;
          break;
        }
        if (start_val == 0 && stop_val == parent_dim) {
          // Full range: use 0:N slice
          ExprDoc start_doc = LiteralDoc::Int(0, p->Attr("elem_offset"));
          ExprDoc stop_doc =
              d->AsDoc<ExprDoc>(parent->shape[i], p->Attr("buffer")->Attr("shape")->ArrayItem(i));
          slices.push_back(SliceDoc(start_doc, stop_doc, std::nullopt));
        } else {
          ExprDoc start_doc = LiteralDoc::Int(start_val, p->Attr("elem_offset"));
          ExprDoc stop_doc = LiteralDoc::Int(stop_val, p->Attr("elem_offset"));
          slices.push_back(SliceDoc(start_doc, stop_doc, std::nullopt));
        }
      }
      if (remaining == 0 && in_bounds) {
        return pdoc[slices];
      }
    }
    return std::nullopt;
  }

  // --- (b) Local: parent has thread axes, child has storage layout (non-thread part) ---
  if (same_elem_offset && same_dtype && !parent_is_default && parent->layout.defined()) {
    if (auto* parent_tile = parent->layout.value().as<tirx::TileLayoutNode>()) {
      if (parent_tile->HasThreadAxis()) {
        // Check if child's layout matches the storage layout (parent layout with thread axes
        // removed). Compute expected storage layout by filtering non-thread shard iters.
        std::vector<tirx::Iter> storage_shard;
        std::vector<tirx::Iter> storage_replica;
        ffi::Map<tirx::Axis, PrimExpr> storage_offset;
        for (const auto& iter : parent_tile->shard) {
          if (!iter->axis->IsThreadAxis()) {
            storage_shard.push_back(iter);
          }
        }
        for (const auto& iter : parent_tile->replica) {
          if (!iter->axis->IsThreadAxis()) {
            storage_replica.push_back(iter);
          }
        }
        for (const auto& [axis, off] : parent_tile->offset) {
          if (!axis->IsThreadAxis()) {
            storage_offset.Set(axis, off);
          }
        }
        tirx::TileLayout expected_storage(
            ffi::Array<tirx::Iter>(storage_shard.begin(), storage_shard.end()),
            ffi::Array<tirx::Iter>(storage_replica.begin(), storage_replica.end()), storage_offset);

        bool child_matches_storage = false;
        if (child->layout.defined()) {
          child_matches_storage =
              StructuralEqual()(child->layout.value(), tirx::Layout(expected_storage));
        }
        if (child_matches_storage) {
          // Compute storage total for auto-infer check
          int64_t total = 1;
          bool all_const = true;
          for (const auto& iter : storage_shard) {
            if (auto* imm = iter->extent.as<IntImmNode>()) {
              total *= imm->value;
            } else {
              all_const = false;
              break;
            }
          }
          // Check if shape can be auto-inferred (single dim matching storage total)
          if (all_const && child->shape.size() == 1) {
            if (auto* child_dim = child->shape[0].as<IntImmNode>()) {
              if (child_dim->value == total) {
                return pdoc->Attr("local")->Call({});
              }
            }
          }
          // Print as parent.local(*shape)
          ffi::Array<ExprDoc> args;
          for (size_t i = 0; i < child->shape.size(); ++i) {
            args.push_back(
                d->AsDoc<ExprDoc>(child->shape[i], p->Attr("buffer")->Attr("shape")->ArrayItem(i)));
          }
          return pdoc->Attr("local")->Call(args);
        }
      }
    }
  }

  // --- (c) View(dtype): different dtype, same elem_offset ---
  if (same_elem_offset && !same_dtype && child->shape.size() == parent->shape.size()) {
    // Verify shape compatibility with dtype reinterpret cast
    int child_bits = child->dtype.bits();
    int parent_bits = parent->dtype.bits();
    bool shapes_compatible = true;
    // All dims except last must match
    for (size_t i = 0; i + 1 < child->shape.size(); ++i) {
      if (!expr_equal(child->shape[i], parent->shape[i])) {
        shapes_compatible = false;
        break;
      }
    }
    if (shapes_compatible && !child->shape.empty()) {
      auto* child_last = child->shape.back().as<IntImmNode>();
      auto* parent_last = parent->shape.back().as<IntImmNode>();
      if (child_last && parent_last) {
        if (child_bits > parent_bits) {
          // Cast up: child_last = parent_last / ratio
          int ratio = child_bits / parent_bits;
          shapes_compatible = (parent_last->value == child_last->value * ratio);
        } else {
          // Cast down: child_last = parent_last * ratio
          int ratio = parent_bits / child_bits;
          shapes_compatible = (child_last->value == parent_last->value * ratio);
        }
      } else {
        shapes_compatible = false;
      }
    }
    // Also verify the parent's layout is compatible with the pack/unpack operation
    if (shapes_compatible && parent->layout.defined()) {
      if (auto* ptile = parent->layout.value().as<tirx::TileLayoutNode>()) {
        if (!ptile->shard.empty() && child_bits > parent_bits) {
          // Cast up requires pack: last shard iter must have stride=1
          // and extent divisible by ratio
          const auto& last_iter = ptile->shard.back();
          auto* last_stride = last_iter->stride.as<IntImmNode>();
          auto* last_extent = last_iter->extent.as<IntImmNode>();
          int ratio = child_bits / parent_bits;
          if (!last_stride || last_stride->value != 1 || !last_extent ||
              last_extent->value % ratio != 0) {
            shapes_compatible = false;
          }
        }
      }
    }
    if (shapes_compatible) {
      ExprDoc dtype_doc =
          LiteralDoc::Str(DType2Str(child->dtype->dtype), p->Attr("buffer")->Attr("dtype"));
      return pdoc->Attr("view")->Call({dtype_doc});
    }
  }

  // --- (d) Permute: child shape is a permutation of parent shape, same elem_offset ---
  if (same_elem_offset && same_dtype && !same_shape &&
      child->shape.size() == parent->shape.size()) {
    // Try to find a permutation
    std::vector<int> perm(child->shape.size(), -1);
    std::vector<bool> used(parent->shape.size(), false);
    bool is_permutation = true;
    for (size_t i = 0; i < child->shape.size(); ++i) {
      bool found = false;
      for (size_t j = 0; j < parent->shape.size(); ++j) {
        if (!used[j] && expr_equal(child->shape[i], parent->shape[j])) {
          perm[i] = j;
          used[j] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        is_permutation = false;
        break;
      }
    }
    // Check it's not identity
    bool is_identity = is_permutation;
    if (is_permutation) {
      for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
          is_identity = false;
          break;
        }
      }
    }
    if (is_permutation && !is_identity) {
      // Verify the layout matches permutation by comparing shard iters directly
      bool layout_matches = false;
      if (parent->layout.defined() && child->layout.defined()) {
        auto* parent_tile = parent->layout.value().as<tirx::TileLayoutNode>();
        auto* child_tile = child->layout.value().as<tirx::TileLayoutNode>();
        if (parent_tile && child_tile && parent_tile->shard.size() == child_tile->shard.size()) {
          StructuralEqual seq;
          layout_matches = true;
          for (size_t i = 0; i < perm.size(); ++i) {
            if (!seq(child_tile->shard[i], parent_tile->shard[perm[i]])) {
              layout_matches = false;
              break;
            }
          }
          // Also check replica and offset are unchanged
          if (layout_matches) {
            layout_matches = seq(child_tile->replica, parent_tile->replica) &&
                             seq(child_tile->offset, parent_tile->offset);
          }
        }
      }
      if (layout_matches) {
        ffi::Array<ExprDoc> args;
        for (int idx : perm) {
          args.push_back(LiteralDoc::Int(idx, p->Attr("buffer")->Attr("shape")));
        }
        return pdoc->Attr("permute")->Call(args);
      }
    }
  }

  // --- (e) Partition: child has 2*parent_ndim dims with grid+tile strides ---
  if (same_elem_offset && same_dtype && !parent->shape.empty() &&
      child->shape.size() == 2 * parent->shape.size() && !child->strides.empty() &&
      child->strides.size() == 2 * parent->shape.size()) {
    size_t ndim = parent->shape.size();
    // Compute parent's row-major strides
    std::vector<int64_t> parent_rm_strides(ndim);
    int64_t stride = 1;
    bool all_const = true;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      parent_rm_strides[i] = stride;
      if (auto* s = parent->shape[i].as<IntImmNode>()) {
        stride *= s->value;
      } else {
        all_const = false;
        break;
      }
    }
    if (all_const) {
      bool is_partition = true;
      for (size_t i = 0; i < ndim; ++i) {
        auto* grid_dim = child->shape[i].as<IntImmNode>();
        auto* tile_dim = child->shape[ndim + i].as<IntImmNode>();
        auto* parent_dim = parent->shape[i].as<IntImmNode>();
        auto* grid_stride = child->strides[i].as<IntImmNode>();
        auto* tile_stride = child->strides[ndim + i].as<IntImmNode>();
        if (!grid_dim || !tile_dim || !parent_dim || !grid_stride || !tile_stride) {
          is_partition = false;
          break;
        }
        // grid × tile == parent dim
        if (grid_dim->value * tile_dim->value != parent_dim->value) {
          is_partition = false;
          break;
        }
        // inner strides match parent's row-major strides
        if (tile_stride->value != parent_rm_strides[i]) {
          is_partition = false;
          break;
        }
        // grid stride == tile_dim × inner stride
        if (grid_stride->value != tile_dim->value * tile_stride->value) {
          is_partition = false;
          break;
        }
      }
      if (is_partition) {
        ffi::Array<ExprDoc> tuple_elems;
        for (size_t i = 0; i < ndim; ++i) {
          tuple_elems.push_back(
              d->AsDoc<ExprDoc>(child->shape[i], p->Attr("buffer")->Attr("shape")->ArrayItem(i)));
        }
        return pdoc->Attr("partition")->Call({}, {"num_tiles"}, {TupleDoc(tuple_elems)});
      }
    }
  }

  // --- (f) View(*shape, layout=L): different shape/layout, same dtype and elem_offset ---
  if (same_elem_offset && same_dtype && !same_shape) {
    // Buffer.view(...) copies the parent's strides onto the child (see
    // python/tvm/tirx/buffer.py:view). If parent has strides but child
    // doesn't (or vice versa), the sugar can't faithfully round-trip
    // through view — fall back to T.decl_buffer where strides is an
    // explicit kwarg.
    bool same_strides = (child->strides.size() == parent->strides.size());
    if (same_strides) {
      for (size_t i = 0; i < child->strides.size(); ++i) {
        if (!expr_equal(child->strides[i], parent->strides[i])) {
          same_strides = false;
          break;
        }
      }
    }
    if (!same_strides) return std::nullopt;

    ffi::Array<ExprDoc> args;
    ffi::Array<ffi::String> kwargs_keys;
    ffi::Array<ExprDoc> kwargs_values;
    for (size_t i = 0; i < child->shape.size(); ++i) {
      args.push_back(
          d->AsDoc<ExprDoc>(child->shape[i], p->Attr("buffer")->Attr("shape")->ArrayItem(i)));
    }
    // Check if layout differs
    bool same_layout = false;
    if (child->layout.defined() && parent->layout.defined()) {
      same_layout = StructuralEqual()(child->layout.value(), parent->layout.value());
    } else if (!child->layout.defined() && !parent->layout.defined()) {
      same_layout = true;
    }
    if (!same_layout && child->layout.defined() && !child_is_default) {
      kwargs_keys.push_back("layout");
      kwargs_values.push_back(
          d->AsDoc<ExprDoc>(child->layout.value(), p->Attr("buffer")->Attr("layout")));
    }
    return pdoc->Attr("view")->Call(args, kwargs_keys, kwargs_values);
  }

  return std::nullopt;
}

/*!
 * \brief Try to produce a DeclBuffer sugar expression, trying all parent buffer candidates.
 */
ffi::Optional<ExprDoc> TryDeclBufferSugar(const tirx::Buffer& child, const AccessPath& p,
                                          const IRDocsifier& d) {
  auto parents = FindParentBuffers(child, d);
  for (const auto& parent : parents) {
    if (auto sugar = TryDeclBufferSugarWithParent(child, p, d, parent)) {
      return sugar;
    }
  }
  return std::nullopt;
}

Doc DeclBufferDoc(tirx::DeclBuffer stmt, AccessPath p, IRDocsifier d,
                  BufferVarDefinition var_definitions) {
  // Try sugar detection when syntax_sugar is enabled
  if (d->cfg->syntax_sugar) {
    if (auto sugar = TryDeclBufferSugar(stmt->buffer, p, d)) {
      ExprDoc lhs = DefineBuffer(stmt->buffer, d->frames.back(), d);
      // Define data pointer inline if needed
      if (!d->IsVarDefined(stmt->buffer->data)) {
        tirx::Buffer buf = stmt->buffer;
        d->Define(stmt->buffer->data, d->frames.back(), [d, buf, p]() {
          return d->AsDoc<ExprDoc>(buf, p->Attr("buffer"))->Attr("data");
        });
      }
      return AssignDoc(lhs, sugar.value(), std::nullopt);
    }
  }
  ExprDoc rhs = BufferDecl(stmt->buffer, "decl_buffer", {}, p->Attr("buffer"), d->frames.back(), d,
                           var_definitions);
  ExprDoc lhs = DefineBuffer(stmt->buffer, d->frames.back(), d);
  return AssignDoc(lhs, rhs, std::nullopt);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::DeclBuffer>(  //
        "", [](tirx::DeclBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          return DeclBufferDoc(stmt, p, d, BufferVarDefinition::None);
        });

namespace {
Doc AllocBufferDoc(tirx::AllocBuffer stmt, AccessPath p, IRDocsifier d) {
  if (d->cfg->syntax_sugar && stmt->buffer.IsScalar(true)) {
    ExprDoc lhs = DefineBuffer(stmt->buffer, d->frames.back(), d);
    if (!d->IsVarDefined(stmt->buffer->data)) {
      tirx::Buffer buf = stmt->buffer;
      d->Define(stmt->buffer->data, d->frames.back(),
                [d, buf, p]() { return d->AsDoc<ExprDoc>(buf, p->Attr("buffer"))->Attr("data"); });
    }
    ExprDoc type_ann = TIR(d, DType2Str(stmt->buffer->dtype->dtype));
    return AssignDoc(lhs, std::nullopt, type_ann);
  }
  ExprDoc rhs = BufferDecl(stmt->buffer, "alloc_buffer", {}, p->Attr("buffer"), d->frames.back(), d,
                           BufferVarDefinition::DataPointer);
  // alloc_buffer carries an `annotations` field on the IR node that BufferDecl
  // doesn't know about. When non-empty, append it as an `annotations=...`
  // kwarg on the emitted call so round-trip preserves the annotation map.
  if (!stmt->annotations.empty()) {
    if (const auto* call = rhs.as<CallDocNode>()) {
      ffi::Array<ffi::String> new_keys = call->kwargs_keys;
      ffi::Array<ExprDoc> new_values = call->kwargs_values;
      new_keys.push_back("annotations");
      new_values.push_back(d->AsDoc<ExprDoc>(stmt->annotations, p->Attr("annotations")));
      rhs = CallDoc(call->callee, call->args, new_keys, new_values);
    }
  }
  ExprDoc lhs = DefineBuffer(stmt->buffer, d->frames.back(), d);
  return AssignDoc(lhs, rhs, std::nullopt);
}

}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AllocBuffer>(  //
        "", [](tirx::AllocBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          return AllocBufferDoc(stmt, p, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::IfThenElse>(  //
        "", [](tirx::IfThenElse stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          ffi::Array<StmtDoc> then_branch;
          ffi::Array<StmtDoc> else_branch;
          if (stmt->then_case.defined()) {
            With<TIRFrame> f(d, stmt->then_case);
            AsDocBody(stmt->then_case, p->Attr("then_case"), f->get(), d);
            then_branch = (*f)->stmts;
          }
          if (stmt->else_case.defined()) {
            With<TIRFrame> f(d, stmt->else_case);
            AsDocBody(stmt->else_case.value(), p->Attr("else_case"), f->get(), d);
            else_branch = (*f)->stmts;
          }
          return IfDoc(cond, then_branch, else_branch);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SeqStmt>("", [](tirx::SeqStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt, p, f->get(), d);
      return StmtBlockDoc((*f)->stmts);
    });

void InsertEnvThread(const tirx::IterVar& iter_var, const AccessPath& iter_var_p,
                     const IRDocsifier& d) {
  Frame f = FindLowestVarDef(iter_var->var, d).value();
  DefineVar(iter_var->var, f, d);
  ExprDoc rhs = TIR(d, "env_thread")
                    ->Call({LiteralDoc::Str(iter_var->thread_tag,  //
                                            iter_var_p->Attr("thread_tag"))});
  ExprDoc lhs = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  f->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
}

ExprDoc DocsifyLaunchThread(const tirx::AttrStmt& attr_stmt, const AccessPath& attr_stmt_p,
                            ffi::Optional<tirx::Var>* define_var, const IRDocsifier& d) {
  tirx::IterVar iter_var = attr_stmt->node.as_or_throw<tirx::IterVar>();
  AccessPath iter_var_p = attr_stmt_p->Attr("node");

  ExprDoc var_doc{ffi::UnsafeInit()};
  if (d->IsVarDefined(iter_var->var)) {
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  } else if (IsAncestorOfAllVarUse(attr_stmt, iter_var->var, d)) {
    var_doc = LiteralDoc::Str(iter_var->thread_tag, iter_var_p->Attr("thread_tag"));
    *define_var = iter_var->var;
  } else {
    InsertEnvThread(iter_var, iter_var_p, d);
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  }
  return TIR(d, "launch_thread")
      ->Call({
          var_doc,
          d->AsDoc<ExprDoc>(attr_stmt->value, attr_stmt_p->Attr("value")),
      });
}

/*! \brief Check whether an AttrStmt has node=IntImm(int32, 0) (the dict-attr pattern). */
static bool IsDictAttrPattern(const tirx::AttrStmt& stmt) {
  if (auto int_imm = stmt->node.as<IntImmNode>()) {
    return int_imm->ty.as_or_throw<PrimType>() == PrimType::Int(32) && int_imm->value == 0;
  }
  return false;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AttrStmt>(  //
        "", [](tirx::AttrStmt stmt, AccessPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          ffi::Optional<ExprDoc> lhs = std::nullopt;
          ffi::Optional<ExprDoc> rhs = std::nullopt;
          ffi::Optional<tirx::Var> define_var = std::nullopt;
          tirx::Stmt body = stmt->body;
          AccessPath body_p = stmt_p->Attr("body");
          if (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread") {
            if (stmt->node.as<tirx::IterVarNode>()) {
              rhs = DocsifyLaunchThread(stmt, stmt_p, &define_var, d);
            }
          }
          if (stmt->attr_key == "tirx_hint") {
            if (auto map_node = stmt->node.as<ffi::Map<ffi::String, ffi::Any>>()) {
              ffi::Array<ExprDoc> args;
              ffi::Array<ffi::String> kwargs_keys;
              ffi::Array<ExprDoc> kwargs_values;
              for (const auto& [k, v] : map_node.value()) {
                if (k == "message") {
                  auto s = v.as<ffi::String>().value();
                  args.push_back(LiteralDoc::Str(s, stmt_p->Attr("node")));
                } else {
                  kwargs_keys.push_back(k);
                  kwargs_values.push_back(d->AsDoc<ExprDoc>(v, stmt_p->Attr("node")));
                }
              }
              rhs = TIR(d, "hint")->Call(args, kwargs_keys, kwargs_values);
            }
          }
          if (!rhs.defined()) {
            // Try to collapse consecutive dict-attr-pattern AttrStmts into T.attr({...})
            if (IsDictAttrPattern(stmt)) {
              ffi::Array<ExprDoc> keys;
              ffi::Array<ExprDoc> values;
              tirx::AttrStmt cur = stmt;
              AccessPath cur_p = stmt_p;
              while (true) {
                keys.push_back(LiteralDoc::Str(cur->attr_key, cur_p->Attr("attr_key")));
                values.push_back(d->AsDoc<ExprDoc>(cur->value, cur_p->Attr("value")));
                if (auto next = cur->body.as<tirx::AttrStmt>()) {
                  if (IsDictAttrPattern(next.value())) {
                    cur = next.value();
                    cur_p = cur_p->Attr("body");
                    continue;
                  }
                }
                body = cur->body;
                body_p = cur_p->Attr("body");
                break;
              }
              rhs = TIR(d, "attr")->Call({DictDoc(keys, values)});
            } else {
              rhs = TIR(d, "attr")->Call({
                  d->AsDoc<ExprDoc>(stmt->node, stmt_p->Attr("node")),
                  LiteralDoc::Str(stmt->attr_key, stmt_p->Attr("attr_key")),
                  d->AsDoc<ExprDoc>(stmt->value, stmt_p->Attr("value")),
              });
            }
          }
          With<TIRFrame> f(d, stmt);
          if (define_var.defined()) {
            lhs = DefineVar(define_var.value(), *f, d);
          }
          AsDocBody(body, body_p, f->get(), d);
          return DoConciseScoping(lhs, rhs.value(), &(*f)->stmts, concise);
        });

TVM_SCRIPT_REPR(tirx::BindNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AttrStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AssertStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::WhileNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AllocBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BreakNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ContinueNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DeclBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SeqStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::IfThenElseNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::EvaluateNode, ReprPrintTIR);
}  // namespace printer
}  // namespace script
}  // namespace tvm
