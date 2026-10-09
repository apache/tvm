/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_permuted_layout.cc
 * \brief The pass injects permuted layout for shared memory buffers to avoid bank conflicts.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>

#include "../../runtime/thread_storage_scope.h"
#include "../../s_tir/ir/ir_mutator_with_analyzer.h"
#include "../../support/utils.h"
#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using namespace sym;
using namespace runtime;

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallNode>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

class PermutedLayoutInjector : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;

  static Function Transform(Function func) {
    Analyzer analyzer;

    auto new_body = ffi::make_object<PermutedLayoutInjector>(func, analyzer)
                        ->Mutate(func->body)
                        .ValueOrUnchanged(func->body);
    auto func_node = func.CopyOnWrite();
    func_node->body = new_body;
    return func;
  }

  explicit PermutedLayoutInjector(Function func, const Analyzer& analyzer)
      : IRMutatorWithAnalyzer(analyzer) {
    for (const Var& param : func->params) {
      if (auto buffer = param.as<TensorVar>()) {
        buffer_map_.insert({buffer.value().var(), buffer.value()});
      }
    }
  }

 private:
  ffi::Array<PrimExpr> PermuteIndices(PrimExpr row_idx, PrimExpr col_idx, int row_size) {
    TVM_FFI_ICHECK(permute_);
    // Index after vectorizing by 8
    PrimExpr col_idx_outer = floordiv(col_idx, VECTORIZE_FACTOR),
             col_idx_inner = floormod(col_idx, VECTORIZE_FACTOR);
    PrimExpr new_col_idx_outer{ffi::UnsafeInit{}};
    if (row_size % 64 == 0) {
      // Use 8 * 8 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 32 banks
      // 0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
      // 0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
      // 0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
      // 0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
      // 0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
      // 0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
      // 0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
      // 0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8);
      new_col_idx_outer = col_idx_outer ^ row_idx_sub;
    } else {
      TVM_FFI_ICHECK(row_size % 32 == 0);
      // Use 8 * 4 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 16 banks
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    3  2  1  0
      // 0  1  2  3    ==>    3  2  1  0
      // View with 8 elements per row:
      // 0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
      // 0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
      // 0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
      // 0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8);
      new_col_idx_outer = col_idx_outer ^ floordiv(row_idx_sub, 2);
    }
    return {row_idx, analyzer_->Simplify(new_col_idx_outer * 8 + col_idx_inner)};
  }

  static bool CheckAnnotation(const Any& annotation) {
    if (auto opt_str = annotation.as<ffi::String>()) {
      // Support string annotation for backward compatibility
      return *opt_str != "";
    } else if (auto* node = annotation.as<IntImmNode>()) {
      return node->value != 0;
    } else if (auto opt_val = annotation.try_cast<int64_t>()) {
      return *opt_val != 0;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid permuted layout annotation: " << annotation;
    }
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) final {
    // Record the mapping from buffer identity to buffer for later lookup.
    for (auto buffer : op->alloc_buffers) {
      buffer_map_.insert({buffer.var(), buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer.var(), match_buffer->buffer});
    }

    if (op->annotations.count("permuted_layout") == 0 ||
        !CheckAnnotation(op->annotations.at("permuted_layout"))) {
      return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
    }

    auto prev_permute = permute_;
    permute_ = true;

    SBlock block = IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    permute_ = prev_permute;

    // Erase the permuted_layout annotation after the pass
    auto block_node = block.CopyOnWrite();
    block_node->annotations.erase("permuted_layout");
    return block;
  }

  int CheckAndGetBufferRowSize(TensorVar buffer) {
    TVM_FFI_ICHECK(buffer->shape.size() >= 2)
        << "The dimension of TensorVar \"" << buffer.name() << "\" with shape " << buffer->shape
        << " should be at least 2";

    auto dim = buffer->shape.size();
    auto buffer_row_size = buffer->shape[dim - 1].as<IntImmNode>()->value;
    auto buffer_col_size = buffer->shape[dim - 2].as<IntImmNode>()->value;

    if (buffer_row_size % 64 != 0) {
      TVM_FFI_ICHECK(buffer_row_size % 32 == 0)
          << "Permuted SLayout for TensorVar \"" << buffer.name() << "\" with shape "
          << buffer->shape << " is not supported since its second dimension is not divisible by 32";
      TVM_FFI_ICHECK(buffer_col_size % 2 == 0)
          << "Permuted SLayout for TensorVar \"" << buffer.name() << "\" with shape "
          << buffer->shape
          << " is not supported since its first dimension is not divisible by 2 and second "
             "dimension is not divisible by 64";
    }

    return buffer_row_size.as<int>().value();
  }

  ffi::Array<PrimExpr> HandleTensorIndices(TensorVar buffer, ffi::Array<PrimExpr> indices) {
    auto buffer_row_size = CheckAndGetBufferRowSize(buffer);

    // Mutate the last two indices
    auto indices_size = indices.size();
    PrimExpr row_idx = indices[indices_size - 2];
    PrimExpr col_idx = indices[indices_size - 1];
    auto new_indices = PermuteIndices(row_idx, col_idx, buffer_row_size);
    indices.Set(indices_size - 2, new_indices[0]);
    indices.Set(indices_size - 1, new_indices[1]);
    return indices;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    // Rewrite write from global to shared.dyn or shared
    // We assume the shape of the shared memory is [..., row_size, col_size],
    // where row_size is divisible by 64, or divisible by 32 and col_size is divisible by 2.
    auto store = IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();

    if (!permute_ || store->dest.as_or_throw<TensorVar>()->shape.size() < 2) {
      return store;
    }

    auto scope = StorageScope::Create(store->dest.as_or_throw<TensorVar>().scope());
    if (scope.rank != StorageRank::kShared) {
      return store;
    }

    auto store_node = store.CopyOnWrite();
    store_node->indices =
        HandleTensorIndices(store_node->dest.as_or_throw<TensorVar>(), store_node->indices);
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    // Rewrite load from shared or shared.dyn to global
    auto load = IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                    .as_or_throw<TensorLoad>();

    if (!permute_ || load->source.as_or_throw<tvm::tirx::TensorVar>()->shape.size() < 2) {
      return load;
    }

    auto scope = StorageScope::Create(load->source.as_or_throw<tvm::tirx::TensorVar>().scope());
    if (scope.rank != StorageRank::kShared) {
      return load;
    }

    return MakeTensorLoad(
        load->source.as_or_throw<tvm::tirx::TensorVar>(),
        HandleTensorIndices(load->source.as_or_throw<tvm::tirx::TensorVar>(), load->indices),
        load->span);
  }

  // Decode physical byte additions around a logical tensor address.  Keep the
  // logical load untouched here: the intrinsic's explicit offset must be added
  // before applying the permutation, and visiting the load first would swizzle twice.
  TensorVar DecodePointer(Expr pointer, PrimExpr* byte_offset) {
    if (const auto* call = pointer.as<CallNode>()) {
      if (call->op.same_as(tirx::ptr_byte_offset_op())) {
        *byte_offset =
            *byte_offset +
            Mutate(call->args[1]).ValueOrUnchanged(call->args[1]).as_or_throw<PrimExpr>();
        return DecodePointer(call->args[0], byte_offset);
      }
      if (call->op.same_as(tirx::reinterpret_op())) {
        return DecodePointer(call->args[0], byte_offset);
      }
      if (call->op.same_as(tirx::address_of_op())) {
        const auto* load = call->args[0].as<TensorLoadNode>();
        TVM_FFI_ICHECK(load) << "Expected a tensor address for permuted layout";
        TensorVar buffer = load->source.as_or_throw<TensorVar>();
        auto indices = Mutate(load->indices)
                           .ValueOrUnchanged(load->indices)
                           .as_or_throw<ffi::Array<PrimExpr>>();
        auto flat_indices = buffer->ElemOffset(indices);
        if (buffer->layout.has_value()) {
          auto coordinates = buffer->layout.value()->Canonicalize()->Apply(indices, buffer->shape);
          TVM_FFI_ICHECK_EQ(coordinates.size(), 1U);
          flat_indices = {(*coordinates.begin()).second + buffer->elem_offset};
        }
        TVM_FFI_ICHECK_EQ(flat_indices.size(), 1U);
        PrimType dtype = buffer->dtype;
        int bytes = (dtype.bits() * dtype.lanes() + 7) / 8;
        *byte_offset = *byte_offset + flat_indices[0] * bytes;
        return buffer;
      }
    }
    auto data_var = GetBufferDataVar(pointer);
    TVM_FFI_ICHECK(data_var.has_value()) << "Expected a tensor pointer, received " << pointer;
    auto it = buffer_map_.find(data_var.value());
    TVM_FFI_ICHECK(it != buffer_map_.end()) << "Unknown tensor pointer: " << pointer;
    return it->second;
  }

  Expr PermutePointer(Expr pointer, ffi::Optional<PrimExpr> offset = std::nullopt) {
    PrimExpr byte_offset = PrimExpr(0);
    TensorVar buffer = DecodePointer(pointer, &byte_offset);
    PrimType dtype = buffer->dtype;
    int bytes = (dtype.bits() * dtype.lanes() + 7) / 8;
    int row_size = CheckAndGetBufferRowSize(buffer);
    PrimExpr smem_offset = floordiv(byte_offset, bytes) + offset.value_or(PrimExpr(0));
    auto indices =
        PermuteIndices(floordiv(smem_offset, row_size), floormod(smem_offset, row_size), row_size);
    PrimExpr new_offset = analyzer_->Simplify(indices[0] * row_size + indices[1]);
    PrimExpr new_bytes = new_offset * bytes;
    PrimExpr remainder = analyzer_->Simplify(floormod(byte_offset, bytes));
    if (!prim::IsZero(remainder)) new_bytes = new_bytes + remainder;
    return Call(pointer->ty, tirx::ptr_byte_offset_op(), {buffer.data(), new_bytes});
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    static const Op ptx_ldmatrix_op = Op::Get("tirx.ptx_legacy.ldmatrix");
    static const Op mma_store_op = Op::Get("tirx.cuda.mma_store");
    if (!permute_ || (!op->op.same_as(ptx_ldmatrix_op) && !op->op.same_as(mma_store_op))) {
      return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
    }
    bool is_ldmatrix = op->op.same_as(ptx_ldmatrix_op);
    int pointer_index = is_ldmatrix ? 5 : 2;
    ffi::Array<Expr> args;
    for (int i = 0; i < static_cast<int>(op->args.size()); ++i) {
      args.push_back(i == pointer_index ? op->args[i]
                                        : Mutate(op->args[i]).ValueOrUnchanged(op->args[i]));
    }
    if (is_ldmatrix) {
      PrimExpr offset = args[6].as_or_throw<PrimExpr>();
      args.Set(pointer_index, PermutePointer(args[pointer_index], offset));
      args.Set(6, IntImm(offset.ty(), 0));
    } else {
      args.Set(pointer_index, PermutePointer(args[pointer_index]));
    }
    return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->span);
  }

  static constexpr size_t VECTORIZE_FACTOR = 8;
  static constexpr size_t BANK_SIZE_BYTES = 128;

  // Mapping from data Var of a TensorVar to TensorVar, for lookup
  std::unordered_map<Var, TensorVar> buffer_map_;
  bool permute_ = false;
};

namespace transform {

Pass InjectPermutedLayout() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return PermutedLayoutInjector::Transform(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InjectPermutedLayout");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectPermutedLayout", InjectPermutedLayout);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
