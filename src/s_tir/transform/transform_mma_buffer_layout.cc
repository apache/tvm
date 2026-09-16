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

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Rewriter for all m16n8k8.matrix[A/B/C] buffer. This pass mainly do two things:
 *     1. Lower m16n8k8.matrix[A/B/C] buffer to local registers, where each thread holds their
 *        own part of the matrix;
 *     2. Rewrite access of m16n8k8.matrixC so it can access the correct part of the matrix.
 *   The reason why access of m16n8k8.matrix[A/B] buffer doesn't need this kind of rewrite is
 *   that their access is through opaque access inside ldmatrix and mma_sync. Please refer to
 *   get_index_[A/B] in python/tvm/tirx/tensor_intrin/cuda.py.
 *   We cannot use this kind of opaque access in matrixC too since the ptx stmatrix is only
 *   supported for sm90 or higher. Therefore, writeback of matrixC is limited to the
 *   transparent way.
 */
class MmaBufferLayoutTransformer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
    SBlock block = ffi::GetRef<SBlock>(op);
    auto* n = block.CopyOnWrite();
    auto fmutate = [this](const BufferVar& buffer) {
      // m16n8k8.matrix[A/B/C] buffers are composed ofseveral small blocks. Assume the block's
      // shape is [bi, bj]. Inside each small block, we have 8 threads in stride dimension and 4
      // threads in contiguous dimension, so we change the buffer's shape from [i, j]
      // to [i // bi, j // bj, bi // 8, bj // 4].
      if (buffer.scope() == "m16n8k8.matrixC") {
        // m16n8k8.matrixC
        // bi = 16, bj = 8
        size_t size = buffer->shape.size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 16 == 0 && dim1->value % 8 == 0);

        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 16), IntImm::Int32(dim1->value / 8), 2, 2});

        BufferVar new_buffer =
            decl_buffer(std::move(new_shape), buffer->dtype, buffer.name(), "local");
        VarRemapSet(buffer, new_buffer);
        return new_buffer;

      } else if (buffer.scope() == "m16n8k8.matrixA") {
        // m16n8k8.matrixA
        // bi = 32, bj = 8
        size_t size = buffer->shape.size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 32 == 0 && dim1->value % 8 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 32), IntImm::Int32(dim1->value / 8), 4, 2});

        BufferVar new_buffer =
            decl_buffer(std::move(new_shape), buffer->dtype, buffer.name(), "local");
        VarRemapSet(buffer, new_buffer);
        return new_buffer;

      } else if (buffer.scope() == "m16n8k8.matrixB") {
        // m16n8k8.matrixB
        // bj = 8, bj = 32
        size_t size = buffer->shape.size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 8 == 0 && dim1->value % 32 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 8), IntImm::Int32(dim1->value / 32), 1, 8});

        BufferVar new_buffer =
            decl_buffer(std::move(new_shape), buffer->dtype, buffer.name(), "local");
        VarRemapSet(buffer, new_buffer);
        return new_buffer;
      }
      return buffer;
    };
    n->alloc_buffers.MutateByApply(fmutate);
    n->body = Mutate(n->body, inplace_mode).ValueOrUnchanged(n->body);
    return block;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) {
    BufferVar original_buffer = op->buffer;
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore store = ffi::GetRef<BufferStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    if (auto replacement = VarRemapGet(original_buffer).as<BufferVar>()) {
      auto* n = store.CopyOnWrite();
      if (original_buffer.scope() == "m16n8k8.matrixC") {
        const auto index_map_func = tvm::ffi::Function::GetGlobal("tirx.index_map_m16n8k8.matrixC");
        TVM_FFI_ICHECK(index_map_func.has_value());
        auto index_map = IndexMap::FromFunc(2, *index_map_func);
        auto new_indices = index_map->MapIndices(store->indices, analyzer);
        n->buffer = replacement.value();
        n->indices = std::move(new_indices);
      } else if (original_buffer.scope() == "m16n8k8.matrixA" ||
                 original_buffer.scope() == "m16n8k8.matrixB") {
        TVM_FFI_ICHECK(false)
            << "TransformMmaBufferLayout requires " << original_buffer.scope()
            << " buffers to be accessed through opaque ldmatrix/mma_sync operations, but found "
               "an explicit BufferStore.";
      }
    }
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) {
    BufferVar buffer = op->source.as_or_throw<BufferVar>();
    // Remap the source together with its indices below, after the scope checks.
    auto indices_result =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices_result.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices_result).ValueUnchecked();
    }
    if (auto replacement = VarRemapGet(buffer).as<BufferVar>()) {
      ffi::Array<PrimExpr> indices = load->indices;
      if (buffer.scope() == "m16n8k8.matrixC") {
        const auto index_map_func = tvm::ffi::Function::GetGlobal("tirx.index_map_m16n8k8.matrixC");
        TVM_FFI_ICHECK(index_map_func.has_value());
        auto index_map = IndexMap::FromFunc(2, *index_map_func);
        indices = index_map->MapIndices(load->indices, analyzer);
      } else {
        TVM_FFI_ICHECK(false)
            << "TransformMmaBufferLayout requires " << buffer.scope()
            << " buffers to be accessed through opaque ldmatrix/mma_sync operations, but found "
               "an explicit TensorLoad.";
      }
      return BufferLoad(replacement.value(), indices, load->span);
    }
    return load;
  }

 private:
  arith::Analyzer analyzer;
};

namespace transform {

Pass TransformMmaBufferLayout() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<MmaBufferLayoutTransformer>()
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.TransformMmaBufferLayout", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.TransformMmaBufferLayout", TransformMmaBufferLayout);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
