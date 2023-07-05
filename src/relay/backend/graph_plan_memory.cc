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
 * \file relay/backend/graph_plan_memory.cc
 * \brief Memory index assignment pass for executing
 *   the program in the graph executor.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/container/array.h>
#include <tvm/tir/op.h>

#include "../../runtime/texture.h"
#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/memory.h"
#include "../transforms/device_aware_visitors.h"
#include "./token_allocator.h"
#include "./utils.h"

namespace tvm {
namespace relay {

using TargetsMap = Map<Integer, Target>;
using Texture2DShape = runtime::Texture2DShape<int64_t>;
constexpr auto Is2DStorage = runtime::IsTextureStorage;

using backend::StaticMemoryPlan;
using backend::StorageInfo;
using IntegerArray = Array<Integer>;

class StorageAllocaBaseVisitor : public transform::DeviceAwareExprVisitor {
 public:
  StorageAllocaBaseVisitor() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}

  // run the visitor on a global function.
  void Run(const Function& func) { VisitExpr(func); }

  using transform::DeviceAwareExprVisitor::VisitExpr_;

  void VisitExpr_(const ConstantNode* op) final { this->CreateToken(op, false); }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void DeviceAwareVisitExpr_(const FunctionNode* func_node) final {
    if (function_nesting() > 1) {
      // do not recurse into sub functions.
      return;
    }
    if (func_node->HasNonzeroAttr(attr::kPrimitive)) {
      // No storage needed for primitive functions.
      return;
    }
    for (const auto& param : func_node->params) {
      CreateToken(param.get(), /*can_realloc=*/false);
    }
    // Process the function body, and make sure all result tokens are considered 'alive'.
    for (StorageToken* tok : GetToken(func_node->body)) {
      tok->ref_counter += 1;
    }
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<StorageToken*> fields;
    for (Expr field : op->fields) {
      auto tokens = GetToken(field);
      fields.insert(fields.end(), tokens.begin(), tokens.end());
    }
    token_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& tok = GetToken(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), tok.size());
    token_map_[op] = {tok[op->index]};
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void PreVisitLetBinding_(const Var& var, const Expr& value) final {
    token_map_[var.get()] = GetToken(value);
  }

  void PostVisitLet_(const LetNode* let_node) final {
    token_map_[let_node] = GetToken(let_node->body);
  }

 protected:
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*>> token_map_;
  /*! \brief empty token map */
  const std::vector<StorageToken*> no_tokens_;

  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<StorageToken*>& GetToken(const Expr& expr) {
    this->VisitExpr(expr);
    // See through on_device calls.
    Expr real_expr = IgnoreOnDevice(expr);

    // Functions don't require data storage, represented by the empty token
    if (real_expr->checked_type().as<FuncTypeNode>()) {
      return no_tokens_;
    }
    this->VisitExpr(real_expr);
    auto it = token_map_.find(real_expr.get());
    ICHECK(it != token_map_.end()) << "Expression not found in storage map:" << std::endl
                                   << PrettyPrint(real_expr);
    return it->second;
  }

  /*!
   * \brief Allocates (or reuses if \p can_realloc is true) a storage token for holding
   * the result of evaluating \p op.
   */
  void CreateToken(const ExprNode* expr_node, bool can_realloc) {
    return CreateTokenOnDevice(expr_node, GetVirtualDevice(GetRef<Expr>(expr_node)), can_realloc);
  }

  /*!
   * \brief Allocates (or reuses if \p can_realloc is true) a storage token for holding
   * the result of evaluating \p op on \p device_type.
   */
  virtual void CreateTokenOnDevice(const ExprNode* op, const VirtualDevice& virtual_device,
                                   bool can_realloc) = 0;
};

/*! \brief Associate storage with every expression without any concern for sharing. */
class StorageAllocaInit : protected StorageAllocaBaseVisitor {
 public:
  explicit StorageAllocaInit(support::Arena* arena) : arena_(arena) {}

  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*>> GetInitTokenMap(
      const Function& func) {
    this->Run(func);
    return std::move(token_map_);
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  void CreateTokenOnDevice(const ExprNode* op, const VirtualDevice& virtual_device,
                           bool can_realloc) override {
    ICHECK(!token_map_.count(op));
    std::vector<StorageToken*> tokens;
    for (const auto& ttype : FlattenTupleType(op->checked_type())) {
      auto* token = arena_->make<StorageToken>();
      token->ttype = ttype;
      token->virtual_device = virtual_device;
      tokens.push_back(token);
    }
    token_map_[op] = tokens;
  }

  using StorageAllocaBaseVisitor::DeviceAwareVisitExpr_;

  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    // create token for the call node.
    CreateToken(call_node, true);

    // for each input, visit argument token.
    for (Expr arg : call_node->args) {
      for (StorageToken* tok : GetToken(arg)) {
        tok->ref_counter += 1;
      }
    }
  }

 private:
  // allocator
  support::Arena* arena_;
  Map<Expr, Array<String>> node_storage_map_;
};

/*! \brief Associate storage with every expression, reusing storage where possible. */
class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  StorageAllocator() = default;

  /*!
   * \return total number of bytes allocated
   */
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (const auto* p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // Run storage allocation for a function.
  StaticMemoryPlan Plan(const Function& func) {
    VLOG_CONTEXT << "StorageAllocator";
    VLOG(1) << "planning:" << std::endl << PrettyPrint(func);
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);
    this->Run(func);

    // The value of smap contains two integer arrays where the first array
    // contains the planned storage ids and the second holds the device types.
    Map<Expr, backend::StorageInfo> smap;
    int num_annotated_nodes = 0;
    int num_nodes = 0;

    for (const auto& kv : token_map_) {
      std::vector<int64_t> storage_ids;
      storage_ids.reserve(kv.second.size());
      std::vector<VirtualDevice> virtual_devices;
      virtual_devices.reserve(kv.second.size());
      std::vector<int64_t> sid_sizes_byte;
      sid_sizes_byte.reserve(kv.second.size());

      for (StorageToken* tok : kv.second) {
        VLOG(1) << "token: " << tok->ToString();
        if (tok->is_valid()) {
          num_annotated_nodes++;
        }
        num_nodes++;
        storage_ids.push_back(tok->storage_id);
        virtual_devices.push_back(tok->virtual_device);
        sid_sizes_byte.push_back(allocator_.GetMemorySize(tok));
      }
      auto storage_info = backend::StorageInfo(std::move(storage_ids), std::move(virtual_devices),
                                               std::move(sid_sizes_byte));
      smap.Set(GetRef<Expr>(kv.first), storage_info);
    }
    // Either all or none of the nodes should be annotated.
    VLOG(1) << "num annotated nodes / num_nodes: " << num_annotated_nodes << " / " << num_nodes
            << std::endl;
    if (num_annotated_nodes != 0 && num_annotated_nodes != num_nodes) {
      LOG(FATAL) << num_annotated_nodes << " out of " << num_nodes
                 << "expressions are assigned with virtual device types. Either all "
                    "or none of the expressions are expected to be annotated.";
    }
    return backend::StaticMemoryPlan(smap);
  }

 protected:
  // override create token by getting token as prototype requirements.
  void CreateTokenOnDevice(const ExprNode* op, const VirtualDevice& virtual_device,
                           bool can_realloc) final {
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    std::vector<StorageToken*> tokens;

    for (StorageToken* tok : it->second) {
      ICHECK(tok->virtual_device == virtual_device);
      if (can_realloc) {
        tokens.push_back(allocator_.Request(tok));
      } else {
        // Allocate a new token,
        StorageToken* allocated_tok = allocator_.Alloc(tok);
        allocated_tok->virtual_device = tok->virtual_device;
        // ensure it never get de-allocated.
        allocated_tok->ref_counter += 1;
        tokens.push_back(allocated_tok);
      }
    }
    token_map_[op] = tokens;
  }

  // Mark op to reuse the input_token
  // tie the two memories together
  void ReuseInputToken(const ExprNode* op, StorageToken* input_token) {
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    ICHECK_EQ(it->second.size(), 1U);
    StorageToken* prototype = it->second[0];
    // add the reference counter of the output
    // so the input token can only be deleted after references
    // to both are expired
    input_token->ref_counter += prototype->ref_counter;
    // reuse the input token
    token_map_[op] = {input_token};
  }

  using StorageAllocaBaseVisitor::DeviceAwareVisitExpr_;

  // The call map
  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    std::vector<StorageToken*> args;
    // for each input, visit argument token.

    for (const Expr& arg : call_node->args) {
      // Note: GetToken skips GlobalVars and handles tuples properly, so we don't need to treat
      // call_lowered specially.
      for (StorageToken* tok : GetToken(arg)) {
        args.push_back(tok);
      }
    }

    // Under the flat-memory setting.
    // we can force aliasing the input and output of reshape
    // to make it an nop. Note that this is not true
    // for non-flat memory case. Given the current graph plan memory
    // only works for flat memory case, we will go with this choice
    //
    // TODO(tvm-team) Update checks of flat memory enablement when we support
    // opaque-nd memory planning to skip this path.
    // TODO(mbs): "reshape" cleanup.
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    if (call_lowered_props.lowered_func.defined() && IsReshapeOnly(call_lowered_props)) {
      ICHECK_EQ(call_lowered_props.arguments.size(), 1U);
      ReuseInputToken(call_node, args[0]);
    } else {
      // create token for the call node.
      CreateToken(call_node, true);
    }

    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : token_map_.at(call_node)) {
      allocator_.CheckForRelease(tok);
    }
    for (StorageToken* tok : args) {
      tok->ref_counter -= 1;
      allocator_.CheckForRelease(tok);
    }
  }

  class TokenAllocator {
   public:
    StorageToken* Alloc(StorageToken* proto) {
      return Is2DStorage(proto) ? token_2d_.Alloc(proto, storage_ids_++)
                                : token_1d_.Alloc(proto, storage_ids_++);
    }
    StorageToken* Request(StorageToken* proto) {
      StorageToken* token =
          Is2DStorage(proto) ? token_2d_.Request(proto) : token_1d_.Request(proto);
      return token ? token : this->Alloc(proto);
    }
    void CheckForRelease(StorageToken* tok) {
      return Is2DStorage(tok) ? token_2d_.CheckForRelease(tok) : token_1d_.CheckForRelease(tok);
    }

    size_t GetMemorySize(StorageToken* tok) {
      // TODO(amalyshe): figure out who requries sizes and for what
      // size in case of texture is not enough - we can return any value if it
      // assumed to be used for memory allocatoion or we can return real size
      // if it is just for information
      return Is2DStorage(tok) ? 0 : token_1d_.GetMemorySize(tok);
    }
    static bool Is2DStorage(StorageToken* tok) {
      return relay::Is2DStorage(tok->virtual_device->memory_scope);
    }

   private:
    int64_t storage_ids_{0};
    TokenAllocator1D token_1d_;
    TokenAllocator2D token_2d_;
  };

 private:
  // allocator
  support::Arena arena_;
  // scale used for rough match
  // size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*>> prototype_;
  /*! \brief token allocator for optimizing 1d and 2d token alloc requests */
  TokenAllocator allocator_;
};

StaticMemoryPlan GraphPlanMemory(const Function& func) { return StorageAllocator().Plan(func); }

TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory").set_body_typed(GraphPlanMemory);

}  // namespace relay
}  // namespace tvm
