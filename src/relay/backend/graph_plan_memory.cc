/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/graph_mem_alloca.cc
 * \brief Memory index assignment pass for executing
 *   the program in the graph runtime.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include "../../common/arena.h"

namespace tvm {
namespace relay {

struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type node. */
  const TensorTypeNode* ttype{nullptr};
  /*! \brief virtual device index */
  int device_id{0};
  /*! \brief The storage id */
  int64_t storage_id{-1};
};

class StorageAllocaBaseVisitor : public ExprVisitor {
 public:
  // run the visitor on a function.
  void Run(const Function& func) {
    for (Var param : func->params) {
      CreateToken(param.operator->(), false);
    }
    this->VisitExpr(func->body);
  }

  void VisitExpr_(const ConstantNode* op) final {
    this->CreateToken(op, false);
  }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recursive into sub function.
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
      auto tok = GetToken(field);
      CHECK_EQ(tok.size(), 1U);
      fields.push_back(tok[0]);
    }
    token_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& tok = GetToken(op->tuple);
    CHECK_LT(static_cast<size_t>(op->index), tok.size());
    token_map_[op] = {tok[op->index]};
  }

  void VisitExpr_(const IfNode* op) final {
    LOG(FATAL) << "if is not supported.";
  }

  void VisitExpr_(const LetNode* op) final {
    auto token = GetToken(op->value);
    token_map_[op->var.operator->()] = token;
    token_map_[op] = GetToken(op->body);
  }

 protected:
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > token_map_;

  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<StorageToken*>& GetToken(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = token_map_.find(expr.operator->());
    CHECK(it != token_map_.end());
    return it->second;
  }
  /*!
   * \brief Populate the token map to set op's tokens
   * \param op The node to be processed.
   * \param can_realloc Whether we can re-allocate the memory.
   */
  virtual void CreateToken(const ExprNode* op, bool can_realloc) = 0;
};


class StorageAllocaInit : protected StorageAllocaBaseVisitor {
 public:
  explicit StorageAllocaInit(common::Arena* arena)
      : arena_(arena) {}


  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> >
  GetInitTokenMap(const Function& func) {
    this->Run(func);
    return std::move(token_map_);
  }


 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  void CreateToken(const ExprNode* op, bool can_realloc)  final {
    CHECK(!token_map_.count(op));
    std::vector<StorageToken*> tokens;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        CHECK(ttype);
        StorageToken* token = arena_->make<StorageToken>();
        token->ttype = ttype;
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      CHECK(ttype);
      StorageToken* token = arena_->make<StorageToken>();
      token->ttype = ttype;
      tokens.push_back(token);
    }
    token_map_[op] = tokens;
  }

  void VisitExpr_(const CallNode* op) final {
    // create token for the call node.
    CreateToken(op, true);
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        tok->ref_counter += 1;
      }
    }
  }

 private:
  // allocator
  common::Arena* arena_;
};


class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  /*!
   * \return totoal number of bytes allocated
   */
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (const auto* p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // Run storage allocation for a function.
  Map<Expr, Array<Integer> > Plan(const Function& func) {
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);
    this->Run(func);

    Map<Expr, Array<Integer> > smap;

    for (const auto& kv : token_map_) {
      Array<Integer> vec;
      for (StorageToken* tok : kv.second) {
        vec.push_back(tok->storage_id);
      }
      smap.Set(GetRef<Expr>(kv.first), vec);
    }
    return smap;
  }


 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;
  // override create token by getting token as prototype requirements.
  void CreateToken(const ExprNode* op, bool can_realloc)  final {
    CHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    CHECK(it != prototype_.end());
    std::vector<StorageToken*> tokens;
    for (StorageToken* tok : it->second) {
      if (can_realloc) {
        tokens.push_back(Request(tok));
      } else {
        // Allocate a new token,
        StorageToken* allocated_tok = Alloc(tok, GetMemorySize(tok));
        // ensure it never get de-allocated.
        allocated_tok->ref_counter += 1;
        tokens.push_back(allocated_tok);
      }
    }
    token_map_[op] = tokens;
  }
  // The call map
  void VisitExpr_(const CallNode* op) final {
    std::vector<StorageToken*> args;
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        args.push_back(tok);
      }
    }
    // create token for the call node.
    CreateToken(op, true);
    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : token_map_.at(op)) {
      CheckForRelease(tok);
    }
    for (StorageToken* tok : args) {
      tok->ref_counter -= 1;
      CheckForRelease(tok);
    }
  }
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }
  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  size_t GetMemorySize(StorageToken* prototype) {
    const TensorTypeNode* ttype = prototype->ttype;
    CHECK(ttype != nullptr);
    size_t size = 1;
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = as_const_int(dim);
      CHECK(pval != nullptr)
          << "Cannot allocate memory symbolic tensor shape "
          << ttype->shape;
      size *= static_cast<size_t>(pval[0]);
    }
    size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
    return size;
  }
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype) {
    // calculate the size;
    size_t size = GetMemorySize(prototype);
    // search memory block in [size / match_range_, size * match_range_)
    if (match_range_ == 0) {
      return this->Alloc(prototype, size);
    }
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageToken *tok = it->second;
      if (tok->device_id != prototype->device_id) continue;
      CHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      free_.erase(it);
      return tok;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken *tok = it->second;
      if (tok->device_id != prototype->device_id) continue;
      CHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // erase from map and return
      free_.erase(it);
      return tok;
    }
    // cannot find anything return a new one.
    return this->Alloc(prototype, size);
  }
  /*!
   * \brief Allocate a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, size_t size) {
    prototype->max_bytes = size;
    prototype->storage_id = static_cast<int64_t>(data_.size());
    data_.push_back(prototype);
    return prototype;
  }
  /*!
   * \brief Check if we can release token.
   * \tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok) {
    CHECK_GE(tok->storage_id, 0);
    CHECK_GE(tok->ref_counter, 0);
    if (tok->ref_counter == 0) {
      free_.insert({tok->max_bytes, tok});
    }
  }

 private:
  // allocator
  common::Arena arena_;
  // scale used for rough match
  size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > prototype_;
};


Map<Expr, Array<Integer> > GraphPlanMemory(const Function& func) {
  return StorageAllocator().Plan(func);
}

TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory")
.set_body_typed<Map<Expr, Array<Integer> >(const Function&)>(GraphPlanMemory);

}  // namespace relay
}  // namespace tvm
