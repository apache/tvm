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
 * \file backend/cuda/transforms/lower_iket.cc
 * \brief Lower frontend-only TIRx IKET annotations to CUDA tracing helpers.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace transform {

namespace {

constexpr int kMaxDeclarations = 30;
constexpr int kOfficialMetaInfoBytes = 48;
constexpr int kOfficialEventAttributesBytes = 60;
constexpr int kOfficialRangeAttributesBytes = 72;
constexpr size_t kMaxTokenAnalysisStates = 256;
constexpr size_t kMaxTokenAnalysisIterations = 256;
constexpr size_t kMaxConvergenceAnalysisIterations = 256;

enum class DeclarationKind : int {
  kRange = 1,
  kPush = 2,
  kMark = 3,
};

const Op& IketMarkOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_mark");
  return op;
}

const Op& IketRangeStartOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_range_start");
  return op;
}

const Op& IketRangeEndOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_range_end");
  return op;
}

const Op& IketRangePushOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_range_push");
  return op;
}

const Op& IketRangePopOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_range_pop");
  return op;
}

const Op& IketSentinelOp() {
  static const Op& op = Op::Get("tirx.cuda.iket_sentinel_token");
  return op;
}

bool IsIketOp(const ffi::ObjectRef& value) {
  auto op = value.as<Op>();
  if (!op.has_value()) return false;
  return op.value().same_as(IketMarkOp()) || op.value().same_as(IketRangeStartOp()) ||
         op.value().same_as(IketRangeEndOp()) || op.value().same_as(IketRangePushOp()) ||
         op.value().same_as(IketRangePopOp()) || op.value().same_as(IketSentinelOp());
}

bool IsTokenProducer(const CallNode* call) {
  return call->op.same_as(IketRangeStartOp()) || call->op.same_as(IketSentinelOp());
}

bool IsValidUTF8(const std::string& value) {
  size_t i = 0;
  while (i < value.size()) {
    uint8_t first = static_cast<uint8_t>(value[i]);
    if (first <= 0x7f) {
      ++i;
      continue;
    }
    int continuation = 0;
    uint32_t codepoint = 0;
    if ((first & 0xe0) == 0xc0) {
      continuation = 1;
      codepoint = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
      continuation = 2;
      codepoint = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
      continuation = 3;
      codepoint = first & 0x07;
    } else {
      return false;
    }
    if (i + continuation >= value.size()) return false;
    for (int j = 1; j <= continuation; ++j) {
      uint8_t byte = static_cast<uint8_t>(value[i + j]);
      if ((byte & 0xc0) != 0x80) return false;
      codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    if ((continuation == 1 && codepoint < 0x80) || (continuation == 2 && codepoint < 0x800) ||
        (continuation == 3 && codepoint < 0x10000) || codepoint > 0x10ffff ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
      return false;
    }
    i += continuation + 1;
  }
  return true;
}

std::string GetName(const CallNode* call, size_t index = 0) {
  TVM_FFI_CHECK(call->args.size() > index, ValueError)
      << call->op.as<Op>().value()->name << " requires a literal event name";
  const auto* name = call->args[index].as<StringImmNode>();
  TVM_FFI_CHECK(name != nullptr, TypeError)
      << call->op.as<Op>().value()->name << " requires a string-literal event name";
  std::string result = name->value;
  TVM_FFI_CHECK(!result.empty(), ValueError) << "IKET event names must not be empty";
  TVM_FFI_CHECK(result.size() <= 32, ValueError)
      << "IKET event name exceeds 32 UTF-8 bytes: " << result;
  TVM_FFI_CHECK(IsValidUTF8(result), ValueError) << "IKET event name is not valid UTF-8";
  return result;
}

std::string ValidatePayload(const Expr& payload) {
  DLDataType dtype = payload.as_or_throw<PrimExpr>().ty()->dtype;
  TVM_FFI_CHECK_EQ(dtype.lanes, 1, TypeError) << "IKET payload must be a scalar value";
  TVM_FFI_CHECK_GT(dtype.bits, 0, TypeError) << "IKET payload must have a concrete bit width";
  TVM_FFI_CHECK_LE(dtype.bits, 64, TypeError) << "IKET payload must be at most 64 bits";
  bool supported = dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLFloat ||
                   dtype.code == kDLBfloat;
  TVM_FFI_CHECK(supported, TypeError) << "IKET payload must be bool, int, uint, or float, but got "
                                      << ffi::DLDataTypeToString(dtype);
  CallEffectKind effect = SideEffect(payload.as_or_throw<PrimExpr>());
  TVM_FFI_CHECK(effect <= CallEffectKind::kReadState, ValueError)
      << "IKET payload expressions may read state but must not update state; got " << effect;
  return ffi::DLDataTypeToString(dtype);
}

bool IsScalarBufferAccess(const BufferVar& buffer, const ffi::Array<PrimExpr>& indices) {
  if (buffer->shape.size() != 1 || indices.size() != 1) return false;
  const auto* extent = buffer->shape[0].as<IntImmNode>();
  const auto* index = indices[0].as<IntImmNode>();
  return extent && extent->value == 1 && index && index->value == 0;
}

const char* DeclarationKindName(DeclarationKind kind) {
  switch (kind) {
    case DeclarationKind::kRange:
      return "range";
    case DeclarationKind::kPush:
      return "push_pop";
    case DeclarationKind::kMark:
      return "mark";
  }
  return "unknown";
}

struct DeclarationKey {
  DeclarationKind kind;
  std::string name;

  bool operator<(const DeclarationKey& other) const {
    return std::tie(kind, name) < std::tie(other.kind, other.name);
  }
};

struct Declaration {
  DeclarationKey key;
  bool payload_schema_set{false};
  bool has_payload{false};
  std::string payload_dtype;
  bool end_payload_schema_set{false};
  bool end_has_payload{false};
  std::string end_payload_dtype;
  uint32_t event_id{0};
};

class AnnotationCollector : public StmtExprVisitor {
 public:
  std::map<DeclarationKey, Declaration> declarations;
  bool has_annotations{false};

 private:
  void AddDeclaration(DeclarationKind kind, const CallNode* call, bool sentinel = false) {
    std::string name = GetName(call);
    TVM_FFI_CHECK(call->args.size() == 1 || call->args.size() == 2, TypeError)
        << call->op.as<Op>().value()->name << " expects a name and optional payload";
    bool has_payload = call->args.size() == 2;
    std::string dtype = has_payload ? ValidatePayload(call->args[1]) : std::string();
    auto [name_it, name_inserted] = declaration_kinds_.emplace(name, kind);
    TVM_FFI_CHECK(name_inserted || name_it->second == kind, ValueError)
        << "IKET declaration " << name << " changes event kind from "
        << DeclarationKindName(name_it->second) << " to " << DeclarationKindName(kind);
    DeclarationKey key{kind, std::move(name)};
    auto [it, inserted] = declarations.emplace(key, Declaration{key});
    Declaration& declaration = it->second;
    if (!sentinel) {
      if (declaration.payload_schema_set) {
        TVM_FFI_CHECK_EQ(declaration.has_payload, has_payload, ValueError)
            << "IKET declaration " << declaration.key.name
            << " changes payload presence between sites";
        TVM_FFI_CHECK_EQ(declaration.payload_dtype, dtype, TypeError)
            << "IKET declaration " << declaration.key.name
            << " changes payload dtype between sites";
      } else {
        declaration.payload_schema_set = true;
        declaration.has_payload = has_payload;
        declaration.payload_dtype = std::move(dtype);
      }
    }
  }

  void VisitExpr_(const CallNode* call) final {
    if (!IsIketOp(call->op)) {
      StmtExprVisitor::VisitExpr_(call);
      return;
    }
    has_annotations = true;
    if (call->op.same_as(IketMarkOp())) {
      AddDeclaration(DeclarationKind::kMark, call);
    } else if (call->op.same_as(IketRangeStartOp())) {
      TVM_FFI_CHECK(call->ty.as_or_throw<PrimType>()->dtype.code == kDLUInt &&
                        call->ty.as_or_throw<PrimType>()->dtype.bits == 32,
                    TypeError)
          << "IKET range_start must return uint32";
      AddDeclaration(DeclarationKind::kRange, call);
    } else if (call->op.same_as(IketSentinelOp())) {
      TVM_FFI_CHECK_EQ(call->args.size(), 1, TypeError)
          << "IKET sentinel_token expects one event name";
      TVM_FFI_CHECK(call->ty.as_or_throw<PrimType>()->dtype.code == kDLUInt &&
                        call->ty.as_or_throw<PrimType>()->dtype.bits == 32,
                    TypeError)
          << "IKET sentinel_token must return uint32";
      AddDeclaration(DeclarationKind::kRange, call, true);
    } else if (call->op.same_as(IketRangePushOp())) {
      AddDeclaration(DeclarationKind::kPush, call);
    } else if (call->op.same_as(IketRangeEndOp())) {
      TVM_FFI_CHECK(call->args.size() == 1 || call->args.size() == 2, TypeError)
          << "IKET range_end expects a token and optional payload";
      DLDataType token_dtype = call->args[0].as_or_throw<PrimExpr>().ty()->dtype;
      TVM_FFI_CHECK(token_dtype.code == kDLUInt && token_dtype.bits == 32 && token_dtype.lanes == 1,
                    TypeError)
          << "IKET RangeToken must have dtype uint32";
      if (call->args.size() == 2) ValidatePayload(call->args[1]);
    } else if (call->op.same_as(IketRangePopOp())) {
      TVM_FFI_CHECK_EQ(call->args.size(), 0, TypeError) << "IKET range_pop takes no arguments";
    }
  }

  std::unordered_map<std::string, DeclarationKind> declaration_kinds_;
};

using TokenBufferSet = std::unordered_set<const VarNode*>;

class TokenBufferCollector : public StmtExprVisitor {
 public:
  explicit TokenBufferCollector(TokenBufferSet* buffers) : buffers_(buffers) {}

  bool changed{false};

 private:
  void VisitStmt_(const BufferStoreNode* store) final {
    bool is_token_value = false;
    if (const auto* call = store->value.as<CallNode>()) {
      is_token_value = IsTokenProducer(call);
    } else if (const auto* load = store->value.as<BufferLoadNode>()) {
      is_token_value = buffers_->count(load->buffer.get());
    }
    if (is_token_value && buffers_->insert(store->buffer.get()).second) changed = true;
    StmtExprVisitor::VisitStmt_(store);
  }

  TokenBufferSet* buffers_;
};

TokenBufferSet CollectTokenBuffers(const Stmt& body) {
  TokenBufferSet buffers;
  bool changed = true;
  while (changed) {
    TokenBufferCollector collector(&buffers);
    collector(body);
    changed = collector.changed;
  }
  return buffers;
}

using TokenDeclarationMap = std::unordered_map<const VarNode*, std::set<DeclarationKey>>;

class TokenDeclarationCollector : public StmtExprVisitor {
 public:
  explicit TokenDeclarationCollector(TokenDeclarationMap* declarations)
      : declarations_(declarations) {}

  bool changed{false};

 private:
  void VisitStmt_(const BufferStoreNode* store) final {
    std::set<DeclarationKey> possible;
    if (const auto* call = store->value.as<CallNode>(); call && IsTokenProducer(call)) {
      possible.insert(DeclarationKey{DeclarationKind::kRange, GetName(call)});
    } else if (const auto* load = store->value.as<BufferLoadNode>()) {
      auto it = declarations_->find(load->buffer.get());
      if (it != declarations_->end()) possible = it->second;
    }
    if (!possible.empty()) {
      auto& target = (*declarations_)[store->buffer.get()];
      size_t old_size = target.size();
      target.insert(possible.begin(), possible.end());
      changed = changed || target.size() != old_size;
    }
    StmtExprVisitor::VisitStmt_(store);
  }

  TokenDeclarationMap* declarations_;
};

TokenDeclarationMap CollectTokenDeclarations(const Stmt& body) {
  TokenDeclarationMap declarations;
  bool changed = true;
  while (changed) {
    TokenDeclarationCollector collector(&declarations);
    collector(body);
    changed = collector.changed;
  }
  return declarations;
}

class RangeEndSchemaVerifier : public StmtExprVisitor {
 public:
  RangeEndSchemaVerifier(const TokenDeclarationMap& token_declarations,
                         std::map<DeclarationKey, Declaration>* declarations)
      : token_declarations_(token_declarations), declarations_(declarations) {}

 private:
  void VisitExpr_(const CallNode* call) final {
    if (!call->op.same_as(IketRangeEndOp())) {
      StmtExprVisitor::VisitExpr_(call);
      return;
    }
    const auto* token = call->args[0].as<BufferLoadNode>();
    TVM_FFI_ICHECK(token != nullptr);
    auto possible_it = token_declarations_.find(token->buffer.get());
    TVM_FFI_ICHECK(possible_it != token_declarations_.end());
    bool has_payload = call->args.size() == 2;
    std::string dtype = has_payload ? ValidatePayload(call->args[1]) : std::string();
    for (const DeclarationKey& key : possible_it->second) {
      auto declaration_it = declarations_->find(key);
      TVM_FFI_ICHECK(declaration_it != declarations_->end());
      Declaration& declaration = declaration_it->second;
      if (declaration.end_payload_schema_set) {
        TVM_FFI_CHECK_EQ(declaration.end_has_payload, has_payload, ValueError)
            << "range_end for " << key.name << " changes payload presence between sites";
        TVM_FFI_CHECK_EQ(declaration.end_payload_dtype, dtype, TypeError)
            << "range_end for " << key.name << " changes payload dtype between sites";
      } else {
        declaration.end_payload_schema_set = true;
        declaration.end_has_payload = has_payload;
        declaration.end_payload_dtype = dtype;
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  const TokenDeclarationMap& token_declarations_;
  std::map<DeclarationKey, Declaration>* declarations_;
};

class TokenVerifier : public StmtExprVisitor {
 public:
  explicit TokenVerifier(const TokenBufferSet& token_buffers) : token_buffers_(token_buffers) {}

 private:
  void VisitStmt_(const BufferStoreNode* store) final {
    if (!token_buffers_.count(store->buffer.get())) {
      StmtExprVisitor::VisitStmt_(store);
      return;
    }
    TVM_FFI_CHECK(store->buffer->dtype->dtype.code == kDLUInt &&
                      store->buffer->dtype->dtype.bits == 32 &&
                      store->buffer->dtype->dtype.lanes == 1,
                  TypeError)
        << "IKET RangeToken storage must have dtype uint32";
    TVM_FFI_CHECK(
        store->buffer.scope() == "local" && IsScalarBufferAccess(store->buffer, store->indices),
        ValueError)
        << "IKET RangeToken must use element zero of a local uint32[1] buffer";
    bool valid_value = false;
    if (const auto* call = store->value.as<CallNode>()) {
      valid_value = IsTokenProducer(call);
      allow_producer_ = valid_value;
      VisitExpr(store->value);
      allow_producer_ = false;
    } else if (const auto* load = store->value.as<BufferLoadNode>()) {
      valid_value = token_buffers_.count(load->buffer.get());
      allow_token_load_ = valid_value;
      VisitExpr(store->value);
      allow_token_load_ = false;
    }
    TVM_FFI_CHECK(valid_value, ValueError)
        << "RangeToken may only be assigned another token, range_start, or sentinel_token";
    for (const PrimExpr& index : store->indices) VisitExpr(index);
  }

  void VisitExpr_(const BufferLoadNode* load) final {
    if (token_buffers_.count(load->buffer.get())) {
      TVM_FFI_CHECK(allow_token_load_, ValueError)
          << "RangeToken may only be assigned or passed directly to range_end";
    }
    StmtExprVisitor::VisitExpr_(load);
  }

  void VisitExpr_(const CallNode* call) final {
    if (IsTokenProducer(call)) {
      TVM_FFI_CHECK(allow_producer_, ValueError)
          << "range_start and sentinel_token results must be assigned to a RangeToken";
      bool old_allow_producer = allow_producer_;
      allow_producer_ = false;
      for (const Expr& arg : call->args) VisitExpr(arg);
      allow_producer_ = old_allow_producer;
      return;
    }
    if (call->op.same_as(IketRangeEndOp())) {
      TVM_FFI_CHECK_GE(call->args.size(), 1, TypeError) << "range_end requires a RangeToken";
      const auto* token = call->args[0].as<BufferLoadNode>();
      TVM_FFI_CHECK(token != nullptr && token_buffers_.count(token->buffer.get()), ValueError)
          << "range_end requires a directly loaded RangeToken";
      allow_token_load_ = true;
      VisitExpr(call->args[0]);
      allow_token_load_ = false;
      for (size_t i = 1; i < call->args.size(); ++i) VisitExpr(call->args[i]);
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  const TokenBufferSet& token_buffers_;
  bool allow_token_load_{false};
  bool allow_producer_{false};
};

enum class TokenValueKind : uint8_t {
  kSentinel,
  kActiveRange,
  kConsumed,
};

struct TokenValue {
  TokenValueKind kind{TokenValueKind::kSentinel};
  std::string name;

  bool operator==(const TokenValue& other) const {
    return kind == other.kind && name == other.name;
  }

  bool operator<(const TokenValue& other) const {
    return std::tie(kind, name) < std::tie(other.kind, other.name);
  }
};

struct TokenAnalysisState {
  std::map<const VarNode*, TokenValue> token_values;
  std::set<std::string> active_ranges;

  bool operator==(const TokenAnalysisState& other) const {
    return token_values == other.token_values && active_ranges == other.active_ranges;
  }

  bool operator<(const TokenAnalysisState& other) const {
    return std::tie(token_values, active_ranges) <
           std::tie(other.token_values, other.active_ranges);
  }
};

using TokenAnalysisStates = std::set<TokenAnalysisState>;

class TokenOperationFinder : public StmtExprVisitor {
 public:
  bool found{false};

 private:
  void VisitExpr_(const CallNode* call) final {
    if (IsTokenProducer(call) || call->op.same_as(IketRangeEndOp())) {
      found = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
};

/*! \brief Prove the strict token alternation required by NVIDIA IKET.
 *
 * Any state explosion, unsupported control flow, unknown token, second start of
 * an active name, or end of an inactive name rejects the annotated module.
 */
class OfficialTokenAnalyzer {
 public:
  explicit OfficialTokenAnalyzer(const TokenBufferSet& token_buffers)
      : token_buffers_(token_buffers) {}

  bool Prove(const Stmt& body) {
    TokenAnalysisStates initial{TokenAnalysisState{}};
    TokenAnalysisStates output = Process(body, initial);
    if (!valid_) return false;
    for (const TokenAnalysisState& state : output) {
      if (!state.active_ranges.empty()) return false;
    }
    return true;
  }

 private:
  TokenAnalysisStates Limit(TokenAnalysisStates states) {
    if (states.size() > kMaxTokenAnalysisStates) valid_ = false;
    return valid_ ? std::move(states) : TokenAnalysisStates{};
  }

  TokenAnalysisStates Union(const TokenAnalysisStates& lhs, const TokenAnalysisStates& rhs) {
    TokenAnalysisStates result = lhs;
    result.insert(rhs.begin(), rhs.end());
    return Limit(std::move(result));
  }

  TokenAnalysisStates ProcessLoop(const Stmt& body, const TokenAnalysisStates& input) {
    TokenAnalysisStates closure = input;
    for (size_t iteration = 0; valid_ && iteration < kMaxTokenAnalysisIterations; ++iteration) {
      TokenAnalysisStates after_body = Process(body, closure);
      if (!valid_) return {};
      TokenAnalysisStates next = Union(closure, after_body);
      if (!valid_) return {};
      if (next == closure) return closure;
      closure = std::move(next);
    }
    valid_ = false;
    return {};
  }

  TokenAnalysisStates ProcessTokenStore(const BufferStoreNode* store,
                                        const TokenAnalysisStates& input) {
    TokenAnalysisStates result;
    for (const TokenAnalysisState& old_state : input) {
      TokenAnalysisState state = old_state;
      if (const auto* call = store->value.as<CallNode>(); call && IsTokenProducer(call)) {
        if (call->op.same_as(IketSentinelOp())) {
          state.token_values[store->buffer.get()] = TokenValue{TokenValueKind::kSentinel, {}};
        } else {
          std::string name = GetName(call);
          if (state.active_ranges.count(name)) {
            valid_ = false;
            return {};
          }
          state.active_ranges.insert(name);
          state.token_values[store->buffer.get()] =
              TokenValue{TokenValueKind::kActiveRange, std::move(name)};
        }
      } else if (const auto* load = store->value.as<BufferLoadNode>();
                 load && token_buffers_.count(load->buffer.get())) {
        auto source = state.token_values.find(load->buffer.get());
        if (source == state.token_values.end()) {
          valid_ = false;
          return {};
        }
        state.token_values[store->buffer.get()] = source->second;
      } else {
        valid_ = false;
        return {};
      }
      result.insert(std::move(state));
      if (store->predicate.has_value()) result.insert(old_state);
    }
    return Limit(std::move(result));
  }

  TokenAnalysisStates ProcessRangeEnd(const CallNode* call, const TokenAnalysisStates& input) {
    const auto* load = call->args[0].as<BufferLoadNode>();
    if (!load || !token_buffers_.count(load->buffer.get())) {
      valid_ = false;
      return {};
    }
    TokenAnalysisStates result;
    for (const TokenAnalysisState& old_state : input) {
      auto token = old_state.token_values.find(load->buffer.get());
      if (token == old_state.token_values.end()) {
        valid_ = false;
        return {};
      }
      if (token->second.kind == TokenValueKind::kConsumed) {
        valid_ = false;
        return {};
      }
      TokenAnalysisState state = old_state;
      if (token->second.kind == TokenValueKind::kActiveRange) {
        auto active = state.active_ranges.find(token->second.name);
        if (active == state.active_ranges.end()) {
          valid_ = false;
          return {};
        }
        state.active_ranges.erase(active);
      }
      state.token_values[load->buffer.get()] = TokenValue{TokenValueKind::kConsumed, {}};
      result.insert(std::move(state));
    }
    return Limit(std::move(result));
  }

  TokenAnalysisStates Process(const Stmt& stmt, const TokenAnalysisStates& input) {
    if (!valid_ || input.empty()) return input;
    if (const auto* sequence = stmt.as<SeqStmtNode>()) {
      TokenAnalysisStates states = input;
      for (const Stmt& item : sequence->seq) {
        states = Process(item, states);
        if (!valid_ || states.empty()) break;
      }
      return states;
    }
    if (const auto* store = stmt.as<BufferStoreNode>()) {
      if (token_buffers_.count(store->buffer.get())) {
        return ProcessTokenStore(store, input);
      }
      return input;
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>()) {
      if (const auto* call = evaluate->value.as<CallNode>()) {
        if (call->op.same_as(IketRangeEndOp())) return ProcessRangeEnd(call, input);
        if (call->op.same_as(builtin::thread_return())) {
          for (const TokenAnalysisState& state : input) {
            if (!state.active_ranges.empty()) {
              valid_ = false;
              break;
            }
          }
          return {};
        }
      }
      return input;
    }
    if (const auto* branch = stmt.as<IfThenElseNode>()) {
      TokenAnalysisStates then_states = Process(branch->then_case, input);
      TokenAnalysisStates else_states =
          branch->else_case.has_value() ? Process(branch->else_case.value(), input) : input;
      return Union(then_states, else_states);
    }
    if (const auto* loop = stmt.as<ForNode>()) return ProcessLoop(loop->body, input);
    if (const auto* loop = stmt.as<WhileNode>()) return ProcessLoop(loop->body, input);
    if (const auto* attr_stmt = stmt.as<AttrStmtNode>()) {
      return Process(attr_stmt->body, input);
    }
    if (const auto* block = stmt.as<SBlockNode>()) {
      TokenAnalysisStates states = input;
      if (block->init.has_value()) states = Union(states, Process(block->init.value(), states));
      return Process(block->body, states);
    }
    if (const auto* realize = stmt.as<SBlockRealizeNode>()) {
      return Union(input, Process(realize->block, input));
    }
    if (stmt.as<BreakNode>() || stmt.as<ContinueNode>()) {
      valid_ = false;
      return {};
    }
    // Unknown statement forms are harmless only if they cannot hide token
    // operations.  Otherwise their control flow has not been proved.
    TokenOperationFinder finder;
    finder(stmt);
    if (finder.found) {
      valid_ = false;
      return {};
    }
    return input;
  }

  const TokenBufferSet& token_buffers_;
  bool valid_{true};
};

using OfficialStackState = std::vector<std::string>;
using OfficialStackStates = std::set<OfficialStackState>;

class StackOperationFinder : public StmtExprVisitor {
 public:
  bool found{false};

 private:
  void VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(IketRangePushOp()) || call->op.same_as(IketRangePopOp())) {
      found = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
};

/*! \brief Prove balanced LIFO push/pop behavior required by NVIDIA IKET. */
class OfficialStackAnalyzer {
 public:
  bool Prove(const Stmt& body) {
    OfficialStackStates output = Process(body, OfficialStackStates{OfficialStackState{}});
    if (!valid_) return false;
    for (const OfficialStackState& state : output) {
      if (!state.empty()) return false;
    }
    return true;
  }

 private:
  OfficialStackStates Limit(OfficialStackStates states) {
    if (states.size() > kMaxTokenAnalysisStates) valid_ = false;
    return valid_ ? std::move(states) : OfficialStackStates{};
  }

  OfficialStackStates Union(const OfficialStackStates& lhs, const OfficialStackStates& rhs) {
    OfficialStackStates result = lhs;
    result.insert(rhs.begin(), rhs.end());
    return Limit(std::move(result));
  }

  OfficialStackStates ProcessLoop(const Stmt& body, const OfficialStackStates& input) {
    OfficialStackStates closure = input;
    for (size_t iteration = 0; valid_ && iteration < kMaxTokenAnalysisIterations; ++iteration) {
      OfficialStackStates after_body = Process(body, closure);
      if (!valid_) return {};
      OfficialStackStates next = Union(closure, after_body);
      if (!valid_) return {};
      if (next == closure) return closure;
      closure = std::move(next);
    }
    valid_ = false;
    return {};
  }

  OfficialStackStates Process(const Stmt& stmt, const OfficialStackStates& input) {
    if (!valid_ || input.empty()) return input;
    if (const auto* sequence = stmt.as<SeqStmtNode>()) {
      OfficialStackStates states = input;
      for (const Stmt& item : sequence->seq) {
        states = Process(item, states);
        if (!valid_ || states.empty()) break;
      }
      return states;
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>()) {
      if (const auto* call = evaluate->value.as<CallNode>()) {
        if (call->op.same_as(IketRangePushOp())) {
          OfficialStackStates result;
          for (OfficialStackState state : input) {
            state.push_back(GetName(call));
            result.insert(std::move(state));
          }
          return Limit(std::move(result));
        }
        if (call->op.same_as(IketRangePopOp())) {
          OfficialStackStates result;
          for (OfficialStackState state : input) {
            if (state.empty()) {
              valid_ = false;
              return {};
            }
            state.pop_back();
            result.insert(std::move(state));
          }
          return Limit(std::move(result));
        }
        if (call->op.same_as(builtin::thread_return())) {
          for (const OfficialStackState& state : input) {
            if (!state.empty()) {
              valid_ = false;
              break;
            }
          }
          return {};
        }
      }
      return input;
    }
    if (const auto* branch = stmt.as<IfThenElseNode>()) {
      OfficialStackStates then_states = Process(branch->then_case, input);
      OfficialStackStates else_states =
          branch->else_case.has_value() ? Process(branch->else_case.value(), input) : input;
      return Union(then_states, else_states);
    }
    if (const auto* loop = stmt.as<ForNode>()) return ProcessLoop(loop->body, input);
    if (const auto* loop = stmt.as<WhileNode>()) return ProcessLoop(loop->body, input);
    if (const auto* attr_stmt = stmt.as<AttrStmtNode>()) return Process(attr_stmt->body, input);
    if (const auto* block = stmt.as<SBlockNode>()) {
      OfficialStackStates states = input;
      if (block->init.has_value()) states = Union(states, Process(block->init.value(), states));
      return Process(block->body, states);
    }
    if (const auto* realize = stmt.as<SBlockRealizeNode>()) {
      return Union(input, Process(realize->block, input));
    }
    if (stmt.as<BreakNode>() || stmt.as<ContinueNode>()) {
      valid_ = false;
      return {};
    }
    StackOperationFinder finder;
    finder(stmt);
    if (finder.found) {
      valid_ = false;
      return {};
    }
    return input;
  }

  bool valid_{true};
};

class StripIket : public StmtExprMutator {
 public:
  explicit StripIket(TokenBufferSet token_buffers) : token_buffers_(std::move(token_buffers)) {}

 private:
  Stmt VisitStmt_(const AllocBufferNode* alloc) final {
    if (token_buffers_.count(alloc->buffer.get())) return Evaluate(0);
    return StmtExprMutator::VisitStmt_(alloc);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    if (token_buffers_.count(store->buffer.get())) return Evaluate(0);
    return StmtExprMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const EvaluateNode* evaluate) final {
    if (const auto* call = evaluate->value.as<CallNode>(); call && IsIketOp(call->op)) {
      return Evaluate(0);
    }
    return StmtExprMutator::VisitStmt_(evaluate);
  }

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(IketRangeStartOp()) || call->op.same_as(IketSentinelOp())) {
      return IntImm(PrimType::UInt(32), 0);
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  TokenBufferSet token_buffers_;
};

bool IsEvaluateZero(const Stmt& stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate ? evaluate->value.as<IntImmNode>() : nullptr;
  return value && value->value == 0;
}

class RemoveStrippedIketNoOps : public StmtExprMutator {
 private:
  Stmt PreserveConditionEffects(const PrimExpr& condition) {
    return SideEffect(condition) > CallEffectKind::kReadState ? Evaluate(condition) : Evaluate(0);
  }

  Stmt VisitStmt_(const SeqStmtNode* sequence) final {
    ffi::Array<Stmt> items;
    for (const Stmt& item : sequence->seq) {
      Stmt rewritten = VisitStmt(item);
      if (!IsEvaluateZero(rewritten)) items.push_back(std::move(rewritten));
    }
    return SeqStmt::Flatten(items);
  }

  Stmt VisitStmt_(const AttrStmtNode* attr_stmt) final {
    Stmt body = VisitStmt(attr_stmt->body);
    if (IsEvaluateZero(body)) return body;
    if (body.same_as(attr_stmt->body)) return ffi::GetRef<Stmt>(attr_stmt);
    return AttrStmt(attr_stmt->node, attr_stmt->attr_key,
                    VisitExpr(attr_stmt->value).as_or_throw<PrimExpr>(), body, attr_stmt->span);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt body = VisitStmt(loop->body);
    if (IsEvaluateZero(body)) return body;
    if (body.same_as(loop->body)) return ffi::GetRef<Stmt>(loop);
    return For(loop->loop_var, VisitExpr(loop->min).as_or_throw<PrimExpr>(),
               VisitExpr(loop->extent).as_or_throw<PrimExpr>(), loop->kind, body,
               loop->thread_binding, loop->annotations, loop->step, loop->span);
  }

  Stmt VisitStmt_(const WhileNode* loop) final {
    Stmt body = VisitStmt(loop->body);
    if (IsEvaluateZero(body)) return body;
    if (body.same_as(loop->body)) return ffi::GetRef<Stmt>(loop);
    return While(VisitExpr(loop->condition).as_or_throw<PrimExpr>(), body, loop->span);
  }

  Stmt VisitStmt_(const IfThenElseNode* branch) final {
    PrimExpr condition = VisitExpr(branch->condition).as_or_throw<PrimExpr>();
    Stmt then_case = VisitStmt(branch->then_case);
    if (!branch->else_case.has_value()) {
      if (IsEvaluateZero(then_case)) return PreserveConditionEffects(condition);
      return IfThenElse(condition, then_case, std::nullopt, branch->span);
    }
    Stmt else_case = VisitStmt(branch->else_case.value());
    bool empty_then = IsEvaluateZero(then_case);
    bool empty_else = IsEvaluateZero(else_case);
    if (empty_then && empty_else) return PreserveConditionEffects(condition);
    if (empty_else) return IfThenElse(condition, then_case, std::nullopt, branch->span);
    if (empty_then) return IfThenElse(!condition, else_case, std::nullopt, branch->span);
    return IfThenElse(condition, then_case, else_case, branch->span);
  }
};

bool IsCudaDeviceFunction(const PrimFunc& function) {
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  CallingConv calling_conv =
      function->GetAttr<CallingConv>(tvm::attr::kCallingConv, CallingConv::kDefault).value();
  return target.has_value() && target.value()->kind->name == "cuda" &&
         calling_conv == CallingConv::kDeviceKernelLaunch;
}

bool IsSm90OrNewer(const PrimFunc& function) {
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value()) return false;
  auto arch = target.value()->GetAttr<ffi::String>("arch");
  if (!arch.has_value()) return false;
  std::string value = arch.value();
  if (!value.starts_with("sm_")) return false;
  size_t end = 3;
  while (end < value.size() && value[end] >= '0' && value[end] <= '9') ++end;
  if (end == 3) return false;
  return std::stoi(value.substr(3, end - 3)) >= 90;
}

bool HasAnyPayload(const std::map<DeclarationKey, Declaration>& declarations) {
  for (const auto& item : declarations) {
    const Declaration& declaration = item.second;
    if (declaration.has_payload ||
        (declaration.end_payload_schema_set && declaration.end_has_payload)) {
      return true;
    }
  }
  return false;
}

std::string FunctionName(const GlobalVar& global_var, const PrimFunc& function) {
  if (auto symbol = function->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    return symbol.value();
  }
  return global_var->name_hint;
}

struct KernelIketInfo {
  GlobalVar global_var;
  PrimFunc function;
  std::string name;
  std::map<DeclarationKey, Declaration> declarations;
};

void PutOfficialU32(std::vector<uint8_t>* bytes, size_t offset, uint32_t value) {
  TVM_FFI_ICHECK_LE(offset + sizeof(value), bytes->size());
  for (size_t index = 0; index < sizeof(value); ++index) {
    (*bytes)[offset + index] = static_cast<uint8_t>(value >> (index * 8));
  }
}

uint32_t OfficialRangeId(const std::string& name) {
  uint32_t value = 2166136261U;
  for (uint8_t byte : name) {
    value ^= byte;
    value *= 16777619U;
  }
  return value;
}

std::string OfficialSymbolName(const std::string& name) {
  std::string result;
  result.reserve(name.size());
  for (uint8_t byte : name) {
    bool valid = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                 (byte >= '0' && byte <= '9') || byte == '_';
    result.push_back(valid ? static_cast<char>(byte) : '_');
  }
  return result;
}

std::string OfficialByteArray(const std::string& symbol, const std::vector<uint8_t>& bytes) {
  std::ostringstream os;
  os << "__device__ __align__(1) unsigned char " << symbol << '[' << bytes.size() << "] = {";
  for (size_t index = 0; index < bytes.size(); ++index) {
    if (index) os << ',';
    os << static_cast<unsigned int>(bytes[index]);
  }
  os << "};\n";
  return os.str();
}

std::map<DeclarationKey, Declaration> CollectOfficialDeclarations(
    const std::vector<KernelIketInfo>& kernels) {
  std::map<DeclarationKey, Declaration> declarations;
  for (const KernelIketInfo& kernel : kernels) {
    for (const auto& [key, declaration] : kernel.declarations) {
      auto [it, inserted] = declarations.emplace(key, declaration);
      TVM_FFI_ICHECK(inserted || it->second.event_id == declaration.event_id);
    }
  }
  return declarations;
}

std::string BuildOfficialDeviceSource(const std::vector<KernelIketInfo>& kernels) {
  std::map<DeclarationKey, Declaration> declarations = CollectOfficialDeclarations(kernels);
  std::ostringstream os;
  os << R"IKET(
extern "C" {
)IKET";

  std::vector<uint8_t> meta(kOfficialMetaInfoBytes);
  PutOfficialU32(&meta, 0, kOfficialMetaInfoBytes);
  PutOfficialU32(&meta, 4, 0);
  PutOfficialU32(&meta, 8, 5);
  PutOfficialU32(&meta, 12, 31);
  PutOfficialU32(&meta, 16, 32);
  PutOfficialU32(&meta, 20, kOfficialEventAttributesBytes);
  PutOfficialU32(&meta, 24, 0xbabef19dU);
  PutOfficialU32(&meta, 28, 0);
  PutOfficialU32(&meta, 32, 3);
  os << OfficialByteArray("__iket_meta_info", meta);

  std::unordered_map<uint32_t, std::string> range_ids;
  for (const auto& [key, declaration] : declarations) {
    uint32_t range_id = key.kind == DeclarationKind::kMark ? 0 : OfficialRangeId(key.name);
    if (range_id != 0) {
      auto [it, inserted] = range_ids.emplace(range_id, key.name);
      TVM_FFI_CHECK(inserted || it->second == key.name, ValueError)
          << "NVIDIA IKET range-id collision between " << it->second << " and " << key.name;
    }

    std::vector<uint8_t> event(kOfficialEventAttributesBytes);
    PutOfficialU32(&event, 0, kOfficialEventAttributesBytes);
    PutOfficialU32(&event, 4, declaration.event_id);
    PutOfficialU32(&event, 8, 3);
    PutOfficialU32(&event, 12, 0);
    uint32_t event_position =
        key.kind == DeclarationKind::kRange ? 4 : (key.kind == DeclarationKind::kPush ? 1 : 0);
    PutOfficialU32(&event, 16, event_position);
    PutOfficialU32(&event, 20, range_id);
    PutOfficialU32(&event, 24, key.name.size());
    std::copy(key.name.begin(), key.name.end(), event.begin() + 28);
    std::string event_symbol = "__iket_evt_decl_" + OfficialSymbolName(key.name) + "_" +
                               std::to_string(declaration.event_id) + "_attrs";
    os << OfficialByteArray(event_symbol, event);

    if (range_id != 0) {
      std::vector<uint8_t> range(kOfficialRangeAttributesBytes);
      PutOfficialU32(&range, 0, kOfficialRangeAttributesBytes);
      PutOfficialU32(&range, 4, 0);
      PutOfficialU32(&range, 8, range_id);
      PutOfficialU32(&range, 12, 0xffffffffU);
      PutOfficialU32(&range, 16, key.kind == DeclarationKind::kRange ? 1 : 2);
      PutOfficialU32(&range, 20, key.kind == DeclarationKind::kRange ? 1 : 0);
      PutOfficialU32(&range, 32, key.name.size());
      std::copy(key.name.begin(), key.name.end(), range.begin() + 36);
      std::string range_symbol = "__iket_range_decl_" + OfficialSymbolName(key.name) + "_" +
                                 std::to_string(range_id) + "_attrs";
      os << OfficialByteArray(range_symbol, range);
    }
  }
  os << R"IKET(
}

template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_impl() {
  asm volatile(
      "{\n"
      ".reg .b32 %%r, %%t;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "or.b32 %%t, %%t, %0;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "st.weak.shared.u32 [%%r], %%t;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId)
      : "memory");
}

__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id) {
  switch (event_id) {
)IKET";
  for (const auto& [key, declaration] : declarations) {
    os << "    case " << declaration.event_id << ":\n"
       << "      tvm_builtin_iket_official_event_impl<" << declaration.event_id << ">();\n"
       << "      break;\n";
  }
  os << R"IKET(    case 31:
      tvm_builtin_iket_official_event_impl<31>();
      break;
    default:
      break;
  }
  return event_id;
}
)IKET";
  return os.str();
}

using UniformBufferSet = std::unordered_set<const VarNode*>;
using DivergentVarSet = std::unordered_set<const VarNode*>;

UniformBufferSet IntersectUniformBuffers(const UniformBufferSet& lhs, const UniformBufferSet& rhs) {
  UniformBufferSet result;
  for (const VarNode* buffer : lhs) {
    if (rhs.count(buffer)) result.insert(buffer);
  }
  return result;
}

DivergentVarSet UnionDivergentVars(const DivergentVarSet& lhs, const DivergentVarSet& rhs) {
  DivergentVarSet result = lhs;
  result.insert(rhs.begin(), rhs.end());
  return result;
}

class UniformExprChecker : public ExprVisitor {
 public:
  UniformExprChecker(const DivergentVarSet& divergent_vars, const UniformBufferSet& uniform_buffers)
      : divergent_vars_(divergent_vars), uniform_buffers_(uniform_buffers) {}

  bool IsUniform(const Expr& expr) {
    uniform_ = true;
    operator()(expr);
    return uniform_;
  }

 private:
  void VisitExpr_(const VarNode* var) final {
    if (divergent_vars_.count(var)) uniform_ = false;
  }

  void VisitExpr_(const BufferLoadNode* load) final {
    if (!uniform_buffers_.count(load->buffer.get()) ||
        !IsScalarBufferAccess(load->buffer, load->indices)) {
      uniform_ = false;
      return;
    }
    for (const PrimExpr& index : load->indices) VisitExpr(index);
  }

  void VisitExpr_(const CallNode* call) final {
    // Calls may read lane-local or device state even when their explicit
    // arguments are uniform.  Treat them conservatively, except for likely(),
    // which is only an annotation around its argument.
    if (call->op.same_as(builtin::likely())) {
      ExprVisitor::VisitExpr_(call);
    } else if (IsTokenProducer(call)) {
      // Both producers return a declaration id (or sentinel zero) that is
      // independent of a potentially lane-varying payload.
      return;
    } else if (call->op.same_as(builtin::tvm_warp_shuffle()) && call->args.size() == 5) {
      // A shuffle from one warp-uniform source lane is a broadcast.  Its
      // value may depend on threadIdx, but every active lane observes the
      // selected lane's value.
      VisitExpr(call->args[0]);
      for (size_t i = 2; i < call->args.size(); ++i) VisitExpr(call->args[i]);
    } else if (call->op.same_as(Op::Get("tirx.cuda.__shfl_sync")) && call->args.size() == 4) {
      // CUDA's explicit __shfl_sync(mask, value, src_lane, width) has the
      // same broadcast semantics when mask/src_lane/width are uniform.
      // Ignore the lane-local value, but prove the control operands uniform.
      VisitExpr(call->args[0]);
      VisitExpr(call->args[2]);
      VisitExpr(call->args[3]);
    } else if (call->op.same_as(builtin::bitwise_and()) ||
               call->op.same_as(builtin::bitwise_or()) ||
               call->op.same_as(builtin::bitwise_xor()) ||
               call->op.same_as(builtin::bitwise_not())) {
      // These integer/boolean operators are pure.  A composed guard remains
      // uniform exactly when each operand is uniform.
      ExprVisitor::VisitExpr_(call);
    } else {
      uniform_ = false;
    }
  }

  const DivergentVarSet& divergent_vars_;
  const UniformBufferSet& uniform_buffers_;
  bool uniform_{true};
};

class LoopControlFinder : public StmtExprVisitor {
 public:
  bool found{false};

 private:
  void VisitStmt_(const BreakNode* op) final { found = true; }
  void VisitStmt_(const ContinueNode* op) final { found = true; }
  void VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(builtin::break_loop()) || call->op.same_as(builtin::continue_loop())) {
      found = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
};

class AnnotationFinder : public StmtExprVisitor {
 public:
  bool found{false};

 private:
  void VisitExpr_(const CallNode* call) final {
    if (IsIketOp(call->op)) {
      found = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
};

class IketConvergenceVerifier : public StmtExprVisitor {
 private:
  bool IsUniform(const Expr& expr) const {
    return UniformExprChecker(divergent_vars_, uniform_buffers_).IsUniform(expr);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    bool uniform_store = !divergent_context_ && IsScalarBufferAccess(op->buffer, op->indices) &&
                         IsUniform(op->value);
    for (const PrimExpr& index : op->indices) {
      uniform_store = uniform_store && IsUniform(index);
      VisitExpr(index);
    }
    VisitExpr(op->value);
    if (uniform_store) {
      uniform_buffers_.insert(op->buffer.get());
    } else {
      uniform_buffers_.erase(op->buffer.get());
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    const VarNode* thread_var = nullptr;
    if (op->attr_key == attr::thread_extent) {
      std::string thread_tag;
      if (auto iter_var = op->node.as<IterVar>()) {
        thread_tag = iter_var.value()->thread_tag;
        thread_var = iter_var.value()->var.get();
      } else if (auto var = op->node.as<Var>()) {
        thread_tag = var.value()->name;
        thread_var = var.value().get();
      }
      if (!thread_tag.starts_with("threadIdx.")) thread_var = nullptr;
    }
    bool inserted = thread_var && divergent_vars_.insert(thread_var).second;
    VisitExpr(op->value);
    VisitStmt(op->body);
    if (inserted) divergent_vars_.erase(thread_var);
  }

  void VisitStmt_(const BindNode* op) final {
    if (!IsUniform(op->value)) divergent_vars_.insert(op->var.get());
    VisitExpr(op->value);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    VisitExpr(op->condition);
    bool old_divergent = divergent_context_;
    bool condition_uniform = IsUniform(op->condition);
    divergent_context_ = divergent_context_ || !condition_uniform;
    auto old_vars = divergent_vars_;
    auto old_buffers = uniform_buffers_;
    VisitStmt(op->then_case);
    auto then_buffers = uniform_buffers_;
    divergent_vars_ = old_vars;
    uniform_buffers_ = old_buffers;
    if (op->else_case.has_value()) {
      VisitStmt(op->else_case.value());
    }
    uniform_buffers_ = IntersectUniformBuffers(then_buffers, uniform_buffers_);
    divergent_vars_ = std::move(old_vars);
    divergent_context_ = old_divergent;
  }

  void VisitStmt_(const ForNode* op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    bool uniform_loop = IsUniform(op->min) && IsUniform(op->extent);
    AnnotationFinder annotations;
    annotations(op->body);
    LoopControlFinder loop_control;
    loop_control(op->body);
    TVM_FFI_CHECK(!(annotations.found && loop_control.found), ValueError)
        << "IKET event sites are not allowed in a loop containing break or continue";

    bool old_divergent = divergent_context_;
    DivergentVarSet old_vars = divergent_vars_;
    UniformBufferSet old_buffers = uniform_buffers_;
    DivergentVarSet loop_entry_vars = old_vars;
    if (!uniform_loop) loop_entry_vars.insert(op->loop_var.get());

    DivergentVarSet loop_head_vars = loop_entry_vars;
    UniformBufferSet loop_head_buffers = old_buffers;
    bool converged = false;
    for (size_t iteration = 0; iteration < kMaxConvergenceAnalysisIterations; ++iteration) {
      divergent_vars_ = loop_head_vars;
      uniform_buffers_ = loop_head_buffers;
      divergent_context_ = old_divergent || !uniform_loop;
      VisitStmt(op->body);

      DivergentVarSet next_vars = UnionDivergentVars(loop_entry_vars, divergent_vars_);
      UniformBufferSet next_buffers = IntersectUniformBuffers(old_buffers, uniform_buffers_);
      if (next_vars == loop_head_vars && next_buffers == loop_head_buffers) {
        converged = true;
        break;
      }
      loop_head_vars = std::move(next_vars);
      loop_head_buffers = std::move(next_buffers);
    }
    TVM_FFI_CHECK(converged, ValueError)
        << "IKET convergence analysis did not reach a loop fixed point";
    uniform_buffers_ = std::move(loop_head_buffers);
    divergent_vars_ = std::move(old_vars);
    divergent_context_ = old_divergent;
  }

  void VisitStmt_(const WhileNode* op) final {
    AnnotationFinder annotations;
    annotations(op->body);
    LoopControlFinder loop_control;
    loop_control(op->body);
    TVM_FFI_CHECK(!(annotations.found && loop_control.found), ValueError)
        << "IKET event sites are not allowed in a loop containing break or continue";
    bool old_divergent = divergent_context_;
    DivergentVarSet old_vars = divergent_vars_;
    UniformBufferSet old_buffers = uniform_buffers_;
    DivergentVarSet loop_head_vars = old_vars;
    UniformBufferSet loop_head_buffers = old_buffers;
    bool converged = false;
    for (size_t iteration = 0; iteration < kMaxConvergenceAnalysisIterations; ++iteration) {
      divergent_vars_ = loop_head_vars;
      uniform_buffers_ = loop_head_buffers;
      VisitExpr(op->condition);
      divergent_context_ = old_divergent || !IsUniform(op->condition);
      VisitStmt(op->body);

      DivergentVarSet next_vars = UnionDivergentVars(old_vars, divergent_vars_);
      UniformBufferSet next_buffers = IntersectUniformBuffers(old_buffers, uniform_buffers_);
      if (next_vars == loop_head_vars && next_buffers == loop_head_buffers) {
        converged = true;
        break;
      }
      loop_head_vars = std::move(next_vars);
      loop_head_buffers = std::move(next_buffers);
    }
    TVM_FFI_CHECK(converged, ValueError)
        << "IKET convergence analysis did not reach a loop fixed point";
    uniform_buffers_ = std::move(loop_head_buffers);
    divergent_vars_ = std::move(old_vars);
    divergent_context_ = old_divergent;
  }

  void VisitExpr_(const CallNode* call) final {
    if (IsIketOp(call->op)) {
      if (divergent_context_) {
        TVM_FFI_THROW(ValueError) << "IKET event site may be reached by a divergent set of lanes: "
                                  << GetRef<Expr>(call);
      }
      if (call->op.same_as(IketRangeEndOp())) {
        TVM_FFI_CHECK(IsUniform(call->args[0]), ValueError)
            << "IKET range_end requires a warp-uniform RangeToken";
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  DivergentVarSet divergent_vars_;
  UniformBufferSet uniform_buffers_;
  bool divergent_context_{false};
};

class InstrumentOfficialKernel : public StmtExprMutator {
 public:
  InstrumentOfficialKernel(const KernelIketInfo& info, std::string device_source)
      : info_(info), device_source_(std::move(device_source)) {}

  PrimFunc Run() {
    PrimFunc result = info_.function;
    result.CopyOnWrite()->body = operator()(info_.function->body);
    return result;
  }

 private:
  const Declaration& Lookup(DeclarationKind kind, const CallNode* call) const {
    DeclarationKey key{kind, GetName(call)};
    auto it = info_.declarations.find(key);
    TVM_FFI_ICHECK(it != info_.declarations.end()) << "Missing official IKET declaration";
    return it->second;
  }

  PrimExpr Event(PrimExpr event_id) const {
    static const Op& event_op = Op::Get("tirx.cuda.iket_official_event");
    return Call(PrimType::UInt(32), event_op,
                {cast(PrimType::UInt(32), event_id), StringImm(device_source_)});
  }

  Stmt VisitStmt_(const EvaluateNode* evaluate) final {
    if (const auto* call = evaluate->value.as<CallNode>();
        call && call->op.same_as(IketRangeEndOp())) {
      PrimExpr token = VisitExpr(call->args[0]).as_or_throw<PrimExpr>();
      return Evaluate(Event(token));
    }
    return StmtExprMutator::VisitStmt_(evaluate);
  }

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(IketRangeStartOp())) {
      return Event(IntImm(PrimType::UInt(32), Lookup(DeclarationKind::kRange, call).event_id));
    }
    if (call->op.same_as(IketSentinelOp())) return IntImm(PrimType::UInt(32), 0);
    if (call->op.same_as(IketMarkOp())) {
      return Event(IntImm(PrimType::UInt(32), Lookup(DeclarationKind::kMark, call).event_id));
    }
    if (call->op.same_as(IketRangePushOp())) {
      return Event(IntImm(PrimType::UInt(32), Lookup(DeclarationKind::kPush, call).event_id));
    }
    if (call->op.same_as(IketRangePopOp())) {
      return Event(IntImm(PrimType::UInt(32), 31));
    }
    if (call->op.same_as(IketRangeEndOp())) {
      TVM_FFI_THROW(ValueError) << "range_end must be emitted in statement position";
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  const KernelIketInfo& info_;
  std::string device_source_;
};

bool IketEnabled(const IRModule& module) { return module->HasNonzeroAttr("tirx.iket.enabled"); }

IRModule LowerIketImpl(IRModule module) {
  if (!IketEnabled(module)) {
    for (const auto& [global_var, base_function] : module->functions) {
      const auto* prim_func = base_function.as<PrimFuncNode>();
      if (!prim_func) continue;
      PrimFunc function = ffi::GetRef<PrimFunc>(prim_func);
      AnnotationCollector collector;
      collector(function->body);
      TokenBufferSet tokens = CollectTokenBuffers(function->body);
      if (!collector.has_annotations && tokens.empty()) continue;
      if (collector.has_annotations) {
        TokenVerifier verifier(tokens);
        verifier(function->body);
        TokenDeclarationMap token_declarations = CollectTokenDeclarations(function->body);
        RangeEndSchemaVerifier schema_verifier(token_declarations, &collector.declarations);
        schema_verifier(function->body);
      }
      StripIket strip(std::move(tokens));
      Stmt body = RemoveStrippedIketNoOps()(strip(function->body));
      if (!body.same_as(function->body)) {
        function.CopyOnWrite()->body = body;
        module->Update(global_var, function);
      }
    }
    return module;
  }

  std::vector<KernelIketInfo> kernels;
  for (const auto& [global_var, base_function] : module->functions) {
    const auto* prim_func = base_function.as<PrimFuncNode>();
    if (!prim_func) continue;
    PrimFunc function = ffi::GetRef<PrimFunc>(prim_func);
    AnnotationCollector collector;
    collector(function->body);
    if (!collector.has_annotations) continue;

    std::string function_name = FunctionName(global_var, function);
    TVM_FFI_CHECK(IsCudaDeviceFunction(function), ValueError)
        << "IKET annotations are only valid in a split CUDA device kernel";
    TVM_FFI_CHECK_LE(collector.declarations.size(), kMaxDeclarations, ValueError)
        << "NVIDIA IKET supports at most " << kMaxDeclarations << " declarations per kernel";

    TokenBufferSet tokens = CollectTokenBuffers(function->body);
    TokenVerifier verifier(tokens);
    verifier(function->body);
    TokenDeclarationMap token_declarations = CollectTokenDeclarations(function->body);
    RangeEndSchemaVerifier schema_verifier(token_declarations, &collector.declarations);
    schema_verifier(function->body);
    IketConvergenceVerifier convergence_verifier;
    convergence_verifier(function->body);

    TVM_FFI_CHECK(IsSm90OrNewer(function), ValueError)
        << "NVIDIA IKET requires SM90 or newer for kernel " << function_name;
    TVM_FFI_CHECK(!HasAnyPayload(collector.declarations), ValueError)
        << "NVIDIA IKET does not support payloads in kernel " << function_name;
    TVM_FFI_CHECK(OfficialTokenAnalyzer(tokens).Prove(function->body), ValueError)
        << "NVIDIA IKET requires token ranges to be provably strictly alternating "
           "and closed on every exit in kernel "
        << function_name;
    TVM_FFI_CHECK(OfficialStackAnalyzer().Prove(function->body), ValueError)
        << "NVIDIA IKET requires balanced range_push/range_pop paths in kernel " << function_name;

    kernels.push_back(KernelIketInfo{global_var, function, std::move(function_name),
                                     std::move(collector.declarations)});
  }

  std::sort(
      kernels.begin(), kernels.end(),
      [](const KernelIketInfo& lhs, const KernelIketInfo& rhs) { return lhs.name < rhs.name; });
  for (size_t i = 1; i < kernels.size(); ++i) {
    TVM_FFI_CHECK_NE(kernels[i - 1].name, kernels[i].name, ValueError)
        << "IKET device kernels must have unique global symbols: " << kernels[i].name;
  }

  std::map<DeclarationKey, uint32_t> event_ids;
  std::unordered_map<std::string, DeclarationKind> event_kinds;
  for (const KernelIketInfo& kernel : kernels) {
    for (const auto& [key, declaration] : kernel.declarations) {
      auto [kind_it, kind_inserted] = event_kinds.emplace(key.name, key.kind);
      TVM_FFI_CHECK(kind_inserted || kind_it->second == key.kind, ValueError)
          << "NVIDIA IKET declaration " << key.name << " changes event kind across kernels";
      event_ids.emplace(key, 0);
    }
  }
  TVM_FFI_CHECK_LE(event_ids.size(), kMaxDeclarations, ValueError)
      << "NVIDIA IKET supports at most " << kMaxDeclarations
      << " distinct declarations in one CUDA module";

  uint32_t event_id = 1;
  for (auto& [key, id] : event_ids) id = event_id++;
  for (KernelIketInfo& kernel : kernels) {
    for (auto& [key, declaration] : kernel.declarations) {
      declaration.event_id = event_ids.at(key);
    }
  }

  std::string device_source = BuildOfficialDeviceSource(kernels);
  for (const KernelIketInfo& kernel : kernels) {
    module->Update(kernel.global_var, InstrumentOfficialKernel(kernel, device_source).Run());
  }
  return module;
}

}  // namespace

Pass LowerIket() {
  auto pass_func = [](IRModule module, tvm::transform::PassContext) {
    return LowerIketImpl(std::move(module));
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.backend.cuda.LowerIket", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.backend.cuda.transforms.LowerIket", LowerIket);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
