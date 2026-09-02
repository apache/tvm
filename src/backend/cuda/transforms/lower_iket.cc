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
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

constexpr uint32_t kNativeMaxDeclarations = 30;
constexpr uint32_t kExtendedMaxDeclarations = 4032;
constexpr uint32_t kNativeFirstEventId = 1;
constexpr uint32_t kExtendedFirstEventId = 64;
constexpr uint32_t kRangePopEventId = 31;
constexpr uint32_t kNativeMaxEventId = 31;
constexpr uint32_t kExtendedMaxEventId = 4095;
constexpr int kOfficialMetaInfoBytes = 48;
constexpr int kOfficialEventAttributesBytes = 60;
constexpr int kOfficialRangeAttributesBytes = 72;

enum class DeclarationKind : int {
  kRange = 1,
  kPush = 2,
  kMark = 3,
};

enum class InstrumentMode : uint32_t {
  kNativeDump = 3,
  kExtendedNativeDump = 5,
};

enum class PayloadType : uint32_t {
  kNone = 0,
  kI8 = 1,
  kUI8 = 2,
  kI16 = 3,
  kUI16 = 4,
  kI32 = 5,
  kUI32 = 6,
  kI64 = 7,
  kFP32 = 13,
  kFP64 = 14,
  kUI64 = 16,
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
  const auto* name = call->args[index].as<prim::StringImmNode>();
  TVM_FFI_CHECK(name != nullptr, TypeError)
      << call->op.as<Op>().value()->name << " requires a string-literal event name";
  std::string result = name->value;
  TVM_FFI_CHECK(!result.empty(), ValueError) << "IKET event names must not be empty";
  TVM_FFI_CHECK(result.size() <= 32, ValueError)
      << "IKET event name exceeds 32 UTF-8 bytes: " << result;
  TVM_FFI_CHECK(IsValidUTF8(result), ValueError) << "IKET event name is not valid UTF-8";
  return result;
}

PayloadType ValidatePayload(const Expr& payload) {
  ffi::Optional<PrimExpr> value_optional = payload.as<PrimExpr>();
  TVM_FFI_CHECK(value_optional.has_value(), TypeError)
      << "IKET payload must be a scalar numeric value, but got " << payload;
  PrimExpr value = value_optional.value();
  ffi::Optional<PrimType> type = value->ty.as<PrimType>();
  TVM_FFI_CHECK(type.has_value(), TypeError)
      << "IKET payload must be a scalar numeric value, but got " << value->ty;
  DLDataType dtype = type.value()->dtype;
  TVM_FFI_CHECK_EQ(dtype.lanes, 1, TypeError) << "IKET payload must be a scalar value";
  PayloadType payload_type;
  if (dtype.code == kDLBool) {
    payload_type = PayloadType::kUI8;
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    payload_type = PayloadType::kI8;
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    payload_type = PayloadType::kUI8;
  } else if (dtype.code == kDLInt && dtype.bits == 16) {
    payload_type = PayloadType::kI16;
  } else if (dtype.code == kDLUInt && dtype.bits == 16) {
    payload_type = PayloadType::kUI16;
  } else if (dtype.code == kDLInt && dtype.bits == 32) {
    payload_type = PayloadType::kI32;
  } else if (dtype.code == kDLUInt && dtype.bits == 32) {
    payload_type = PayloadType::kUI32;
  } else if (dtype.code == kDLInt && dtype.bits == 64) {
    payload_type = PayloadType::kI64;
  } else if (dtype.code == kDLUInt && dtype.bits == 64) {
    payload_type = PayloadType::kUI64;
  } else if (dtype.code == kDLFloat && dtype.bits == 32) {
    payload_type = PayloadType::kFP32;
  } else if (dtype.code == kDLFloat && dtype.bits == 64) {
    payload_type = PayloadType::kFP64;
  } else {
    TVM_FFI_THROW(TypeError)
        << "IKET payload supports only bool, int/uint 8/16/32/64, float32, and float64; got "
        << ffi::DLDataTypeToString(dtype);
  }
  CallEffectKind effect = SideEffect(value);
  TVM_FFI_CHECK(effect <= CallEffectKind::kReadState, ValueError)
      << "IKET payload expressions may read state but must not update state; got " << effect;
  return payload_type;
}

const char* PayloadTypeName(PayloadType type) {
  switch (type) {
    case PayloadType::kNone:
      return "NoPayload";
    case PayloadType::kI8:
      return "I8";
    case PayloadType::kUI8:
      return "UI8";
    case PayloadType::kI16:
      return "I16";
    case PayloadType::kUI16:
      return "UI16";
    case PayloadType::kI32:
      return "I32";
    case PayloadType::kUI32:
      return "UI32";
    case PayloadType::kI64:
      return "I64";
    case PayloadType::kFP32:
      return "FP32";
    case PayloadType::kFP64:
      return "FP64";
    case PayloadType::kUI64:
      return "UI64";
  }
  return "Unknown";
}

bool Is64BitPayload(PayloadType type) {
  return type == PayloadType::kI64 || type == PayloadType::kUI64 || type == PayloadType::kFP64;
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
  PayloadType payload_type{PayloadType::kNone};
  bool end_payload_schema_set{false};
  bool end_has_payload{false};
  PayloadType end_payload_type{PayloadType::kNone};
  uint32_t event_id{0};
};

class AnnotationCollector : public StmtExprVisitor {
 public:
  std::map<DeclarationKey, Declaration> declarations;
  bool has_annotations{false};
  bool has_payload_calls{false};

 private:
  void AddDeclaration(DeclarationKind kind, const CallNode* call) {
    std::string name = GetName(call);
    TVM_FFI_CHECK(call->args.size() == 1 || call->args.size() == 2, TypeError)
        << call->op.as<Op>().value()->name << " expects a name and optional payload";
    bool has_payload = call->args.size() == 2;
    has_payload_calls = has_payload_calls || has_payload;
    PayloadType payload_type = has_payload ? ValidatePayload(call->args[1]) : PayloadType::kNone;
    auto [name_it, name_inserted] = declaration_kinds_.emplace(name, kind);
    TVM_FFI_CHECK(name_inserted || name_it->second == kind, ValueError)
        << "IKET declaration " << name << " changes event kind from "
        << DeclarationKindName(name_it->second) << " to " << DeclarationKindName(kind);
    DeclarationKey key{kind, std::move(name)};
    auto [it, inserted] = declarations.emplace(key, Declaration{key});
    Declaration& declaration = it->second;
    if (declaration.payload_schema_set) {
      TVM_FFI_CHECK_EQ(declaration.has_payload, has_payload, ValueError)
          << "IKET declaration " << declaration.key.name
          << " changes payload presence between sites";
      TVM_FFI_CHECK(declaration.payload_type == payload_type, TypeError)
          << "IKET declaration " << declaration.key.name << " changes payload type from "
          << PayloadTypeName(declaration.payload_type) << " to " << PayloadTypeName(payload_type)
          << " between sites";
    } else {
      declaration.payload_schema_set = true;
      declaration.has_payload = has_payload;
      declaration.payload_type = payload_type;
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
      // A sentinel carries only token-flow identity.  It emits no runtime
      // event and therefore must not create metadata or consume an event ID.
      GetName(call);
    } else if (call->op.same_as(IketRangePushOp())) {
      AddDeclaration(DeclarationKind::kPush, call);
    } else if (call->op.same_as(IketRangeEndOp())) {
      TVM_FFI_CHECK(call->args.size() == 1 || call->args.size() == 2, TypeError)
          << "IKET range_end expects a token and optional payload";
      DLDataType token_dtype = call->args[0].as_or_throw<PrimExpr>().ty()->dtype;
      TVM_FFI_CHECK(token_dtype.code == kDLUInt && token_dtype.bits == 32 && token_dtype.lanes == 1,
                    TypeError)
          << "IKET RangeToken must have dtype uint32";
      if (call->args.size() == 2) {
        has_payload_calls = true;
        ValidatePayload(call->args[1]);
      }
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
    } else if (const auto* load = store->value.as<TensorLoadNode>()) {
      is_token_value = buffers_->count(load->source.as_or_throw<tvm::tirx::BufferVar>().get());
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
    } else if (const auto* load = store->value.as<TensorLoadNode>()) {
      auto it = declarations_->find(load->source.as_or_throw<tvm::tirx::BufferVar>().get());
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
    const auto* token = call->args[0].as<TensorLoadNode>();
    TVM_FFI_ICHECK(token != nullptr);
    auto possible_it = token_declarations_.find(token->source.as_or_throw<BufferVar>().get());
    TVM_FFI_ICHECK(possible_it != token_declarations_.end());
    bool has_payload = call->args.size() == 2;
    PayloadType payload_type = has_payload ? ValidatePayload(call->args[1]) : PayloadType::kNone;
    for (const DeclarationKey& key : possible_it->second) {
      auto declaration_it = declarations_->find(key);
      // A sentinel-only identity has no real declaration and no payload
      // schema.  If the same token can also carry a real start, that real
      // declaration is still checked below.
      if (declaration_it == declarations_->end()) continue;
      Declaration& declaration = declaration_it->second;
      if (declaration.end_payload_schema_set) {
        TVM_FFI_CHECK_EQ(declaration.end_has_payload, has_payload, ValueError)
            << "range_end for " << key.name << " changes payload presence between sites";
        TVM_FFI_CHECK(declaration.end_payload_type == payload_type, TypeError)
            << "range_end for " << key.name << " changes payload type from "
            << PayloadTypeName(declaration.end_payload_type) << " to "
            << PayloadTypeName(payload_type) << " between sites";
      } else {
        declaration.end_payload_schema_set = true;
        declaration.end_has_payload = has_payload;
        declaration.end_payload_type = payload_type;
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  const TokenDeclarationMap& token_declarations_;
  std::map<DeclarationKey, Declaration>* declarations_;
};

void ValidateRangeSchemas(const std::map<DeclarationKey, Declaration>& declarations) {
  for (const auto& [key, declaration] : declarations) {
    if (key.kind != DeclarationKind::kRange || !declaration.end_payload_schema_set) continue;
    TVM_FFI_CHECK_EQ(declaration.has_payload, declaration.end_has_payload, ValueError)
        << "IKET token range " << key.name
        << " must use payloads at both range_start and range_end, or at neither endpoint";
    TVM_FFI_CHECK(declaration.payload_type == declaration.end_payload_type, TypeError)
        << "IKET token range " << key.name << " changes payload type from "
        << PayloadTypeName(declaration.payload_type) << " at range_start to "
        << PayloadTypeName(declaration.end_payload_type) << " at range_end";
  }
}

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
    } else if (const auto* load = store->value.as<TensorLoadNode>()) {
      valid_value = token_buffers_.count(load->source.as_or_throw<tvm::tirx::BufferVar>().get());
      allow_token_load_ = valid_value;
      VisitExpr(store->value);
      allow_token_load_ = false;
    }
    TVM_FFI_CHECK(valid_value, ValueError)
        << "RangeToken may only be assigned another token, range_start, or sentinel_token";
    for (const PrimExpr& index : store->indices) VisitExpr(index);
  }

  void VisitExpr_(const TensorLoadNode* load) final {
    if (token_buffers_.count(load->source.as_or_throw<tvm::tirx::BufferVar>().get())) {
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
      const auto* token = call->args[0].as<TensorLoadNode>();
      TVM_FFI_CHECK(
          token != nullptr && token_buffers_.count(token->source.as_or_throw<BufferVar>().get()),
          ValueError)
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
  bool has_payload_calls{false};
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

std::string BuildOfficialDeviceSource(const std::vector<KernelIketInfo>& kernels,
                                      InstrumentMode mode) {
  std::map<DeclarationKey, Declaration> declarations = CollectOfficialDeclarations(kernels);
  uint32_t max_event_id =
      mode == InstrumentMode::kNativeDump ? kNativeMaxEventId : kExtendedMaxEventId;
  std::ostringstream os;
  os << R"IKET(
extern "C" {
)IKET";

  std::vector<uint8_t> meta(kOfficialMetaInfoBytes);
  PutOfficialU32(&meta, 0, kOfficialMetaInfoBytes);
  PutOfficialU32(&meta, 4, 0);
  PutOfficialU32(&meta, 8, 5);
  PutOfficialU32(&meta, 12, max_event_id);
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
    PutOfficialU32(&event, 8, static_cast<uint32_t>(mode));
    PutOfficialU32(&event, 12, static_cast<uint32_t>(declaration.payload_type));
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
  bool has_payload_calls =
      std::any_of(kernels.begin(), kernels.end(),
                  [](const KernelIketInfo& kernel) { return kernel.has_payload_calls; });
  if (mode == InstrumentMode::kNativeDump && !has_payload_calls) {
    // Keep this source byte-for-byte compatible with the original NativeDump
    // no-payload helper.  Existing MegaKernel placeholders and SASS must not
    // move merely because payload and ExtendedNativeDump support exists.
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

  os << R"IKET(
}
)IKET";

  if (mode == InstrumentMode::kNativeDump) {
    os << R"IKET(
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

template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_payload32_impl(
    unsigned int payload) {
  asm volatile(
      "{\n"
      ".reg .pred %%p;\n"
      ".reg .b32 %%r, %%t, %%mask, %%payload32;\n"
      "activemask.b32 %%mask;\n"
      "elect.sync _|%%p, %%mask;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "or.b32 %%t, %%t, %0;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "mov.b32 %%payload32, %1;\n"
      "@%%p st.weak.shared.u32 [%%r], %%t;\n"
      "@%%p st.weak.shared.b32 [%%r+4], %%payload32;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId), "r"(payload)
      : "memory");
}

template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_payload64_impl(
    unsigned long long payload) {
  asm volatile(
      "{\n"
      ".reg .pred %%p;\n"
      ".reg .b32 %%r, %%t, %%mask;\n"
      ".reg .b64 %%payload64;\n"
      "activemask.b32 %%mask;\n"
      "elect.sync _|%%p, %%mask;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "or.b32 %%t, %%t, %0;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "mov.b64 %%payload64, %1;\n"
      "@%%p st.weak.shared.u32 [%%r], %%t;\n"
      "@%%p st.weak.shared.b64 [%%r+8], %%payload64;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId), "l"(payload)
      : "memory");
}
)IKET";
  } else {
    os << R"IKET(
template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_impl() {
  asm volatile(
      "{\n"
      ".reg .b32 %%r, %%t, %%evtid;\n"
      ".reg .b64 %%ts_evtid;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "mov.b32 %%evtid, %0;\n"
      "mov.b64 %%ts_evtid, {%%t, %%evtid};\n"
      "st.weak.shared.u64 [%%r], %%ts_evtid;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId)
      : "memory");
}

template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_payload32_impl(
    unsigned int payload) {
  asm volatile(
      "{\n"
      ".reg .pred %%p;\n"
      ".reg .b32 %%r, %%t, %%mask, %%evtid, %%payload32;\n"
      ".reg .b64 %%ts_evtid;\n"
      "activemask.b32 %%mask;\n"
      "elect.sync _|%%p, %%mask;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "mov.b32 %%evtid, %0;\n"
      "mov.b32 %%payload32, %1;\n"
      "mov.b64 %%ts_evtid, {%%t, %%evtid};\n"
      "@%%p st.weak.shared.u64 [%%r], %%ts_evtid;\n"
      "@%%p st.weak.shared.b32 [%%r+8], %%payload32;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId), "r"(payload)
      : "memory");
}

template <unsigned int EventId>
__forceinline__ __device__ void tvm_builtin_iket_official_event_payload64_impl(
    unsigned long long payload) {
  asm volatile(
      "{\n"
      ".reg .pred %%p;\n"
      ".reg .b32 %%r, %%t, %%mask, %%evtid;\n"
      ".reg .b64 %%ts_evtid, %%payload64;\n"
      "activemask.b32 %%mask;\n"
      "elect.sync _|%%p, %%mask;\n"
      "mov.b32 %%r, %%cluster_ctarank;\n"
      "mov.u32 %%t, %%globaltimer_lo;\n"
      "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n"
      "mov.b32 %%evtid, %0;\n"
      "mov.b64 %%payload64, %1;\n"
      "mov.b64 %%ts_evtid, {%%t, %%evtid};\n"
      "@%%p st.weak.shared.u64 [%%r], %%ts_evtid;\n"
      "@%%p st.weak.shared.b64 [%%r+8], %%payload64;\n"
      "pmevent.mask %0;\n"
      "}\n"
      :
      : "n"(EventId), "l"(payload)
      : "memory");
}
)IKET";
  }

  os << R"IKET(
__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id) {
  switch (event_id) {
)IKET";
  for (const auto& [key, declaration] : declarations) {
    if (declaration.has_payload) continue;
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

__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id, unsigned int payload) {
  switch (event_id) {
)IKET";
  for (const auto& [key, declaration] : declarations) {
    if (!declaration.has_payload || Is64BitPayload(declaration.payload_type)) continue;
    os << "    case " << declaration.event_id << ":\n"
       << "      tvm_builtin_iket_official_event_payload32_impl<" << declaration.event_id
       << ">(payload);\n"
       << "      break;\n";
  }
  os << R"IKET(    default:
      break;
  }
  return event_id;
}

__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id, unsigned long long payload) {
  switch (event_id) {
)IKET";
  for (const auto& [key, declaration] : declarations) {
    if (!declaration.has_payload || !Is64BitPayload(declaration.payload_type)) continue;
    os << "    case " << declaration.event_id << ":\n"
       << "      tvm_builtin_iket_official_event_payload64_impl<" << declaration.event_id
       << ">(payload);\n"
       << "      break;\n";
  }
  os << R"IKET(    default:
      break;
  }
  return event_id;
}

template <typename Payload>
__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id, Payload payload) {
  if constexpr (sizeof(Payload) <= 4) {
    unsigned int payload_bits = static_cast<unsigned int>(payload);
    if constexpr (sizeof(Payload) < 4) {
      payload_bits &= (1U << (sizeof(Payload) * 8)) - 1U;
    }
    return tvm_builtin_iket_official_event(event_id, payload_bits);
  } else {
    return tvm_builtin_iket_official_event(
        event_id, static_cast<unsigned long long>(payload));
  }
}

__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id, float payload) {
  return tvm_builtin_iket_official_event(event_id, __float_as_uint(payload));
}

__forceinline__ __device__ unsigned int tvm_builtin_iket_official_event(
    unsigned int event_id, double payload) {
  return tvm_builtin_iket_official_event(
      event_id, static_cast<unsigned long long>(__double_as_longlong(payload)));
}
)IKET";
  return os.str();
}

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
                {cast(PrimType::UInt(32), event_id), prim::StringImm(device_source_)});
  }

  PrimExpr Event(PrimExpr event_id, PrimExpr payload) const {
    static const Op& event_op = Op::Get("tirx.cuda.iket_official_event");
    return Call(
        PrimType::UInt(32), event_op,
        {cast(PrimType::UInt(32), event_id), prim::StringImm(device_source_), std::move(payload)});
  }

  PrimExpr NormalizePayload(PrimExpr payload, PayloadType type) const {
    TVM_FFI_ICHECK(type != PayloadType::kNone);
    return payload;
  }

  Stmt VisitStmt_(const EvaluateNode* evaluate) final {
    if (const auto* call = evaluate->value.as<CallNode>();
        call && call->op.same_as(IketRangeEndOp())) {
      PrimExpr token = VisitExpr(call->args[0]).as_or_throw<PrimExpr>();
      if (call->args.size() == 2) {
        PayloadType payload_type = ValidatePayload(call->args[1]);
        PrimExpr payload =
            NormalizePayload(VisitExpr(call->args[1]).as_or_throw<PrimExpr>(), payload_type);
        return IfThenElse(token != 0, Evaluate(Event(token, std::move(payload))));
      }
      return Evaluate(Event(token));
    }
    return StmtExprMutator::VisitStmt_(evaluate);
  }

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(IketRangeStartOp())) {
      const Declaration& declaration = Lookup(DeclarationKind::kRange, call);
      PrimExpr event_id = IntImm(PrimType::UInt(32), declaration.event_id);
      if (declaration.has_payload) {
        return Event(event_id, NormalizePayload(VisitExpr(call->args[1]).as_or_throw<PrimExpr>(),
                                                declaration.payload_type));
      }
      return Event(event_id);
    }
    if (call->op.same_as(IketSentinelOp())) return IntImm(PrimType::UInt(32), 0);
    if (call->op.same_as(IketMarkOp())) {
      const Declaration& declaration = Lookup(DeclarationKind::kMark, call);
      PrimExpr event_id = IntImm(PrimType::UInt(32), declaration.event_id);
      if (declaration.has_payload) {
        return Event(event_id, NormalizePayload(VisitExpr(call->args[1]).as_or_throw<PrimExpr>(),
                                                declaration.payload_type));
      }
      return Event(event_id);
    }
    if (call->op.same_as(IketRangePushOp())) {
      const Declaration& declaration = Lookup(DeclarationKind::kPush, call);
      PrimExpr event_id = IntImm(PrimType::UInt(32), declaration.event_id);
      if (declaration.has_payload) {
        return Event(event_id, NormalizePayload(VisitExpr(call->args[1]).as_or_throw<PrimExpr>(),
                                                declaration.payload_type));
      }
      return Event(event_id);
    }
    if (call->op.same_as(IketRangePopOp())) {
      return Event(IntImm(PrimType::UInt(32), kRangePopEventId));
    }
    if (call->op.same_as(IketRangeEndOp())) {
      TVM_FFI_THROW(ValueError) << "range_end must be emitted in statement position";
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  const KernelIketInfo& info_;
  std::string device_source_;
};

bool IketEnabled(const IRModule& module) {
  if (module->HasNonzeroAttr("tirx.iket.enabled")) return true;
  const char* child_enable = std::getenv("TVM_IKET_INJECTED_CHILD_ENABLE");
  const char* profile = std::getenv("TVM_IKET_OFFICIAL_PROFILE");
  const char* injection = std::getenv("CUDA_INJECTION64_PATH");
  const char* injection_config = std::getenv("SMODEL_INJECTION_CONFIG");
  return child_enable && std::string(child_enable) == "1" && profile &&
         std::string(profile) == "cutlass-4.6.0" && injection && injection[0] != '\0' &&
         injection_config && injection_config[0] != '\0';
}

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
        ValidateRangeSchemas(collector.declarations);
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

    TokenBufferSet tokens = CollectTokenBuffers(function->body);
    TokenVerifier verifier(tokens);
    verifier(function->body);
    TokenDeclarationMap token_declarations = CollectTokenDeclarations(function->body);
    RangeEndSchemaVerifier schema_verifier(token_declarations, &collector.declarations);
    schema_verifier(function->body);
    ValidateRangeSchemas(collector.declarations);
    TVM_FFI_CHECK(IsSm90OrNewer(function), ValueError)
        << "NVIDIA IKET requires SM90 or newer for kernel " << function_name;

    kernels.push_back(KernelIketInfo{global_var, function, std::move(function_name),
                                     std::move(collector.declarations),
                                     collector.has_payload_calls});
  }

  std::sort(
      kernels.begin(), kernels.end(),
      [](const KernelIketInfo& lhs, const KernelIketInfo& rhs) { return lhs.name < rhs.name; });
  for (size_t i = 1; i < kernels.size(); ++i) {
    TVM_FFI_CHECK_NE(kernels[i - 1].name, kernels[i].name, ValueError)
        << "IKET device kernels must have unique global symbols: " << kernels[i].name;
  }

  std::map<DeclarationKey, Declaration> module_declarations;
  std::unordered_map<std::string, DeclarationKind> event_kinds;
  for (const KernelIketInfo& kernel : kernels) {
    for (const auto& [key, declaration] : kernel.declarations) {
      auto [kind_it, kind_inserted] = event_kinds.emplace(key.name, key.kind);
      TVM_FFI_CHECK(kind_inserted || kind_it->second == key.kind, ValueError)
          << "NVIDIA IKET declaration " << key.name << " changes event kind across kernels";
      auto [declaration_it, declaration_inserted] = module_declarations.emplace(key, declaration);
      if (!declaration_inserted) {
        const Declaration& previous = declaration_it->second;
        TVM_FFI_CHECK_EQ(previous.has_payload, declaration.has_payload, ValueError)
            << "NVIDIA IKET declaration " << key.name << " changes payload presence across kernels";
        TVM_FFI_CHECK(previous.payload_type == declaration.payload_type, TypeError)
            << "NVIDIA IKET declaration " << key.name << " changes payload type from "
            << PayloadTypeName(previous.payload_type) << " to "
            << PayloadTypeName(declaration.payload_type) << " across kernels";
      }
    }
  }
  TVM_FFI_CHECK_LE(module_declarations.size(), kExtendedMaxDeclarations, ValueError)
      << "NVIDIA IKET supports at most " << kExtendedMaxDeclarations
      << " distinct user declarations in one CUDA module; got " << module_declarations.size();

  InstrumentMode mode = module_declarations.size() <= kNativeMaxDeclarations
                            ? InstrumentMode::kNativeDump
                            : InstrumentMode::kExtendedNativeDump;
  if (mode == InstrumentMode::kExtendedNativeDump) {
    LOG(WARNING) << "NVIDIA IKET is using ExtendedNativeDump for " << module_declarations.size()
                 << " declarations; records are wider and instrumentation overhead increases";
  }

  uint32_t event_id =
      mode == InstrumentMode::kNativeDump ? kNativeFirstEventId : kExtendedFirstEventId;
  std::map<DeclarationKey, uint32_t> event_ids;
  for (const auto& [key, declaration] : module_declarations) {
    event_ids.emplace(key, event_id++);
  }
  for (KernelIketInfo& kernel : kernels) {
    for (auto& [key, declaration] : kernel.declarations) {
      declaration.event_id = event_ids.at(key);
    }
  }

  std::string device_source = BuildOfficialDeviceSource(kernels, mode);
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
