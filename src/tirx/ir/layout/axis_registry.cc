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

/*
 * Axis definitions, attributes, fusers/splitters, and registrations.
 */
#include "utils.h"

namespace tvm {
namespace tirx {

/**************** Axis ****************/
// AxisNode
ffi::ObjectPtr<ffi::Object> CreateAxis(const std::string& name) {
  // Hack use ffi::Any as exchange
  auto axis = Axis::Get(name);
  TVM_FFI_ICHECK(axis.defined()) << "Cannot find axis '" << name << '\'';
  return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<ffi::Object>(axis);
}

bool AxisNode::IsThreadAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return thread_attr_map[ffi::GetRef<Axis>(this)];
}

bool AxisNode::IsMemoryAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return !thread_attr_map[ffi::GetRef<Axis>(this)];
}

ffi::Optional<ExecScope> AxisNode::GetScope() const {
  static const auto& scope_attr_map = Axis::GetAttrMap<ffi::Optional<ExecScope>>("scope");
  return scope_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<ExecScope> AxisNode::GetSubscope() const {
  static const auto& subscope_attr_map = Axis::GetAttrMap<ffi::Optional<ExecScope>>("subscope");
  return subscope_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<FAxisFuser> AxisNode::GetFuser() const {
  static const auto& fuser_attr_map = Axis::GetAttrMap<ffi::Optional<FAxisFuser>>("fuser");
  return fuser_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<FAxisSplitter> AxisNode::GetSplitter() const {
  static const auto& splitter_attr_map = Axis::GetAttrMap<ffi::Optional<FAxisSplitter>>("splitter");
  return splitter_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisIsThreadAxis", [](Axis axis) { return axis->IsThreadAxis(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisIsMemoryAxis", [](Axis axis) { return axis->IsMemoryAxis(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisGetScope", [](Axis axis) { return axis->GetScope(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisGetSubscope", [](Axis axis) { return axis->GetSubscope(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<AxisNode>()
      .def("__data_to_json__", [](const AxisNode* node) -> ffi::String { return node->name; })
      .def("__data_from_json__", [](const ffi::String& name) -> Axis { return Axis::Get(name); });
}

// Axis
Axis Axis::Get(const ffi::String& name) {
  const AxisRegEntry* reg = AxisRegistry::Global()->Get(name);
  if (reg != nullptr) {
    return reg->axis_;
  }
  // Auto-register unknown axes on the fly
  return AxisRegEntry::RegisterOrGet(name).axis_;
}

template <typename ValueType>
inline AxisAttrMap<ValueType> Axis::GetAttrMap(const ffi::String& attr_name) {
  return AxisAttrMap<ValueType>(AxisRegistry::Global()->GetAttrMap(attr_name));
}

// AxisRegEntry
inline AxisNode* AxisRegEntry::get() { return const_cast<AxisNode*>(axis_.operator->()); }

AxisRegEntry::AxisRegEntry(uint32_t index) {
  ffi::ObjectPtr<AxisNode> n = ffi::make_object<AxisNode>();
  n->index_ = index;
  axis_ = Axis(n);
}

AxisRegEntry& AxisRegEntry::RegisterOrGet(const ffi::String& name) {
  auto& entry = AxisRegistry::Global()->RegisterOrGet(name);
  entry.get()->name = name;
  return entry;
}

ffi::Array<ffi::String> AxisRegEntry::ListAxisNames() {
  return AxisRegistry::Global()->ListAllNames();
}

template <typename ValueType>
inline AxisRegEntry& AxisRegEntry::set_attr(const ffi::String& key, const ValueType& value,
                                            int plevel) {
  TVM_FFI_ICHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  ffi::Any rv;
  rv = value;
  UpdateAttr(key, rv, plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_scope(const ffi::String& scope_name, int plevel) {
  set_attr<ffi::Optional<ExecScope>>("scope", ExecScope(scope_name), plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_subscope(const ffi::String& subscope_name, int plevel) {
  set_attr<ffi::Optional<ExecScope>>("subscope", ExecScope(subscope_name), plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_fuser(const FAxisFuser& fuser) {
  set_attr<ffi::Optional<FAxisFuser>>("fuser", fuser);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_splitter(const FAxisSplitter& splitter) {
  set_attr<ffi::Optional<FAxisSplitter>>("splitter", splitter);
  return *this;
}

void AxisRegEntry::UpdateAttr(const ffi::String& key, ffi::Any value, int plevel) {
  AxisRegistry::Global()->UpdateAttr(key, axis_, value, plevel);
}

// register thread axis split/fuse helpers
ffi::Array<Iter> SplitterGen(const Iter& iter, const Axis& axis_outer, const Axis& axis_inner,
                             const PrimExpr& e_inner) {
  arith::Analyzer analyzer;
  if (analyzer->CanProve(iter->extent * iter->stride < e_inner)) {
    return {Iter(iter->extent, iter->stride, axis_inner)};
  } else if (analyzer->CanProveEqual(floormod(e_inner, iter->stride), 0) &&
             analyzer->CanProveEqual(floormod(iter->extent * iter->stride, e_inner), 0)) {
    const auto& d = analyzer->Simplify(floordiv(e_inner, iter->stride));
    const auto& c = analyzer->Simplify(floordiv(iter->extent, d));
    return {Iter(c, IntImm(e_inner.ty(), 1), axis_outer), Iter(d, iter->stride, axis_inner)};
  } else if (analyzer->CanProveEqual(floormod(iter->stride, e_inner), 0)) {
    const auto& d = analyzer->Simplify(floordiv(iter->stride, e_inner));
    return {Iter(iter->extent, d, axis_outer)};
  }
  return {};
}

// register thread axes
TVM_REGISTER_AXIS("bx").set_attr<bool>("thread", true).set_scope("thread").set_subscope("cta");
TVM_REGISTER_AXIS("by").set_attr<bool>("thread", true).set_scope("thread").set_subscope("cta");
TVM_REGISTER_AXIS("bz").set_attr<bool>("thread", true).set_scope("thread").set_subscope("cta");
TVM_REGISTER_AXIS("cbx").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("cby").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("cbz").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("tx")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // tx -> warpid, laneid
          return SplitterGen(iter, Axis::Get("warpid"), Axis::Get("laneid"), 32);
        } else if (scope == "warpgroup") {
          // tx -> wgid, tid_in_wg
          return SplitterGen(iter, Axis::Get("wgid"), Axis::Get("tid_in_wg"), 128);
        }
        LOG(FATAL) << "Cannot split cta->thread axis into cta->" << scope << "->thread";
      }
      return {};
    });
TVM_REGISTER_AXIS("warpid")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("warp")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        // cta->warp ===> cta->thread (tx)
        if (subscope == "thread" && scope == "cta") {
          return Iter(iter->extent, 32 * iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // warpid -> wgid, wid_in_wg
          return SplitterGen(iter, Axis::Get("wgid"), Axis::Get("wid_in_wg"), 4);
        }
        LOG(FATAL) << "Cannot split cta->warp axis into cta->" << scope << "->warp";
      }
      return {};
    });
TVM_REGISTER_AXIS("laneid")
    .set_attr<bool>("thread", true)
    .set_scope("warp")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "warpgroup") {
          // warp->thread ===> warpgroup->thread (tid_in_wg)
          return Iter(iter->extent, iter->stride, Axis::Get("tid_in_wg"));
        } else if (subscope == "thread" && scope == "cta") {
          // warp->thread ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "laneid can not be split any more";
      }
      return {};
    });
TVM_REGISTER_AXIS("wgid")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("warpgroup")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "cta") {
          // cta->warpgroup ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride * 128, Axis::Get("tx"));
        } else if (subscope == "warp" && scope == "cta") {
          // cta->warpgroup ===> cta->warp (warpid)
          return Iter(iter->extent, iter->stride * 4, Axis::Get("wgid"));
        }
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "wgid can not be split any more";
      }
      return {};
    });
TVM_REGISTER_AXIS("tid_in_wg")
    .set_attr<bool>("thread", true)
    .set_scope("warpgroup")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "cta") {
          // warpgroup->thread ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // tid_in_wg -> wid_in_wg, laneid
          return SplitterGen(iter, Axis::Get("wid_in_wg"), Axis::Get("laneid"), 32);
        }
        LOG(FATAL) << "Cannot split warpgroup->thread axis into warpgroup->" << scope << "->thread";
      }
      return {};
    });
TVM_REGISTER_AXIS("wid_in_wg")
    .set_attr<bool>("thread", true)
    .set_scope("warpgroup")
    .set_subscope("warp")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "warpgroup") {
          // warpgroup->warp ===> warpgroup->thread (tid_in_wg)
          return Iter(iter->extent, iter->stride * 32, Axis::Get("tid_in_wg"));
        } else if (subscope == "thread" && scope == "cta") {
          // warpgroup->warp ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride * 32, Axis::Get("tx"));
        } else if (subscope == "warp" && scope == "cta") {
          // warpgroup->warp ===> cta->warp (warpid)
          return Iter(iter->extent, iter->stride, Axis::Get("warpid"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "wid_in_wg can not be split any more";
      }
      return {};
    });

// register memory axis
TVM_REGISTER_AXIS("m").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("P").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("F").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("Bank").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TCol").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TLane").set_attr<bool>("thread", false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisGet", [](ffi::String name) -> Axis { return Axis::Get(name); });
}

}  // namespace tirx
}  // namespace tvm
