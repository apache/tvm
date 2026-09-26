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
#include <tvm/ffi/reflection/enum_def.h>

#include <mutex>

#include "utils.h"

namespace tvm {
namespace tirx {
namespace refl = tvm::ffi::reflection;

/**************** Axis ****************/
ffi::ObjectPtr<ffi::Object> CreateAxis(const std::string& name) {
  auto axis = Axis::Get(name);
  return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<ffi::Object>(axis);
}

template <typename T>
ffi::Optional<T> AxisAttr(const AxisNode* axis, const char* key) {
  static refl::TypeAttrColumn state_column(refl::type_attr::kEnumState);
  ffi::EnumState state =
      state_column[AxisNode::_GetOrAllocRuntimeTypeIndex()].cast<ffi::EnumState>();
  auto column = state->attrs.Get(ffi::String(key));
  if (!column) return std::nullopt;
  auto value = column.value().Get(ffi::GetRef<Axis>(axis));
  if (!value) return std::nullopt;
  return value.value().cast<T>();
}

bool AxisNode::IsThreadAxis() const {
  auto thread = AxisAttr<bool>(this, "thread");
  TVM_FFI_ICHECK(thread.has_value()) << "Axis '" << _str_index << "' has no thread classification";
  return thread.value();
}

bool AxisNode::IsMemoryAxis() const {
  auto thread = AxisAttr<bool>(this, "thread");
  TVM_FFI_ICHECK(thread.has_value()) << "Axis '" << _str_index << "' has no thread classification";
  return !thread.value();
}

ffi::Optional<ExecScope> AxisNode::GetScope() const { return AxisAttr<ExecScope>(this, "scope"); }

ffi::Optional<ExecScope> AxisNode::GetSubscope() const {
  return AxisAttr<ExecScope>(this, "subscope");
}

ffi::Optional<FAxisFuser> AxisNode::GetFuser() const { return AxisAttr<FAxisFuser>(this, "fuser"); }

ffi::Optional<FAxisSplitter> AxisNode::GetSplitter() const {
  return AxisAttr<FAxisSplitter>(this, "splitter");
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
  AxisNode::RegisterReflection();
  refl::TypeAttrDef<AxisNode>()
      .def("__data_to_json__", [](const AxisNode* node) -> ffi::String { return node->_str_index; })
      .def("__data_from_json__", [](const ffi::String& name) -> Axis { return Axis::Get(name); });
}

// Axis
Axis Axis::Get(const ffi::String& name) {
  static std::mutex registration_mutex;
  std::lock_guard<std::mutex> lock(registration_mutex);
  static refl::TypeAttrColumn state_column(refl::type_attr::kEnumState);
  ffi::EnumState state =
      state_column[AxisNode::_GetOrAllocRuntimeTypeIndex()].cast<ffi::EnumState>();
  auto existing = state->indexes.Get(ffi::Any(name));
  if (!existing) {
    refl::EnumDef<AxisNode>(name.c_str());
    existing = state->indexes.Get(ffi::Any(name));
  }
  return ffi::Any(existing.value()).cast<Axis>();
}

// register thread axis split/fuse helpers
ffi::Array<Iter> SplitterGen(const Iter& iter, const Axis& axis_outer, const Axis& axis_inner,
                             const PrimExpr& e_inner) {
  sym::Analyzer analyzer;
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
[[maybe_unused]] static auto axis_entry_1 = refl::EnumDef<AxisNode>("bx")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("thread"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_2 = refl::EnumDef<AxisNode>("by")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("thread"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_3 = refl::EnumDef<AxisNode>("bz")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("thread"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_4 = refl::EnumDef<AxisNode>("cbx")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("cluster"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_5 = refl::EnumDef<AxisNode>("cby")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("cluster"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_6 = refl::EnumDef<AxisNode>("cbz")
                                                .set_attr("thread", true)
                                                .set_attr("scope", ExecScope("cluster"))
                                                .set_attr("subscope", ExecScope("cta"));
[[maybe_unused]] static auto axis_entry_7 =
    refl::EnumDef<AxisNode>("tx")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("cta"))
        .set_attr("subscope", ExecScope("thread"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
                                         Iter iter) -> ffi::Optional<Iter> {
                    if (target->kind->default_device_type == kDLCUDA) {
                      return std::nullopt;
                    }
                    return std::nullopt;
                  }))
        .set_attr(
            "splitter",
            FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
              sym::Analyzer analyzer;
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
            }));
[[maybe_unused]] static auto axis_entry_8 =
    refl::EnumDef<AxisNode>("warpid")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("cta"))
        .set_attr("subscope", ExecScope("warp"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
                                         Iter iter) -> ffi::Optional<Iter> {
                    if (target->kind->default_device_type == kDLCUDA) {
                      // cta->warp ===> cta->thread (tx)
                      if (subscope == "thread" && scope == "cta") {
                        return Iter(iter->extent, 32 * iter->stride, Axis::Get("tx"));
                      }
                      return std::nullopt;
                    }
                    return std::nullopt;
                  }))
        .set_attr("splitter", FAxisSplitter([](Target target, ffi::String scope,
                                               Iter iter) -> ffi::Array<Iter> {
                    sym::Analyzer analyzer;
                    if (target->kind->default_device_type == kDLCUDA) {
                      if (scope == "warp") {
                        // warpid -> wgid, wid_in_wg
                        return SplitterGen(iter, Axis::Get("wgid"), Axis::Get("wid_in_wg"), 4);
                      }
                      LOG(FATAL) << "Cannot split cta->warp axis into cta->" << scope << "->warp";
                    }
                    return {};
                  }));
[[maybe_unused]] static auto axis_entry_9 =
    refl::EnumDef<AxisNode>("laneid")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("warp"))
        .set_attr("subscope", ExecScope("thread"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
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
                  }))
        .set_attr("splitter", FAxisSplitter([](Target target, ffi::String scope,
                                               Iter iter) -> ffi::Array<Iter> {
                    sym::Analyzer analyzer;
                    if (target->kind->default_device_type == kDLCUDA) {
                      LOG(FATAL) << "laneid can not be split any more";
                    }
                    return {};
                  }));
[[maybe_unused]] static auto axis_entry_10 =
    refl::EnumDef<AxisNode>("wgid")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("cta"))
        .set_attr("subscope", ExecScope("warpgroup"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
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
                  }))
        .set_attr("splitter", FAxisSplitter([](Target target, ffi::String scope,
                                               Iter iter) -> ffi::Array<Iter> {
                    sym::Analyzer analyzer;
                    if (target->kind->default_device_type == kDLCUDA) {
                      LOG(FATAL) << "wgid can not be split any more";
                    }
                    return {};
                  }));
[[maybe_unused]] static auto axis_entry_11 =
    refl::EnumDef<AxisNode>("tid_in_wg")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("warpgroup"))
        .set_attr("subscope", ExecScope("thread"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
                                         Iter iter) -> ffi::Optional<Iter> {
                    if (target->kind->default_device_type == kDLCUDA) {
                      if (subscope == "thread" && scope == "cta") {
                        // warpgroup->thread ===> cta->thread (tx)
                        return Iter(iter->extent, iter->stride, Axis::Get("tx"));
                      }
                      return std::nullopt;
                    }
                    return std::nullopt;
                  }))
        .set_attr("splitter", FAxisSplitter([](Target target, ffi::String scope,
                                               Iter iter) -> ffi::Array<Iter> {
                    sym::Analyzer analyzer;
                    if (target->kind->default_device_type == kDLCUDA) {
                      if (scope == "warp") {
                        // tid_in_wg -> wid_in_wg, laneid
                        return SplitterGen(iter, Axis::Get("wid_in_wg"), Axis::Get("laneid"), 32);
                      }
                      LOG(FATAL) << "Cannot split warpgroup->thread axis into warpgroup->" << scope
                                 << "->thread";
                    }
                    return {};
                  }));
[[maybe_unused]] static auto axis_entry_12 =
    refl::EnumDef<AxisNode>("wid_in_wg")
        .set_attr("thread", true)
        .set_attr("scope", ExecScope("warpgroup"))
        .set_attr("subscope", ExecScope("warp"))
        .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
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
                  }))
        .set_attr("splitter", FAxisSplitter([](Target target, ffi::String scope,
                                               Iter iter) -> ffi::Array<Iter> {
                    sym::Analyzer analyzer;
                    if (target->kind->default_device_type == kDLCUDA) {
                      LOG(FATAL) << "wid_in_wg can not be split any more";
                    }
                    return {};
                  }));

// register memory axis
[[maybe_unused]] static auto axis_entry_13 = refl::EnumDef<AxisNode>("m").set_attr("thread", false);
[[maybe_unused]] static auto axis_entry_14 = refl::EnumDef<AxisNode>("P").set_attr("thread", false);
[[maybe_unused]] static auto axis_entry_15 = refl::EnumDef<AxisNode>("F").set_attr("thread", false);
[[maybe_unused]] static auto axis_entry_16 =
    refl::EnumDef<AxisNode>("Bank").set_attr("thread", false);
[[maybe_unused]] static auto axis_entry_17 =
    refl::EnumDef<AxisNode>("TCol").set_attr("thread", false);
[[maybe_unused]] static auto axis_entry_18 =
    refl::EnumDef<AxisNode>("TLane").set_attr("thread", false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AxisGet", [](ffi::String name) -> Axis { return Axis::Get(name); });
}

}  // namespace tirx
}  // namespace tvm
