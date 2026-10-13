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

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/tirx/layout.h>

#include "../../../tirx/ir/layout/axis_registry.h"
#include "../../../tirx/ir/layout/utils.h"
namespace tvm::backend::cuda {
using namespace tvm::tirx;
namespace refl = tvm::ffi::reflection;
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

void RegisterLayoutAxes() {
  // Register thread axes in declaration order.
  AxisDef("bx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "kernel"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("by")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "kernel"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("bz")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "kernel"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("cbx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cluster"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("cby")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cluster"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("cbz")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cluster"))
      .set_attr("subscope", ExecScope("cuda", "cta"));

  AxisDef("tx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cta"))
      .set_attr("subscope", ExecScope("cuda", "thread"))
      .set_attr("fuser", FAxisFuser([](Target target, ffi::String subscope, ffi::String scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    return std::nullopt;
                  }
                  return std::nullopt;
                }))
      .set_attr("splitter",
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

  AxisDef("warpid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cta"))
      .set_attr("subscope", ExecScope("cuda", "warp"))
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
      .set_attr("splitter",
                FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
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

  AxisDef("laneid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "warp"))
      .set_attr("subscope", ExecScope("cuda", "thread"))
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
      .set_attr("splitter",
                FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
                  sym::Analyzer analyzer;
                  if (target->kind->default_device_type == kDLCUDA) {
                    LOG(FATAL) << "laneid can not be split any more";
                  }
                  return {};
                }));

  AxisDef("wgid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "cta"))
      .set_attr("subscope", ExecScope("cuda", "warpgroup"))
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
      .set_attr("splitter",
                FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
                  sym::Analyzer analyzer;
                  if (target->kind->default_device_type == kDLCUDA) {
                    LOG(FATAL) << "wgid can not be split any more";
                  }
                  return {};
                }));

  AxisDef("tid_in_wg")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "warpgroup"))
      .set_attr("subscope", ExecScope("cuda", "thread"))
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
      .set_attr("splitter",
                FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
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

  AxisDef("wid_in_wg")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ExecScope("cuda", "warpgroup"))
      .set_attr("subscope", ExecScope("cuda", "warp"))
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
      .set_attr("splitter",
                FAxisSplitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
                  sym::Analyzer analyzer;
                  if (target->kind->default_device_type == kDLCUDA) {
                    LOG(FATAL) << "wid_in_wg can not be split any more";
                  }
                  return {};
                }));

  for (const char* name : {"TCol", "TLane"}) {
    AxisDef(name).set_attr("backend", ffi::String("cuda")).set_attr("thread", false);
  }
}
TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tirx.cuda.RegisterLayoutAxes", RegisterLayoutAxes);
}
}  // namespace tvm::backend::cuda
