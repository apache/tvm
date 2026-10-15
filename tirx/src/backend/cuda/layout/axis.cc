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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/layout.h>

#include "../../../tirx/ir/layout/axis_registry.h"
namespace tvm::backend::cuda {
using namespace tvm::tirx;
namespace refl = tvm::ffi::reflection;
void RegisterLayoutAxes() {
  // Register thread axes in declaration order.
  AxisDef("bx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.kernel"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("by")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.kernel"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("bz")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.kernel"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("cbx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cluster"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("cby")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cluster"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("cbz")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cluster"))
      .set_attr("subscope", ThreadScope("cuda.cta"));

  AxisDef("tx")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cta"))
      .set_attr("subscope", ThreadScope("cuda.thread"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    return std::nullopt;
                  }
                  return std::nullopt;
                }));

  AxisDef("warpid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cta"))
      .set_attr("subscope", ThreadScope("cuda.warp"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    // cta->warp ===> cta->thread (tx)
                    if (subscope.same_as(ThreadScope("cuda.thread")) &&
                        scope.same_as(ThreadScope("cuda.cta"))) {
                      return Iter(iter->extent, 32 * iter->stride, Axis::Get("tx"));
                    }
                    return std::nullopt;
                  }
                  return std::nullopt;
                }));

  AxisDef("laneid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.warp"))
      .set_attr("subscope", ThreadScope("cuda.thread"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    if (subscope.same_as(ThreadScope("cuda.thread")) &&
                        scope.same_as(ThreadScope("cuda.warpgroup"))) {
                      // warp->thread ===> warpgroup->thread (tid_in_wg)
                      return Iter(iter->extent, iter->stride, Axis::Get("tid_in_wg"));
                    } else if (subscope.same_as(ThreadScope("cuda.thread")) &&
                               scope.same_as(ThreadScope("cuda.cta"))) {
                      // warp->thread ===> cta->thread (tx)
                      return Iter(iter->extent, iter->stride, Axis::Get("tx"));
                    }
                    return std::nullopt;
                  }
                  return std::nullopt;
                }));

  AxisDef("wgid")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.cta"))
      .set_attr("subscope", ThreadScope("cuda.warpgroup"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    if (subscope.same_as(ThreadScope("cuda.thread")) &&
                        scope.same_as(ThreadScope("cuda.cta"))) {
                      // cta->warpgroup ===> cta->thread (tx)
                      return Iter(iter->extent, iter->stride * 128, Axis::Get("tx"));
                    } else if (subscope.same_as(ThreadScope("cuda.warp")) &&
                               scope.same_as(ThreadScope("cuda.cta"))) {
                      // cta->warpgroup ===> cta->warp (warpid)
                      return Iter(iter->extent, iter->stride * 4, Axis::Get("wgid"));
                    }
                  }
                  return std::nullopt;
                }));

  AxisDef("tid_in_wg")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.warpgroup"))
      .set_attr("subscope", ThreadScope("cuda.thread"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    if (subscope.same_as(ThreadScope("cuda.thread")) &&
                        scope.same_as(ThreadScope("cuda.cta"))) {
                      // warpgroup->thread ===> cta->thread (tx)
                      return Iter(iter->extent, iter->stride, Axis::Get("tx"));
                    }
                    return std::nullopt;
                  }
                  return std::nullopt;
                }));

  AxisDef("wid_in_wg")
      .set_attr("backend", ffi::String("cuda"))
      .set_attr("thread", true)
      .set_attr("scope", ThreadScope("cuda.warpgroup"))
      .set_attr("subscope", ThreadScope("cuda.warp"))
      .set_attr("fuser", FAxisFuser([](Target target, ThreadScope subscope, ThreadScope scope,
                                       Iter iter) -> ffi::Optional<Iter> {
                  if (target->kind->default_device_type == kDLCUDA) {
                    if (subscope.same_as(ThreadScope("cuda.thread")) &&
                        scope.same_as(ThreadScope("cuda.warpgroup"))) {
                      // warpgroup->warp ===> warpgroup->thread (tid_in_wg)
                      return Iter(iter->extent, iter->stride * 32, Axis::Get("tid_in_wg"));
                    } else if (subscope.same_as(ThreadScope("cuda.thread")) &&
                               scope.same_as(ThreadScope("cuda.cta"))) {
                      // warpgroup->warp ===> cta->thread (tx)
                      return Iter(iter->extent, iter->stride * 32, Axis::Get("tx"));
                    } else if (subscope.same_as(ThreadScope("cuda.warp")) &&
                               scope.same_as(ThreadScope("cuda.cta"))) {
                      // warpgroup->warp ===> cta->warp (warpid)
                      return Iter(iter->extent, iter->stride, Axis::Get("warpid"));
                    }
                    return std::nullopt;
                  }
                  return std::nullopt;
                }));

  for (const char* name : {"TCol", "TLane"}) {
    AxisDef(name).set_attr("backend", ffi::String("cuda")).set_attr("thread", false);
  }
}
TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tirx.cuda.RegisterLayoutAxes", RegisterLayoutAxes);
}
}  // namespace tvm::backend::cuda
