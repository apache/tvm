
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
 * \file src/ffi/ffi_api.cc
 * \brief Extra ffi apis for frontend to access containers.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/reflection.h>

namespace tvm {
namespace ffi {

// Favor struct outside function scope as MSVC may have bug for in fn scope struct.
class MapForwardIterFunctor {
 public:
  MapForwardIterFunctor(ffi::MapObj::iterator iter, ffi::MapObj::iterator end)
      : iter_(iter), end_(end) {}
  // 0 get current key
  // 1 get current value
  // 2 move to next: return true if success, false if end
  Any operator()(int command) const {
    if (command == 0) {
      return (*iter_).first;
    } else if (command == 1) {
      return (*iter_).second;
    } else {
      ++iter_;
      if (iter_ == end_) {
        return false;
      }
      return true;
    }
  }

 private:
  mutable ffi::MapObj::iterator iter_;
  ffi::MapObj::iterator end_;
};

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("ffi.Array",
                  [](ffi::PackedArgs args, Any* ret) {
                    *ret = Array<Any>(args.data(), args.data() + args.size());
                  })
      .def("ffi.ArrayGetItem", [](const ffi::ArrayObj* n, int64_t i) -> Any { return n->at(i); })
      .def("ffi.ArraySize",
           [](const ffi::ArrayObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def_packed("ffi.Map",
                  [](ffi::PackedArgs args, Any* ret) {
                    TVM_FFI_ICHECK_EQ(args.size() % 2, 0);
                    Map<Any, Any> data;
                    for (int i = 0; i < args.size(); i += 2) {
                      data.Set(args[i], args[i + 1]);
                    }
                    *ret = data;
                  })
      .def("ffi.MapSize",
           [](const ffi::MapObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def("ffi.MapGetItem", [](const ffi::MapObj* n, const Any& k) -> Any { return n->at(k); })
      .def("ffi.MapCount",
           [](const ffi::MapObj* n, const Any& k) -> int64_t { return n->count(k); })
      .def("ffi.MapForwardIterFunctor", [](const ffi::MapObj* n) -> ffi::Function {
        return ffi::Function::FromTyped(MapForwardIterFunctor(n->begin(), n->end()));
      });
});
}  // namespace ffi
}  // namespace tvm
