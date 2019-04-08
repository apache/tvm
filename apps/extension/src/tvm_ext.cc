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
 *  Copyright (c) 2017 by Contributors
 * \brief Example package that uses TVM.
 * \file tvm_ext.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/device_api.h>

namespace tvm_ext {
using IntVector = std::vector<int>;
class NDSubClass;
}  // namespace tvm_ext

namespace tvm {
namespace runtime {
template<>
struct extension_type_info<tvm_ext::IntVector> {
  static const int code = 17;
};
template<>
struct array_type_info<tvm_ext::NDSubClass> {
  static const int code = 1;
};
}  // namespace tvm
}  // namespace runtime

using namespace tvm;
using namespace tvm::runtime;

namespace tvm_ext {
/*!
 * \brief A subclass of TVM's NDArray.
 *
 * To use this extension, an external library should
 *
 * 1) Inherit TVM's NDArray and NDArray container,
 *    and define the trait `array_type_info` for this class.
 *
 * 2) Define a constructor in the inherited class that accepts
 *    a pointer to TVM's Container, which is nullable.
 *
 * 3) On Python frontend, inherit `tvm.nd.NDArrayBase`,
 *    define the class attribute `_array_type_code` consistent to
 *    the C++ type trait, and register the subclass using `tvm.register_extension`.
 */
class NDSubClass : public tvm::runtime::NDArray {
 public:
  class SubContainer : public NDArray::Container {
   public:
    SubContainer(int addtional_info) :
      addtional_info_(addtional_info) {
      array_type_code_ = array_type_info<NDSubClass>::code;
    }
    static bool Is(NDArray::Container *container) {
      SubContainer *c = static_cast<SubContainer*>(container);
      return c->array_type_code_ == array_type_info<NDSubClass>::code;
    }
    int addtional_info_{0};
  };
  NDSubClass(NDArray::Container *container) {
    if (container == nullptr) {
      data_ = nullptr;
      return;
    }
    CHECK(SubContainer::Is(container));
    container->IncRef();
    data_ = container;
  }
  ~NDSubClass() {
    this->reset();
  }
  NDSubClass AddWith(const NDSubClass &other) const {
    SubContainer *a = static_cast<SubContainer*>(data_);
    SubContainer *b = static_cast<SubContainer*>(other.data_);
    CHECK(a != nullptr && b != nullptr);
    return NDSubClass(new SubContainer(a->addtional_info_ + b->addtional_info_));
  }
  int get_additional_info() const {
    SubContainer *self = static_cast<SubContainer*>(data_);
    CHECK(self != nullptr);
    return self->addtional_info_;
  }
};
}  // namespace tvm_ext

namespace tvm_ext {

TVM_REGISTER_EXT_TYPE(IntVector);

TVM_REGISTER_GLOBAL("tvm_ext.ivec_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    IntVector vec;
    for (int i = 0; i < args.size(); ++i) {
      vec.push_back(args[i].operator int());
    }
    *rv = vec;
  });

TVM_REGISTER_GLOBAL("tvm_ext.ivec_get")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = args[0].AsExtension<IntVector>()[args[1].operator int()];
  });


TVM_REGISTER_GLOBAL("tvm_ext.bind_add")
.set_body([](TVMArgs args_, TVMRetValue *rv_) {
    PackedFunc pf = args_[0];
    int b = args_[1];
    *rv_ = PackedFunc([pf, b](TVMArgs args, TVMRetValue *rv) {
        *rv = pf(b, args[0]);
      });
  });

TVM_REGISTER_GLOBAL("tvm_ext.sym_add")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    Var a = args[0];
    Var b = args[1];
    *rv = a + b;
  });

TVM_REGISTER_GLOBAL("device_api.ext_dev")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = (*tvm::runtime::Registry::Get("device_api.cpu"))();
  });

TVM_REGISTER_GLOBAL("tvm_ext.nd_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  int addtional_info = args[0];
  *rv = NDSubClass(new NDSubClass::SubContainer(addtional_info));
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_add_two")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  NDSubClass a = args[0];
  NDSubClass b = args[1];
  *rv = a.AddWith(b);
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_get_addtional_info")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  NDSubClass a = args[0];
  *rv = a.get_additional_info();
});

}  // namespace tvm_ext

// External function exposed to runtime.
extern "C" float TVMTestAddOne(float y) {
  return y + 1;
}

// This callback approach allows extension allows tvm to extract
// This way can be helpful when we want to use a header only
// minimum version of TVM Runtime.
extern "C" int TVMExtDeclare(TVMFunctionHandle pregister) {
  const PackedFunc& fregister =
      *static_cast<PackedFunc*>(pregister);
  auto mul = [](TVMArgs args, TVMRetValue *rv) {
    int x = args[0];
    int y = args[1];
    *rv = x * y;
  };
  fregister("mul", PackedFunc(mul));
  return 0;
}
