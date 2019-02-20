
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
struct extension_class_info<tvm_ext::IntVector> {
  static const int code = 17;
};
template<>
struct array_type_index<tvm_ext::NDSubClass> {
  static const int code = 1;
};
}  // namespace tvm
}  // namespace runtime

using namespace tvm;
using namespace tvm::runtime;

namespace tvm_ext {
class NDSubClass : public tvm::runtime::NDArray {
 public:
  class SubContainer : public NDArray::Container {
   public:
    SubContainer(bool is_tracing) {
      array_type_index_ = array_type_index<NDSubClass>::code;
      is_tracing_ = is_tracing;
    }
    static bool Is(NDArray::Container *container) {
      SubContainer *c = static_cast<SubContainer*>(container);
      return c->array_type_index_ == array_type_index<NDSubClass>::code;
    }
    bool is_tracing_{false};
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
  NDSubClass addWith(const NDSubClass &other) const {
    SubContainer *a = static_cast<SubContainer*>(data_);
    SubContainer *b = static_cast<SubContainer*>(other.data_);
    CHECK(a != nullptr && b != nullptr);
    return NDSubClass(new SubContainer(a->is_tracing_ || b->is_tracing_));
  }
  bool get_tracing() const {
    SubContainer *self = static_cast<SubContainer*>(data_);
    CHECK(self != nullptr);
    return self->is_tracing_;
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
  bool is_tracing = args[0];
  *rv = NDSubClass(new NDSubClass::SubContainer(is_tracing));
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_add_two")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  NDSubClass a = args[0];
  NDSubClass b = args[1];
  *rv = a.addWith(b);
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_get_tracing")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  NDSubClass a = args[0];
  *rv = (bool)(a.get_tracing());
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
