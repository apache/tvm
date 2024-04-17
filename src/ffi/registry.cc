#include <memory>
#include <string>
#include <tvm/ffi/ffi.hpp>
#include <unordered_map>
#include <vector>

namespace {

using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Func;
using tvm::ffi::Ref;
using tvm::ffi::TypeIndexTraits;
using tvm::ffi::TypeTraits;
using tvm::ffi::details::InitTypeTable;

struct TypeInfoImpl : public TVMFFITypeInfo {
  std::string type_key_data;
  std::vector<int32_t> type_ancestors_data;
};

struct TypeTable {
  int32_t num_types;
  std::vector<std::shared_ptr<TypeInfoImpl>> type_table;
  std::unordered_map<std::string, std::shared_ptr<TypeInfoImpl>> type_key_to_info;
  std::unordered_map<std::string, std::vector<Any>> type_attrs;

  static TypeTable *New();
  static TypeTable *Global() {
    static std::unique_ptr<TypeTable> instance(TypeTable::New());
    return instance.get();
  }
};

int32_t TypeDef(TypeTable *self, int32_t type_index, const char *type_key, int32_t type_depth,
                const int32_t *type_ancestors) {
  if (self == nullptr) {
    TVM_FFI_THROW(ValueError) << "TypeTable is provided as nullptr.";
  }
  if (self->type_key_to_info.count(type_key) != 0) {
    TVM_FFI_THROW(ValueError) << "Type key is already registered.";
  }
  if (type_index == -1) {
    type_index = self->num_types++;
  }
  std::shared_ptr<TypeInfoImpl> info = std::make_shared<TypeInfoImpl>();
  {
    info->type_key_data = std::string(type_key);
    info->type_ancestors_data = std::vector<int32_t>(type_ancestors, type_ancestors + type_depth);
    info->type_index = type_index;
    info->type_key = info->type_key_data.c_str();
    info->type_depth = type_depth;
    info->type_ancestors = info->type_ancestors_data.data();
  }
  if (type_index >= static_cast<int32_t>(self->type_table.size())) {
    self->type_table.resize((type_index / 1024 + 1) * 1024);
  }
  self->type_table.at(type_index) = info;
  self->type_key_to_info[type_key] = info;
  return type_index;
}

struct InitFunctor {
  template <enum TVMFFITypeIndex type_index, typename ObjectType>
  void RegisterType(TypeTable *self) {
    TypeDef(self, static_cast<int32_t>(type_index), TypeIndexTraits<type_index>::type_key,
            std::max<int32_t>(ObjectType::_type_depth, 0), ObjectType::_type_ancestors.data());
  }

  template <enum TVMFFITypeIndex type_index, typename ObjectType>
  void RegisterStr(TypeTable *self) {
    Any value = Any(Ref<Func>::New(TypeTraits<ObjectType>::__str__));
    TVMFFIDynTypeSetAttr(self, static_cast<int32_t>(type_index), "__str__", &value);
  }
};

TypeTable *TypeTable::New() {
  TypeTable *self = new TypeTable();
  self->type_table.resize(1024);
  self->type_key_to_info.reserve(1024);
  self->num_types = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDynObjectBegin);
  InitTypeTable(InitFunctor(), self);
  return self;
}

} // namespace

TVM_FFI_API void TVMFFIDynTypeDef(TVMFFITypeTableHandle _self, const char *type_key,
                                  int32_t type_depth, const int32_t *type_ancestors,
                                  int32_t *type_index) {
  TypeTable *self = static_cast<TypeTable *>(_self);
  if (self == nullptr) {
    self = TypeTable::Global();
  }
  *type_index = TypeDef(self, -1, type_key, type_depth, type_ancestors);
}

TVM_FFI_API void TVMFFIDynTypeIndex2Info(TVMFFITypeTableHandle _self, int32_t type_index,
                                         TVMFFITypeInfoHandle *ret) {
  TypeTable *self = static_cast<TypeTable *>(_self);
  if (self == nullptr) {
    self = TypeTable::Global();
  }
  if (type_index < 0 || type_index >= static_cast<int32_t>(self->type_table.size())) {
    *ret = nullptr;
  } else {
    *ret = self->type_table.at(type_index).get();
  }
}

TVM_FFI_API void TVMFFIDynTypeSetAttr(TVMFFITypeTableHandle _self, int32_t type_index,
                                      const char *key, TVMFFIAnyHandle value) {
  TypeTable *self = static_cast<TypeTable *>(_self);
  if (self == nullptr) {
    self = TypeTable::Global();
  }
  std::vector<Any> &attrs = self->type_attrs[key];
  if (type_index >= static_cast<int32_t>(attrs.size())) {
    attrs.resize((type_index / 1024 + 1) * 1024);
  }
  attrs.at(type_index) = Any(*static_cast<AnyView *>(value));
}

TVM_FFI_API void TVMFFIDynTypeGetAttr(TVMFFITypeTableHandle _self, int32_t type_index,
                                      const char *key, TVMFFIAnyHandle *ret) {
  TypeTable *self = static_cast<TypeTable *>(_self);
  if (self == nullptr) {
    self = TypeTable::Global();
  }
  std::vector<Any> &attrs = self->type_attrs[key];
  if (type_index >= static_cast<int32_t>(attrs.size())) {
    *ret = nullptr;
  } else {
    *ret = &attrs.at(type_index);
  }
}

TVM_FFI_API void TVMFFIDynTypeTypeTableCreate(TVMFFITypeTableHandle *ret) {
  *ret = TypeTable::New();
}

TVM_FFI_API void TVMFFIDynTypeTypeTableDestroy(TVMFFITypeTableHandle handle) {
  delete static_cast<TypeTable *>(handle);
}
