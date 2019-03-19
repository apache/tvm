#include "datatype_registry.h"
#include <tvm/api_registry.h>
#include "topi/detail/extern.h"

namespace tvm {

TVM_REGISTER_GLOBAL("_register_datatype")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DatatypeRegistry::Global()->RegisterDatatype(
          args[0], (uint8_t)args[1].operator int(), args[2].operator size_t());
    });

TVM_REGISTER_GLOBAL("_get_type_code")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeCode(args[0]);
    });

TVM_REGISTER_GLOBAL("_get_type_name")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeName(args[0].operator int());
    });

TVM_REGISTER_GLOBAL("_get_storage_size")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      // TODO(gus) I cast this down to an int so that it can be automatically
      // converted to TVMRetValue. This is dumb and should be fixed somehow.
      *ret = (int)DatatypeRegistry::Global()->GetStorageSize(args[0].operator int());
    });

void DatatypeRegistry::RegisterDatatype(const std::string& type_name,
                                        uint8_t type_code,
                                        size_t storage_size) {
  code_to_name[type_code] = type_name;
  name_to_code[type_name] = type_code;
  code_to_storage_size[type_code] = storage_size;
}

uint8_t DatatypeRegistry::GetTypeCode(const std::string& type_name) {
  return name_to_code[type_name];
}

std::string DatatypeRegistry::GetTypeName(uint8_t type_code) {
  return code_to_name[type_code];
}

size_t DatatypeRegistry::GetStorageSize(uint8_t type_code) {
  return code_to_storage_size[type_code];
}

const runtime::PackedFunc* GetCastLowerFunc(const std::string& target,
                                            uint8_t type_code,
                                            uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.datatypes.lower.";
  ss << target << ".";
  ss << "cast"
     << ".";

  if (DatatypeRegistry::Global()->DatatypeRegistered(type_code)) {
    ss << DatatypeRegistry::Global()->GetTypeName(type_code);
  } else {
    ss << runtime::TypeCode2Str(type_code);
  }

  ss << ".";

  if (DatatypeRegistry::Global()->DatatypeRegistered(src_type_code)) {
    ss << DatatypeRegistry::Global()->GetTypeName(src_type_code);
  } else {
    ss << runtime::TypeCode2Str(src_type_code);
  }

  return runtime::Registry::Get(ss.str());
}

TVM_REGISTER_GLOBAL("_register_cast")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      const std::string target = args[0];
      const std::string type = args[1];
      const std::string src_type = args[2];
      const std::string extern_func_name = args[3];
      auto lower_cast_name =
          "tvm.datatypes.lower." + target + ".cast." + type + "." + src_type;
      runtime::Registry::Register(lower_cast_name)
          .set_body([extern_func_name](TVMArgs args, TVMRetValue* rv) {
            Expr e = args[0];
            const ir::Cast* cast = e.as<ir::Cast>();
            internal_assert(cast);
            // TODO(gus) UInt(32) here is the resulting storage class, maybe.
            // They should probably be able to specify this themselves. Or it should
            // be given when the datatype is originally registered.
            *rv = ir::Call::make(UInt(32), extern_func_name, {cast->value},
                                 ir::Call::Extern);
          });
      });

TVM_REGISTER_GLOBAL("_register_op").set_body([](TVMArgs args, TVMRetValue* rv) {
  const std::string target = args[0];
  const std::string op = args[1];
  PackedFunc ext_func = args[2];
  auto ext_func_name = "tvm.datatypes.lower." + target + "." + op + ".external";
  runtime::Registry::Register(ext_func_name).set_body(ext_func);
  runtime::Registry::Register("tvm.datatypes.lower." + target + "." + op)
      .set_body([ext_func_name](TVMArgs args, TVMRetValue* rv) {
        Expr e = args[0];
        const ir::Add* add = e.as<ir::Add>();
        internal_assert(add);
        return topi::detail::call_packed({Expr(ext_func_name), add->a, add->b});
      });
});
}  // namespace tvm
