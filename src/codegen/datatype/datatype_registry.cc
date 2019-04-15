#include "datatype_registry.h"
#include <tvm/api_registry.h>

namespace tvm {

TVM_REGISTER_GLOBAL("_datatype_register")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      DatatypeRegistry::Global()->RegisterDatatype(
          args[0], static_cast<uint8_t>(args[1].operator int()),
          args[2].operator size_t());
    });

TVM_REGISTER_GLOBAL("_datatype_get_type_code")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeCode(args[0]);
    });

TVM_REGISTER_GLOBAL("_datatype_get_type_name")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeName(args[0].operator int());
    });

TVM_REGISTER_GLOBAL("_datatype_registered")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = DatatypeRegistry::Global()->DatatypeRegistered(
          args[0].operator int());
    });

void DatatypeRegistry::RegisterDatatype(const std::string &type_name,
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

const runtime::PackedFunc* GetCastLowerFunc(const std::string& target,
                                            uint8_t type_code,
                                            uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.lower.";
  ss << target << ".";
  ss << "Cast"
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

TVM_REGISTER_GLOBAL("_register_Cast")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
      const std::string target = args[0];
      const std::string type = args[1];
      const std::string src_type = args[2];
      const std::string extern_func_name = args[3];

      auto lower_cast_name =
          "tvm.datatype.lower." + target + ".Cast." + type + "." + src_type;
      runtime::Registry::Register(lower_cast_name)
          .set_body([extern_func_name](TVMArgs args, TVMRetValue *rv) {
            Expr e = args[0];
            const ir::Cast *cast = e.as<ir::Cast>();
            CHECK(cast) << "Expected cast";
            // Custom datatypes should get cast to their underlying storage
            // type.
            // TODO(gus) I'm not using the width registered originally; I'm
            // using the bits() attached to the type (where does this come
            // from?)
            auto return_type = DatatypeRegistry::Global()->DatatypeRegistered(
                                   cast->type.code())
                                   ? UInt(cast->type.bits())
                                   : cast->type;
            *rv = ir::Call::make(return_type, extern_func_name, {cast->value},
                                 ir::Call::Extern);
          });
    });

#define REGISTER_OP_A_B(OP)                                                    \
  TVM_REGISTER_GLOBAL("_register_" #OP)                                        \
      .set_body([](TVMArgs args, TVMRetValue *rv) {                            \
        const std::string target = args[0];                                    \
        const std::string type = args[1];                                      \
        const std::string extern_func_name = args[2];                          \
        auto lower_op_name =                                                   \
            "tvm.datatype.lower." + target + "." #OP "." + type;              \
        runtime::Registry::Register(lower_op_name)                             \
            .set_body([extern_func_name](TVMArgs args, TVMRetValue *rv) {      \
              Expr e = args[0];                                                \
              const ir::OP *op = e.as<ir::OP>();                               \
              CHECK(op) << "Expected " #OP;                                    \
              *rv = ir::Call::make(UInt(op->type.bits()), extern_func_name,    \
                                   {op->a, op->b}, ir::Call::Extern);          \
            });                                                                \
      });

REGISTER_OP_A_B(Add);
}  // namespace tvm
