#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/var.h>
#include "text_printer.h"

namespace tvm {
namespace printer {

class ModelLibraryFormatPrinter : public ::tvm::runtime::ModuleNode {
 public:
  ModelLibraryFormatPrinter(bool show_meta_data, const runtime::TypedPackedFunc<std::string(ObjectRef)>& annotate, bool show_warning) :
      text_printer_{show_meta_data, annotate, show_warning} {}

  const char* type_key() const override {
    return "model_library_format_printer";
  }

  std::string Print(const ObjectRef& node) {
    Doc doc;
    doc << text_printer_.PrintFinal(node);
    return doc.str();
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "print") {
      return TypedPackedFunc<std::string(ObjectRef)>([sptr_to_self, this](ObjectRef node) { return Print(node); });
    } else if (name == "get_var_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
                          ICHECK_EQ(args.size(), 1) << "usage: get_var_name(Var v)";

                          std::string var_name;
                          if (text_printer_.GetVarName(args[0], &var_name)) {
                            *rv = var_name;
                          }
                        });
    } else {
      return PackedFunc();
    }
  }

 private:
  TextPrinter text_printer_;
};

TVM_REGISTER_GLOBAL("tir.ModelLibraryFormatPrinter").set_body_typed(
  [](bool show_meta_data, const runtime::TypedPackedFunc<std::string(ObjectRef)>& annotate, bool show_warning) {
    return ObjectRef(make_object<ModelLibraryFormatPrinter>(show_meta_data, annotate, show_warning));
  });

}  // namespace printer
}  // namespace tvm
