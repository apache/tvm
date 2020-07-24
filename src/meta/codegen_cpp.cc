#include <tvm/runtime/registry.h>
#include <tvm/meta/functor.h>
#include <string>
#include <sstream>

namespace tvm {
namespace meta {

class CodeGenCPP : public ExprFunctor<std::string(const MetaIR&)> {
 public:
  std::string VisitMetaIR(const MetaIR& n) override {
    using TParent = ExprFunctor<std::string(const MetaIR&)>;
    return TParent::VisitMetaIR(n);
  }
  
  std::string VisitMetaIR_(const VarDefNode* n) override {
    return " ";
  }

  std::string VisitMetaIR_(const ObjectDefNode* n) override {
    auto* pbase = static_cast<const ObjectDefNode*>(n->base.get());
    CHECK(pbase);
    std::string base_name = pbase->name;
    std::string base_ref_name = pbase->ref_name;

    // generate object declaration
    stream_ << "class " << n->name << " : public " << base_name << " {\n";
    stream_ << " public: \n"; 
    stream_ << "  static constexpr const char* _type_key = " << n->name << ";\n";
    stream_ << "  TVM_DECLARE_BASE_OBJECT_INFO(" << n->name << ", " << base_name << ");\n";
    stream_ << "};\n";

    // generate object reference declaration
    stream_ << "class " << n->ref_name << " : public " << base_ref_name << " {\n";
    stream_ << " public: \n"; 
    stream_ << "  TVM_DEFINE_OBJECT_REF_METHODS(" << n->ref_name << ", "
            << base_ref_name << ", " << n->ref_name <<");\n"; 
    stream_ << "};\n";
    return stream_.str();
  }
 private:
  std::stringstream stream_;
};

TVM_REGISTER_GLOBAL("meta.GenerateCPP").set_body_typed(
  [](MetaIR ir) {
    return CodeGenCPP().VisitMetaIR(ir);
});



}  // namespace meta
}  // namespace tvm
