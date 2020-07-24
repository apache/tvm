#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

using tvm::runtime::Object;
using tvm::runtime::ObjectRef;
using tvm::runtime::String;

namespace tvm {
namespace meta {

class MetaIRNode : public Object {
 public:
  static constexpr const char* _type_key = "meta.MetaIR";
  TVM_DECLARE_BASE_OBJECT_INFO(MetaIRNode, Object);
};

class MetaIR : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(MetaIR, ObjectRef, MetaIRNode);
};

class VarDefNode : public MetaIRNode {
 public:
  String name;
  String type_name; // ObjectDef ?

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("type_info", &type_info);
  }

  static constexpr const char* _type_key = "meta.VarDef";
  TVM_DECLARE_BASE_OBJECT_INFO(VarDefNode, MetaIRNode);
};

class VarDef : public MetaIR {
 public:
  TVM_DLL VarDef(String name, MetaIR type_info);
  TVM_DEFINE_OBJECT_REF_METHODS(VarDef, MetaIR, VarDefNode);
};

class ObjectDefNode : public MetaIRNode {
 public:
  String name;
  String ref_name;
  String nmspace;
  MetaIR base;  // ObjectDef ?
  Array<VarDef> variables;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("ref_name", &ref_name);
    v->Visit("nmspace", &nmspace);
    v->Visit("base", &base);
    v->Visit("variables", &variables);
  }

  static constexpr const char* _type_key = "meta.ObjectDef";
  TVM_DECLARE_BASE_OBJECT_INFO(ObjectDefNode, MetaIRNode);
};

class ObjectDef : public MetaIR {
 public:
  TVM_DLL ObjectDef(String name, String ref_name, String nmspace,
    MetaIR base, Array<VarDef> variables);
  TVM_DEFINE_OBJECT_REF_METHODS(ObjectDef, MetaIR, ObjectDefNode);
};

}  // namespace meta
}  // namespace tvm
