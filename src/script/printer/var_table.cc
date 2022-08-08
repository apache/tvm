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

#include <tvm/node/object_path.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/var_table.h>

namespace tvm {
namespace script {
namespace printer {

String GenerateUniqueName(const String& name_hint, std::unordered_set<String>* defined_names) {
  String name = name_hint;
  for (int i = 1; !defined_names->insert(name).second; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

IdDoc VarTableNode::Define(const ObjectRef& obj, const String& name_hint,
                           const ObjectPath& object_path, const Frame& frame) {
  String name = GenerateUniqueName(name_hint, &this->defined_names);
  DocFactory doc_factory = [name]() { return IdDoc(name); };

  auto result = obj2info.insert({obj, VariableInfo{std::move(doc_factory), name}});
  ICHECK(result.second) << "Duplicated object: " << obj;

  IdDoc def_doc(name);
  def_doc->source_paths.push_back(object_path);

  frame->AddExitCallback([this, obj]() { this->RemoveVar(obj); });

  return def_doc;
}

void VarTableNode::DefineByDoc(const ObjectRef& obj, DocFactory doc_factory, const Frame& frame) {
  ICHECK(obj2info.find(obj) == obj2info.end()) << "Duplicated object: " << obj;

  ICHECK(!doc_factory()->IsInstance<IdDocNode>())
      << "VarTableNode::Define cannot be used for variable that's mapped to IdDoc.";

  obj2info.insert({obj, VariableInfo{std::move(doc_factory), NullOpt}});

  frame->AddExitCallback([this, obj]() { this->RemoveVar(obj); });
}

Optional<ExprDoc> VarTableNode::GetVarDoc(const ObjectRef& obj,
                                          const ObjectPath& object_path) const {
  auto it = obj2info.find(obj);
  if (it == obj2info.end()) {
    return NullOpt;
  }
  ExprDoc doc = it->second.doc_factory();
  doc->source_paths.push_back(object_path);
  return doc;
}

bool VarTableNode::IsVarDefined(const ObjectRef& obj) const { return obj2info.count(obj); }

void VarTableNode::RemoveVar(const ObjectRef& obj) {
  auto it = obj2info.find(obj);
  ICHECK(it != obj2info.end()) << "No such object: " << obj;

  if (it->second.name.defined()) {
    defined_names.erase(it->second.name.value());
  }
  obj2info.erase(it);
}

VarTable::VarTable() { data_ = make_object<VarTableNode>(); }

TVM_REGISTER_NODE_TYPE(VarTableNode);
TVM_REGISTER_GLOBAL("script.printer.VarTable").set_body_typed([]() { return VarTable(); });
TVM_REGISTER_GLOBAL("script.printer.VarTableDefine")
    .set_body_method<VarTable, VarTableNode, IdDoc, const ObjectRef&, const String&,
                     const ObjectPath&, const Frame&>(&VarTableNode::Define);
TVM_REGISTER_GLOBAL("script.printer.VarTableDefineByDoc")
    .set_body_typed([](VarTable var_table, const ObjectRef& obj, runtime::PackedFunc factory,
                       Frame frame) {
      var_table->DefineByDoc(
          obj, [f = std::move(factory)]() { return f(); }, frame);
    });
TVM_REGISTER_GLOBAL("script.printer.VarTableGetVarDoc")
    .set_body_method<VarTable>(&VarTableNode::GetVarDoc);
TVM_REGISTER_GLOBAL("script.printer.VarTableIsVarDefined")
    .set_body_method<VarTable>(&VarTableNode::IsVarDefined);

}  // namespace printer
}  // namespace script
}  // namespace tvm
