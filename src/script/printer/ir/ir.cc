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
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_REGISTER_NODE_TYPE(IRFrameNode);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IRModule>("", [](IRModule mod, ObjectPath p, IRDocsifier d) -> Doc {
      std::vector<std::pair<GlobalVar, BaseFunc>> functions{mod->functions.begin(),
                                                            mod->functions.end()};
      // print "main" first
      std::sort(functions.begin(), functions.end(), [](const auto& lhs, const auto& rhs) {
        String lhs_name = lhs.first->name_hint;
        String rhs_name = rhs.first->name_hint;
        if (lhs_name == "main") {
          lhs_name = "";
        }
        if (rhs_name == "main") {
          rhs_name = "";
        }
        return lhs_name < rhs_name;
      });
      ICHECK(!d->mod.defined());
      d->mod = mod;
      {
        With<IRFrame> f(d);
        (*f)->AddDispatchToken(d, "ir");
        for (const auto& kv : functions) {
          GlobalVar gv = kv.first;
          BaseFunc func = kv.second;
          (*f)->stmts.push_back(d->AsDoc<FunctionDoc>(func, p->Attr("functions")->MapValue(gv)));
        }
        return ClassDoc(IdDoc("Module"), {IR(d)}, (*f)->stmts);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<DictAttrs>("", [](DictAttrs attrs, ObjectPath p, IRDocsifier d) -> Doc {
      return d->AsDoc(attrs->dict, p->Attr("dict"));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<GlobalVar>("", [](GlobalVar gv, ObjectPath p, IRDocsifier d) -> Doc {
      return IdDoc("GlobalVar")->Call({LiteralDoc::Str(gv->name_hint)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Op>("", [](Op op, ObjectPath p, IRDocsifier d) -> Doc {
      return IdDoc("Op")->Call({LiteralDoc::Str(op->name)});
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
