/*!
 *  Copyright (c) 2018 by Contributors
 * \file  environment.cc
 * \brief The global environment in Relay.
 */
#include <sstream>
#include "tvm/relay/compiler/environment.h"
// #include "tvm/relay/alpha_eq.h"
// #include "tvm/relay/debug.h"
// #include "tvm/relay/typeck/typechecker.h"
// #include "tvm/relay/util/rang.h"
// #include "tvm/runtime/packed_func_ext.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

Environment EnvironmentNode::make(
      std::unordered_map<GlobalVar, Function, NodeHash> global_funcs) {
  std::shared_ptr<EnvironmentNode> n = std::make_shared<EnvironmentNode>();
  n->items = std::move(global_funcs);
  return Environment(n);
}

GlobalVar EnvironmentNode::GetGlobalVar(const std::string &str) {
  auto global_id = global_map_.find(str);
  if (global_id != global_map_.end()) {
      return (*global_id).second;
  } else {
      auto id = GlobalVarNode::make(str);
      this->global_map_.Set(str, id);
      return id;
  }
}

// // Add a new item to the global environment
// // throws an exception if the item already
// // exists.
// void EnvironmentNode::add(const Item &unchecked_item, bool update) {
//   // Type check the item before we add it to the environment.
//   auto env = GetRef<Environment>(this);
//   Item item = check(env, unchecked_item);

//   if (const OperatorNode *op_node = item.as<OperatorNode>()) {
//     Operator op = GetRef<Operator>(op_node);
//     auto type = op->type;
//     if (operators.find(op->id) != operators.end()) {
//       if (!update) {
//         throw dmlc::Error("already have definition for XXXX.");
//       }

//       auto old_type = operators[op->id]->type;

//       if (!alpha_eq(type, old_type)) {
//         throw dmlc::Error(
//             "Environment#update changes type, not possible in this mode.");
//       }

//       operators.insert({op->id, op});
//     } else {
//       operators.insert({op->id, op});
//     }
//   } else if (const FunctionNode *d = item.as<FunctionNode>()) {
//     auto def = GetRef<Function>(d);
//     auto type = def->type;
//     if (items.find(def->id) != items.end()) {
//       if (!update) {
//         throw dmlc::Error("already have definition for XXXX.");
//       }

//       auto old_type = items[def->id].as<FunctionNode>()->type;

//       if (!alpha_eq(type, old_type)) {
//         throw dmlc::Error(
//             "Environment#update changes type, not possible in this mode.");
//       }

//       this->items.insert({def->id, def});
//     } else {
//       this->items.insert({def->id, def});
//     }
//   } else {
//     throw EnvError("internal error: unknown item type, unreachable code");
//   }
// }

// void EnvironmentNode::update(const Item &item) { return this->add(item, true); }

// void EnvironmentNode::remove(const GlobalId &id) { this->items.erase(id); }

Function EnvironmentNode::Lookup(const GlobalVar &var) {
  if (items.find(var) != items.end()) {
    return items.at(var);
  } else {
    throw Error(std::string("there is no definition of ") + var->name_hint);
  }
}

Function EnvironmentNode::Lookup(const std::string &str) {
  GlobalVar id = this->GetGlobalVar(str);
  return this->Lookup(id);
}

// inline FileId EnvironmentNode::add_source(std::string file_name,
//                                           std::string source) {
//   return this->source_map_.add_source(file_name, source);
// }

// void EnvironmentNode::report_error(std::string msg, Span sp) {
//   this->errors_.push_back(Error(msg, sp));
// }

// void EnvironmentNode::display_errors() {
//   for (auto err : this->errors_) {
//     auto sp = err.sp;
//     auto source_file = this->source_map_.GetSource(err.sp->file_id);
//     auto file_name = source_file.file_name;
//     auto source_at_span = source_file.SourceAt(err.sp, 1);
//     std::string error_marker = "error:";
//     auto line_info =
//         std::to_string(sp->lineno) + ":" + std::to_string(sp->col_offset);

//     std::cout << rang::style::bold << rang::fg::red << error_marker
//               << rang::fg::reset << file_name << ":" << line_info
//               << rang::style::reset << " " << source_at_span << std::endl;

//     // Build the cursor.

//     // Fix this code, hardwired to compute alignment of pointer.
//     size_t spaces = error_marker.size() + line_info.size() + file_name.size() +
//                     sp->col_offset - 3;

//     std::string cursor = "~~~~^~~~~";
//     for (size_t i = 0; i < spaces; i++) {
//       std::cout << " ";
//     }
//     std::cout << rang::fg::red << cursor << " " << err.msg << rang::style::reset
//               << std::endl;
//   }
// }

TVM_REGISTER_API("relay._make.Environment")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = EnvironmentNode::make({});
    });

// TVM_REGISTER_API("relay._env.Environment_add")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       Item item = args[1];
//       env->add(item, true);  // REMOVE ME
//     });

// TVM_REGISTER_API("relay._env.Environment_lookup_global")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       GlobalId id = args[1];
//       *ret = env->lookup(id);
//     });

// TVM_REGISTER_API("relay._env.Environment_lookup_operator")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       OperatorId id = args[1];
//       *ret = env->lookup(id);
//     });

// // TVM_REGISTER_API("relay._env.Environment_remove_global")
// //     .set_body([](TVMArgs args, TVMRetValue *ret) {
// //       Environment env = args[0];
// //       GlobalId id = args[1];
// //       env->remove(id);
// //     });

// TVM_REGISTER_API("relay._env.Environment_global_id")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       std::string str = args[1];
//       *ret = env->global_id(str);
//     });

// TVM_REGISTER_API("relay._env.Environment_operator_id")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       std::string str = args[1];
//       *ret = env->operator_id(str);
//     });

// TVM_REGISTER_API("relay._env.Environment_register_shape_ext")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       ShapeExtension ext = args[1];
//       env->register_shape_ext(ext);
//     });

// TVM_REGISTER_API("relay._env.Environment_register_primitive")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       std::string str = args[1];
//       *ret = env->global_id(str);
//     });

// TVM_REGISTER_API("relay._env.Environment_add_source")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       std::string file_name = args[1];
//       std::string source_name = args[2];
//       *ret = env->add_source(file_name, source_name);
//     });

// TVM_REGISTER_API("relay._env.Environment_report_error")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       std::string msg = args[1];
//       Span sp = args[2];
//       env->report_error(msg, sp);
//     });

// TVM_REGISTER_API("relay._env.Environment_display_errors")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       return env->display_errors();
//     });

// TVM_REGISTER_API("relay._env.Environment_get_operators")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       *ret = env->get_operators();
//     });

// TVM_REGISTER_API("relay._env.Environment_get_defns")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       *ret = env->get_defns();
//     });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<EnvironmentNode>([](const EnvironmentNode *node,
                                      tvm::IRPrinter *p) {
      p->stream << "EnvironmentNode(todo)";  // << node->items << ")";
    });

}  // namespace relay
}  // namespace tvm
