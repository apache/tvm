/*!
 *  Copyright (c) 2018 by Contributors
 * \file  environment.cc
 * \brief The global environment in Relay.
 */
#include <sstream>
#include "tvm/relay/environment.h"
// #include "tvm/relay/util/rang.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

Environment EnvironmentNode::make(
      tvm::Map<GlobalVar, Function> global_funcs) {
  std::shared_ptr<EnvironmentNode> n = std::make_shared<EnvironmentNode>();
  n->functions = std::move(global_funcs);
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

/*! \brief Add a new item to the global environment
 * \note if the update flag is not set adding a duplicate
 * definition will trigger an exception, otherwise we will
 * update the definition if and only if it is type compatible.
 */
void EnvironmentNode::Add(const GlobalVar& var, const Function & func, bool update) {
  throw Error("NYI");
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
}

void EnvironmentNode::Update(const GlobalVar& var, const Function & func) {
  this->Add(var, func, true);
}

void EnvironmentNode::Remove(const GlobalVar&) { 
  // Clarify with @tqchen about how to use COW to do this.
  throw Error("NYI");
  // this->items.erase(id); 
}

Function EnvironmentNode::Lookup(const GlobalVar &var) {
  auto func = functions.find(var);
  if (func != functions.end()) {
    return (*func).second;
  } else {
    throw Error(std::string("there is no definition of ") + var->name_hint);
  }
}

Function EnvironmentNode::Lookup(const std::string &str) {
  GlobalVar id = this->GetGlobalVar(str);
  return this->Lookup(id);
}

inline SourceName EnvironmentNode::AddSource(std::string file_name,
                                          std::string source) {
  throw Error("need to restore error handling");
  // return this->source_map_.add_source(file_name, source);
}

void EnvironmentNode::ReportError(std::string msg, Span sp) {
  throw Error("need to restore error handling");
  // this->errors_.push_back(Error(msg, sp));
}

void EnvironmentNode::DisplayErrors() {
  throw Error("need to restore error printing");
  // for (auto err : this->errors_) {
  //   auto sp = err.sp;
  //   auto source_file = this->source_map_.GetSource(err.sp->file_id);
  //   auto file_name = source_file.file_name;
  //   auto source_at_span = source_file.SourceAt(err.sp, 1);
  //   std::string error_marker = "error:";
  //   auto line_info =
  //       std::to_string(sp->lineno) + ":" + std::to_string(sp->col_offset);

  //   std::cout << rang::style::bold << rang::fg::red << error_marker
  //             << rang::fg::reset << file_name << ":" << line_info
  //             << rang::style::reset << " " << source_at_span << std::endl;

  //   // Build the cursor.

  //   // Fix this code, hardwired to compute alignment of pointer.
  //   size_t spaces = error_marker.size() + line_info.size() + file_name.size() +
  //                   sp->col_offset - 3;

  //   std::string cursor = "~~~~^~~~~";
  //   for (size_t i = 0; i < spaces; i++) {
  //     std::cout << " ";
  //   }
  //   std::cout << rang::fg::red << cursor << " " << err.msg << rang::style::reset
  //             << std::endl;
  // }
}

TVM_REGISTER_API("relay._make.Environment")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = EnvironmentNode::make(args[0]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<EnvironmentNode>([](const EnvironmentNode *node,
                                      tvm::IRPrinter *p) {
      p->stream << "EnvironmentNode( " << node->functions << ")";
    });

}  // namespace relay
}  // namespace tvm
