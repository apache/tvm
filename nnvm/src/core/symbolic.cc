/*!
 *  Copyright (c) 2016 by Contributors
 * \file symbolic.cc
 * \brief Symbolic graph composition API.
 */
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <nnvm/op_attr_types.h>

namespace nnvm {

namespace symbol_constants {
const char *kNamespaceSeparator = "$";
}  // namespace symbol_constants

// auxililary version attribute in variable.
struct VariableParam {
  uint32_t version{0};
};

NodePtr CreateVariableNode(const std::string& name) {
  NodePtr n = Node::Create();
  n->attrs.op = nullptr;
  n->attrs.name = name;
  n->attrs.parsed = VariableParam();
  return n;
}

// scan over a node's input, update the version to latest
// If the node's op mutates a certain input variable,
// The version of that varaible will increase
// version is used to implicitly order the mutation sequences
inline void UpdateNodeVersion(Node *n) {
  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  for (NodeEntry& e : n->inputs) {
    if (e.node->is_variable()) {
      e.version = nnvm::get<VariableParam>(e.node->attrs.parsed).version;
    }
  }
  if (fmutate_inputs.count(n->op()) != 0) {
    for (uint32_t i : fmutate_inputs[n->op()](n->attrs)) {
      NodeEntry& e = n->inputs[i];
      CHECK(e.node->is_variable())
          << "Mutation target can only be Variable";
      // increase the version of the variable.
      e.version = ++nnvm::get<VariableParam>(e.node->attrs.parsed).version;
    }
  }
}

inline std::string DefaultVarName(const std::string &op_name,
                                  const std::string &arg_name) {
  if (op_name.length() == 0) {
    return arg_name;
  } else {
    return op_name + '_' + arg_name;
  }
}

inline void KeywordArgumentMismatch(const char *source,
                                    const std::vector<std::string>& user_args,
                                    const array_view<std::string>& args) {
  std::unordered_set<std::string> keys(args.begin(), args.end());
  std::ostringstream head, msg;
  msg << "\nCandidate arguments:\n";
  for (size_t i = 0; i < args.size(); ++i) {
    msg << "\t[" << i << ']' << args[i] << '\n';
  }

  for (const auto& key : user_args) {
    if (keys.count(key) == 0) {
      LOG(FATAL) << source
                 << "Keyword argument name " << key << " not found."
                 << msg.str();
    }
  }
}

template<typename T>
inline std::vector<std::string> GetKeys(
    const std::unordered_map<std::string, T>& kwargs) {
  std::vector<std::string> keys(kwargs.size());
  std::transform(kwargs.begin(), kwargs.end(), keys.begin(),
                 [](decltype(*kwargs.begin())& kv) { return kv.first; });
  return keys;
}

// whether the symbol is atomic functor
inline bool IsAtomic(const std::vector<NodeEntry>& outputs) {
  return outputs[0].node->inputs.size() == 0 &&
      outputs[0].node->control_deps.size() == 0;
}

// public functions
Symbol Symbol::Copy() const {
  std::unordered_map<Node*, NodePtr> old_new;
  // use DFSVisit to copy all the nodes
  DFSVisit(this->outputs, [&old_new](const NodePtr& node) {
      NodePtr np = Node::Create();
      np->attrs = node->attrs;
      old_new[node.get()] = std::move(np);
    });
  // connect nodes of new graph
  for (const auto &kv : old_new) {
    for (const NodeEntry& e : kv.first->inputs) {
      Node *ptr = e.node.get();
      kv.second->inputs.emplace_back(NodeEntry{old_new[ptr], e.index, e.version});
    }
    for (const NodePtr& p : kv.first->control_deps) {
      kv.second->control_deps.emplace_back(old_new[p.get()]);
    }
  }
  // set the head
  Symbol ret;
  for (const NodeEntry &e : outputs) {
    ret.outputs.emplace_back(NodeEntry{old_new[e.node.get()], e.index, e.version});
  }
  return ret;
}

void Symbol::Print(std::ostream &os) const {
  if (outputs.size() == 1 &&
      outputs[0].node->inputs.size() == 0 &&
      outputs[0].node->control_deps.size() == 0) {
    if (outputs[0].node->is_variable()) {
      os << "Variable:" << outputs[0].node->attrs.name << '\n';
    } else {
      os << "AtomicFunctor "<< " Op:" << outputs[0].node->op()->name << '\n';
    }
  } else {
    // use DFSVisit to copy all the nodes
    os << "Symbol Outputs:\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
      os << "\toutput[" << i << "]=" << outputs[i].node->attrs.name
         << '(' << outputs[i].index << ")\n";
    }
    DFSVisit(this->outputs, [&os](const NodePtr& node) {
        if (node->is_variable()) {
          os << "Variable:" << node->attrs.name << '\n';
        } else {
          os << "--------------------\n";
          os << "Op:" << node->op()->name << ", Name=" << node->attrs.name << '\n'
             << "Inputs:\n";
          for (size_t i = 0; i < node->inputs.size(); ++i) {
            const NodeEntry& e = node->inputs[i];
            os << "\targ[" << i << "]=" << e.node->attrs.name
               << '(' << e.index << ")";
            if (e.node->is_variable()) {
              os << " version=" << e.version << '\n';
            } else {
              os << '\n';
            }
          }
          if (!node->attrs.dict.empty()) {
            os << "Attrs:\n";
            // make an ordered copy because unordered_map doesn't guarantee order.
            std::map<std::string, std::string> sorted_dict(
              node->attrs.dict.begin(), node->attrs.dict.end());
            for (auto &kv : sorted_dict) {
              os << '\t' << kv.first << '=' << kv.second << '\n';
            }
          }
          if (node->control_deps.size() != 0) {
            os << "Control deps:\n";
            for (size_t i = 0; i < node->control_deps.size(); ++i) {
              os << "\tcdep[" << i << "]=" << node->control_deps[i]->attrs.name << '\n';
            }
          }
        }
      });
  }
}

Symbol Symbol::operator[] (size_t index) const {
  size_t nreturn = outputs.size();
  CHECK_LT(index, nreturn) << "Symbol only accept nonnegative index";
  if (nreturn == 1) {
    return *this;
  } else {
    Symbol s;
    s.outputs.push_back(outputs[index]);
    return s;
  }
}

std::vector<NodePtr> Symbol::ListInputs(ListInputOption option) const {
  std::vector<NodePtr> ret;
  if (option == kAll) {
    DFSVisit(this->outputs, [&ret](const NodePtr &node) {
        if (node->is_variable()) {
          ret.push_back(node);
        }
      });
  } else {
    std::unordered_set<Node*> mutable_set;
    std::vector<NodePtr> vlist;
    static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
    DFSVisit(this->outputs, [&ret, &mutable_set, &vlist](const NodePtr &node) {
        if (node->is_variable()) {
          vlist.push_back(node);
        } else if (fmutate_inputs.count(node->op())) {
          for (uint32_t i : fmutate_inputs[node->op()](node->attrs)){
            mutable_set.insert(node->inputs[i].node.get());
          }
        }
      });
    for (const NodePtr& node : vlist) {
      if ((option == kReadOnlyArgs && mutable_set.count(node.get()) == 0) ||
          (option == kAuxiliaryStates && mutable_set.count(node.get()) != 0)) {
        ret.emplace_back(node);
      }
    }
  }
  return ret;
}

std::vector<std::string> Symbol::ListInputNames(ListInputOption option) const {
  std::vector<NodePtr> inputs = ListInputs(option);
  std::vector<std::string> ret(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    ret[i] = inputs[i]->attrs.name;
  }
  return ret;
}

std::vector<std::string> Symbol::ListOutputNames() const {
  static auto& flist_ouputs = Op::GetAttr<FListOutputNames>("FListOutputNames");

  std::vector<std::string> ret;
  for (auto &head : outputs) {
    if (head.node->is_variable()) {
      ret.push_back(head.node->attrs.name);
    } else {
      const std::string& hname = head.node->attrs.name;
      std::string rname;
      FListOutputNames fn = flist_ouputs.get(head.node->op(), nullptr);
      if (fn != nullptr) {
        rname = fn(head.node->attrs)[head.index];
      } else {
        rname = "output";
        if (head.node->num_outputs() != 1) {
          std::ostringstream os;
          os << rname << head.index;
          rname = os.str();
        }
      }
      if (hname.length() == 0) {
        ret.push_back(std::move(rname));
      } else {
        ret.push_back(hname + '_' + rname);
      }
    }
  }
  return ret;
}

// compositional logic
void Symbol::Compose(const array_view<const Symbol*>& args,
                     const std::unordered_map<std::string, const Symbol*>& kwargs,
                     const std::string& name) {
  static auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
  static auto& fset_attrs = Op::GetAttr<FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose");

  CHECK(!outputs[0].node->is_variable()) << "Variable cannot be composed";
  // parameter check.
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_EQ(args[i]->outputs.size(), 1U)
        << "Argument " << i << " is a tuple, single value is required";
  }
  for (const auto& kv : kwargs) {
    CHECK_EQ(kv.second->outputs.size(), 1U)
        << "Keyword Argument " << kv.first << " is a tuple, single value is required";
  }
  // assign new name
  outputs[0].node->attrs.name = name;

  // Atomic functor composition.
  if (IsAtomic(outputs)) {
    Node* n = outputs[0].node.get();
    uint32_t n_req = n->num_inputs();

    if (n_req != kVarg) {
      n->inputs.resize(n_req);
      CHECK_LE(args.size(), n_req)
          << "Incorrect number of arguments, requires " << n_req
          << ", provided " << args.size();
      for (size_t i = 0; i < args.size(); ++i) {
        n->inputs[i] = args[i]->outputs[0];
      }
      // switch to keyword argument matching
      if (args.size() != n_req) {
        FListInputNames fn = flist_inputs.get(n->op(), nullptr);
        auto arg_names = (fn == nullptr) ? std::vector<std::string>{"data"} : fn(n->attrs);
        if (arg_names.size() != n_req) {
          LOG(FATAL) << "Not enough argument to call operator " << outputs[0].node->op()->name;
        }
        size_t nmatched = 0;
        for (size_t i = args.size(); i < n_req; ++i) {
          auto it = kwargs.find(arg_names[i]);
          if (it != kwargs.end() && it->first == arg_names[i]) {
            n->inputs[i] = it->second->outputs[0];
            ++nmatched;
          } else {
            n->inputs[i] = NodeEntry{
              CreateVariableNode(DefaultVarName(name, arg_names[i])), 0, 0};
            // copy attribute of parent over automatically created variables
            n->inputs[i].node->attrs.dict = n->attrs.dict;
          }
        }

        if (nmatched != kwargs.size()) {
          n->inputs.clear();
          std::vector<std::string> keys = GetKeys(kwargs);
          array_view<std::string> view(dmlc::BeginPtr(arg_names) + args.size(),
                                       dmlc::BeginPtr(arg_names) + arg_names.size());
          KeywordArgumentMismatch("Symbol.Compose", keys, view);
        }
      }
    } else {
      CHECK_EQ(kwargs.size(), 0U) << "Variable length function do not accept kwargs";
      n->inputs.reserve(args.size());
      for (const Symbol* s : args) {
        n->inputs.push_back(s->outputs[0]);
      }
    }
    UpdateNodeVersion(n);

    FSetInputVarAttrOnCompose fn = fset_attrs.get(n->op(), nullptr);
    if (fn != nullptr) {
      for (size_t i = 0; i < n->inputs.size(); ++i) {
        if (n->inputs[i].node->is_variable()) {
          fn(n->attrs, n->inputs[i].node, i);
        }
      }
    }
  } else {
    // general composition
    CHECK_EQ(args.size(), 0U)
        << "General composition only support kwargs for now";
    size_t nmatched = 0;
    size_t arg_counter = 0;
    std::unordered_map<Node *, const NodeEntry*> replace_map;
    // replace map stores the existing replacement plan for arguments node
    auto find_replace_map = [&nmatched, &arg_counter, &args, &kwargs, &replace_map]
        (const NodePtr &node) {
      if (node->is_variable()) {
        if (arg_counter < args.size()) {
          replace_map[node.get()] = &(args[arg_counter]->outputs[0]);
          ++arg_counter;
        } else {
            // match kwargs
          auto kit = kwargs.find(node->attrs.name);
          if (kit != kwargs.end()) {
            replace_map[node.get()] = &(kit->second->outputs[0]);
            ++nmatched;
          }
        }
      }
    };
    DFSVisit(this->outputs, find_replace_map);

    if (nmatched == kwargs.size() && arg_counter <= args.size()) {
      std::vector<Node*> update_nodes;
      std::vector<std::pair<NodeEntry*, const NodeEntry*> > replace_plan;
      auto find_replace_plan = [&replace_map, &replace_plan, &update_nodes]
          (const NodePtr &node) {
        // visit all the childs, find possible replacement
        bool repl = false;
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          NodeEntry *e = &(node->inputs[i]);
          if (e->node->is_variable()) {
            auto iter = replace_map.find(e->node.get());
            if (iter != replace_map.end()) {
              replace_plan.push_back(std::make_pair(e, iter->second));
              repl = true;
            }
          }
        }
        if (repl) update_nodes.push_back(node.get());
      };
      DFSVisit(this->outputs, find_replace_plan);

      for (const auto& kv : replace_plan) {
        *(kv.first) = *(kv.second);
      }
      for (Node* n : update_nodes) {
        UpdateNodeVersion(n);
      }
    } else {
      std::vector<std::string> keys = GetKeys(kwargs);
      std::vector<std::string> arg_names = ListInputNames(kAll);
      array_view<std::string> view(dmlc::BeginPtr(arg_names) + arg_counter,
                                   dmlc::BeginPtr(arg_names) + arg_names.size());
      KeywordArgumentMismatch("Symbol.Compose", keys, arg_names);
    }
  }
}

Symbol Symbol::operator () (const array_view<const Symbol*>& args,
                            const std::unordered_map<std::string, const Symbol*>& kwargs,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(args, kwargs, name);
  return s;
}

void Symbol::AddControlDeps(const Symbol& src) {
  CHECK_EQ(outputs.size(), 1U)
      << "AddControlDeps only works for nongrouped symbol";
  Node* n = outputs[0].node.get();
  for (const NodeEntry& sp : src.outputs) {
    n->control_deps.push_back(sp.node);
  }
}

Symbol Symbol::GetInternals() const {
  static auto& fnum_vis_output = Op::GetAttr<FNumVisibleOutputs>("FNumVisibleOutputs");
  Symbol ret;
  DFSVisit(this->outputs, [&ret](const NodePtr& node) {
      Node* n = node.get();
      if (n->is_variable()) {
        // grab version from variable.
        VariableParam& param = nnvm::get<VariableParam>(n->attrs.parsed);
        ret.outputs.emplace_back(NodeEntry{node, 0, param.version});
      } else {
        uint32_t nout = n->num_outputs();
        if (fnum_vis_output.count(n->op())) {
          nout = fnum_vis_output[n->op()](n->attrs);
        }
        for (uint32_t i = 0; i < nout; ++i) {
          ret.outputs.emplace_back(NodeEntry{node, i, 0});
        }
      }
    });
  return ret;
}

Symbol Symbol::GetChildren() const {
  static auto& fnum_vis_output = Op::GetAttr<FNumVisibleOutputs>("FNumVisibleOutputs");
  Symbol ret;
  std::unordered_set<Node*> visited;
  for (const auto& p : this->outputs) {
    Node* node = p.node.get();
    if (visited.count(node)) continue;
    visited.insert(node);
    ret.outputs.insert(ret.outputs.end(), node->inputs.begin(), node->inputs.end());
  }
  return ret;
}

void Symbol::SetAttrs(const std::vector<std::pair<std::string, std::string> >& attrs) {
  Node* node = outputs[0].node.get();
  for (const NodeEntry& e : outputs) {
    CHECK(node == e.node.get())
        << "Symbol.SetAttrs only works for non-grouped symbol";
  }
  for (const auto& kv : attrs) {
    if (kv.first == "name") {
      node->attrs.name = kv.second;
    } else {
      node->attrs.dict[kv.first] = kv.second;
    }
  }
  if (node->op() != nullptr && node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
}

bool Symbol::GetAttr(const std::string& key, std::string* out) const {
  Node* node = outputs[0].node.get();
  for (const NodeEntry& e : outputs) {
    if (node != e.node.get()) return false;
  }
  if (key == "name") {
    *out = node->attrs.name;
    return true;
  }
  auto it = node->attrs.dict.find(key);
  if (it == node->attrs.dict.end()) return false;
  *out = it->second;
  return true;
}

std::unordered_map<std::string, std::string> Symbol::ListAttrs(ListAttrOption option) const {
  if (option == kRecursive) {
    std::unordered_map<std::string, std::string> ret;
    DFSVisit(this->outputs, [&ret](const NodePtr& n) {
        for (const auto& it : n->attrs.dict) {
          ret[n->attrs.name + symbol_constants::kNamespaceSeparator + it.first] = it.second;
        }
      });
    return ret;
  } else {
    return outputs[0].node->attrs.dict;
  }
}

std::vector<std::tuple<std::string, std::string, std::string> >
    Symbol::ListAttrsRecursive() const {
  std::vector<std::tuple<std::string, std::string, std::string> > ret;
  DFSVisit(this->outputs, [&ret](const NodePtr& n) {
      for (const auto& it : n->attrs.dict) {
        ret.emplace_back(std::make_tuple(n->attrs.name, it.first, it.second));
      }
    });
  return ret;
}

Symbol Symbol::CreateFunctor(const Op* op,
                             std::unordered_map<std::string, std::string> attrs) {
  static auto& fnum_vis_output = Op::GetAttr<FNumVisibleOutputs>("FNumVisibleOutputs");
  Symbol s;
  NodePtr n = Node::Create();
  n->attrs.op = op;
  n->attrs.dict = std::move(attrs);
  if (n->op()->attr_parser != nullptr) {
    n->op()->attr_parser(&(n->attrs));
  }

  uint32_t nout = n->num_outputs();
  if (fnum_vis_output.count(n->op())) {
    nout = fnum_vis_output[n->op()](n->attrs);
  }
  for (uint32_t i = 0; i < nout; ++i) {
    s.outputs.emplace_back(NodeEntry{n, i, 0});
  }
  return s;
}

Symbol Symbol::CreateFunctor(const NodeAttrs& attrs) {
  static auto& fnum_vis_output = Op::GetAttr<FNumVisibleOutputs>("FNumVisibleOutputs");
  Symbol s;
  NodePtr n = Node::Create();
  n->attrs = attrs;

  uint32_t nout = n->num_outputs();
  if (fnum_vis_output.count(n->op())) {
    nout = fnum_vis_output[n->op()](n->attrs);
  }
  for (uint32_t i = 0; i < nout; ++i) {
    s.outputs.emplace_back(NodeEntry{n, i, 0});
  }
  return s;
}

Symbol Symbol::CreateGroup(const std::vector<Symbol> &symbols) {
  Symbol ret;
  for (const auto &s : symbols) {
    ret.outputs.insert(ret.outputs.end(), s.outputs.begin(), s.outputs.end());
  }
  return ret;
}

Symbol Symbol::CreateVariable(const std::string& name) {
  Symbol s;
  s.outputs.emplace_back(NodeEntry{CreateVariableNode(name), 0, 0});
  return s;
}

}  // namespace nnvm
