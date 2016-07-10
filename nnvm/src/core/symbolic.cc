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
const char *kNamespaceSeparator = "_";
}  // namespace symbol_constants


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
  return outputs.size() == 1 && outputs[0].node->inputs.size() == 0;
}

// public functions
Symbol Symbol::Copy() const {
  std::unordered_map<Node*, std::shared_ptr<Node> > old_new;
  // use DFSVisit to copy all the nodes
  DFSVisit(this->outputs, [&old_new](const std::shared_ptr<Node>& node) {
      old_new[node.get()] =  std::make_shared<Node>(*node);
    });
  // connect nodes of new graph
  for (const auto &kv : old_new) {
    for (const NodeEntry& e : kv.first->inputs) {
      Node *ptr = e.node.get();
      kv.second->inputs.emplace_back(NodeEntry{old_new[ptr], e.index});
    }
  }
  // set the head
  Symbol ret;
  for (const NodeEntry &e : outputs) {
    ret.outputs.emplace_back(NodeEntry{old_new[e.node.get()], e.index});
  }
  return ret;
}

void Symbol::Print(std::ostream &os) const {
  if (outputs.size() == 1 && outputs[0].node->inputs.size() == 0) {
    os << "AtomicFunctor "<< " Op:" << outputs[0].node->op->name << '\n';
  } else {
    // use DFSVisit to copy all the nodes
    os << "Outputs:\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
      os << "\toutput[" << i << "]=" << outputs[i].node->attrs.name
         << '(' << outputs[i].index << ")\n";
    }
    DFSVisit(this->outputs, [&os](const std::shared_ptr<Node>& node) {
        if (node->is_variable()) {
          os << "Variable:" << node->attrs.name << '\n';
        } else {
          os << "Name: " << node->attrs.name << " Op:" << node->op->name << '\n'
             << "Inputs:\n";
          for (size_t i = 0; i < node->inputs.size(); ++i) {
            os << "\targ[" << i << "]=" << node->inputs[i].node->attrs.name
               << '(' << node->inputs[i].index << ")\n";
          }
          os << "Attrs:\n";
          for (auto &kv : node->attrs.dict) {
            os << '\t' << kv.first << '=' << kv.second << '\n';
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

std::vector<std::string> Symbol::ListArguments() const {
  std::vector<std::string> ret;
  DFSVisit(this->outputs, [&ret](const std::shared_ptr<Node> &node) {
      if (node->is_variable()) {
        ret.push_back(node->attrs.name);
      }
    });
  return ret;
}

std::vector<std::string> Symbol::ListOutputs() const {
  static auto& flist_ouputs = Op::GetAttr<FListOutputNames>("FListOutputNames");
  std::vector<std::string> ret;
  for (auto &head : outputs) {
    if (head.node->is_variable()) {
      ret.push_back(head.node->attrs.name);
    } else {
      const std::string& hname = head.node->attrs.name;
      std::string rname;
      FListOutputNames fn = flist_ouputs.get(head.node->op, nullptr);
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
void Symbol::Compose(const std::vector<Symbol>& args,
                     const std::unordered_map<std::string, Symbol>& kwargs,
                     const std::string& name) {
  CHECK_EQ(outputs.size(), 1)
      << "Only composition of value function is supported currently";
  CHECK(!outputs[0].node->is_variable()) << "Variable cannot be composed";
  // parameter check.
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_EQ(args[i].outputs.size(), 1)
        << "Argument " << i << " is a tuple, single value is required";
  }
  for (const auto& kv : kwargs) {
    CHECK_EQ(kv.second.outputs.size(), 1)
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
        n->inputs[i] = args[i].outputs[0];
      }
      // switch to keyword argument matching
      if (args.size() != n_req) {
        static auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
        FListInputNames fn = flist_inputs.get(n->op, nullptr);
        auto arg_names = (fn == nullptr) ? std::vector<std::string>{"data"} : fn(n->attrs);
        CHECK_EQ(arg_names.size(), n_req);

        size_t nmatched = 0;
        for (size_t i = args.size(); i < n_req; ++i) {
          auto it = kwargs.find(arg_names[i]);
          if (it != kwargs.end() && it->first == arg_names[i]) {
            n->inputs[i] = it->second.outputs[0];
            ++nmatched;
          } else {
            n->inputs[i] = NodeEntry{Node::Create(), 0};
            n->inputs[i].node->attrs.name = DefaultVarName(name, arg_names[i]);
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
      CHECK_EQ(kwargs.size(), 0) << "Variable length function do not accept kwargs";
      n->inputs.reserve(args.size());
      for (const Symbol& s : args) {
        n->inputs.push_back(s.outputs[0]);
      }
    }
  } else {
    // general composition
    CHECK_EQ(args.size(), 0)
        << "General composition only support kwargs for now";
    size_t nmatched = 0;
    size_t arg_counter = 0;
    std::unordered_map<Node *, const NodeEntry*> replace_map;
    // replace map stores the existing replacement plan for arguments node
    auto find_replace_map = [&nmatched, &arg_counter, &args, &kwargs, &replace_map]
        (const std::shared_ptr<Node> &node) {
      if (node->is_variable()) {
        if (arg_counter < args.size()) {
          replace_map[node.get()] = &(args[arg_counter].outputs[0]);
          ++arg_counter;
        } else {
            // match kwargs
          auto kit = kwargs.find(node->attrs.name);
          if (kit != kwargs.end()) {
            replace_map[node.get()] = &(kit->second.outputs[0]);
            ++nmatched;
          }
        }
      }
    };
    DFSVisit(this->outputs, find_replace_map);

    if (nmatched == kwargs.size() && arg_counter < args.size()) {
      std::vector<std::pair<NodeEntry*, const NodeEntry*> > replace_plan;
      auto find_replace_plan = [&replace_map, &replace_plan]
          (const std::shared_ptr<Node> &node) {
        // visit all the childs, find possible replacement
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          NodeEntry *e = &(node->inputs[i]);
          if (e->node->is_variable()) {
            auto iter = replace_map.find(e->node.get());
            if (iter != replace_map.end()) {
              replace_plan.push_back(std::make_pair(e, iter->second));
            }
          }
        }
      };
      DFSVisit(this->outputs, find_replace_plan);

      for (const auto& kv : replace_plan) {
        *(kv.first) = *(kv.second);
      }
    } else {
      std::vector<std::string> keys = GetKeys(kwargs);
      std::vector<std::string> arg_names = ListArguments();
      array_view<std::string> view(dmlc::BeginPtr(arg_names) + arg_counter,
                                   dmlc::BeginPtr(arg_names) + arg_names.size());
      KeywordArgumentMismatch("Symbol.Compose", keys, ListArguments());
    }
  }
}

Symbol Symbol::operator () (const std::vector<Symbol>& args,
                            const std::unordered_map<std::string, Symbol>& kwargs,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(args, kwargs, name);
  return s;
}

void Symbol::AddControlDeps(const Symbol& src) {
  CHECK_EQ(outputs.size(), 1)
      << "AddControlDeps only works for nongrouped symbol";
  Node* n = outputs[0].node.get();
  for (const NodeEntry& sp : src.outputs) {
    n->control_deps.push_back(sp.node);
  }
}

Symbol Symbol::GetInternals() const {
  Symbol ret;
  DFSVisit(this->outputs, [&ret](const std::shared_ptr<Node>& node) {
      Node* n = node.get();
      uint32_t nout = n->num_outputs();
      for (uint32_t i = 0; i < nout; ++i) {
        ret.outputs.emplace_back(NodeEntry{node, i});
      }
    });
  return ret;
}

void Symbol::SetAttrs(const std::vector<std::pair<std::string, std::string> >& attrs) {
  CHECK_EQ(outputs.size(), 1)
      << "SetAttrs only works for nongrouped symbol";
  Node* n = outputs[0].node.get();
  for (const auto& kv : attrs) {
    n->attrs.dict[kv.first] = kv.second;
  }
  if (n->op->attr_parser != nullptr) {
    (*n->op->attr_parser)(&(n->attrs));
  }
}

std::unordered_map<std::string, std::string> Symbol::ListAttrs(ListAttrOption option) const {
  if (option == kRecursive) {
    std::unordered_map<std::string, std::string> ret;
    DFSVisit(this->outputs, [&ret](const std::shared_ptr<Node>& n) {
        for (const auto& it : n->attrs.dict) {
          ret[n->attrs.name + symbol_constants::kNamespaceSeparator + it.first] = it.second;
        }
      });
    return ret;
  } else {
    return outputs[0].node->attrs.dict;
  }
}

Symbol Symbol::CreateFunctor(const Op* op,
                             std::unordered_map<std::string, std::string>&& attrs) {
  Symbol s;
  std::shared_ptr<Node> n = Node::Create();
  n->op = op;
  n->attrs.dict = std::move(attrs);
  if (n->op->attr_parser != nullptr) {
    (*n->op->attr_parser)(&(n->attrs));
  }
  s.outputs.emplace_back(NodeEntry{std::move(n), 0});
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
  std::shared_ptr<Node> n = Node::Create();
  n->op = nullptr;
  n->attrs.name = name;
  s.outputs.emplace_back(NodeEntry{std::move(n), 0});
  return s;
}

}  // namespace nnvm
