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
#ifndef TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
#define TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_

#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

//////////////////////// Frame ////////////////////////

class IRDocsifier;
class IRDocsifierNode;

/*!
 * Frame is the core data structure for semantic information
 * when printing IR graph into TVMScript code.
 */
class FrameNode : public Object {
 public:
  /*! The docs generated in the frame */
  Array<StmtDoc> stmts;
  /*! The corresponding IRDocsifier */
  IRDocsifierNode* d;
  /*! The callbacks that are going to be invoked when the frame exits */
  std::vector<std::function<void()>> callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("stmts", &stmts);
    // `d` is not visited
    // `callbacks` is not visited
  }

  static constexpr const char* _type_key = "script.printer.Frame";
  TVM_DECLARE_BASE_OBJECT_INFO(FrameNode, Object);

 public:
  virtual ~FrameNode() = default;

  /*!
   * \brief Add a callback function to be called when this frame exits.
   * \param cb The callback function. It should have signature void().
   */
  template <typename TCallback>
  void AddExitCallback(TCallback&& cb) {
    callbacks.emplace_back(std::forward<TCallback>(cb));
  }
  /*!
   * \brief Add a dispatch token to the docsifier, and a callback that pops the token when this
   * frame exits.
   * \param d The docsifier.
   * \param token The token to be added.
   */
  void AddDispatchToken(const IRDocsifier& d, const String& token);
  /*!
   * \brief Method that's called when Frame enters the scope.
   */
  virtual void EnterWithScope();
  /*!
   * \brief Method that's called when Frame exits the scope.
   */
  virtual void ExitWithScope();
};

/*!
 * \brief Reference type of FrameNode
 */
class Frame : public ObjectRef {
 protected:
  Frame() = default;

 public:
  virtual ~Frame() = default;

  /*! \brief Method that's called when Frame enters the scope. */
  void EnterWithScope() { get()->EnterWithScope(); }

  /*! \brief Method that's called when Frame exits the scope. */
  void ExitWithScope() { get()->ExitWithScope(); }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Frame, ObjectRef, FrameNode);
};

//////////////////////// IRDocsifier ////////////////////////

/*!
 * \brief IRDocsifier is the top-level interface in the IR->Doc process.
 *
 * It provides methods to convert IR node object to Doc, operate on Frame
 * objects and change dispatch tokens.
 */
class IRDocsifierNode : public Object {
 public:
  /*! \brief A function that creates the doc for a variable */
  using DocCreator = std::function<ExprDoc()>;
  /*! \brief Information about a variable, including its optional name and its doc creator */
  struct VariableInfo {
    /*! \brief The creator */
    DocCreator creator;
    /*! \brief The name of the variable */
    Optional<String> name;
  };
  /*! \brief The configuration of the printer */
  PrinterConfig cfg{nullptr};
  /*!
   * \brief The stack of frames.
   * \sa FrameNode
   */
  Array<Frame> frames;
  /*!
   * \brief The stack of dispatch tokens.
   *
   * The dispatch token on the top decides which dispatch function to use
   * when converting IR node object to Doc.
   */
  Array<String> dispatch_tokens;
  /*! \brief Mapping from a var to its info */
  std::unordered_map<ObjectRef, VariableInfo, ObjectPtrHash, ObjectPtrEqual> obj2info;
  /*! \brief Metadata printing */
  std::unordered_map<String, Array<ObjectRef>> metadata;
  /*! \brief GlobalInfo printing */
  std::unordered_map<String, Array<GlobalInfo>> global_infos;
  /*! \brief The variable names used already */
  std::unordered_set<String> defined_names;
  /*! \brief Common prefixes of variable usages */
  std::unordered_map<const Object*, std::vector<const Object*>> common_prefix;
  /*! \brief The IR usages for headers printing */
  std::unordered_set<std::string> ir_usage;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("frames", &frames);
    v->Visit("dispatch_tokens", &dispatch_tokens);
    // `obj2info` is not visited
    // `metadata` is not visited
    // `defined_names` is not visited
    // `common_prefix` is not visited
    // `ir_usage` is not visited
  }

  static constexpr const char* _type_key = "script.printer.IRDocsifier";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRDocsifierNode, Object);

 public:
  /*!
   * \brief Define variable by name.
   * \param obj The variable object.
   * \param frame The frame that this variable is defined in.
   * \param name_hint The hint for variable name.
   *
   * \return The id doc for this variable.
   *
   * This function will rename the variable to avoid name conflict with other variables
   * in the table.
   */
  IdDoc Define(const ObjectRef& obj, const Frame& frame, const String& name_hint);

  /*!
   * \brief Define variable by doc factory.
   * \param obj The variable object.
   * \param frame The frame that this variable is defined in.
   * \param doc_factory The function to return an ExprDoc object for this variable.
   *
   * This function is a special form of `Define`. Variable is mapped to ExprDoc rather
   * than IdDoc. It's useful when a variable is implicitly defined without a name, like
   * the buf->data in TIR, which should be mapped to `AttrDoc(IdDoc("<buffer_name>"), "data")`.
   *
   * This function takes a DocFactory instead of Doc. It's because GetVarDoc needs to
   * return a new Doc object every time it's called, as the returned doc will have
   * different `source_path`. Currently there isn't a good way to deep copy a TVMObject
   * so VarTable needs to call a factory function to get a freshly-constructed Doc object
   * every time GetVarDoc is called.
   */
  void Define(const ObjectRef& obj, const Frame& frame, DocCreator doc_factory);

  /*!
   * \brief Get the doc for variable.
   * \param obj The variable object.
   *
   * \return The doc for variable, if it exists in the table. Otherwise it returns NullOpt.
   */
  Optional<ExprDoc> GetVarDoc(const ObjectRef& obj) const;
  /*! \brief Add a TVM object to the metadata section*/
  ExprDoc AddMetadata(const ObjectRef& obj);
  /*! \brief Add a GlobalInfo to the global_infos map.
   * \param name The name of key of global_infos.
   * \param ginfo The GlobalInfo to be added.
   */
  void AddGlobalInfo(const String& name, const GlobalInfo& ginfo);
  /*!
   * \brief Check if a variable exists in the table.
   * \param obj The variable object.
   *
   * \return a boolean for whether variable exists.
   */
  bool IsVarDefined(const ObjectRef& obj) const;
  /*! \brief Remove the variable defined */
  void RemoveVar(const ObjectRef& obj);
  /*!
   * \brief Set the common prefix information of variable usage.
   * \param root The root of the AST.
   * \param is_var A function that returns true if the given object is considered a variable.
   */
  void SetCommonPrefix(const ObjectRef& root, runtime::TypedPackedFunc<bool(ObjectRef)> is_var);
  /*!
   * \brief Transform the input object into TDoc.
   * \param obj The object to be transformed.
   * \param path The path to this object.
   *
   * \return The Doc object.
   */
  template <class TDoc = Doc>
  inline TDoc AsDoc(const ObjectRef& obj, const ObjectPath& path) const;
};

/*!
 * \brief Reference type of IRDocsifierNode.
 */
class IRDocsifier : public ObjectRef {
 public:
  using FType = IRDocsifierFunctor<printer::Doc, ObjectPath, IRDocsifier>;
  /*! \brief Create a IRDocsifier. */
  explicit IRDocsifier(const PrinterConfig& cfg);
  /*! \brief The registration table for IRDocsifier. */
  TVM_DLL static FType& vtable();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRDocsifier, ObjectRef, IRDocsifierNode);
};

//////////////////////// Implementation ////////////////////////

inline void FrameNode::EnterWithScope() {
  if (d != nullptr) {
    d->frames.push_back(GetRef<Frame>(this));
  }
}

inline void FrameNode::ExitWithScope() {
  for (const std::function<void()>& callback : callbacks) {
    callback();
  }
  callbacks.clear();
  if (d != nullptr) {
    d->frames.pop_back();
  }
}

template <class TDoc>
inline static void AddDocDecoration(const Doc& d, const ObjectRef& obj, const ObjectPath& path,
                                    const PrinterConfig& cfg) {
  if (cfg->obj_to_annotate.count(obj)) {
    if (const auto* stmt = d.as<StmtDocNode>()) {
      if (stmt->comment.defined()) {
        stmt->comment = stmt->comment.value() + "\n" + cfg->obj_to_annotate.at(obj);
      } else {
        stmt->comment = cfg->obj_to_annotate.at(obj);
      }
    } else {
      LOG(WARNING) << "Expect StmtDoc to be annotated for object " << obj << ", but got "
                   << Downcast<TDoc>(d)->_type_key;
    }
  }
  for (const ObjectRef& o : cfg->obj_to_underline) {
    if (o.same_as(obj)) {
      cfg->path_to_underline.push_back(path);
    }
  }
  for (const auto& pair : cfg->path_to_annotate) {
    ObjectPath p = pair.first;
    String attn = pair.second;
    if (p->IsPrefixOf(path) && path->IsPrefixOf(p)) {
      if (const auto* stmt = d.as<StmtDocNode>()) {
        if (stmt->comment.defined()) {
          stmt->comment = stmt->comment.value() + "\n" + attn;
        } else {
          stmt->comment = attn;
        }
      } else {
        LOG(WARNING) << "Expect StmtDoc to be annotated at object path " << p << ", but got "
                     << Downcast<TDoc>(d)->_type_key;
      }
    }
  }
}

template <class TDoc>
inline TDoc IRDocsifierNode::AsDoc(const ObjectRef& obj, const ObjectPath& path) const {
  if (obj.defined()) {
    Doc d = IRDocsifier::vtable()(dispatch_tokens.back(), obj, path, GetRef<IRDocsifier>(this));
    d->source_paths.push_back(path);
    AddDocDecoration<TDoc>(d, obj, path, cfg);
    return Downcast<TDoc>(d);
  }
  return Downcast<TDoc>(LiteralDoc::None(path));
}

inline void FrameNode::AddDispatchToken(const IRDocsifier& d, const String& token) {
  d->dispatch_tokens.push_back(token);
  this->AddExitCallback([doc = d.get()]() { doc->dispatch_tokens.pop_back(); });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_H_
