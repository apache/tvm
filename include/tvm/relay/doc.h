/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/doc.h
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#ifndef TVM_RELAY_DOC_H_
#define TVM_RELAY_DOC_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

// ADT
class Doc;

class DocNode : public Node {
  public:
    static constexpr const char* _type_key = "Doc";
    TVM_DECLARE_BASE_NODE_INFO(DocNode, Node);
};

class Doc : public NodeRef {
  public:
    Doc() {}
    explicit Doc(NodePtr<Node> n) : NodeRef(n) {}
    const DocNode* operator->() const {
      return static_cast<const DocNode*>(node_.get());
    }

    using ContainerType = DocNode;
};

class Nil_;

class Nil_Node : public DocNode {
  public:
    Nil_Node() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {}

    TVM_DLL static Nil_ make();

    static constexpr const char* _type_key = "Nil_";
    TVM_DECLARE_NODE_TYPE_INFO(Nil_Node, DocNode);
};

RELAY_DEFINE_NODE_REF(Nil_, Nil_Node, Doc);

class Text_;

class Text_Node : public DocNode {
  public:
    std::string str;
    Doc doc;

    Text_Node() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("str", &str);
      v->Visit("doc", &doc);
    }

    TVM_DLL static Text_ make(std::string str, Doc doc);

    static constexpr const char* _type_key = "Text_";
    TVM_DECLARE_NODE_TYPE_INFO(Text_Node, DocNode);
};

RELAY_DEFINE_NODE_REF(Text_, Text_Node, Doc);

class Line_;

class Line_Node : public DocNode {
  public:
    int indent;
    Doc doc;

    Line_Node() {}

    void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("indent", &indent);
      v->Visit("doc", &doc);
    }

    TVM_DLL static Line_ make(int indent, Doc doc);

    static constexpr const char* _type_key = "Line_";
    TVM_DECLARE_NODE_TYPE_INFO(Line_Node, DocNode);
};

RELAY_DEFINE_NODE_REF(Line_, Line_Node, Doc);

// DSL functions

// empty doc
Doc Nil();
// lift string to text
Doc Text(const std::string str);
// new line
Doc Line();
// concat two docs
Doc Concat(const Doc& left, const Doc& right);
// sugar for Concat
Doc operator+(const Doc& left, const Doc& right);
// indent a doc
Doc Nest(int indent, const Doc& doc);
// convert doc to a string
std::string Layout(const Doc& doc);
// render array-like things: e.g. (1, 2, 3)
Doc PrintArray(const Doc& open, const tvm::Array<Doc>& arr, const Doc& sep, const Doc& close);
// Print constant bool value.
Doc PrintBool(bool value);
// special method to print out const scalar
template<typename T>
Doc PrintConstScalar(DataType dtype, const T* data);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DOC_H_
