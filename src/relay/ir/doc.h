/*!
 *  Copyright (c) 2018 by Contributors
 * \file doc.h
 * \brief A pretty printer DSL for constructing (Doc) and formatting (RDoc) documents.
 *        It is based heavily on Philip Wadler's "A prettier printer."
 *        See https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf
 *        for more details.
 *
 * Since the original paper uses call by value for efficiency, everything doc function is maximally lazy.
 * You can probably yank speed by doing strict analysis and removing some Lazy (if this is bottleneck).
 */
#ifndef TVM_RELAY_IR_DOC_H_
#define TVM_RELAY_IR_DOC_H_

#include <tvm/relay/error.h>
#include <unordered_map>
#include <utility>
#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <ostream>
#include <map>

namespace tvm {
namespace relay {

/*! \brief A Document represent structured text.
 * beside having unstructured string, it capture different ways to compose them -
 * line break, space, indentation, representation choice.
 */
struct Doc;

/*! \brief RDoc represent rendered document.
 * all the high level detail on the document, such as indentation, choice, has been removed.
 * there is only one single, straight forward way to print it.
 */
struct RDoc;

//! \brief Empty document
inline Doc Nil();

//! \brief Concatenate two documents
inline Doc App(const Doc& l, const Doc& r);

//! \brief Indent a document
inline Doc Nest(size_t width, const Doc& doc);

//! \brief Lift string to a document
inline Doc DocOfStr(const std::string& text);

//! \brief New line
inline Doc Endl();

//! \brief Remove all line break from the Document.
inline Doc Flatten(const Doc& d);

/*! \brief Choose between two possible layouts.
 * assume Flatten(l) == Flatten(r), and l need to be more compact.
 */
inline Doc Choose(const Doc& l, const Doc& r);

//! \brief Use a single line if possible
inline Doc Group(const Doc& d);

//! \brief print an RDoc
inline std::ostream& operator<<(std::ostream& os, const RDoc& rdoc);

/*! \brief Joins a vector of documents with a given separator document
 *  \example Join(["a", "b, "c"], ", ") => "a, b, c"
 *  \param vec the vector of documents
 *  \param sep the separator between documents
 */
inline Doc Join(const std::vector<Doc>& vec, const Doc& sep);

/*! \brief Creates an indented block.
 *  \param indent the indentation size
 *  \param open the opening string
 *  \param body the body of the block
 *  \param close the closing string
 */
inline Doc Block(size_t indent, const std::string& open,
                 const Doc& body, const std::string& close);

/*! \brief Creates a comma-separated sequence with opening and closing strings.
 *  \param open the opening string
 *  \param body the body of the Block
 *  \param close the closing string
 */
inline Doc Seq(const std::string& open,
               const std::vector<Doc>& body, const std::string& close);

//! \brief Either a space or a new line
inline Doc Sep();

/*! \brief Layout a document to a given width
 *  \param d the document to render
 *  \param width the line width
 */
inline RDoc Layout(const Doc& d, size_t width = 80);

// end of API, start of implementation

template<typename T>
struct LazyNode {
  mutable std::function<T()> thunk;
  explicit LazyNode(const std::function<T()>& thunk) : thunk(thunk) { }
};

//! \brief denote a value that will be computed (at most once) on need.
template<typename T>
struct Lazy {
  std::shared_ptr<LazyNode<T> > lazy_node;
  explicit Lazy(const std::function<T()>& thunk) :
    lazy_node(std::make_shared<LazyNode<T>>(thunk)) { }
  explicit Lazy(const T& value) : Lazy([=]() { return value; }) { }
  explicit Lazy(const Lazy<Lazy<T>>& thunk) : Lazy([=]() { return thunk.get().get(); }) { }
  // calculate the result.
  // memoize it by replacing the thunk with a constant function which immediate return.
  T get() const {
    T res = lazy_node->thunk();
    lazy_node->thunk = [=]() { return res; };
    return res;
  }
  template<typename R>
  Lazy<R> map(const std::function<R(const T&)>& func) const {
    Lazy<T> self(*this);
    return Lazy<R>([=]() -> R { return func(self.get()); });
  }
};

struct NilNode;
struct AppNode;
struct NestNode;
struct TextNode;
struct LineNode;
struct ChoiceNode;

/*! \brief The inner representation of Doc.
 * a doc represent structured text,
 * and can be rendered onto screen while keeping the structure.
 */
struct DocNode {
  /* a docnode is a union of the below node.
   * exactly one of them will be non null.
   * their meaning is denoted by the construction function of the same name.
   * so for example, the meaning of AppNode is exactly a node construct by App.
   */
  std::shared_ptr<NilNode> nil;
  std::shared_ptr<AppNode> app;
  std::shared_ptr<NestNode> nest;
  std::shared_ptr<TextNode> text;  // construct by DocOfStr
  std::shared_ptr<LineNode> line;
  std::shared_ptr<ChoiceNode> choice;
  DocNode(std::shared_ptr<NilNode> nil,
           std::shared_ptr<AppNode> app,
           std::shared_ptr<NestNode> nest,
           std::shared_ptr<TextNode> text,
           std::shared_ptr<LineNode> line,
           std::shared_ptr<ChoiceNode> choice) :
    nil(nil),
    app(app),
    nest(nest),
    text(text),
    line(line),
    choice(choice) { }
};

struct Doc {
  Lazy<DocNode> doc;
  explicit Doc(const DocNode& ed) : doc(ed) { }
  explicit Doc(const Lazy<Doc>& ldoc) :
    doc(ldoc.map<Lazy<DocNode> >([](const Doc& d){ return d.doc; })) { }

  Doc operator+(const Doc& r) const {
    return App(*this, r);
  }

  template<typename T>
  Lazy<T> Match(
    const std::function<T()>& nilf,
    const std::function<T(const Doc&, const Doc&)>& appf,
    const std::function<T(size_t, const Doc&)>& nestf,
    const std::function<T(const std::string&)>& textf,
    const std::function<T()>& linef,
    const std::function<T(const Doc&, const Doc&)>& choicef) const;
};

struct NilNode { };

struct AppNode {
  Doc left, right;
  AppNode(const Doc& left, const Doc& right) : left(left), right(right) { }
};

struct NestNode {
  size_t space;
  Doc doc;
  NestNode(size_t space, const Doc& doc) : space(space), doc(doc) { }
};

struct TextNode {
  std::string text;
  explicit TextNode(const std::string& text) : text(text) { }
};

struct LineNode { };

struct ChoiceNode {
  Doc left, right;
  ChoiceNode(const Doc& left, const Doc& right) : left(left), right(right) { }
};

template<typename T>
Lazy<T> Doc::Match(
    const std::function<T()>& nilf,
    const std::function<T(const Doc&, const Doc&)>& appf,
    const std::function<T(size_t, const Doc&)>& nestf,
    const std::function<T(const std::string&)>& textf,
    const std::function<T()>& linef,
    const std::function<T(const Doc&, const Doc&)>& choicef) const {
    return doc.map<T>([=](const DocNode& d) {
      if (d.nil) {
        return nilf();
      } else if (d.app) {
        return appf(d.app->left, d.app->right);
      } else if (d.nest) {
        return nestf(d.nest->space, d.nest->doc);
      } else if (d.text) {
        return textf(d.text->text);
      } else if (d.line) {
        return linef();
      } else {
        return choicef(d.choice->left, d.choice->right);
      }
    });
}

//! \brief Empty document
inline Doc Nil() {
  return Doc(DocNode(std::make_shared<NilNode>(), nullptr, nullptr, nullptr, nullptr, nullptr));
}

//! \brief Concatenate two documents
inline Doc App(const Doc& l, const Doc& r) {
  return Doc(DocNode(
    nullptr,
    std::make_shared<AppNode>(l, r),
    nullptr,
    nullptr,
    nullptr,
    nullptr));
}

//! \brief Indent a document
inline Doc Nest(size_t width, const Doc& doc) {
  auto x = std::make_shared<NestNode>(width, doc);
  return Doc(DocNode(
    nullptr,
    nullptr,
    std::make_shared<NestNode>(width, doc),
    nullptr,
    nullptr,
    nullptr));
}

//! \brief Lift string to a document
inline Doc DocOfStr(const std::string& text) {
  return Doc(DocNode(nullptr, nullptr, nullptr,
    std::make_shared<TextNode>(text), nullptr, nullptr));
}

//! \brief New line
inline Doc Endl() {
  return Doc(DocNode(nullptr, nullptr, nullptr, nullptr, std::make_shared<LineNode>(), nullptr));
}

/*! \brief Choose between two possible layouts.
 * assume Flatten(l) == Flatten(r), and l need to be more compact.
 */
inline Doc Choose(const Doc& l, const Doc& r) {
  return Doc(DocNode(
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    std::make_shared<ChoiceNode>(l, r)));
}

//! \brief Remove new line from the whole document.
inline Doc Flatten(const Doc& d) {
  return Doc(d.Match<Doc>(
    []() { return Nil(); },
    [](const Doc& l, const Doc& r) { return Flatten(l) + Flatten(r); },
    [](size_t space, const Doc& doc) { return Flatten(doc); },
    [](const std::string& str) { return DocOfStr(str); },
    []() { return DocOfStr(" "); },
    [](const Doc& l, const Doc& r) { return Flatten(l); }));
}

//! \brief Use a single line if possible
inline Doc Group(const Doc& d) {
  return Choose(Flatten(d), d);
}

struct RNilNode;
struct RTextNode;
struct RLineNode;

struct RDocNode {
  std::shared_ptr<RNilNode> rnil;
  std::shared_ptr<RTextNode> rtext;
  std::shared_ptr<RLineNode> rline;
  RDocNode(const std::shared_ptr<RNilNode>& rnil,
           const std::shared_ptr<RTextNode>& rtext,
           const std::shared_ptr<RLineNode>& rline) :
    rnil(rnil), rtext(rtext), rline(rline) { }
};

/*! \brief RDoc represent rendered document.
 * all the high level detail on the document, such as indentation, alternative, has been removed.
 * there is only one single, straight forward way to print it.
 */
struct RDoc {
  Lazy<RDocNode> doc;
  explicit RDoc(const RDocNode& d) : doc(d) { }
  explicit RDoc(const Lazy<RDoc>& ldoc) :
    doc(ldoc.map<Lazy<RDocNode>>([](const RDoc& d){ return d.doc; })) { }
  template<typename T>
  Lazy<T> Match(
    const std::function<T()> &rnilf,
    const std::function<T(const std::string&, const RDoc&)>& rtextf,
    const std::function<T(size_t, const RDoc&)>& rlinef) const;
};

inline std::ostream& operator<<(std::ostream& os, const RDoc& rdoc) {
  return *rdoc.Match<std::ostream*>(
    [&]() { return & os; },
    [&](const std::string& text, const RDoc& r) {
      return & (os << text << r);
    },
    [&](size_t space, const RDoc& r) {
      return & (os << std::endl << std::string(space, ' ') << r);
    }).get();
}

struct RNilNode { };

struct RTextNode {
  std::string text;
  RDoc rest;
  RTextNode(const std::string& text, const RDoc& rest) : text(text), rest(rest) { }
};

struct RLineNode {
  size_t space;
  RDoc rest;
  RLineNode(size_t space, const RDoc& rest) : space(space), rest(rest) { }
};

//! \brief Empty RDoc
inline RDoc RNil() { return RDoc(RDocNode(std::make_shared<RNilNode>(), nullptr, nullptr)); }

//! \brief RDoc that begin with std::string
inline RDoc RText(const std::string& text, const RDoc& rest) {
  return RDoc(RDocNode(nullptr, std::make_shared<RTextNode>(text, rest), nullptr));
}

//! \brief RDoc that begin with a new line, followed by space
inline RDoc RLine(size_t space, const RDoc& rest) {
  return RDoc(RDocNode(nullptr, nullptr, std::make_shared<RLineNode>(space, rest)));
}

template<typename T>
Lazy<T> RDoc::Match(
  const std::function<T()>& rnilf,
  const std::function<T(const std::string&, const RDoc&)>& rtextf,
  const std::function<T(size_t, const RDoc&)>& rlinef) const {
  return doc.map<T>([=](const RDocNode& rdoc) {
    if (rdoc.rnil) {
      return rnilf();
    } else if (rdoc.rtext) {
      return rtextf(rdoc.rtext->text, rdoc.rtext->rest);
    } else {
      return rlinef(rdoc.rline->space, rdoc.rline->rest);
    }
  });
}

template<typename T>
struct List;

template<typename T>
struct EagerList {
  const std::shared_ptr<std::pair<T, List<T>>> cons;
};

//! \brief lazy list
template<typename T>
struct List {
  Lazy<EagerList<T> > l;
  List() : l([]() { return EagerList<T>({nullptr}); }) { }
  List(const T& t, const List<T>& l) :
    l([=]() { return EagerList<T>({std::make_shared<std::pair<T, List<T>>>(t, l)}); }) { }
  template<typename R>
  Lazy<R> Match(const std::function<R()>& nullf,
                const std::function<R(const T&, const List<T>&)>& consf) const {
    return l.template map<R>([=](const EagerList<T>& l) {
        if (l.cons) {
          return consf(l.cons->first, l.cons->second);
        } else {
          return nullf();
        }
    });
  }
};

//! \brief Does x fit into line of size w?
inline bool Fits(int w, const RDoc& x) {
  return (w >= 0) && x.Match<bool>(
    []() { return true; },
    [=](const std::string& s, const RDoc& x) { return Fits(w - s.size(), x); },
    [](size_t space, const RDoc& x) { return true; }).get();
}

//! \brief Choose the one that fits best.
inline RDoc Better(size_t w, size_t k, const RDoc& x, const RDoc& y) {
  return Fits(w-k, x) ? x : y;
}

typedef std::pair<size_t/*indent size*/, Doc> best_arg;
inline RDoc Best(size_t w/*wrap width*/, size_t k/*space used*/,
  const List<best_arg>& l/*to be rendered*/) {
  return RDoc(l.Match<RDoc>(
    []() { return RNil(); },
    [=](const best_arg& p, const List<best_arg>& z) {
      return RDoc(p.second.Match<RDoc>(
        [=]() { return Best(w, k, z); },
        [=](const Doc& x, const Doc& y) {
          return Best(
            w,
            k,
            List<best_arg>(best_arg(p.first, x), List<best_arg>(best_arg(p.first, y), z))); },
        [=](size_t j, const Doc& x) {
          return Best(w, k, List<best_arg>(best_arg(p.first + j, x), z)); },
        [=](const std::string& text) { return RText(text, Best(w, k + text.size(), z)); },
        [=]() { return RLine(p.first, Best(w, p.first, z)); },
        [=](const Doc& x, const Doc& y) {
          return Better(
            w,
            k,
            Best(w, k, List<best_arg>(best_arg(p.first, x), z)),
            Best(w, k, List<best_arg>(best_arg(p.first, y), z))); }));
    }));
}

/*! \brief Joins a vector of documents with a given separator document
 *  \example Join(["a", "b, "c"], ", ") => "a, b, c"
 *  \param vec the vector of documents
 *  \param sep the separator between documents
 */
inline Doc Join(const std::vector<Doc>& vec, const Doc& sep) {
  // https://www.safaribooksonline.com/library/view/c-cookbook/0596007612/ch04s09.html
  Doc output = Nil();
  for (auto p = vec.begin(); p != vec.end(); ++p) {
    output = output + *p;
    if (p != vec.end() - 1) {
      output = output + sep;
    }
  }

  return output;
}

/*! \brief Creates an indented block.
 *  \param indent the indentation size
 *  \param open the opening string
 *  \param body the body of the block
 *  \param close the closing string
 */
inline Doc Block(size_t indent, const std::string& open,
  const Doc& body, const std::string& close) {
  return DocOfStr(open) + Nest(indent, Endl() + body) + Endl() + DocOfStr(close);
}

/*! \brief Creates a comma-separated sequence with opening and closing strings.
 *  \param open the opening string
 *  \param body the body of the Block
 *  \param close the closing string
 */
inline Doc Seq(const std::string& open,
  const std::vector<Doc>& body, const std::string& close) {
  return Group(DocOfStr(open) +
               Nest(open.size(), Join(body, DocOfStr(",") + Endl())) +
               DocOfStr(close));
}

//! \brief Either a space or a new line
inline Doc Sep() {
  return Choose(DocOfStr(" "), Endl());
}

/*! \brief Layout a document to a given width
 *  \param d the document to render
 *  \param width the line width
 */
inline RDoc Layout(const Doc& d, size_t width) {
  return Best(width, 0, List<best_arg>(best_arg(0, d), List<best_arg>()));
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_IR_DOC_H_
