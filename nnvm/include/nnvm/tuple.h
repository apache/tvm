/*!
 *  Copyright (c) 2016 by Contributors
 * \file tuple.h
 * \brief Data structure Tuple and TShape to store dynamic sized shapes.
 */
#ifndef NNVM_TUPLE_H_
#define NNVM_TUPLE_H_

#include <vector>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <iostream>
#include "./base.h"

namespace nnvm {

/*! \brief data type to store array index */
typedef uint32_t index_t;

/*!
 * \brief A dynamic sized array data strcuture
 *  that is optimized for storing small number of elements with same type.
 *  Data will be stored in stack when number of elements is small.
 *
 *  It is suitable to hold Shape of Tensor.
 *
 * \tparam ValueType The type of data stored inside tuple.
 * \sa TShape
 */
template<typename ValueType>
class Tuple {
 public:
  // Tuple requires the content to be simple data type.
  static_assert(std::is_pod<ValueType>::value,
                "Tuple only support simple data type like int");
  /*! \brief default constructor */
  Tuple() = default;
  /*! \brief destructor */
  inline ~Tuple() {
    delete [] data_heap_;
  }
  /*!
   * \brief copy constructor from another tuple
   * \param s the source tuple
   */
  inline Tuple(const Tuple<ValueType>& s) {
    this->assign(s.begin(), s.end());
  }
  /*!
   * \brief constructor from initializer list
   * \param init the initializer_list
   */
  inline Tuple(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
  }
  /*!
   * \brief move constructor from Tuple
   * \param src the source shape
   */
  inline Tuple(Tuple<ValueType>&& src) {
    this->swap(src);
  }
  /*!

   * \param ndim the number of dimension of the Tuple
   * \param v The value to fill.
   */
  inline Tuple(index_t ndim, ValueType v) {
    this->SetDim(ndim);
    std::fill_n(begin(), ndim, v);
  }
  /*!
   * \brief construct the Tuple from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline Tuple(RandomAccessIterator begin,
               RandomAccessIterator end) {
    this->assign(begin, end);
  }
  /*!
   * \brief Assign content to tuple from iterator.
   * \param begin the beginning of iteratro
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline void assign(RandomAccessIterator begin,
                     RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, this->begin());
  }
  /*!
   * \brief Swap current object with other
   * \param other another object to be swapped.
   */
  inline void swap(Tuple<ValueType>& other) noexcept {  // NOLINT(*)
    std::swap(ndim_, other.ndim_);
    std::swap(num_heap_allocated_, other.num_heap_allocated_);
    std::swap(data_stack_, other.data_stack_);
    std::swap(data_heap_, other.data_heap_);
  }
  /*!
   * \brief assignment from another tuple.
   * \param src source tuple
   * \return reference of self
   */
  inline Tuple<ValueType>& operator=(const Tuple<ValueType>& src) {
    this->assign(src.begin(), src.end());
    return *this;
  }
  /*!
   * \brief assignment from rvalue of another tuple.
   * \param src source tuple
   * \return reference of self
   */
  inline Tuple<ValueType>& operator=(Tuple<ValueType>&& src) {
    Tuple<ValueType>(std::move(src)).swap(*this);
    return *this;
  }
  /*!
   * \brief assignment from initializer list
   * \param init the source initializer list
   * \return reference of self
   */
  inline Tuple<ValueType> &operator=(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
    return *this;
  }
  /*!
   * \return whether two tuple equals
   * \param s the tuple to compare against
   */
  inline bool operator==(const Tuple<ValueType> &s) const {
    if (ndim_ != s.ndim_) return false;
    return std::equal(begin(), end(), s.begin());
  }
  /*!
   * \return whether two tuple not equal
   * \param s the tuple to compare against
   */
  inline bool operator!=(const Tuple<ValueType> &s) const {
    return !(*this == s);
  }
  /*! \return the begin data pointer to content of the tuple */
  inline const ValueType *begin() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the begin data pointer to content of the tuple */
  inline ValueType *begin() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data pointer to end of the tuple */
  inline const ValueType* end() const {
    return ndim_ <= kStackCache ? (data_stack_ + ndim_): (data_heap_ + ndim_);
  }
  /*! \return the data pointer to end the tuple */
  inline ValueType* end() {
    return ndim_ <= kStackCache ? (data_stack_ + ndim_): (data_heap_ + ndim_);
  }
  /*! \return number of dimension of the tuple */
  inline index_t ndim() const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline ValueType& operator[](index_t i) {
    return begin()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const ValueType& operator[](index_t i) const {
    return begin()[i];
  }
  /*!
   * \brief Save Tuple to JSON.
   * \param writer JSONWriter
   */
  inline void Save(dmlc::JSONWriter* writer) const {
    std::vector<ValueType> tmp(begin(), end());
    writer->Write(tmp);
  }
  /*!
   * \brief Load Tuple from JSON.
   * \param reader JSONReader
   */
  inline void Load(dmlc::JSONReader* reader) {
    std::vector<ValueType> tmp;
    reader->Read(&tmp);
    this->assign(tmp.begin(), tmp.end());
  }
  /*!
   * \brief allow output string of tuple to ostream
   * \param os the output stream
   * \param t the tuple
   * \return the ostream
   */
  friend std::ostream &operator<<(std::ostream &os, const Tuple<ValueType> &t) {
    os << '(';
    const ValueType* begin = t.begin();
    const ValueType* end = t.end();
    for (const ValueType* it = begin; it != end; ++it) {
      if (it != begin) os << ',';
      os << *it;
    }
    // python style tuple
    if (t.ndim() == 1) os << ',';
    os << ')';
    return os;
  }
  /*!
   * \brief read tuple from the istream
   * \param is the input stream
   * \param t The tuple
   * \return the istream
   */
  friend std::istream &operator>>(std::istream &is, Tuple<ValueType> &t) {
    // get (
    while (true) {
      char ch = is.peek();
      if (isdigit(ch)) {
        ValueType idx;
        if (is >> idx) {
          t.assign(&idx, &idx + 1);
        }
        return is;
      }
      is.get();
      if (ch == '(') break;
      if (!isspace(ch)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    }
    index_t idx;
    std::vector<ValueType> tmp;
    while (is >> idx) {
      tmp.push_back(idx);
      char ch;
      do {
        ch = is.get();
      } while (isspace(ch));
      if (std::is_integral<ValueType>::value && ch == 'L') {
        ch = is.get();
      }
      if (ch == ',') {
        while (true) {
          ch = is.peek();
          if (isspace(ch)) {
            is.get(); continue;
          }
          if (ch == ')') {
            is.get(); break;
          }
          break;
        }
        if (ch == ')') break;
      } else if (ch == ')') {
        break;
      } else {
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    t.assign(tmp.begin(), tmp.end());
    return is;
  }

 private:
  // stack cache size
  static const uint32_t kStackCache = 4;
  /*! \brief number of dimension of the tuple */
  index_t ndim_{0};
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_{0};
  /*! \brief in stack space used to store shape when it is small */
  ValueType data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  ValueType* data_heap_{nullptr};
  // internal function to change the dimension
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
        dim > num_heap_allocated_) {
      delete [] data_heap_;
      data_heap_ = new ValueType[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*!
 * \brief A Shape class that is used to represent shape of each tensor.
 */
class TShape : public Tuple<index_t> {
 public:
  // inheritate other constructors from Tuple
  using Tuple<index_t>::Tuple;
  /*! \brief default constructor */
  TShape() = default;
  /*!
   * \brief copy constructor of TShape
   * \param s source shape.
   */
  inline TShape(const Tuple<index_t>& s)  // NOLINT(*)
      : Tuple<index_t>(s) {}
  /*!
   * \brief move constructor.
   * \param s source shape.
   */
  inline TShape(Tuple<index_t>&& s) {  // NOLINT(*)
    this->swap(s);
  }
  /*!
   * \brief assignment function from tshape
   * \param src source shape.
   * \return self.
   */
  inline TShape& operator=(const Tuple<index_t>& src) {
    this->assign(src.begin(), src.end());
    return *this;
  }
  /*!
   * \brief move assignment function from tshape
   * \param src source shape.
   * \return self.
   */
  inline TShape& operator=(Tuple<index_t>&& src) {  // NOLINT(*)
    TShape(std::move(src)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \return total number of elements in the shape */
  inline size_t Size() const {
    size_t size = 1;
    const index_t* start = begin(), *fin = end();
    for (const index_t* it = start; it != fin; ++it) {
      size *= *it;
    }
    return size;
  }
};

}  // namespace nnvm

#endif  // NNVM_TUPLE_H_
