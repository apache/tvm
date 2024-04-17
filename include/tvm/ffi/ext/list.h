#ifndef TVM_FFI_LIST_H_
#define TVM_FFI_LIST_H_
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <tvm/ffi/core/core.h>
#include <vector>

namespace tvm {
namespace ffi {
namespace details {
struct _ListHeaeder {
  int64_t list_capacity; // sizeof(TVMFFIAny*) per element
  int64_t list_length;
  int64_t pool_capacity; // sizeof(TVMFFIAny) per element
  int64_t pool_length;
};
using ListHeader = AnyWithExtra<_ListHeaeder>;
static_assert(sizeof(ListHeader) == sizeof(TVMFFIList));
static_assert(offsetof(ListHeader, _extra.list_capacity) == offsetof(TVMFFIList, list_capacity));
static_assert(offsetof(ListHeader, _extra.list_length) == offsetof(TVMFFIList, list_length));
static_assert(offsetof(ListHeader, _extra.pool_capacity) == offsetof(TVMFFIList, pool_capacity));
static_assert(offsetof(ListHeader, _extra.pool_length) == offsetof(TVMFFIList, pool_length));
void ListRangeCheck(int64_t begin, int64_t end, int64_t length);
} // namespace details

#define TVM_FFI_LIST_EXEC_IF_POD(ListCls, TypeIndex, Exec)                                         \
  if constexpr (ListCls::all_items_are_pod) {                                                      \
    Exec;                                                                                          \
  } else if constexpr (!ListCls::all_items_are_obj) {                                              \
    if (details::IsTypeIndexPOD(TypeIndex)) {                                                      \
      Exec;                                                                                        \
    }                                                                                              \
  }

template <typename ListClass>
struct ListMethods {
  using ListItem = TVMFFIAny *;
  using PoolItem = TVMFFIAny;

private:
  // Section 1: Iterator and utility classes

  struct Iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = AnyView;
    using pointer = AnyView;
    using reference = AnyView;

    Iterator() = default;
    Iterator(const Iterator &) = default;
    Iterator(Iterator &&) = default;
    Iterator &operator=(const Iterator &) = default;
    Iterator &operator=(Iterator &&) = default;

    Iterator &operator++() {
      ++index;
      return *this;
    }
    Iterator &operator--() {
      --index;
      return *this;
    }
    Iterator operator++(int) {
      Iterator ret = *this;
      ++index;
      return ret;
    }
    Iterator operator--(int) {
      Iterator ret = *this;
      --index;
      return ret;
    }
    Iterator operator+(difference_type n) const {
      Iterator ret = *this;
      ret.index += n;
      return ret;
    }
    Iterator operator-(difference_type n) const {
      Iterator ret = *this;
      ret.index -= n;
      return ret;
    }
    Iterator &operator+=(difference_type n) {
      index += n;
      return *this;
    }
    Iterator &operator-=(difference_type n) {
      index -= n;
      return *this;
    }
    AnyView operator[](difference_type n) const { return self->GetItem(index + n); }
    difference_type operator-(const Iterator &other) const { return index - other.index; }
    bool operator==(const Iterator &other) const { return index == other.index; }
    bool operator!=(const Iterator &other) const { return index != other.index; }
    reference operator*() const { return self->GetItem(index); }
    pointer operator->() const = delete;

  protected:
    friend struct ListMethods;
    Iterator(ListClass *self, int64_t index) : self(self), index(index) {}
    ListClass *self;
    int64_t index;
  };

  struct ReverseIterator : public std::reverse_iterator<Iterator> {
    using std::reverse_iterator<Iterator>::reverse_iterator;
  };

  TVM_FFI_INLINE auto *Get() { return &(static_cast<ListClass *>(this)->_extra); }
  TVM_FFI_INLINE auto *Get() const { return &(static_cast<const ListClass *>(this)->_extra); }
  TVM_FFI_INLINE auto ListBegin() const {
    return reinterpret_cast<const ListItem *>(static_cast<const ListClass *>(this) + 1);
  }
  TVM_FFI_INLINE auto PoolBegin() const {
    return reinterpret_cast<const PoolItem *>(ListBegin() + Get()->list_capacity);
  }
  TVM_FFI_INLINE auto ListBegin() {
    return reinterpret_cast<ListItem *>(static_cast<ListClass *>(this) + 1);
  }
  TVM_FFI_INLINE auto PoolBegin() {
    return reinterpret_cast<PoolItem *>(ListBegin() + Get()->list_capacity);
  }

public:
  // Section 2: Interfaces - exposed to Ref<ListClass>
  void inplace_init_many(int64_t numel, Any *first) {
    this->ReplaceRangeInplace(0, 0, numel, first);
  }
  void inplace_push_back(Any value) {
    this->ReplaceRangeInplace(Get()->list_length, Get()->list_length, 1, &value);
  }
  void inplace_insert(int64_t index, Any value) {
    this->ReplaceRangeInplace(index, index, 1, &value);
  }
  void inplace_insert_many(int64_t index, int64_t numel, Any *first) {
    this->ReplaceRangeInplace(index, index, numel, first);
  }
  void inplace_move_from(ListClass *other) {
    // Exception safety is guaranteed even if there is an exception thrown during
    // this process, because neither `list_length` is modified.
    int64_t &new_pool_len = Get()->pool_length;
    int64_t &new_list_len = Get()->list_length;
    ListItem *new_list_base = this->ListBegin();
    PoolItem *new_pool_base = this->PoolBegin();
    ListItem *old_list_base = other->ListBegin();
    for (int64_t i = 0; i < other->_extra.list_length; ++i) {
      ListItem &e = new_list_base[i] = old_list_base[i];
      if (e) {
        TVM_FFI_LIST_EXEC_IF_POD(ListClass, e->type_index, {
          // Copy POD data to the new POD pool
          e = &(new_pool_base[new_pool_len++] = *e);
        });
      }
    }
    // Critical section: no exception be thrown here. Otherwise ref counting would be wrong.
    { std::swap(other->_extra.list_length, new_list_len); }
  }
  void inplace_resize(int64_t new_size) {
    int64_t &list_length = Get()->list_length;
    if (new_size < list_length) {
      this->RangeDecRef(new_size, list_length);
    } else {
      ListItem *b = this->ListBegin();
      for (int64_t i = list_length; i < new_size; ++i) {
        b[i] = nullptr;
      }
    }
    list_length = new_size;
  }
  void clear() {
    this->RangeDecRef(0, Get()->list_length);
    Get()->list_length = Get()->pool_length = 0;
  }
  void pop_back() {
    int64_t n = Get()->list_length;
    this->ReplaceRangeInplace(n - 1, n, 0, nullptr);
  }
  int64_t size() const { return Get()->list_length; }
  int64_t capacity() const { return Get()->list_capacity; }
  bool empty() const { return Get()->list_length == 0; }
  void erase(int64_t index) { this->ReplaceRangeInplace(index, index + 1, 0, nullptr); }
  void SetItem(int64_t i, Any data) { this->ReplaceRangeInplace(i, i + 1, 1, &data); }
  AnyView GetItem(int64_t index) const {
    details::ListRangeCheck(index, index + 1, Get()->list_length);
    const AnyView *data = static_cast<const AnyView *>(this->ListBegin()[index]);
    if (data == nullptr) {
      return AnyView();
    }
    TVM_FFI_LIST_EXEC_IF_POD(ListClass, data->type_index, { return *data; });
    AnyView ret;
    ret.type_index = data->type_index;
    ret.v_obj = const_cast<AnyView *>(data);
    return ret;
  }
  Iterator begin() { return Iterator(static_cast<ListClass *>(this), 0); }
  Iterator end() { return Iterator(static_cast<ListClass *>(this), Get()->list_length); }
  ReverseIterator rbegin() { return ReverseIterator(end()); }
  ReverseIterator rend() { return ReverseIterator(begin()); }

public:
  // Section 3: Core methods
  void RangeDecRef(int64_t begin, int64_t end) {
    if constexpr (ListClass::all_items_are_pod) {
      return;
    }
    ListItem *b = this->ListBegin();
    for (int64_t i = begin; i < end; ++i) {
      ListItem &data = b[i];
      if (data) {
        TVM_FFI_LIST_EXEC_IF_POD(ListClass, data->type_index, { continue; });
        details::DecRef(data);
      }
    }
  }
  void ReplaceRangeInplace(int64_t range_begin, int64_t range_end, int64_t numel, Any *first) {
    int64_t &list_len = Get()->list_length;
    int64_t &pool_len = Get()->pool_length;
    int64_t delta = numel - (range_end - range_begin);
    details::ListRangeCheck(range_begin, range_end, list_len);
    RangeDecRef(range_begin, range_end);
    ListItem *b = this->ListBegin();
    // Step 1. Move [range_end, list_len) to [range_end + delta, list_len + delta)
    // so there are exaclty `num_elems` free slots in [range_begin, range_end + delta)
    std::memmove(b + range_end + delta, b + range_end, (list_len - range_end) * sizeof(ListItem));
    list_len += delta;
    // Step 2. Copy `num_elems` items to [range_begin, range_end + delta)
    PoolItem *p = this->PoolBegin();
    for (int64_t i = 0; i < numel; ++i) {
      Any v = std::move(first[i]);
      ListItem &d = b[range_begin + i] = nullptr;
      if (details::IsTypeIndexNone(v.type_index)) {
        continue;
      }
      TVM_FFI_LIST_EXEC_IF_POD(ListClass, v.type_index, {
        d = &(p[pool_len++] = v);
        continue;
      });
      // N.B. Transfer the ownership of `v` to the list
      std::swap(d, v.v_obj);
      v.type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone);
    }
  }
  Ref<ListClass> MaybeMoveToMoreCapacity(int64_t list_len_delta, int64_t pool_len_delta) {
    ListClass *self = static_cast<ListClass *>(this);
    int64_t list_cap = self->_extra.list_capacity, pool_cap = self->_extra.pool_capacity;
    int64_t new_list_cap = GrowCapacity(list_cap, self->_extra.list_length + list_len_delta);
    int64_t new_pool_cap = GrowCapacity(pool_cap, self->_extra.pool_length + pool_len_delta);
    if (new_list_cap > list_cap || new_pool_cap > pool_cap) {
      Ref<ListClass> ret = ListClass::Allocator::New(new_list_cap, new_pool_cap);
      ret->inplace_move_from(self);
      return ret;
    }
    return Ref<ListClass>(self);
  }
  static int64_t GrowCapacity(int64_t current, int64_t required) {
    for (; current < required; current = std::max(current * 2, ListClass::kMinCapacity)) {
    }
    return current;
  }
};

struct List : private details::ListHeader, private ListMethods<List> {
  TVM_FFI_DEF_STATIC_TYPE(List, Object, TVMFFITypeIndex::kTVMFFIList);
  TVM_FFI_DEF_TYPE_FRIENDS();
  constexpr static int64_t kMinCapacity = 4;
  struct Allocator;

  List(int64_t list_capacity, int64_t pool_capacity) {
    this->_extra.list_capacity = list_capacity;
    this->_extra.list_length = 0;
    this->_extra.pool_capacity = pool_capacity;
    this->_extra.pool_length = 0;
  }
  using TBase = ListMethods<List>;
  using TBase::begin;
  using TBase::capacity;
  using TBase::clear;
  using TBase::empty;
  using TBase::end;
  using TBase::erase;
  using TBase::GetItem;
  using TBase::inplace_init_many;
  using TBase::inplace_insert;
  using TBase::inplace_insert_many;
  using TBase::inplace_push_back;
  using TBase::inplace_resize;
  using TBase::pop_back;
  using TBase::rbegin;
  using TBase::rend;
  using TBase::SetItem;
  using TBase::size;

private:
  constexpr static bool all_items_are_pod = false;
  constexpr static bool all_items_are_obj = false;
  ~List() { this->clear(); }

  template <typename>
  friend struct ListMethods;
};

struct List::Allocator {
  static TVM_FFI_INLINE List *New() { return New(0, 0); }
  static TVM_FFI_INLINE List *New(int64_t list_capacity, int64_t pool_capacity) {
    static_assert(sizeof(PoolItem) % sizeof(ListItem) == 0);
    constexpr int64_t ratio = sizeof(PoolItem) / sizeof(ListItem);
    static_assert(ratio == 2);
    return DefaultObjectAllocator<List>::NewWithPad<ListItem>(list_capacity + pool_capacity * ratio,
                                                              list_capacity, pool_capacity);
  }
};

template <>
struct Ref<List> : public RefBase<List> {
  TVM_FFI_DEF_TYPE_FRIENDS();
  TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(Ref<List>, RefBase<List>)

private:
  struct AccessProxy {
    AccessProxy &operator=(Any data) {
      self->SetItem(index, data);
      return *this;
    }
    template <typename Type>
    operator Type() const {
      return Type(self->get()->GetItem(index));
    }
    operator AnyView() const { return self->get()->GetItem(index); }

    friend std::ostream &operator<<(std::ostream &os, const AccessProxy &self) {
      return os << self.self->get()->GetItem(self.index);
    }

  private:
    friend struct Ref<List>;
    AccessProxy(Ref<List> *self, int64_t index) : self(self), index(index) {}
    Ref<List> *self;
    int64_t index;
  };

public:
  Ref() : TBase(List::Allocator::New(0, 0)) {}
  Ref(std::initializer_list<Any> data) : Ref(data.begin(), data.end()) {}
  template <typename Iter>
  Ref<List>(Iter first, Iter last) : Ref(std::distance(first, last)) {
    std::vector<Any> elems = this->EnsureCapacityWithList(first, last);
    int64_t numel = static_cast<int64_t>(elems.size());
    this->get()->inplace_init_many(numel, elems.data());
  }
  template <typename Iter>
  void insert(int64_t index, Iter first, Iter last) {
    std::vector<Any> elems = this->EnsureCapacityWithList(first, last);
    int64_t numel = static_cast<int64_t>(elems.size());
    this->get()->inplace_insert_many(index, numel, elems.data());
  }
  void push_back(Any data) {
    this->EnsureCapacityWithOneMoreElement(data)->inplace_push_back(data);
  }
  void insert(int64_t index, Any data) {
    this->EnsureCapacityWithOneMoreElement(data)->inplace_insert(index, data);
  }
  void resize(int64_t new_size) { this->EnsureCapacityAtLeast(new_size)->inplace_resize(new_size); }
  void reserve(int64_t capacity) { this->EnsureCapacityAtLeast(capacity); }
  void clear() { this->get()->clear(); }
  void erase(int64_t index) { this->get()->erase(index); }
  void pop_back() { this->get()->pop_back(); }
  int64_t size() const { return this->get()->size(); }
  int64_t capacity() const { return this->get()->capacity(); }
  bool empty() const { return this->get()->empty(); }
  AnyView front() const { return this->get()->GetItem(0); }
  AnyView back() const { return this->get()->GetItem(this->get()->size() - 1); }
  auto operator[](int64_t index) { return AccessProxy(this, index); }
  auto operator[](int64_t index) const { return this->get()->GetItem(index); }
  auto begin() { return this->get()->begin(); }
  auto end() { return this->get()->end(); }
  auto rbegin() { return this->get()->rbegin(); }
  auto rend() { return this->get()->rend(); }
  void SetItem(int64_t i, Any data) {
    TVM_FFI_LIST_EXEC_IF_POD(List, data.type_index,
                             { this->get()->MaybeMoveToMoreCapacity(0, 1).Swap(*this); });
    this->get()->SetItem(i, data);
  }

protected:
  friend struct List;
  using RefBase<List>::operator*;
  using RefBase<List>::operator->;
  using TSelf = Ref<List>;
  using TBase = RefBase<List>;
  friend struct List;
  template <typename>
  friend struct ListMethods;

  Ref<List>(int64_t size) : TBase(List::Allocator::New(size, size)) {}

  List *EnsureCapacityWithOneMoreElement(const Any &data) {
    int64_t pool_len_delta = 0;
    TVM_FFI_LIST_EXEC_IF_POD(List, data.type_index, { pool_len_delta = 1; });
    this->get()->MaybeMoveToMoreCapacity(1, pool_len_delta).Swap(*this);
    return this->get();
  }

  List *EnsureCapacityAtLeast(int64_t capacity) {
    List *self = this->get();
    int64_t list_len_delta = std::max<int64_t>(capacity - self->_extra.list_capacity, 0);
    int64_t pool_len_delta = 0;
    if constexpr (!List::all_items_are_obj) {
      pool_len_delta = std::max<int64_t>(capacity - self->_extra.pool_capacity, 0);
    }
    this->get()->MaybeMoveToMoreCapacity(list_len_delta, pool_len_delta).Swap(*this);
    return this->get();
  }

  template <typename Iter>
  std::vector<Any> EnsureCapacityWithList(Iter first, Iter last) {
    std::vector<Any> ret(first, last);
    int64_t list_len_delta = ret.size();
    int64_t pool_len_delta = 0;
    if constexpr (List::all_items_are_pod) {
      pool_len_delta = ret.size();
    } else if constexpr (!List::all_items_are_obj) {
      for (const Any &v : ret) {
        if (details::IsTypeIndexPOD(v.type_index)) {
          ++pool_len_delta;
        }
      }
    }
    this->get()->MaybeMoveToMoreCapacity(list_len_delta, pool_len_delta).Swap(*this);
    return ret;
  }
};

namespace details {
TVM_FFI_INLINE void ListRangeCheck(int64_t begin, int64_t end, int64_t length) {
  if (begin > end) {
    TVM_FFI_THROW(IndexError) << "Invalid range [" << begin << ", " << end
                              << ") when indexing a list";
  }
  if (begin < 0 || end > length) {
    if (begin == end || begin + 1 == end) {
      TVM_FFI_THROW(IndexError) << "Indexing `" << begin << "` of a list of size " << length;
    } else {
      TVM_FFI_THROW(IndexError) << "Indexing [" << begin << ", " << end << ") of a list of size "
                                << length;
    }
  }
}
} // namespace details
} // namespace ffi
} // namespace tvm

#undef TVM_FFI_LIST_EXEC_IF_POD

#endif // TVM_FFI_LIST_H_
