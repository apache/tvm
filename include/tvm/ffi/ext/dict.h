#ifndef TVM_FFI_DICT_H_
#define TVM_FFI_DICT_H_
#include <iterator>
#include <tvm/ffi/core/core.h>
#include <type_traits>
#include <unordered_map>

namespace tvm {
namespace ffi {
namespace details {
struct _DictHeader {
  int64_t capacity;
  int64_t size;
};
using DictHeader = AnyWithExtra<_DictHeader>;
static_assert(sizeof(DictHeader) == sizeof(TVMFFIDict));
static_assert(offsetof(DictHeader, _extra.capacity) == offsetof(TVMFFIDict, capacity));
static_assert(offsetof(DictHeader, _extra.size) == offsetof(TVMFFIDict, size));
int32_t CountLeadingZeros(uint64_t);
uint64_t BitCeil(uint64_t);
} // namespace details

/*!
 * Dict is a flat hash table that uses quadratic/triangluar probing [1] for collision resolution.
 * Its underlying data structure is a consecutive array of `Block`s, where each `Block`
 * contains `kCapacity = 16` key-value pairs and their metadata (see also `Block`).
 * The capacity of a Dict is always a power of 2, and Fibonacci Hashing [2] is used to avoid integer
 * modulo and help improve the distribution of the hash values.
 *
 * Probing is conducted only within a fixed set of 125 possible offsets (see also
 * `kNextProbeLocation`), which effectively organizes the elements of collision chains in a linked
 * list. With the help of metadata, we can easily iterate through the linked list and relocate
 * elements whenever necessary.
 *
 * [1] Triangular numbers. https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
 * [2] Fibonacci Hashing: https://programmingpraxis.com/2018/06/19/fibonacci-hash/
 */
template <typename Hash, typename Equal>
struct DictBase : protected details::DictHeader {
protected:
  /* clang-format off */
  inline static const constexpr uint64_t kNextProbeLocation[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    /* triangle numbers for quadratic probing */ 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628,
    /* larger triangle numbers */ 8515, 19110, 42778, 96141, 216153, 486591, 1092981, 2458653, 5532801, 12442566, 27993903, 62983476, 141717030, 318844378, 717352503, 1614057336, 3631522476, 8170957530, 18384510628, 41364789378, 93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695, 5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000, 309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701, 17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251, 457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
  };
  /* clang-format on */
  static constexpr int32_t kBlockCapacity = 16;
  static constexpr uint8_t kNumProbe = std::size(kNextProbeLocation);
  static constexpr uint8_t kEmptySlot = uint8_t(0b11111111);
  static constexpr uint8_t kProtectedSlot = uint8_t(0b11111110);
  static constexpr uint64_t kMinSize = 7;
  static constexpr uint8_t kNewHead = 0b00000000;
  static constexpr uint8_t kNewTail = 0b10000000;
  using KVPair = std::pair<TVMFFIAny, TVMFFIAny>;

  struct Block {
    /**
     * Each block has a 8-byte key (TVMFFIAny), 8-byte value (TVMFFIAny), and a 1-byte metadata.
     *
     * Metadata can be one of the following three cases:
     * - 1) Empty: 0xFF (0b11111111)_2. The slot is available and can be written in.
     * - 2) Protected: 0xFE (0b11111110)_2. The slot is empty but not allowed to be written. It is
     *   only used during insertion when relocating certain elements.
     * - 3) Normal: (0bXYYYYYYY)_2. The highest bit `X` indicates if it is the head of a linked
     *   list, where `0` means head, and `1` means not head. The next 7 bits `YYYYYYY` are used as
     *   the "next pointer" (i.e. pointer to the next element),  where `kNextProbeLocation[YYYYYYY]`
     *   is the offset to the next element. And if `YYYYYYY == 0`, it means the end of the linked
     *   list.
     */
    uint8_t meta[kBlockCapacity];
    KVPair data[kBlockCapacity];
  };
  static_assert(sizeof(Block) == kBlockCapacity * (1 + sizeof(TVMFFIAny) * 2), "ABI check");
  static_assert(std::is_aggregate_v<Block>, "ABI check");
  static_assert(std::is_trivially_destructible_v<Block>, "ABI check");
  static_assert(std::is_standard_layout_v<Block>, "ABI check");
  static_assert(kNumProbe == 126, "Check assumption");

  struct ListIter {
    static ListIter None() { return ListIter(0, nullptr); }
    static ListIter FromIndex(const DictBase *self, uint64_t i) {
      if (i >= self->Cap()) // TODO: remove this check
        TVM_FFI_THROW(InternalError) << "Indexing " << i << " of length " << self->Cap();
      return ListIter(i, self->Blocks() + (i / kBlockCapacity));
    }
    static ListIter FromHash(const DictBase *self, uint64_t h) {
      return ListIter::FromIndex(self, (11400714819323198485ull * h) >>
                                           (details::CountLeadingZeros(self->Cap()) + 1));
    }
    auto &Data() const { return cur->data[i % kBlockCapacity]; }
    uint8_t &Meta() const { return cur->meta[i % kBlockCapacity]; }
    uint64_t Offset() const { return kNextProbeLocation[Meta() & 0b01111111]; }
    bool IsHead() const { return (Meta() & 0b10000000) == 0b00000000; }
    void SetNext(uint8_t jump) const { (Meta() &= 0b10000000) |= jump; }
    void Advance(const DictBase *self) { *this = WithOffset(self, Offset()); }
    bool IsNone() const { return cur == nullptr; }
    ListIter WithOffset(const DictBase *self, uint64_t offset) const {
      return offset == 0 ? ListIter::None()
                         : ListIter::FromIndex(self, (i + offset) & (self->Cap() - 1));
    }

    uint64_t i;
    Block *cur;

  protected:
    ListIter() = default;
    explicit ListIter(uint64_t i, Block *cur) : i(i), cur(cur) {}
  };

  ListIter Head(uint64_t hash) const {
    ListIter iter = ListIter::FromHash(this, hash);
    return iter.IsHead() ? iter : ListIter::None();
  }

  ListIter Prev(ListIter iter) const {
    ListIter prev = Head(Hash()(iter.Data().first));
    ListIter next = prev;
    for (next.Advance(this); next.i != iter.i; prev = next, next.Advance(this)) {
    }
    return prev;
  }

  uint64_t Cap() const { return static_cast<uint64_t>(this->_extra.capacity); }
  uint64_t Size() const { return this->_extra.size; }
  Block *Blocks() { return reinterpret_cast<Block *>(static_cast<DictBase *>(this) + 1); }
  Block *Blocks() const { return const_cast<DictBase *>(this)->Blocks(); }

  ListIter Probe(ListIter cur, uint8_t *result) const {
    for (uint8_t i = 1; i < kNumProbe && kNextProbeLocation[i] < this->Size(); ++i) {
      ListIter next = cur.WithOffset(this, kNextProbeLocation[i]);
      if (next.Meta() == kEmptySlot) {
        *result = i;
        return next;
      }
    }
    *result = 0;
    return ListIter::None();
  }

  ListIter Lookup(const TVMFFIAny &key) const {
    uint64_t hash = Hash()(key);
    for (ListIter iter = Head(hash); !iter.IsNone(); iter.Advance(this)) {
      if (Equal()(key, iter.Data().first)) {
        return iter;
      }
    }
    return ListIter::None();
  }

  template <typename Pred>
  void IterateAll(Pred pred) {
    Block *blocks_ = this->Blocks();
    int64_t num_blocks = this->_extra.capacity / kBlockCapacity;
    for (int64_t i = 0; i < num_blocks; ++i) {
      for (int j = 0; j < kBlockCapacity; ++j) {
        uint8_t &meta = blocks_[i].meta[j];
        auto &data = blocks_[i].data[j];
        if (meta != kEmptySlot && meta != kProtectedSlot) {
          pred(&meta, &data.first, &data.second);
        }
      }
    }
  }

  void Clear() {
    this->IterateAll([](uint8_t *meta, TVMFFIAny *key, TVMFFIAny *value) {
      static_cast<Any *>(key)->Reset();
      static_cast<Any *>(value)->Reset();
      *meta = kEmptySlot;
    });
    this->_extra.size = 0;
  }

  void Erase(int64_t index) {
    ListIter iter = ListIter::FromIndex(this, index);
    if (uint64_t offset = iter.Offset(); offset != 0) {
      ListIter prev = iter;
      ListIter next = iter.WithOffset(this, offset);
      while ((offset = next.Offset()) != 0) {
        prev = next;
        next = next.WithOffset(this, offset);
      }
      iter.Data() = next.Data();
      next.Meta() = kEmptySlot;
      prev.SetNext(0);
    } else {
      if (!iter.IsHead()) {
        Prev(iter).SetNext(0);
      }
      iter.Meta() = kEmptySlot;
      auto &data = iter.Data();
      static_cast<Any &>(data.first).Reset();
      static_cast<Any &>(data.second).Reset();
    }
    this->_extra.size -= 1;
  }

public:
  struct Iterator;
  friend struct Iterator;
  struct Iterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::pair<const Any, Any>;
    using pointer = value_type *;
    using reference = value_type &;
    static_assert(sizeof(value_type) == sizeof(KVPair), "ABI check");
    static_assert(offsetof(value_type, first) == offsetof(KVPair, first), "ABI check");
    static_assert(offsetof(value_type, second) == offsetof(KVPair, second), "ABI check");

    Iterator() = default;
    Iterator(const Iterator &) = default;
    Iterator(Iterator &&) = default;
    Iterator &operator=(const Iterator &) = default;
    Iterator &operator=(Iterator &&) = default;

    Iterator &operator++() {
      int64_t cap = self->Cap();
      while (++index < cap) {
        if (ListIter::FromIndex(self, index).Meta() != DictBase::kEmptySlot) {
          return *this;
        }
      }
      return *this;
    }
    Iterator &operator--() {
      while (--index >= 0) {
        if (ListIter::FromIndex(self, index).Meta() != DictBase::kEmptySlot) {
          return *this;
        }
      }
      return *this;
    }
    Iterator operator++(int) {
      Iterator copy(*this);
      this->operator++();
      return copy;
    }
    Iterator operator--(int) {
      Iterator copy(*this);
      this->operator--();
      return copy;
    }
    bool operator==(const Iterator &other) const { return index == other.index; }
    bool operator!=(const Iterator &other) const { return index != other.index; }
    reference operator*() const { return *(this->operator->()); }
    pointer operator->() const {
      return reinterpret_cast<pointer>(&ListIter::FromIndex(self, this->index).Data());
    }

  protected:
    friend struct DictBase;
    Iterator(int64_t index, DictBase *self) : index(index), self(self) {}

    int64_t index;
    DictBase *self;
  };

  KVPair *TryInsertOrLookup(TVMFFIAny *key) {
    if (Cap() == Size() || Size() + 1 > Cap() * 0.99) {
      return nullptr;
    }
    // `iter` starts from the head of the linked list
    ListIter iter = ListIter::FromHash(this, Hash()(*key));
    uint8_t new_meta = kNewHead;
    // There are three cases over all:
    // 1) available - `iter` points to an empty slot that we could directly write in;
    // 2) hit - `iter` points to the head of the linked list which we want to iterate through
    // 3) relocate - `iter` points to the body of a different linked list, and in this case, we
    // will have to relocate the elements to make space for the new element
    if (iter.Meta() == kEmptySlot) { // (Case 1) available
      // Do nothing
    } else if (iter.IsHead()) { // (Case 2) hit
      // Point `iter` to the last element of the linked list
      for (ListIter prev = iter;; prev = iter) {
        if (Equal()(*key, iter.Data().first)) {
          return &iter.Data();
        }
        iter.Advance(this);
        if (iter.IsNone()) {
          iter = prev;
          break;
        }
      }
      // Prob at the end of the linked list for the next empty slot
      ListIter prev = iter;
      uint8_t jump;
      iter = Probe(iter, &jump);
      if (jump == 0) {
        return nullptr;
      }
      prev.SetNext(jump);
      new_meta = kNewTail;
    } else {
      // (Case 3) relocate: Chop the list starting from `iter`, and move it to other locations.
      ListIter next = iter, prev = Prev(iter);
      // Loop invariant:
      // - `next` points to the first element of the list we are going to relocate.
      // - `prev` points to the last element of the list that we are relocating to.
      for (uint8_t next_meta = kProtectedSlot; !next.IsNone(); next_meta = kEmptySlot) {
        // Step 1. Prob for the next empty slot `new_next` after `prev`
        uint8_t jump;
        ListIter new_next = Probe(prev, &jump);
        if (jump == 0) {
          return nullptr;
        }
        // Step 2. Relocate `next` to `new_next`
        new_next.Meta() = kNewTail;
        new_next.Data() = {next.Data().first, next.Data().second};
        std::swap(next_meta, next.Meta());
        prev.SetNext(jump);
        // Step 3. Update `prev` -> `new_next`, `next` -> `Advance(next)`
        prev = new_next;
        next = next.WithOffset(this, kNextProbeLocation[next_meta & 0b01111111]);
      }
    }
    this->_extra.size += 1;
    iter.Meta() = new_meta;
    auto &data = iter.Data() = {*key, TVMFFIAny()};
    key->type_index = 0;
    key->v_int64 = 0;
    return &data;
  }

  Any &at(const Any &key) {
    ListIter iter = Lookup(key);
    if (iter.IsNone()) {
      TVM_FFI_THROW(KeyError) << "Key not found";
    }
    return static_cast<Any &>(iter.Data().second);
  }

  const Any &at(const Any &key) const {
    ListIter iter = Lookup(key);
    if (iter.IsNone()) {
      TVM_FFI_THROW(KeyError) << "Key not found";
    }
    return static_cast<const Any &>(iter.Data().second);
  }
  Any &operator[](const Any &key) = delete; // May need rehashing
  const Any &operator[](const Any &key) const { return this->at(key); }
  int64_t size() const { return this->_extra.size; }
  int64_t count(const Any &key) const { return Lookup(key).IsNone() ? 0 : 1; }
  void clear() { this->Clear(); }
  Iterator begin() { return Iterator(-1, this).operator++(); }
  Iterator end() { return Iterator(this->Cap(), this); }
  Iterator find(const Any &key) {
    ListIter iter = Lookup(key);
    return iter.IsNone() ? end() : Iterator(iter.i, this);
  }
  void erase(const Any &key) {
    ListIter iter = Lookup(key);
    if (!iter.IsNone()) {
      Erase(iter.i);
    } else {
      TVM_FFI_THROW(KeyError) << "Key not found";
    }
  }
};

struct AnyHash {
  uint64_t operator()(const TVMFFIAny &a) const {
    if (a.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      return details::TVMFFIStrHash(reinterpret_cast<TVMFFIStr *>(a.v_obj));
    }
    union {
      int64_t i64;
      uint64_t u64;
    } cvt;
    cvt.i64 = a.v_int64;
    return cvt.u64;
  }
};

struct AnyEqual {
  bool operator()(const TVMFFIAny &a, const TVMFFIAny &b) const {
    if (a.type_index != b.type_index) {
      return false;
    }
    if (a.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      return details::TVMFFIStrCmp(reinterpret_cast<TVMFFIStr *>(a.v_obj),
                                   reinterpret_cast<TVMFFIStr *>(b.v_obj)) == 0;
    }
    return a.v_int64 == b.v_int64;
  }
};

struct Dict : protected DictBase<AnyHash, AnyEqual> {
  TVM_FFI_DEF_STATIC_TYPE(Dict, Object, TVMFFITypeIndex::kTVMFFIDict);
  TVM_FFI_DEF_TYPE_FRIENDS();
  struct Allocator;
  friend struct Allocator;

protected:
  explicit Dict(int64_t capacity) {
    this->_extra.capacity = capacity;
    this->_extra.size = 0;
    for (int64_t i = 0; i < capacity / kBlockCapacity; ++i) {
      Block &block = Blocks()[i];
      std::memset(block.meta, kEmptySlot, sizeof(block.meta));
    }
  }
  ~Dict() { this->Clear(); }
};

struct Dict::Allocator {
  static TVM_FFI_INLINE Dict *New(int64_t capacity) {
    if ((capacity & (capacity - 1)) != 0 || capacity % kBlockCapacity != 0) {
      TVM_FFI_THROW(InternalError) << "capacity: " << capacity;
    }
    int64_t num_blocks = capacity / Dict::kBlockCapacity;
    return DefaultObjectAllocator<Dict>::NewWithPad<Dict::Block>(num_blocks, capacity);
  }
  static TVM_FFI_INLINE Dict *New() { return New(0); }
};

template <>
struct Ref<Dict> : public RefBase<Dict> {
  using TSelf = Ref<Dict>;
  using TBase = RefBase<Dict>;
  using Iterator = Dict::Iterator;

  TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(Ref<Dict>, RefBase<Dict>)
  TVM_FFI_DEF_TYPE_FRIENDS();

  Ref() : TBase(Dict::Allocator::New()) {}
  template <typename Iter>
  Ref(Iter begin, Iter end) : TBase(FromRange(begin, end)) {}
  Ref(std::initializer_list<std::pair<Any, Any>> init) : TSelf(init.begin(), init.end()) {}
  template <typename K, typename V, typename Hash, typename Equal>
  Ref(const std::unordered_map<K, V, Hash, Equal> &source) : TSelf(source.begin(), source.end()) {}
  const Any &at(const Any &key) const { return get()->at(key); }
  Any &at(const Any &key) { return get()->at(key); }
  const Any &operator[](const Any &key) const { return get()->at(key); }
  Any &operator[](Any key) { return static_cast<Any &>(InsertOrLookup(std::move(key))->second); }
  int64_t size() const { return get()->size(); }
  int64_t count(const Any &key) const { return get()->count(key); }
  bool empty() const { return size() == 0; }
  void clear() { get()->clear(); }
  Iterator begin() { return get()->begin(); }
  Iterator end() { return get()->end(); }
  Iterator find(const Any &key) { return get()->find(key); }
  void erase(const Any &key) { get()->erase(key); }

protected:
  using TBase::operator*;
  using TBase::operator->;

  Dict::KVPair *InsertOrLookup(Any key) {
    for (;;) {
      if (Dict::KVPair *ret = get()->TryInsertOrLookup(&key)) {
        return ret;
      }
      Ref<Dict>::FromRange(begin(), end(), get()->Cap() * 2).Swap(*this);
    }
    TVM_FFI_UNREACHABLE();
  }

  template <typename Iter>
  static Ref<Dict> FromRange(Iter begin, Iter end) {
    int64_t num = std::distance(begin, end);
    int64_t capacity = details::BitCeil(num * 2);
    capacity = std::max<int64_t>(
        (capacity + Dict::kBlockCapacity - 1) & ~(Dict::kBlockCapacity - 1), Dict::kBlockCapacity);
    Ref<Dict> dict(Dict::Allocator::New(capacity));
    for (; begin != end; ++begin) {
      Dict::KVPair *ret = dict.InsertOrLookup(Any(begin->first));
      static_cast<Any &>(ret->second) = Any(begin->second);
    }
    return dict;
  }

  template <typename Iter>
  static Ref<Dict> FromRange(Iter begin, Iter end, int64_t capacity) {
    capacity = std::max<int64_t>(
        (capacity + Dict::kBlockCapacity - 1) & ~(Dict::kBlockCapacity - 1), Dict::kBlockCapacity);
    Ref<Dict> dict(Dict::Allocator::New(capacity));
    for (; begin != end; ++begin) {
      Dict::KVPair *ret = dict.InsertOrLookup(Any(begin->first));
      static_cast<Any &>(ret->second) = Any(begin->second);
    }
    return dict;
  }
};

namespace details {

TVM_FFI_INLINE int32_t CountLeadingZeros(uint64_t x) {
#if __cplusplus >= 202002L
  return std::countl_zero(x);
#elif defined(_MSC_VER)
  DWORD leading_zero = 0;
  if (_BitScanReverse64(&leading_zero, value)) {
    return static_cast<int32_t>(63 - leading_zero);
  } else {
    return 64;
  }
#else
  return x == 0 ? 64 : __builtin_clzll(x);
#endif
}

TVM_FFI_INLINE uint64_t BitCeil(uint64_t x) {
#if __cplusplus >= 202002L
  return std::bit_ceil(x);
#else
  return x <= 1 ? 1 : static_cast<uint64_t>(1) << (64 - CountLeadingZeros(x - 1));
#endif
}

} // namespace details
} // namespace ffi
} // namespace tvm
#endif // TVM_FFI_DICT_H_
