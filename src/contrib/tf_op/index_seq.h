/**
 * Refer to std::index_sequence (since c++14)
 * Utilities to invoke variadic function with template <size_t N> 
 */
#ifndef __TFTVM_INDEX_SEQ__
#define __TFTVM_INDEX_SEQ__

template <std::size_t ...>
struct IndexSeq {};

template <std::size_t N, std::size_t ... Tail>
struct IndexSeqHelper : public IndexSeqHelper<N-1U, N-1U, Tail...> {};

template <std::size_t ... Tail>
struct IndexSeqHelper<0U, Tail ...> {
    using type = IndexSeq<Tail ...>;
};

template <std::size_t N>
using make_index_sequence = typename IndexSeqHelper<N>::type;


template <typename F, typename T, std::size_t N, std::size_t... Idx>
decltype(auto) apply_variadic_impl(F f, T(&t)[N], IndexSeq<Idx...>) {
    return f(t[Idx]...);
}

template <typename F, typename T, std::size_t N>
decltype(auto) apply_variadic(F f, T(&t)[N]) {
    return apply_variadic_impl(f, t, make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t N, std::size_t... Idx>
decltype(auto) apply_variadic_by_ptrs_impl(F f, T(&t)[N], IndexSeq<Idx...>) {
    return f(&t[Idx]...);
}

template <typename F, typename T, std::size_t N>
decltype(auto) apply_variadic_by_ptrs(F f, T(&t)[N]) {
    return apply_variadic_by_ptrs_impl(f, t, make_index_sequence<N>{});
}

#endif

