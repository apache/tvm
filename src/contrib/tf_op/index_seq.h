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

/*
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
void apply_variadic_impl(F f, T(&t)[N], IndexSeq<Idx...>) {
    f(t[Idx]...);
}

template <typename F, typename T, std::size_t N>
void apply_variadic(F f, T(&t)[N]) {
    apply_variadic_impl(f, t, make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t N, std::size_t... Idx>
void apply_variadic_by_ptrs_impl(F f, T(&t)[N], IndexSeq<Idx...>) {
    f(&t[Idx]...);
}

template <typename F, typename T, std::size_t N>
void apply_variadic_by_ptrs(F f, T(&t)[N]) {
    apply_variadic_by_ptrs_impl(f, t, make_index_sequence<N>{});
}

#endif

