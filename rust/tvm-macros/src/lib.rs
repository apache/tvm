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

use proc_macro::TokenStream;
use proc_macro_error::proc_macro_error;

mod external;
mod import_module;
mod object;
mod util;

#[proc_macro]
pub fn import_module(input: TokenStream) -> TokenStream {
    import_module::macro_impl(input)
}

#[proc_macro_error]
#[proc_macro_derive(Object, attributes(base, ref_name, type_key, no_derive))]
pub fn macro_impl(input: TokenStream) -> TokenStream {
    // let input = proc_macro2::TokenStream::from(input);
    TokenStream::from(object::macro_impl(input))
}

#[proc_macro_error]
#[proc_macro]
pub fn external(input: TokenStream) -> TokenStream {
    external::macro_impl(input)
}
