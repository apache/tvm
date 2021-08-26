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
use proc_macro2::Span;
use quote::quote;
use syn::DeriveInput;
use syn::Ident;

use crate::util::*;

pub fn macro_impl(input: proc_macro::TokenStream) -> TokenStream {
    let tvm_rt_crate = get_tvm_rt_crate();
    let result = quote! { #tvm_rt_crate::function::Result };
    let error = quote! { #tvm_rt_crate::errors::Error };
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
    let payload_id = derive_input.ident.clone();

    let type_key = get_attr(&derive_input, "type_key")
        .map(attr_to_str)
        .expect("Failed to get type_key");

    let derive = get_attr(&derive_input, "no_derive")
        .map(|_| false)
        .unwrap_or(true);

    let ref_id = get_attr(&derive_input, "ref_name")
        .map(|a| Ident::new(attr_to_str(a).value().as_str(), Span::call_site()))
        .unwrap_or_else(|| {
            let id = payload_id.to_string();
            let suffixes = ["Node", "Obj"];
            if let Some(suf) = suffixes
                .iter()
                .find(|&suf| id.len() > suf.len() && id.ends_with(suf))
            {
                Ident::new(&id[..id.len() - suf.len()], payload_id.span())
            } else {
                panic!(
                    "Either 'ref_name' must be given, or the struct name must end one of {:?}",
                    suffixes
                )
            }
        });

    let base_tokens = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().and_then(|f| {
            let (base_id, base_ty) = (f.ident.clone()?, f.ty.clone());
            if base_id == "base" {
                // The transitive case of subtyping
                Some(quote! {
                    impl<O> AsRef<O> for #payload_id
                        where #base_ty: AsRef<O>
                    {
                        fn as_ref(&self) -> &O {
                            self.#base_id.as_ref()
                        }
                    }
                })
            } else {
                None
            }
        }),
        _ => panic!("derive only works for structs"),
    };

    let ref_derives = if derive {
        quote! { #[derive(Debug, Clone)]}
    } else {
        quote! { #[derive(Clone)] }
    };

    let mut expanded = quote! {
        unsafe impl #tvm_rt_crate::object::IsObject for #payload_id {
            const TYPE_KEY: &'static str = #type_key;
        }

        // a silly AsRef impl is necessary for subtyping to work
        impl AsRef<#payload_id> for #payload_id {
            fn as_ref(&self) -> &Self {
                self
            }
        }

        #ref_derives
        pub struct #ref_id(Option<#tvm_rt_crate::object::ObjectPtr<#payload_id>>);

        impl #tvm_rt_crate::object::IsObjectRef for #ref_id {
            type Object = #payload_id;

            fn as_ptr(&self) -> Option<&#tvm_rt_crate::object::ObjectPtr<Self::Object>> {
                self.0.as_ref()
            }

            fn into_ptr(self) -> Option<#tvm_rt_crate::object::ObjectPtr<Self::Object>> {
                self.0
            }

            fn from_ptr(object_ptr: Option<#tvm_rt_crate::object::ObjectPtr<Self::Object>>) -> Self {
                #ref_id(object_ptr)
            }
        }

        impl std::ops::Deref for #ref_id {
            type Target = #payload_id;

            fn deref(&self) -> &Self::Target {
                self.0.as_ref().unwrap()
            }
        }

        impl std::convert::From<#payload_id> for #ref_id {
            fn from(payload: #payload_id) -> Self {
                let ptr = #tvm_rt_crate::object::ObjectPtr::new(payload);
                #tvm_rt_crate::object::IsObjectRef::from_ptr(Some(ptr))
            }
        }

        impl std::convert::From<#tvm_rt_crate::object::ObjectPtr<#payload_id>> for #ref_id {
            fn from(ptr: #tvm_rt_crate::object::ObjectPtr<#payload_id>) -> Self {
                #tvm_rt_crate::object::IsObjectRef::from_ptr(Some(ptr))
            }
        }

        impl std::convert::TryFrom<#tvm_rt_crate::RetValue> for #ref_id {
            type Error = #error;

            fn try_from(ret_val: #tvm_rt_crate::RetValue) -> #result<#ref_id> {
                use std::convert::TryInto;
                let ptr: #tvm_rt_crate::object::ObjectPtr<#payload_id> = ret_val.try_into()?;
                Ok(ptr.into())
            }
        }

        impl<'a> From<&'a #ref_id> for #tvm_rt_crate::ArgValue<'a> {
            fn from(object_ref: &'a #ref_id) -> #tvm_rt_crate::ArgValue<'a> {
                use std::ffi::c_void;
                let object_ptr = &object_ref.0;
                match object_ptr {
                    None => {
                        #tvm_rt_crate::ArgValue::
                            ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
                    }
                    Some(value) => value.into()
                }
            }
        }

        impl<'a> std::convert::TryFrom<#tvm_rt_crate::ArgValue<'a>> for #ref_id {
            type Error = #error;

            fn try_from(arg_value: #tvm_rt_crate::ArgValue<'a>) -> #result<#ref_id> {
                use std::convert::TryInto;
                let optr = arg_value.try_into()?;
                Ok(#ref_id(Some(optr)))
            }
        }


        impl From<#ref_id> for #tvm_rt_crate::RetValue {
            fn from(object_ref: #ref_id) -> #tvm_rt_crate::RetValue {
                use std::ffi::c_void;
                let object_ptr = &object_ref.0;
                match object_ptr {
                    None => {
                        #tvm_rt_crate::RetValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
                    }
                    Some(value) => value.clone().into()
                }
            }
        }
    };

    expanded.extend(base_tokens);

    if derive {
        let derives = quote! {
            impl std::hash::Hash for #ref_id {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    self.0.hash(state)
                }
            }

            impl std::cmp::PartialEq for #ref_id {
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0
                }
            }

            impl std::cmp::Eq for #ref_id {}
        };

        expanded.extend(derives);
    }

    TokenStream::from(expanded)
}
