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

pub fn macro_impl(input: proc_macro::TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
    let payload_id = derive_input.ident;

    let mut type_key = None;
    let mut ref_name = None;
    let base = Some(Ident::new("base", Span::call_site()));

    for attr in derive_input.attrs {
        if attr.path.is_ident("type_key") {
            type_key = Some(attr.parse_meta().expect("foo"))
        }

        if attr.path.is_ident("ref_name") {
            ref_name = Some(attr.parse_meta().expect("foo"))
        }
    }

    let type_key = if let Some(syn::Meta::NameValue(name_value)) = type_key {
        match name_value.lit {
            syn::Lit::Str(type_key) => type_key,
            _ => panic!("foo"),
        }
    } else {
        panic!("bar");
    };

    let ref_name = if let Some(syn::Meta::NameValue(name_value)) = ref_name {
        match name_value.lit {
            syn::Lit::Str(ref_name) => ref_name,
            _ => panic!("foo"),
        }
    } else {
        panic!("bar");
    };

    let ref_id = Ident::new(&ref_name.value(), Span::call_site());
    let base = base.expect("should be present");

    let expanded = quote! {
        unsafe impl tvm_rt::object::IsObject for #payload_id {
            const TYPE_KEY: &'static str = #type_key;

            fn as_object<'s>(&'s self) -> &'s Object {
                &self.#base.as_object()
            }
        }

        #[derive(Clone)]
        pub struct #ref_id(Option<tvm_rt::object::ObjectPtr<#payload_id>>);

        impl tvm_rt::object::ToObjectRef for #ref_id {
            fn to_object_ref(&self) -> ObjectRef {
                ObjectRef(self.0.as_ref().map(|o| o.upcast()))
            }
        }

        impl std::ops::Deref for #ref_id {
            type Target = #payload_id;

            fn deref(&self) -> &Self::Target {
                self.0.as_ref().unwrap()
            }
        }

        impl std::convert::TryFrom<tvm_rt::RetValue> for #ref_id {
            type Error = ::anyhow::Error;

            fn try_from(ret_val: tvm_rt::RetValue) -> Result<#ref_id, Self::Error> {
                use std::convert::TryInto;
                let oref: ObjectRef = ret_val.try_into()?;
                let ptr = oref.0.ok_or(anyhow::anyhow!("null ptr"))?;
                let ptr = ptr.downcast::<#payload_id>()?;
                Ok(#ref_id(Some(ptr)))
            }
        }

        impl<'a> From<#ref_id> for tvm_rt::ArgValue<'a> {
            fn from(object_ref: #ref_id) -> tvm_rt::ArgValue<'a> {
                use std::ffi::c_void;
                let object_ptr = &object_ref.0;
                match object_ptr {
                    None => {
                        tvm_rt::ArgValue::
                            ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
                    }
                    Some(value) => value.clone().into()
                }
            }
        }

        impl<'a> From<&#ref_id> for tvm_rt::ArgValue<'a> {
            fn from(object_ref: &#ref_id) -> tvm_rt::ArgValue<'a> {
                let oref: #ref_id = object_ref.clone();
                tvm_rt::ArgValue::<'a>::from(oref)
            }
        }

        impl<'a> std::convert::TryFrom<tvm_rt::ArgValue<'a>> for #ref_id {
            type Error = anyhow::Error;

            fn try_from(arg_value: tvm_rt::ArgValue<'a>) -> Result<#ref_id, Self::Error> {
                use std::convert::TryInto;
                let optr = arg_value.try_into()?;
                Ok(#ref_id(Some(optr)))
            }
        }

        impl<'a> std::convert::TryFrom<&tvm_rt::ArgValue<'a>> for #ref_id {
            type Error = anyhow::Error;

            fn try_from(arg_value: &tvm_rt::ArgValue<'a>) -> Result<#ref_id, Self::Error> {
                use std::convert::TryInto;
                let optr = arg_value.try_into()?;
                Ok(#ref_id(Some(optr)))
            }
        }

        impl From<#ref_id> for tvm_rt::RetValue {
            fn from(object_ref: #ref_id) -> tvm_rt::RetValue {
                use std::ffi::c_void;
                let object_ptr = &object_ref.0;
                match object_ptr {
                    None => {
                        tvm_rt::RetValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void)
                    }
                    Some(value) => value.clone().into()
                }
            }
        }

    };

    TokenStream::from(expanded)
}

//  impl TryFrom<RetValue> for Var {
//    type Error = anyhow::Error;

//    fn try_from(ret_val: RetValue) -> Result<Var, Self::Error> {
//       let oref: ObjectRef = ret_val.try_into()?;
//       let var_ptr = oref.0.ok_or(anyhow!("null ptr"))?;
//       let var_ptr = var_ptr.downcast::<VarNode>()?;
//       Ok(Var(Some(var_ptr)))
//    }
// }
