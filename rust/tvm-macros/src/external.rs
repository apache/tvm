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
use proc_macro2::Span;
use proc_macro_error::abort;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::parse_quote;

use syn::{
    token::Semi, Attribute, FnArg, Generics, Ident, Lit, Meta, NestedMeta, Pat, ReturnType,
    Signature, Type, Visibility,
};

struct ExternalItem {
    attrs: Vec<Attribute>,
    visibility: Visibility,
    sig: Signature,
}

impl Parse for ExternalItem {
    fn parse(input: ParseStream) -> Result<Self> {
        let item = ExternalItem {
            attrs: input.call(Attribute::parse_outer)?,
            visibility: input.parse()?,
            sig: input.parse()?,
        };
        let _semi: Semi = input.parse()?;
        Ok(item)
    }
}

struct External {
    visibility: Visibility,
    tvm_name: String,
    ident: Ident,
    generics: Generics,
    inputs: Vec<FnArg>,
    ret_type: ReturnType,
}

impl Parse for External {
    fn parse(input: ParseStream) -> Result<Self> {
        let method: ExternalItem = input.parse()?;
        let visibility = method.visibility;
        assert_eq!(method.attrs.len(), 1);
        let sig = method.sig;
        let tvm_name = method.attrs[0].parse_meta()?;
        let tvm_name = match tvm_name {
            Meta::List(meta_list) => {
                let name = meta_list.path.get_ident().expect("name");
                assert_eq!(name.to_string(), "name".to_string());
                match meta_list.nested.first() {
                    Some(NestedMeta::Lit(Lit::Str(lit))) => lit.value(),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        };

        let ident = sig.ident;
        let generics = sig.generics;
        let inputs = sig
            .inputs
            .iter()
            .cloned()
            .map(|param| param.clone())
            .collect();
        let ret_type = sig.output;

        Ok(External {
            visibility,
            tvm_name,
            ident,
            generics,
            inputs,
            ret_type,
        })
    }
}

struct ExternalInput {
    externs: Vec<External>,
}

impl Parse for ExternalInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut externs: Vec<External> = Vec::new();

        loop {
            if input.is_empty() {
                break;
            }
            externs.push(input.parse()?);
        }

        Ok(ExternalInput { externs })
    }
}

/// Returns true if the type is `bool`, false otherwise
fn is_bool(ty: &syn::Type) -> bool {
    match ty {
        Type::Path(ty_path) => ty_path.path.is_ident("bool"),
        _ => false,
    }
}

pub fn macro_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ext_input = syn::parse_macro_input!(input as ExternalInput);

    let tvm_rt_crate = crate::util::get_tvm_rt_crate();

    let result_type = quote! { #tvm_rt_crate::function::Result };

    let mut items = Vec::new();

    for external in &ext_input.externs {
        let visibility = &external.visibility;
        let name = &external.ident;
        let global_name = format!("global_{}", external.ident);
        let global_name = Ident::new(&global_name, Span::call_site());
        let ext_name = &external.tvm_name;

        let ty_params: Vec<syn::TypeParam> = external
            .generics
            .params
            .iter()
            .map(|ty_param| match ty_param {
                syn::GenericParam::Type(param) => param.clone(),
                _ => abort! { ty_param,
                    "Only supports type parameters."
                },
            })
            .collect();

        let args = &external.inputs;

        // Pre-processing of arguments, done before calling the
        // external PackedFunc.
        let mut arg_pre_proc = Vec::<syn::Stmt>::new();

        // Post-processing of arguments, done after calling the
        // external PackedFunc.
        let mut result_post_proc = Vec::<syn::Stmt>::new();

        let (args, arg_tys): (Vec<Ident>, Vec<Type>) = args
            .iter()
            .map(|arg| match arg {
                FnArg::Typed(pat_type) => match &*pat_type.pat {
                    Pat::Ident(pat_ident) => {
                        let ident: Ident = pat_ident.ident.clone();
                        let ty: Type = *pat_type.ty.clone();
                        (ident, ty)
                    }
                    _ => abort! { pat_type,
                        "Only supports type parameters."
                    },
                },
                pat => abort! {
                    pat, "invalid pattern type for function";

                    note = "{:?} is not allowed here", pat;
                },
            })
            .unzip();

        let (external_arg_tys, internal_arg_tys): (Vec<Type>, Vec<Type>) = arg_tys
            .into_iter()
            .zip(args.iter())
            .map(|(ty, arg)| -> (Type, Type) {
                // A boolean is represented in the FFI as `runtime::Bool`, a
                // child class of ObjectRef, to avoid ambiguity with integer
                // types.
                //
                // * The `tvm_sys` crate cannot (easily) call the
                //   `runtime.BoxBool` PackedFunc to construct an instance,
                //   because the argument packing is defined in `tvm_rt`.
                //
                // * The implementation `TryFrom<ArgValue<'_>> for bool`,
                //   cannot be moved to `tvm_rt`, because `ArgValue` is
                //   defined in `tvm_sys`.
                //
                // Instead, defining these conversions as part of the
                // external wrapping.  For boolean arguments, pre-process by
                // wrapping in BoxBool.  For boolean results, post-process
                // by unwrapping the BoxBool.
                let internal_ty: Type = if is_bool(&ty) {
                    let box_bool = parse_quote! {#tvm_rt_crate::boxed_primitive::BoxBool};
                    arg_pre_proc.push(parse_quote! {
                        let #arg: #box_bool = #arg.into();
                    });
                    box_bool
                } else {
                    ty.clone()
                };
                (ty, internal_ty)
            })
            .unzip();

        let ret_type = match &external.ret_type {
            ReturnType::Type(_, rtype) => *rtype.clone(),
            ReturnType::Default => syn::parse_str::<Type>("()").unwrap(),
        };

        let internal_ret_type: Type = if is_bool(&ret_type) {
            let box_bool = parse_quote! {#tvm_rt_crate::boxed_primitive::BoxBool};
            result_post_proc.push(parse_quote! {
                let res: #ret_type = res.into();
            });
            box_bool
        } else {
            ret_type.clone()
        };

        let global = quote! {
            #[allow(non_upper_case_globals)]
            static #global_name: ::once_cell::sync::Lazy<#tvm_rt_crate::Function> =
            ::once_cell::sync::Lazy::new(|| {
                #tvm_rt_crate::Function::get(#ext_name)
                .expect(concat!("unable to load external function", stringify!(#ext_name), "from TVM registry."))
            });
        };

        items.push(global);

        let wrapper = quote! {
            #visibility fn #name<#(#ty_params),*>(#(#args : #external_arg_tys),*) -> #result_type<#ret_type> {
                let func_ref: #tvm_rt_crate::Function = #global_name.clone();
                let func_ref: Box<
                    dyn Fn(#(#internal_arg_tys),*) -> #result_type<#internal_ret_type>
                  > = func_ref.into();

                #(#arg_pre_proc)*
                let res: #internal_ret_type = func_ref(#(#args),*)?;
                #(#result_post_proc)*
                Ok(res)
            }
        };

        items.push(wrapper);
    }

    proc_macro::TokenStream::from(quote! {
        #(#items
        )*
    })
}
