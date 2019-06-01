#![feature(bind_by_move_pattern_guards)]

extern crate proc_macro;

use std::{fs::File, io::Read, path::Path};

use proc_quote::quote;

#[proc_macro]
pub fn import_module(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::LitStr).value();

    let path = Path::new(&input).canonicalize().unwrap();
    let mut fd = File::open(path).unwrap();
    let mut buffer = Vec::new();
    fd.read_to_end(&mut buffer).unwrap();

    let fn_names = match goblin::Object::parse(&buffer).unwrap() {
        goblin::Object::Elf(elf) => elf
            .syms
            .iter()
            .filter_map(|s| {
                if s.st_type() == 0 || goblin::elf::sym::type_to_str(s.st_type()) == "FILE" {
                    return None;
                }
                match elf.strtab.get(s.st_name) {
                    Some(Ok(name)) if name != "" => {
                        Some(syn::Ident::new(name, proc_macro2::Span::call_site()))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>(),
        goblin::Object::Mach(goblin::mach::Mach::Binary(obj)) => {
            obj.symbols()
                .filter_map(|s| match s {
                    Ok((name, nlist))
                        if nlist.is_global()
                            && nlist.n_sect != 0
                            && !name.ends_with("tvm_module_ctx") =>
                    {
                        Some(syn::Ident::new(
                            if name.starts_with('_') {
                                // Mach objects prepend a _ to globals.
                                &name[1..]
                            } else {
                                &name
                            },
                            proc_macro2::Span::call_site(),
                        ))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        }
        _ => panic!("Unsupported object format."),
    };

    let extern_fns = quote! {
        mod ext {
            extern "C" {
                #(
                    pub(super) fn #fn_names(
                        args: *const tvm_runtime::ffi::TVMValue,
                        type_codes: *const std::os::raw::c_int,
                        num_args: std::os::raw::c_int
                    ) -> std::os::raw::c_int;
                )*
            }
        }
    };

    let fns = quote! {
        use tvm_runtime::{ffi::TVMValue, TVMArgValue, TVMRetValue, FuncCallError};
        #extern_fns

        #(
            pub fn #fn_names(args: &[TVMArgValue]) -> Result<TVMRetValue, FuncCallError> {
                let (values, type_codes): (Vec<TVMValue>, Vec<i32>) = args
                   .into_iter()
                   .map(|arg| {
                       let (val, code) = arg.to_tvm_value();
                       (val, code as i32)
                   })
                   .unzip();
                let exit_code = unsafe {
                    ext::#fn_names(values.as_ptr(), type_codes.as_ptr(), values.len() as i32)
                };
                if exit_code == 0 {
                    Ok(TVMRetValue::default())
                } else {
                    Err(FuncCallError::get_with_context(stringify!(#fn_names).to_string()))
                }
            }
        )*
    };

    proc_macro::TokenStream::from(fns)
}
