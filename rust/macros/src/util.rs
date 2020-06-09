use proc_macro2::TokenStream;
use quote::quote;
use std::env;

pub fn get_tvm_rt_crate() -> TokenStream {
    if env::var("CARGO_PKG_NAME").unwrap() == "tvm-rt" {
        quote!(crate)
    } else {
        quote!(tvm_rt)
    }
}
