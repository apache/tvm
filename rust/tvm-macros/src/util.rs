use std::env;
use quote::quote;
use proc_macro2::TokenStream;

pub fn get_tvm_rt_crate() -> TokenStream {
    if env::var("CARGO_PKG_NAME").unwrap() == "tvm-rt" {
        quote!( crate )
    } else {
        quote!( tvm_rt )
    }
}
