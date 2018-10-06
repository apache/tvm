use std::env;

fn main() {
  println!(
    "cargo:rustc-link-search=native={}",
    env::var("BUILD_DIR").unwrap()
  );
  println!("cargo:rustc-link-lib=static=model");
}
