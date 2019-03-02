## Resnet example

This end-to-end example shows how to:
* build `Resnet 18` with `tvm` and `nnvm` from Python
* use the provided Rust frontend API to test for an input image

To run the example, first `tvm`, `nnvm` and `mxnet` must be installed for the python build. To install mxnet for cpu, run `pip install mxnet`
and to install `tvm` and `nnvm` with `llvm` follow the [TVM installation guide](https://docs.tvm.ai/install/index.html).

* **Build the example**: `cargo build`

To have a successful build, note that it is required to instruct Rust compiler to link to the compiled shared library, for example with
`println!("cargo:rustc-link-search=native={}", build_path)`. See the `build.rs` for more details.

* **Run the example**: `cargo run`
