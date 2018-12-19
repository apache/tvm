How to Bundle TVM Modules
=========================

This folder contains an example on how to bundle a TVM module (with the required
interpreter runtime modules such as `runtime::GraphRuntime`, the graph JSON, and
the params) into a single, self-contained shared object (`bundle.so`) which
exposes a C API wrapping the appropriate `runtime::GraphRuntime` instance.

This is useful for cases where we'd like to avoid deploying the TVM runtime
components to the target host in advance - instead, we simply deploy the bundled
shared-object to the host, which embeds both the model and the runtime
components. The bundle should only depend on libc/libc++.

It also contains an example code (`demo.cc`) to load this shared object and
invoke the packaged TVM model instance. This is a dependency-free binary that
uses the functionality packaged in `bundle.so` (which means that `bundle.so` can
be deployed lazily at runtime, instead of at compile time) to invoke TVM
functionality.

Type the following command to run the sample code under the current folder,
after building TVM first.

```bash
make demo
```

This will:

- Download the mobilenet0.25 model from the MXNet Gluon Model Zoo
- Compile the model with NNVM
- Build a `bundle.so` shared object containing the model specification and
  parameters
- Build a `demo` executable that `dlopen`'s `bundle.so`, instantiates the
  contained graph runtime, and invokes the `GraphRuntime::Run` function on a
  random input, then prints the output tensor to `stderr`.
