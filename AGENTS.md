<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# AGENTS.md — TVM TIRx

## What is this project?

TIRx is a compiler foundation for development of ML compilers and agentic compiler
harnesses. It provides shared IR infrastructure, symbolic analysis, program
transformations, code generation, and runtime support. TIRx gives developers and
agents direct access to compiler internals, from shared abstractions to
implementation details. Its components are designed for inspection,
customization, and composition, making it easy to build new compilers and
harnesses on top. The Python package is `tvm_tirx`; the distribution is
`apache-tvm-tirx`.

## Repository layout

```text
include/tvm/        C++ public headers (ir, sym, tirx, target, runtime, backends)
src/                C++ compiler and runtime implementations
python/tvm_tirx/    Python package and reflected C++ bindings
tests/cpp/          GoogleTest C++ tests
tests/python/       pytest tests organized by owning namespace and backend
tests/lint/         ASF header and file-type checks
tests/scripts/      Build, wheel, test, and documentation entry points
docs/               Sphinx Python reference and autosummary templates
cmake/              Build modules, dependency finders, and package configuration
3rdparty/           TVM FFI submodule and compiler support code
ci/jenkins/         Linux CPU, GPU, and ARM pipelines
ci/wheel/           Wheel-build support
.github/            GitHub Actions and reusable build actions
```

## Building

Use `uv` for Python environments and package installation. Run commands from
the repository root; the default virtual environment is `.venv`.

### Prerequisites

- Python 3.10+, a C++20 compiler, CMake 3.26+, and Ninja.
- LLVM development libraries and `llvm-config` for the default LLVM backend.
- Initialize dependencies: `git submodule update --init --recursive`.
- Install the relevant device SDK when enabling a device runtime.

### Python editable install (primary workflow)

```bash
uv venv
uv pip install --verbose --group dev -e .
```

Re-run the editable install after changing C++ code or build configuration.
Pure Python edits are reflected immediately.

Pass native build settings through `CMAKE_ARGS`, for example:

```bash
CMAKE_ARGS="-DTVM_TIRX_USE_LLVM=ON -DTVM_TIRX_USE_CUDA=OFF" \
  uv pip install --force-reinstall --verbose -e .
```

The main switches are `TVM_TIRX_USE_LLVM`, `TVM_TIRX_USE_CUDA`,
`TVM_TIRX_USE_Z3`, and `TVM_TIRX_BUILD_TESTS`. Device runtimes such as Metal,
ROCm, Vulkan, and OpenCL have their own `TVM_TIRX_USE_*` options.

### C++-only build

```bash
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_cpp --parallel --config RelWithDebInfo --target tvm_tirx_compiler
```

The `runtime` target builds the core runtime and enabled device runtimes.

## Testing

### C++ tests

```bash
cmake -S . -B build_test -DTVM_TIRX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_test --parallel --config Debug --target tirx_cpptest
ctest --test-dir build_test -C Debug --output-on-failure
```

CMake uses an installed GoogleTest package when available and fetches it otherwise.

### Python tests

After the editable install with the `dev` or `test` dependency group:

```bash
uv run --no-sync pytest -v tests/python -m "not gpu"
uv run --no-sync pytest -v tests/python/ir/test_type.py
```

Run the relevant GPU tests on a machine with the matching runtime and hardware:

```bash
uv run --no-sync pytest -v tests/python -m gpu
```

Start with the tests for the changed component. Device and architecture skips
describe missing coverage; they do not demonstrate that a backend works.

## Linting

### Pre-commit (primary lint workflow)

```bash
uv run --no-sync pre-commit run --all-files
uv run --no-sync pre-commit run --files path/to/changed_file
```

Hooks include ASF headers, file types, whitespace, YAML/TOML checks, `ruff`,
and `clang-format`. CUDA launch changes also check generated files.

## Code conventions

### C++

- Source files use `.cc`; headers use `.h`.
- Follow `.clang-format`: Google style, 100 columns, pointer-left (`int* ptr`).
- Keep declarations and implementations in their owning IR, runtime, or backend layer.
- Header guards follow the existing `TVM_<PATH>_H_` pattern.
- IR objects use a `FooNode` data class and a `Foo` reference wrapper.
  Declare runtime type information with `TVM_FFI_DECLARE_OBJECT_INFO` or its
  final variant, and expose fields through `ffi::reflection::ObjectDef`.
- Use the existing TVM FFI error and reference-management conventions.
- Public APIs use Doxygen comments. Retain Apache 2.0 license headers.

### Python

- Import the standalone package as `tvm_tirx`.
- Follow the repository's `ruff` configuration: 100 columns and double quotes.
- Bind reflected objects with `@tvm_ffi.register_object("type.Key")`.
- Keep Python exports, reflected C++ APIs, and reference documentation consistent.
- Place tests beside the corresponding namespace or backend tests.

### Commit messages

Use the existing tag-based style, for example `[FIX]`, `[REFACTOR]`, `[FEAT]`,
`[DOCS]`, or `[CI]`, followed by a concise description of the change.

## Key architecture concepts

- **IR**: `tvm_tirx.ir` owns shared expressions, types, statements, functions,
  modules, and source locations.
- **TIRx**: `tvm_tirx.tirx` adds tensor-program constructs, layouts, analysis,
  and lowering passes on top of the shared IR.
- **Symbolic analysis**: `tvm_tirx.sym` owns simplification and symbolic reasoning.
- **Script**: shared parsing and printing live in `tvm_tirx.script`; the TIRx
  dialect provides its builders and tensor-program syntax.
- **Backends**: `tvm_tirx.backend` contains target-specific operations,
  transformations, code generation, and runtime support.
- **Runtime and FFI**: TVM FFI supplies objects, reflection, and packed function
  calls. TIRx supplies its compiler library and device-aware runtime.

## Documentation

After building TIRx, install the documentation dependency group and build:

```bash
uv pip install --group docs
uv run --no-sync python -m sphinx -b html -W docs docs/_build/html
```

Use standard RST and autosummary templates. Keep public namespace labels fully
qualified and related namespaces adjacent in the Python API navigation.

## CI

GitHub Actions runs macOS and Windows checks and a separate lint workflow.
Jenkins runs Linux CPU, GPU, and ARM builds and tests. The GPU pipeline builds
the documentation alongside tests after the wheel build. The executable entry
points are in `tests/scripts/`; consult them for the exact CI environment.

## Further reading

- `docs/README.md` — local documentation build.
- `docs/reference/python/index.rst` — public Python API organization.
- `pyproject.toml` — package metadata, dependency groups, and Python tooling.
- `CMakeLists.txt` — native targets and configuration options.
