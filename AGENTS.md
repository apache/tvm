<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# AGENTS.md

This file provides vendor-neutral guidance for agentic coding tools working
with Apache TVM.

## Repository Overview

Apache TVM is an open-source machine learning compiler stack. The repository
contains the C++ compiler/runtime, Python bindings, TIR/Relax IRs, scheduling
and lowering passes, target code generators, runtime integrations, tests,
documentation, and application examples.

## Repository Structure

- `include/tvm/` - public C++ headers
- `src/` - C++ implementation
- `python/tvm/` - Python package
- `tests/` - C++, Python, integration, and lint tests
- `cmake/` - CMake modules and default configuration
- `3rdparty/` - vendored dependencies and submodules
- `docs/` - documentation source
- `apps/` - application examples
- `.agents/skills/` - reusable agent workflows for this repository

## Build

Use an existing `build/` directory when present:

```bash
cmake --build build --parallel
```

For a fresh checkout, initialize submodules and configure CMake first:

```bash
git submodule update --init --recursive
mkdir -p build
cp cmake/config.cmake build/config.cmake
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
```

Development should use `PYTHONPATH`, not editable installs:

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/.local/python"
```

Do not use `pip install -e` for TVM or `tvm-ffi`; editable installs can make
one worktree silently import another worktree's code.

## Test And Lint

Run the smallest relevant test first, then broaden as needed. Common examples:

```bash
python -m pytest tests/python/all-platform-minimal-test/ -xvs
python -m pytest tests/python/tir-base/test_tir_base.py -xvs
./build/cpptest
```

For lint validation on a pull request, run pre-commit on the files changed by
the branch instead of the whole repository:

```bash
pre-commit run --files <changed-file>...
```

Use `python -m tirx_kernels.bench_suite` in the **tirx-kernels** repo
(`tirx_kernels/bench_suite/`) when that workflow applies.

## Coding Conventions

- Follow the surrounding style before introducing new abstractions.
- Keep changes scoped to the task and avoid unrelated cleanups.
- Prefer explicit tests that show the IR or behavior being changed.
- Use Apache TVM commit tags such as `[REFACTOR][IR]`, `[FIX][TIR]`, or
  `[DOCS]` as appropriate.
- Preserve Apache license headers in new source, script, and documentation
  files when the surrounding tree uses them.
