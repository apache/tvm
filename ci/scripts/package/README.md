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

# TVM wheel packaging

The TVM wheels are built with `cibuildwheel`. Configuration intrinsic to the
package lives in `pyproject.toml`; the GitHub Actions workflow only adds the
per-run environment on top. This directory holds the few helpers `cibuildwheel`
cannot provide itself.

## Build flow

1. **(Linux CUDA wheels only)** the `CIBW_BEFORE_ALL_LINUX` hook runs
   `manylinux_build_libtvm_runtime_cuda.sh`, which installs the CUDA toolkit inside the manylinux
   container and builds `libtvm_runtime_cuda.so`.
2. `cibuildwheel` builds the main wheel with LLVM linked **statically** and CUDA
   off. The prebuilt CUDA runtime is installed into `tvm/lib/` by CMake via
   `TVM_PACKAGE_EXTRA_LIBS` — no post-build wheel rewriting.
3. The wheel is repaired with the standard per-platform tool — `auditwheel`
   (Linux) / `delocate` (macOS) / `delvewheel` (Windows) — excluding `libtvm_ffi`
   (provided by the `apache-tvm-ffi` package) and, on Linux, the bundled
   `libtvm_runtime_cuda.so` and its CUDA driver dependencies.
4. The `tests/python/wheel` suite runs against the installed wheel via the
   `[tool.cibuildwheel]` `test-command`.

The publish workflow builds four wheels — Linux x86_64 and aarch64 (with the
CUDA runtime, in cibuildwheel's default `manylinux_2_28` image), macOS arm64, and
Windows AMD64 — then optionally uploads them with `pypa/gh-action-pypi-publish`
and verifies the uploaded package.

## Where configuration lives

Settings intrinsic to the package go in `pyproject.toml`, so they apply to
*every* build — `pip install .`, a local `cibuildwheel` run, a fork, or an
upstream pipeline. The workflow sets only what a *specific CI run* provides.

`pyproject.toml` (stable):

- `[build-system].requires` — the build toolchain (`scikit-build-core`, `cmake`,
  `ninja`).
- `[tool.scikit-build.cmake.define]` — static CMake options (`USE_CUDA=OFF`,
  `BUILD_TESTING=OFF`, `TVM_BUILD_PYTHON_MODULE=ON`, `ZLIB_USE_STATIC_LIBS=ON`).
- `[tool.cibuildwheel]` and `[tool.cibuildwheel.{linux,macos,windows}]` — `skip`,
  test config, and the per-platform `before-build` / `repair-wheel-command`.

Workflow `env:` (per-run):

- `CIBW_BUILD` / `CIBW_ARCHS_*` — the architecture; the `manylinux_*` tag already
  selects the default `manylinux_2_28` image, so no image override is set.
- `CIBW_ENVIRONMENT` — the `USE_LLVM` config path, `CMAKE_PREFIX_PATH`, and the
  CUDA library path (all depend on where the runner installed things).
- `CIBW_CONTAINER_ENGINE` — bind-mounts the cached `/opt/llvm` prefix.
- `CIBW_BEFORE_ALL_LINUX`, `CIBW_TEST_ENVIRONMENT`, and the optional dist
  name/version override.

Rule of thumb: still correct when the package is built elsewhere → `pyproject.toml`;
only describes this CI run → the workflow.

## Files

- `.github/workflows/publish_wheel.yml` — the platform matrix, upload, and verify.
- `.github/actions/build-wheel-for-publish` — installs the cached LLVM prefix and
  runs `cibuildwheel`.
- `.github/actions/detect-env-vars` — shared environment detection (CPU count).
- `manylinux_build_libtvm_runtime_cuda.sh` — builds the CUDA runtime sidecar
  (run from `CIBW_BEFORE_ALL_LINUX`); the one build step `cibuildwheel` cannot
  do, since its container ships no CUDA toolkit. No-op for CPU-only wheels.
- `set_wheel_dist.py` — overrides the wheel name/version before the build. This is
  a **fork/development convenience for TestPyPI validation only**; it is unused
  for a normal `tvm` release, and the workflow forbids the override when
  publishing to PyPI.
- `tests/python/wheel/` — post-install smoke checks (import `tvm`, a minimal LLVM
  compile, and that the bundled libraries are correct), each gated by a
  `TVM_EXPECT_*` variable. `pytest.ini` isolates them from the repo's root test
  config.

## Testing

Test a locally built wheel with the same suite `cibuildwheel` uses:

```bash
python -m pip install wheelhouse/*.whl pytest numpy
python -m pytest -c tests/python/wheel/pytest.ini tests/python/wheel
```

Run the workflow from a fork without publishing (add
`-f distribution_name=<temp-name>` for a TestPyPI validation build):

```bash
gh workflow run publish_wheel.yml --repo <owner>/<repo> --ref <branch> \
  -f tag=<branch-or-tag> -f publish_repository=none -f verify_from_repository=false
```
