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

# TVM wheel packaging helper

This directory contains the helper scripts used to build, repair, verify, and
publish TVM Python wheels. The GitHub Actions workflow keeps orchestration in
YAML and puts platform-specific packaging behavior in focused composite actions
and shell/Python helpers.

The wheel build flow is:

1. On Linux, when a CUDA runtime is requested, `cibuildwheel`'s
   `CIBW_BEFORE_ALL_LINUX` hook installs the CUDA toolkit inside the manylinux
   container and builds `libtvm_runtime_cuda.so` (see `before_all_linux.sh`).
2. Build the main Python wheel with `cibuildwheel`, LLVM enabled, and CUDA
   disabled. The prebuilt CUDA runtime path is passed to CMake via
   `TVM_PACKAGE_EXTRA_LIBS`, so CMake installs it into `tvm/lib/` as part of the
   normal build — no separate CUDA build job and no post-build wheel rewriting.
3. Repair the wheel with the standard per-platform tool — `auditwheel`
   (Linux), `delocate` (macOS), `delvewheel` (Windows) — excluding the
   tvm-ffi library (resolved from the apache-tvm-ffi package) and, on Linux,
   the intentionally bundled `libtvm_runtime_cuda.so` plus its CUDA driver
   dependencies. LLVM is linked statically, so the final wheel must not bundle
   or dynamically depend on `libLLVM`.
4. Verify the installed wheel with the `tests/python/wheel` pytest suite via the
   `[tool.cibuildwheel]` `test-command`.
5. Optionally upload and verify the uploaded package.

GitHub Actions flow:

1. The `Publish TVM wheels` workflow builds a platform wheel matrix:
   - Linux x86_64 in cibuildwheel's default `manylinux_2_28` container, with the
     CUDA runtime.
   - Linux aarch64 in cibuildwheel's default `manylinux_2_28` container, with the
     CUDA runtime.
   - macOS arm64 CPU-only.
   - Windows AMD64 CPU-only.
2. On Linux the CUDA runtime is built inside the same `cibuildwheel` container
   via the `CIBW_BEFORE_ALL_LINUX` hook, so there is no separate CUDA build job
   or cross-step artifact handoff.
3. The optional publishing jobs upload the artifacts and can verify the package
   from the selected package index. PyPI publishing requires a `refs/tags/<tag>`
   input and keeps post-upload verification enabled.

Linux wheels are built inside manylinux images. This avoids accidentally
publishing a wheel tagged for the GitHub runner's host glibc, such as
`manylinux_2_39`, which would not install on older supported Linux systems.

Where configuration lives:

Package build behaviour is declared in `pyproject.toml` so it applies to *every*
build — `pip install .`, a local `cibuildwheel` run, another CI, a fork, or an
upstream release pipeline — not just this workflow. The GitHub Actions workflow
adds only the environment that a *specific CI run* provides on top.

In `pyproject.toml` (stable, package-intrinsic — correct wherever cibuildwheel /
scikit-build-core runs):

- `[build-system].requires`: the build toolchain (`scikit-build-core`, `cmake`,
  `ninja`).
- `[tool.scikit-build.cmake.define]`: static CMake options shared by every build
  (`TVM_BUILD_PYTHON_MODULE=ON`, `USE_CUDA=OFF`, `BUILD_TESTING=OFF`,
  `ZLIB_USE_STATIC_LIBS=ON`).
- `[tool.cibuildwheel]`: `build-verbosity`, `test-requires`, `test-command`.
- `[tool.cibuildwheel.{linux,macos,windows}]`: the per-platform `before-build`
  and `repair-wheel-command` (the `auditwheel`/`delocate`/`delvewheel` excludes).

In the workflow `env:` (dynamic — describes only this CI run, cannot be static):

- `CIBW_BUILD` / `CIBW_ARCHS_*`: the per-architecture build selector. The
  `manylinux_*` build tag selects cibuildwheel's default `manylinux_2_28` image,
  so no image override is needed.
- `CIBW_CONTAINER_ENGINE`: bind-mounts the cached `/opt/llvm` prefix into the
  Linux build container.
- `CIBW_ENVIRONMENT`: the `USE_LLVM` config path, `CMAKE_PREFIX_PATH`, and the
  CUDA `TVM_PACKAGE_EXTRA_LIBS` path — all depend on where the runner installed
  things.
- `CIBW_BEFORE_ALL_LINUX`: the CUDA-runtime build (its arguments depend on the
  per-run CUDA architecture and whether CUDA is requested).
- `CIBW_TEST_ENVIRONMENT`: the `TVM_EXPECT_*` post-install expectations (they
  track whether this wheel was built with the CUDA runtime).
- The TestPyPI distribution name/version overrides.

Rule of thumb: if a setting is still correct when the package is built in a
different environment, it belongs in `pyproject.toml`; if it only describes what
this particular CI run provides, it stays in the workflow.

Workflow structure:

- `.github/workflows/publish_wheel.yml`: defines the platform matrix,
  artifact upload, optional publishing, and post-upload verification.
- `.github/actions/detect-env-vars`: shared environment detection.
- `.github/actions/build-wheel-for-publish`: installs the cached LLVM prefix
  and runs `pypa/cibuildwheel` for the LLVM-enabled runtime wheel. On Linux the
  CUDA runtime is built in the `CIBW_BEFORE_ALL_LINUX` hook, installed by CMake
  (`TVM_PACKAGE_EXTRA_LIBS`), and the wheel is repaired with standard
  `auditwheel`/`delocate`/`delvewheel`.
- `ci/scripts/package/before_all_linux.sh`: `CIBW_BEFORE_ALL_LINUX` hook that
  installs the CUDA toolkit in the manylinux container and builds
  `libtvm_runtime_cuda.so` (no-op for CPU-only wheels).
- `ci/scripts/package/tvm_wheel_helper.sh`: implements the build pieces
  `cibuildwheel` cannot — building the CUDA runtime sidecar (`cuda`) and
  post-publish package verification (`verify-pypi`).
- `ci/scripts/package/set_wheel_dist.py`: applies optional distribution
  name/version overrides to `[project]` before the build (used for TestPyPI
  validation builds), so the backend produces the desired wheel directly.
- `tests/python/wheel/`: pytest checks run against the installed wheel (via the
  `[tool.cibuildwheel]` `test-command`). They import tvm, run a minimal LLVM
  compile, and assert the bundled libraries are correct (no dynamic LLVM when
  static LLVM is required, CUDA runtime present when expected). Each check is
  gated by a `TVM_EXPECT_*` environment variable.

To test the workflow from a fork without publishing:

```bash
git push origin HEAD:<branch>
git tag -a tvm-wheel-test0 -m "Test TVM wheel workflow"
git push origin tvm-wheel-test0

gh workflow run publish_wheel.yml \
  --repo <owner>/<repo> \
  --ref <branch> \
  -f tag=tvm-wheel-test0 \
  -f publish_repository=none \
  -f distribution_name=<temporary-package-name> \
  -f cuda_architectures=75 \
  -f verify_from_repository=false
```

If the workflow is not visible in the GitHub UI yet, push or merge these files
to the fork's default branch first. GitHub only lists manually dispatched
workflows once the workflow file exists in the repository.

Local debugging:

The main wheel build, repair, and post-install tests are owned by
`cibuildwheel` (see `pyproject.toml` `[tool.cibuildwheel]` and the publish
workflow). The shell helper only covers what `cibuildwheel` cannot: building
the CUDA runtime sidecar and verifying an already-published package. Use
`.github/actions/build-wheel-for-publish/action.yml` as the source of truth for
the exact `cibuildwheel` environment.

To test a locally built wheel under `wheelhouse/`, install it and run the same
suite `cibuildwheel` uses:

```bash
python -m pip install wheelhouse/*.whl pytest numpy
python -m pytest -c tests/python/wheel/pytest.ini tests/python/wheel
```

To verify a package that has already been uploaded to an index, run:

```bash
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/tvm_wheel_helper.sh verify-pypi
```

Publishing itself is handled in the workflow by the standard
`pypa/gh-action-pypi-publish` action (trusted publishing), not this helper.

Useful knobs:

- `TVM_USE_LLVM`: LLVM config for the CIBW build and repair helpers, default
  `llvm-config --link-static`.
- `TVM_USE_CUDA`: CUDA root or `ON` for the CUDA build, default `ON`.
- `TVM_CUDA_RUNTIME_PATH`: explicit path to `libtvm_runtime_cuda.so` for repair.
- `TVM_CUDA_ARCHITECTURES`: CMake CUDA architectures, default `75`.
- `TVM_WHEEL_DIST_NAME`: optional distribution rename for TestPyPI.
- `TVM_WHEEL_DIST_VERSION`: optional distribution version rewrite.
- `TVM_INCLUDE_CUDA_RUNTIME=1`: build `libtvm_runtime_cuda.so`. Do not set this
  to a value that conflicts with `TVM_SKIP_CUDA`.
- `TVM_SKIP_CUDA=1`: skip building `libtvm_runtime_cuda.so`.
- `TVM_KEEP_BUILD_DIRS=1`: reuse the CMake build directories.
- `TVM_CUDA_BUILD_DIR`: build directory for the CUDA runtime, default
  `<repo>/build-wheel-cuda`.
- `TVM_TEST_INDEX_URL`: package index for `verify-pypi`, default TestPyPI.
- `TVM_EXTRA_INDEX_URL`: extra package index for dependencies, default PyPI.
