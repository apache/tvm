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

1. Optionally build `libtvm_runtime_cuda.so` in a CUDA-enabled Linux CMake build.
2. Build the main Python wheel with `cibuildwheel`, LLVM enabled, and CUDA
   disabled.
3. When requested, inject the CUDA runtime DSO into `tvm/lib/` during the
   `cibuildwheel` repair hook.
4. Repair the wheel, excluding CUDA toolkit/driver DSOs and `libtvm_ffi`.
   `libtvm_runtime_cuda.so`, when requested, is the TVM CUDA runtime that is
   intentionally injected into the wheel.
   On Windows, copy the small runtime DLLs required by LLVM support libraries
   into `tvm/lib/` because there is no auditwheel-style repair tool.
5. Validate ELF links so intra-wheel TVM DSOs resolve through relative rpaths.
   LLVM is expected to be linked statically; the final wheel must not bundle
   or dynamically depend on `libLLVM`.
6. Verify the wheel in a fresh virtualenv.
7. Optionally upload and verify the uploaded package.

GitHub Actions flow:

1. The `Publish TVM wheels` workflow builds a platform wheel matrix:
   - Linux x86_64 in a pinned `manylinux_2_28` container, with the CUDA runtime.
   - Linux aarch64 in a pinned `manylinux_2_28` container, with the CUDA runtime.
   - macOS arm64 CPU-only.
   - Windows AMD64 CPU-only.
2. The Linux CUDA runtime action exposes the built DSO path as an action output.
   The wheel action receives that path explicitly and mounts it into the
   `cibuildwheel` container for the repair hook.
3. The optional publishing jobs upload the artifacts and can verify the package
   from the selected package index. PyPI publishing requires a `refs/tags/<tag>`
   input and keeps post-upload verification enabled.

Linux wheels are built inside manylinux images. This avoids accidentally
publishing a wheel tagged for the GitHub runner's host glibc, such as
`manylinux_2_39`, which would not install on older supported Linux systems.

Workflow structure:

- `.github/workflows/publish_wheel.yml`: defines the platform matrix,
  artifact upload, optional publishing, and post-upload verification.
- `.github/actions/detect-env-vars`: shared environment detection.
- `.github/actions/build-cuda`: builds only the optional CUDA runtime library.
  On Linux this action owns the pinned manylinux Docker/CUDA setup and exposes
  the runtime DSO path as an action output.
- `.github/actions/build-wheel-for-publish`: installs the cached LLVM prefix
  and runs `pypa/cibuildwheel` for the LLVM-enabled runtime wheel. Its custom
  repair hook injects the CUDA runtime before `auditwheel`/`delocate`/Windows
  dependency-copy repair.
- `ci/scripts/package/tvm_wheel_helper.sh`: implements reusable local and CI
  entrypoints around the `cibuildwheel` build, such as `cuda`,
  `manylinux-cuda`, `cibw-repair`, `verify`, `upload`, and `verify-pypi`.
- `ci/scripts/package/rewrite_wheel.py`: rewrites wheel metadata and injects
  extra runtime files, including the CUDA runtime library when CUDA is enabled.
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

The main wheel build is owned by `cibuildwheel`. The shell helper is used for
the build pieces around `cibuildwheel`: CUDA runtime construction, the
`CIBW_REPAIR_WHEEL_COMMAND` hook, final wheel verification, and optional
publish verification.

For the exact `cibuildwheel` environment, use
`.github/actions/build-wheel-for-publish/action.yml` as the source of truth.
For local checks after a wheel exists under `wheelhouse/`, run:

```bash
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/tvm_wheel_helper.sh verify

TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/tvm_wheel_helper.sh verify-pypi
```

For a manual or local upload with the helper, leave `TVM_WHEEL_DIST_NAME`
unset and set the normal Twine credentials:

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_TOKEN" \
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/tvm_wheel_helper.sh upload
```

Useful knobs:

- `TVM_USE_LLVM`: LLVM config for the CIBW build and repair helpers, default
  `llvm-config --link-static`.
- `TVM_USE_CUDA`: CUDA root or `ON` for the CUDA build, default `ON`.
- `TVM_CUDA_RUNTIME_PATH`: explicit path to `libtvm_runtime_cuda.so` for repair.
- `TVM_CUDA_ARCHITECTURES`: CMake CUDA architectures, default `75`.
- `TVM_WHEEL_DIST_NAME`: optional distribution rename for TestPyPI.
- `TVM_WHEEL_DIST_VERSION`: optional distribution version rewrite.
- `TVM_INCLUDE_CUDA_RUNTIME=1`: build or repair a wheel with the CUDA runtime.
  Do not set this to a value that conflicts with `TVM_SKIP_CUDA`.
- `TVM_SKIP_REPAIR=1`: leave the injected wheel unrepaired.
- `TVM_SKIP_CUDA=1`: build or repair a wheel without the CUDA runtime.
- `TVM_KEEP_BUILD_DIRS=1`: reuse the CMake build directories.
- `TVM_MANYLINUX_IMAGE`: manylinux image family for `manylinux-cuda`, such as
  `manylinux_2_28`.
- `TVM_MANYLINUX_IMAGE_TAG`: pinned manylinux image tag for `manylinux-cuda`.
- `TVM_ARCH`: target architecture for `manylinux-cuda`, such as `x86_64` or
  `aarch64`.
- `TVM_AUDITWHEEL_PLAT`: optional `auditwheel repair --plat` override.
- `TVM_AUDITWHEEL_LIBRARY_PATH`: optional, explicit library search path for
  `auditwheel repair`.
- `TVM_EXPECT_WHEEL_PLATFORM_TAG`: require the final wheel filename to include
  a specific platform tag, such as `manylinux_2_28_x86_64`.
- `TVM_EXPECT_CUDA_RUNTIME`: verify whether the installed wheel ships a CUDA
  runtime library.
- `TVM_EXPECT_STATIC_LLVM`: verify that the installed wheel does not ship a
  dynamic LLVM library.
- `TVM_TEST_INDEX_URL`: package index for `verify-pypi`, default TestPyPI.
- `TVM_EXTRA_INDEX_URL`: extra package index for dependencies, default PyPI.
