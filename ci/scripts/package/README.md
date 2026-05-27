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

1. Build `libtvm_runtime_cuda.so` in a CUDA-enabled CMake build.
2. Build the main Python wheel with `cibuildwheel`, LLVM enabled, and CUDA
   disabled.
3. Inject the CUDA runtime DSO into `tvm/lib/` during the `cibuildwheel`
   repair hook.
4. Repair the wheel, excluding CUDA driver/runtime DSOs and `libtvm_ffi`.
5. Validate ELF links so intra-wheel TVM DSOs resolve through relative rpaths.
   LLVM is expected to be linked statically; the final wheel must not bundle
   or dynamically depend on `libLLVM`.
6. Verify the wheel in a fresh virtualenv.
7. Upload with `twine`.

GitHub Actions flow:

1. Create a tag that contains these packaging files.
2. Open the `Publish TVM wheel` workflow in GitHub Actions.
3. Fill `tag` with that tag.
4. The workflow builds a platform wheel matrix:
   - Linux x86_64 in a `manylinux_2_28` container, with CUDA enabled.
   - Linux aarch64 in a `manylinux_2_28` container, with CUDA enabled.
   - macOS arm64 CPU-only.
   - Windows AMD64 CPU-only.
5. For a TestPyPI run, set `publish_repository=testpypi` and set
   `distribution_name` to a temporary package name.
6. After the workflow build, upload, and `verify_pypi` jobs pass, run it again
   with the final tag/name and `publish_repository=pypi`.

Linux wheels are built inside manylinux images. This avoids accidentally
publishing a wheel tagged for the GitHub runner's host glibc, such as
`manylinux_2_39`, which would not install on older supported Linux systems.

Workflow structure:

- `.github/workflows/publish_wheel.yml`: defines the platform matrix,
  artifact upload, optional publishing, and post-upload verification.
- `.github/actions/detect-env-vars`: shared environment detection.
- `.github/actions/build-cuda`: builds only the optional CUDA runtime library.
  On Linux this action owns the manylinux Docker/CUDA setup.
- `.github/actions/build-wheel-for-publish`: installs the cached LLVM prefix
  and runs `pypa/cibuildwheel` for the LLVM-enabled runtime wheel. Its custom
  repair hook injects the CUDA runtime before `auditwheel`/`delocate`/copy repair.
- `ci/scripts/package/tvm_wheel_helper.sh`: implements reusable local and CI
  entrypoints around the `cibuildwheel` build, such as `cuda`,
  `manylinux-cuda`, `cibw-repair`, `verify`, `upload`, and `verify-pypi`.
- `ci/scripts/package/inject_cuda_runtime.py`: rewrites wheel metadata and
  injects the CUDA runtime library when CUDA is enabled.
- `ci/scripts/package/verify_tvm_install.py`: imports the installed wheel and
  checks that the platform runtime library is present.

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

For a real PyPI upload, leave `TVM_WHEEL_DIST_NAME` unset and set the normal
Twine credentials:

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
- `TVM_CUDA_ARCHITECTURES`: CMake CUDA architectures, default `75`.
- `TVM_WHEEL_DIST_NAME`: optional distribution rename for TestPyPI.
- `TVM_WHEEL_DIST_VERSION`: optional distribution version rewrite.
- `TVM_SKIP_REPAIR=1`: leave the injected wheel unrepaired.
- `TVM_SKIP_CUDA=1`: build or repair a wheel without the CUDA runtime.
- `TVM_KEEP_BUILD_DIRS=1`: reuse the CMake build directories.
- `TVM_AUDITWHEEL_PLAT`: optional `auditwheel repair --plat` override.
- `TVM_EXPECT_WHEEL_PLATFORM_TAG`: require the final wheel filename to include
  a specific platform tag, such as `manylinux_2_28_x86_64`.
- `TVM_TEST_INDEX_URL`: package index for `verify-pypi`, default TestPyPI.
- `TVM_EXTRA_INDEX_URL`: extra package index for dependencies, default PyPI.
