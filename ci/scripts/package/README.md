# TVM wheel packaging helper

This helper follows the CUDA-sidecar packaging flow used for local release
validation:

1. Build `libtvm_runtime_cuda.so` in a CUDA-enabled CMake build.
2. Build the main Python wheel with LLVM enabled and CUDA disabled.
3. Inject the CUDA runtime DSO into `tvm/lib/` inside the wheel.
4. Repair the wheel, excluding CUDA driver/runtime DSOs from vendoring.
5. Verify the wheel in a fresh virtualenv.
6. Upload with `twine`.

It mirrors the TVM-FFI packaging patterns in:

- `tvm-ffi/.github/actions/build-wheel-for-publish/action.yml`
- `tvm-ffi/.github/workflows/publish_wheel.yml`
- `tvm-ffi/addons/tvm_ffi_orcjit/pyproject.toml`
- `tvm-ffi/addons/torch_c_dlpack_ext/build_aot_wheels.sh`

GitHub Actions flow:

1. Create a tag that contains these packaging files.
2. Open the `Publish TVM wheel` workflow in GitHub Actions.
3. Fill `tag` with that tag.
4. For a TestPyPI run, set `publish_repository=testpypi` and set
   `distribution_name` to a temporary package name such as
   `tvm-yourname-test`.
5. After the workflow build, upload, and `verify_pypi` jobs pass, run it again
   with the final tag/name and `publish_repository=pypi`.

To test this from the fork `tlopex/tvm` without publishing:

```bash
git push mine HEAD:pypi
git tag -a tvm-wheel-test0 -m "Test TVM wheel workflow"
git push mine tvm-wheel-test0

gh workflow run publish_wheel.yml \
  --repo tlopex/tvm \
  --ref pypi \
  -f tag=tvm-wheel-test0 \
  -f publish_repository=none \
  -f distribution_name=tvm-tlopexh-test \
  -f cuda_architectures=75 \
  -f verify_from_repository=false
```

If the workflow is not visible in the GitHub UI yet, push or merge these files
to the fork's default branch first. GitHub only lists manually dispatched
workflows once the workflow file exists in the repository.

Typical TestPyPI dry run:

```bash
python version.py --git-describe
git tag -a v0.25.dev-test0 -m "Test TVM wheel v0.25.dev-test0"

python -m venv /tmp/tvm-wheel-tools
/tmp/tvm-wheel-tools/bin/python -m pip install -U pip build auditwheel twine

TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
TVM_USE_LLVM=/path/to/llvm-config \
TVM_USE_CUDA=/usr/local/cuda-12.8 \
TVM_WHEEL_DIST_NAME=tvm-tlopexh-test \
ci/scripts/package/build_tvm_wheel.sh all

TVM_UPLOAD_REPOSITORY_URL=https://test.pypi.org/legacy/ \
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/build_tvm_wheel.sh upload

TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/build_tvm_wheel.sh verify-pypi
```

For a real PyPI upload, leave `TVM_WHEEL_DIST_NAME` unset and set the normal
Twine credentials:

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_TOKEN" \
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/build_tvm_wheel.sh all

TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_TOKEN" \
TVM_PYTHON=/tmp/tvm-wheel-tools/bin/python \
ci/scripts/package/build_tvm_wheel.sh upload
```

Useful knobs:

- `TVM_USE_LLVM`: LLVM config for the base wheel, default `ON`.
- `TVM_USE_CUDA`: CUDA root or `ON` for the sidecar build, default `ON`.
- `TVM_CUDA_ARCHITECTURES`: CMake CUDA architectures, default `75`.
- `TVM_WHEEL_DIST_NAME`: optional distribution rename for TestPyPI.
- `TVM_WHEEL_DIST_VERSION`: optional distribution version rewrite.
- `TVM_SKIP_REPAIR=1`: leave the injected wheel unrepaired.
- `TVM_SKIP_CUDA=1`: build a base wheel without a CUDA sidecar.
- `TVM_KEEP_BUILD_DIRS=1`: reuse the CMake build directories.
- `TVM_TEST_INDEX_URL`: package index for `verify-pypi`, default TestPyPI.
- `TVM_EXTRA_INDEX_URL`: extra package index for dependencies, default PyPI.
