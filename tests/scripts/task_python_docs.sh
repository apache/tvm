#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -euxo pipefail

source tests/scripts/setup-pytest-env.sh

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1
IS_LOCAL=${IS_LOCAL:-0}
PYTHON_DOCS_ONLY=${PYTHON_DOCS_ONLY:-0}

cleanup()
{
    rm -rf /tmp/$$.log.txt
}
trap cleanup 0

clean_files() {
    # cleanup old states
    rm -rf docs/_build
    rm -rf docs/_staging
    mkdir -p docs/_build/html
    mkdir -p docs/_staging/html
    rm -rf docs/gen_modules
    rm -rf docs/doxygen
    find . -type f -path "*.pyc" | xargs rm -f
}

sphinx_precheck() {
    clean_files
    echo "PreCheck sphinx doc generation WARNINGS.."
    make cython3

    pushd docs
    make clean
    TVM_TUTORIAL_EXEC_PATTERN=none make html 2>&1 | tee /tmp/$$.log.txt
    check_sphinx_warnings "docs"
    popd
}


function join_by { local IFS="$1"; shift; echo "$*"; }


# These warnings are produced during the docs build for various reasons and are
# known to not signficantly affect the output. Don't add anything new to this
# list without special consideration of its effects, and don't add anything with
# a '|' character.
IGNORED_WARNINGS=(
    '__mro__'
    'UserWarning'
    'FutureWarning'
    'tensorflow'
    'Keras'
    'pytorch'
    'TensorFlow'
    'coremltools'
    '403'
    'git describe'
    'scikit-learn version'
    'doing serial write'
    'gen_gallery extension is not safe for parallel'
    'strategy:conv2d NHWC layout is not optimized for x86 with autotvm.'
    'strategy:depthwise_conv2d NHWC layout is not optimized for x86 with autotvm.'
    'strategy:depthwise_conv2d with layout NHWC is not optimized for arm cpu.'
    'strategy:dense is not optimized for arm cpu.'
    'autotvm:Cannot find config for target=llvm -keys=cpu'
    'autotvm:One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.'
    'autotvm:Cannot find config for target=cuda -keys=cuda,gpu'
    'cannot cache unpickable configuration value:'
    'Invalid configuration value found: 'language = None'.'
    # Warning is thrown during TFLite quantization for micro_train tutorial
    'absl:For model inputs containing unsupported operations which cannot be quantized, the `inference_input_type` attribute will default to the original type.'
    'absl:Found untraced functions such as _jit_compiled_convolution_op'
    'You are using pip version'
    # Tutorial READMEs can be ignored, but other docs should be included
    "tutorials/README.rst: WARNING: document isn't included in any toctree"
)

JOINED_WARNINGS=$(join_by '|' "${IGNORED_WARNINGS[@]}")

check_sphinx_warnings() {
    grep -v -E "$JOINED_WARNINGS" < /tmp/$$.log.txt > /tmp/$$.logclean.txt || true
    if grep --quiet -E "WARN" < /tmp/$$.logclean.txt; then
        echo "Lines with 'WARNING' found in the log, please fix them:"
        grep -E "WARN" < /tmp/$$.logclean.txt
        echo "You can reproduce locally by running 'python tests/scripts/ci.py $1'"
        exit 1
    fi
    echo "No WARNINGS to be fixed."
}

# run precheck step first to fast-fail if there are problems with the docs
if [ "$IS_LOCAL" != "1" ]; then
    echo "Running precheck"
    sphinx_precheck
else
    # skip the precheck when doing local builds since it would add overhead to
    # re-runs (and tutorials are usually not enabled anyways)
    echo "Skipping precheck"
fi


clean_files

# cleanup stale log files
find . -type f -path "*.log" | xargs rm -f
find . -type f -path "*.pyc" | xargs rm -f
make cython3

cd docs
PYTHONPATH=$(pwd)/../python make htmldepoly SPHINXOPTS='-j auto' |& tee /tmp/$$.log.txt
if grep -E "failed to execute|Segmentation fault" < /tmp/$$.log.txt; then
    echo "Some of sphinx-gallery item example failed to execute."
    exit 1
fi

check_sphinx_warnings "docs --tutorial-pattern=.*"

cd ..

if [ "$IS_LOCAL" == "1" ] && [ "$PYTHON_DOCS_ONLY" == "1" ]; then
    echo "PYTHON_DOCS_ONLY was set, skipping other doc builds"
    rm -rf _docs
    mv docs/_build/html _docs
    exit 0
fi

# C++ doc
make cppdoc
rm -f docs/doxygen/html/*.map docs/doxygen/html/*.md5

# Java doc
make javadoc

# type doc
cd web
npm install
npm run typedoc
cd ..

# Rust doc
cd rust
# Temp disable rust doc build
# cargo doc --workspace --no-deps
cd ..

# Prepare the doc dir
rm -rf _docs
mv docs/_build/html _docs
rm -f _docs/.buildinfo
mkdir -p _docs/reference/api
mv docs/doxygen/html _docs/reference/api/doxygen
mv jvm/core/target/site/apidocs _docs/reference/api/javadoc
# mv rust/target/doc _docs/api/rust
mv web/dist/docs _docs/reference/api/typedoc
git rev-parse HEAD > _docs/commit_hash

if [ "$IS_LOCAL" != "1" ]; then
    echo "Start creating the docs tarball.."
    # make the tarball
    tar -C _docs -czf docs.tgz .
    echo "Finish creating the docs tarball"
    du -h docs.tgz

    echo "Finish everything"
fi
