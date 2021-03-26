#!/bin/bash
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

set -e
set -u
set -o pipefail

source tests/scripts/setup-pytest-env.sh

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=4

cleanup()
{
    rm -rf /tmp/$$.log.txt
}
trap cleanup 0

# cleanup old states
rm -rf docs/_build
mkdir -p docs/_build/html
rm -rf docs/gen_modules
rm -rf docs/doxygen

# prepare auto scheduler tutorials
rm -rf tutorials/auto_scheduler/*.json
cp -f tutorials/auto_scheduler/ci_logs/*.json tutorials/auto_scheduler

# remove stale tutorials and always build from scratch.
rm -rf docs/tutorials
rm -rf docs/vta/tutorials

# cleanup stale log files
find . -type f -path "*.log" | xargs rm -f
find . -type f -path "*.pyc" | xargs rm -f
make cython3

cd docs
PYTHONPATH=`pwd`/../python make html |& tee /tmp/$$.log.txt
if grep -E "failed to execute|Segmentation fault" < /tmp/$$.log.txt; then
    echo "Some of sphinx-gallery item example failed to execute."
    exit 1
fi
cd ..

# C++ doc
make doc
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
mkdir -p _docs/api
mv docs/doxygen/html _docs/api/doxygen
mv jvm/core/target/site/apidocs _docs/api/javadoc
# mv rust/target/doc _docs/api/rust
mv web/dist/docs _docs/api/typedoc

echo "Start creating the docs tarball.."
# make the tarball
tar -C _docs -czf docs.tgz .
echo "Finish creating the docs tarball"
du -h docs.tgz

echo "Finish everything"
