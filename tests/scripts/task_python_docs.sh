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

mkdir -p docs/_build/html
rm -rf docs/_build/html/jsdoc
rm -rf docs/_build/html/javadoc

# remove stale tutorials and always build from scratch.
rm -rf docs/tutorials

# C++ doc
make doc

# JS doc
jsdoc web/tvm_runtime.js web/README.md
mv out docs/_build/html/jsdoc

# Java doc
make javadoc
mv jvm/core/target/site/apidocs docs/_build/html/javadoc

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

cd docs
PYTHONPATH=`pwd`/../python make html
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
