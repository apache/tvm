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

# NOTE: install docutils < 0.17 to work around https://github.com/readthedocs/sphinx_rtd_theme/issues/1115
pip3 install \
    autodocsumm \
    "commonmark>=0.7.3" \
    "docutils>=0.11,<0.17" \
    Image \
    matplotlib \
    sphinx \
    sphinx_autodoc_annotation \
    sphinx-gallery==0.4.0 \
    sphinx_rtd_theme
