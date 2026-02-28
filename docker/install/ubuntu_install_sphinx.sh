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

uv pip install \
    autodocsumm==0.2.7 \
    commonmark==0.9.1 \
    docutils==0.16 \
    image==1.5.33 \
    matplotlib==3.3.4 \
    sphinx==4.2.0 \
    sphinx_autodoc_annotation~=1.0 \
    git+https://github.com/sphinx-gallery/sphinx-gallery.git@6142f1791151849b5bec4bf3959f75697ba226cd \
    sphinx_rtd_theme==1.0.0 \
    git+https://github.com/tlc-pack/tlcpack-sphinx-addon.git@768ec1dce349fe4708f6ad68be1ebb3f3dabafa1
