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

DOCS_DIR=0
OTHER_DIR=0
DOC_DIR="\docs"

changed_files=`git diff --no-commit-id --name-only -r origin/main`

for file in $changed_files; do
    if grep -q "$DOC_DIR" <<< "$file"; then
        DOCS_DIR=1
    else
        OTHER_DIR=1
    fi
done

if [[ ($DOCS_DIR -eq !$OTHER_DIR) || ($OTHER_DIR -eq 1) ]]; then 
    exit 1
else
    exit 0
fi
