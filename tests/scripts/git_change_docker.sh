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

set -eux

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" == "main" ]; then
    changed_files=$(git diff --no-commit-id --name-only -r HEAD~1)
else
    changed_files=$(git diff --no-commit-id --name-only -r origin/main)
fi

for file in $changed_files; do
    echo "Checking $file"
    if grep -q "docker/" <<< "$file"; then
        exit 1
    fi
done

# No docker changes
exit 0
