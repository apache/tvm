#!/bin/bash -e
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


source "$(dirname $0)/dev_common.sh"

DEFAULT_STEPS=( file_type asf cpplint clang_format pylint jnilint cppdocs )

function run_lint_step() {
    validate_only=0
    if [ "$1" == "--validate-only" ]; then
        validate_only=1
        shift
    fi

    case "$1" in
        file_type)
            cmd=( python3 tests/lint/check_file_type.py )
            ;;
        asf)
            cmd=( tests/lint/check_asf_header.sh --local )
            ;;
        clang_format)
            cmd=( tests/lint/clang_format.sh )
            ;;
        cpplint)
            cmd=( tests/lint/cpplint.sh )
            ;;
        pylint)
            cmd=( tests/lint/pylint.sh )
            ;;
        jnilint)
            cmd=( tests/lint/jnilint.sh )
            ;;
        cppdocs)
            cmd=( tests/lint/cppdocs.sh )
            ;;
        *)
            echo "error: don't know how to run lint step: $1" >&2
            echo "available lint steps: ${DEFAULT_STEPS[@]}"
            exit 2
            ;;
    esac

    if [ $validate_only -eq 0 ]; then
        run_docker "ci_lint" "${cmd[@]}"
    fi
}

if [ $# -eq 0 ]; then
    # NOTE: matches order in tests/scripts/task_lint.sh
    steps=( "${DEFAULT_STEPS[@]}" )
else
    steps=( "$@" )
fi

for step in "${steps[@]}"; do
    run_lint_step --validate-only "$step"
done

for step in "${steps[@]}"; do
    run_lint_step "$step"
done
