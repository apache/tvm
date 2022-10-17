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

SCRIPT_NAME="$0"
DEFAULT_STEPS=( file_type asf clang_format cpplint python_format pylint jnilint cppdocs mypy )

inplace_fix=0

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
            if [ $inplace_fix -eq 1 ]; then
                cmd=( "${cmd[@]}" --fix )
            fi
            ;;
        clang_format)
            if [ $inplace_fix -eq 0 ]; then
                cmd=( tests/lint/git-clang-format.sh )
            else
                # NOTE: need to run git status to update some docker-side cache. Otherwise,
                # git-clang-format will fail with "The following files would be modified but have
                # unstaged changes:"
                cmd=( bash -c 'git status &>/dev/null && tests/lint/git-clang-format.sh -i --rev origin/main' )
            fi
            ;;
        cpplint)
            cmd=( tests/lint/cpplint.sh )
            ;;
        flake8)
            cmd=( tests/lint/flake8.sh )
            ;;
        pylint)
            cmd=( tests/lint/pylint.sh )
            ;;
        python_format)
            if [ $inplace_fix -eq 0 ]; then
                cmd=( tests/lint/git-black.sh )
            else
                cmd=( tests/lint/git-black.sh -i --rev origin/main )
            fi
            ;;
        jnilint)
            cmd=( tests/lint/jnilint.sh )
            ;;
        cppdocs)
            cmd=( tests/lint/cppdocs.sh )
            ;;
        mypy)
            cmd=( tests/scripts/task_mypy.sh )
            ;;
        *)
            echo "error: don't know how to run lint step: $1" >&2
            echo "usage: ${SCRIPT_NAME} [-i] <lint_step>" >&2
            echo >&2
            echo "options:" >&2
            echo " -i    Fix lint errors in-place where possible (modifies non-compliant files)" >&2
            echo >&2
            echo "available lint_step: ${DEFAULT_STEPS[@]}" >&2
            exit 2
            ;;
    esac
    shift

    if [ $validate_only -eq 0 ]; then
        run_docker -it "ci_lint" "${cmd[@]}"
    fi
}

if [ $# -eq 0 ]; then
    # NOTE: matches order in tests/scripts/task_lint.sh
    steps=( "${DEFAULT_STEPS[@]}" )
else
    steps=( "$@" )
fi

validated_steps=( )
for step in "${steps[@]}"; do
    if [ "${step}" == "-i" ]; then
        inplace_fix=1
        continue
    fi
    run_lint_step --validate-only "$step"
    validated_steps=( "${validated_steps[@]}" "$step" )
done

for step in "${validated_steps[@]}"; do
    run_lint_step "$step"
done
