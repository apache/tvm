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

set -euo pipefail

rat_tempdir="$(mktemp -d)"

function cleanup() {
    rm -rf "${rat_tempdir}"
}
trap cleanup EXIT

rat_output="${rat_tempdir}/$$.apache-rat.txt"

fix=0
filter_untracked=0
while [ "x${1-}" != "x" ]; do
    case "$1" in
        --local)
            filter_untracked=1
            ;;
        --fix)
            fix=1
            ;;
        *)
            echo "Unrecognized option: $1"
            exit 2
            ;;
    esac
    shift
done

java -jar /bin/apache-rat.jar -E tests/lint/rat-excludes  -d . | (grep -E "^== File" >"${rat_output}" || true)

# Rat can't be configured to ignore untracked files, so filter them.
if [ ${filter_untracked} -eq 1 ]; then
    echo "NOTE: --local flag present, filtering untracked files"
    processed_rat_output="${rat_output}-processed"
    cat ${rat_output} | sed 's/^== File: //g' | \
        python3 $(dirname "$0")/filter_untracked.py | \
        sed 's/^/== File: /g' >"${processed_rat_output}"
    rat_output="${processed_rat_output}"
fi

grep --quiet -E "File" "${rat_output}" && success=1 || success=0

if [ $success -eq 0 ]; then
    echo "No files violate ASF header check"
else
    if [ $fix -eq 1 ]; then
        python3 tests/lint/add_asf_header.py "${rat_output}"
        echo "Ran add_asf_header.py to fix the following files:"
    else
        echo "Need to add ASF header to the following files."
    fi
    echo "----------------File List----------------"
    cat "${rat_output}"
    echo "-----------------------------------------"
    if [ $fix -eq 0 ]; then
        echo "To fix, run check_asf_header.sh --fix, or use the following manual steps:"
        echo "- Create file_list.txt in your text editor"
        echo "- Copy paste the above content in file-list into file_list.txt"
        echo "- python3 tests/lint/add_asf_header.py file_list.txt"
        echo "When running via docker/lint.sh, add -i to automatically fix these problems."
    fi
    exit 1
fi
