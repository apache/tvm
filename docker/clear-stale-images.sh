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
#
# Remove tvm-related docker images from the local system which
# are not used by the currently-checked-out branch in this git
# repository plus any linked worktrees.

set -euo pipefail

dry_run=0
repositories=( "$(cd $(dirname "$0") && git rev-parse --show-toplevel)" )
skip_confirm=0
verbose=0
while [ "${1+x}" == "x" ]; do
    case "$1" in
        --help|-h)
            echo "usage: $0 [-n] [-v] [-y] <repository> [<repository> ...]"
            echo ""
            echo "Remove tvm-related docker images from the local system which"
            echo "are not used by the currently-checked-out branch in this git"
            echo "repository plus any linked worktrees."
            echo ""
            echo 'This command should remove only docker images beginning with "tlcpack"'
            echo ""
            echo "Options:"
            echo " -n           Perform a dry-run and just print the docker rmi command"
            echo " -v           Verbosely list the images kept and why"
            echo " -y           Skip confirmation"
            echo " <repository> Additional git repositories to consult."
            exit 2
            ;;
        -n)
            dry_run=1
            ;;
        -v)
            verbose=1
            ;;
        -y)
            skip_confirm=1
            ;;
        *)
            repositories=( "${repositories[@]}" "$1" )
            ;;
    esac
    shift
done

declare -a used_images
for r in "${repositories[@]}"; do
    if [ -d "${r}/.git" ]; then
        worktree="${r}"
    else
        worktree="$(cat "${r}/.git")"
        worktree="${worktree##gitdir: }"
    fi
    worktree_list=$(cd "${worktree}" && git worktree list --porcelain | grep '^worktree ')
    while read wt; do
        d="${wt:9:${#wt}}"  # strip "worktree " prefix
        for img in $(cat "${d}/Jenkinsfile" | grep -E '^ci_[a-z]+ = ' | sed -E "s/ci_[a-z]+ = '([^\"]*)'/\1/"); do
            used_images=( "${used_images[@]}" "${img}" )
        done
    done < <(echo -n "${worktree_list}")
done

declare -a to_rm
while read image; do
    if [ "${image}" == "<none>:<none>" ]; then
        continue
    fi
    grep -qE "^tlcpack" < <(echo "$image") && is_tlcpack=1 || is_tlcpack=0
    if [ $is_tlcpack -eq 0 ]; then   # non-tlcpack image
        if [ $verbose -ne 0 ]; then
            echo "skipping (non-tvm): $image"
        fi
        continue
    fi
    grep -q "$image" < <(echo "${used_images[@]}") && is_used=1 || is_used=0
    if [ $is_used -eq 1 ]; then  # Image was found in used_images
        if [ $verbose -ne 0 ]; then
            echo "skipping (image used): $image"
        fi
        continue
    fi
    to_rm=( "${to_rm[@]}" "${image}" )
done < <(docker images --format '{{.Repository}}:{{.Tag}}')

docker_cmd=( docker rmi "${to_rm[@]}" )
if [ ${dry_run} -ne 0 ]; then
    echo "would run: ${docker_cmd[@]}"
else
    if [ $skip_confirm -eq 0 ]; then
        echo "will run: ${docker_cmd[@]}"
        read -p "Proceed? [y/N] " proceed
        if [ "${proceed-}" != "y" -a "${proceed-}" != "Y" ]; then
            echo "Aborted."
            exit 2
        fi
    fi
    "${docker_cmd[@]}"
fi
