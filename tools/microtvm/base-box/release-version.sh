#!/bin/bash -e

SKIP_ADD=( )
if [ "$1" == "--skip-add" ]; then
    SKIP_ADD=( "-var" "skip_add=true" )
    shift
fi

if [ $# -ne 1 ]; then
    echo "usage: $0 [--skip-add] <version>"
    exit 2
fi

cd "$(dirname $0)"
if [ ! -e api-token ]; then
    echo "must create a file named 'api-token' in $(pwd)"
    echo "file contents:"
    echo api_token = "<VAGRANT_CLOUD_API_TOKEN>"
    exit 2
fi

ALL_PROVIDERS=( virtualbox )
for provider in "${ALL_PROVIDERS[@]}"; do
    set -x
    packer build -var-file=api-token "${SKIP_ADD[@]}" -var "provider=${provider}" -var "version=$1" packer.hcl
done
