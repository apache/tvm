#!/bin/bash

function show_usage() {
    cat <<EOF
This script is for running microtvm_api_server.
Usage: launch_microtvm_api_server.sh <microtvm_api_server.py> --read-fd <READ_FD_PATH> --write-fd <WRITE_FD_PATH>
EOF
}

if [ "$#" -lt 5 -o "$1" == "--help" ]; then
    show_usage
    exit -1
fi

west_file_path=$(which west)

# Remove space and extra characters
line=$(head -n 1 ${west_file_path})
line=$(echo ${line} | sed 's/ //')
line=$(echo ${line} | sed 's/!//')
line=$(echo ${line} | sed 's/#//')
PYTHON_CMD=$line

# Run server
$PYTHON_CMD $1 $2 $3 $4 $5 
