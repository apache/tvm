#!/bin/bash
echo "Cleanup data..."
cd $1 && rm -rf Cmake* && cd ..
