#!/bin/bash
cd $1 && cmake .. && make $2 && cd ..
