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

# Command line arguments:
# Arg 1: path to vta root
# Arg 2: path of config param script

if { [llength $argv] eq 4 } {
    set root_dir        [lindex $argv 2]
    set vta_config      [lindex $argv 3]
} else {
    puts "Not enough arguments provided!"
    exit
}

# Derive paths
set src_dir "$root_dir/hardware/xilinx/src"
set sim_dir "$root_dir/hardware/xilinx/sim"
set test_dir "$root_dir/tests/hardware/common"

# C define flags that we want to pass to the compiler
set cflags [exec python $vta_config --cflags]

# Get the VTA configuration paramters
set ::device        [exec python $vta_config --get-fpga-dev]
set ::period        [exec python $vta_config --get-fpga-per]

# Get the VTA SRAM reshape/partition factors to get all memories
# to be of the same axi width.
set ::inp_reshape_factor    [exec python $vta_config --get-inp-mem-axi-ratio]
set ::inp_partition_factor  [exec python $vta_config --get-inp-mem-banks]
set ::wgt_reshape_factor    [exec python $vta_config --get-wgt-mem-axi-ratio]
set ::wgt_partition_factor  [exec python $vta_config --get-wgt-mem-banks]
set ::out_reshape_factor    [exec python $vta_config --get-out-mem-axi-ratio]
set ::out_partition_factor  [exec python $vta_config --get-out-mem-banks]


# Initializes the HLS design and sets HLS pragmas for memory partitioning.
# This is necessary because of a Vivado restriction that doesn't allow for
# buses wider than 1024 bits.
proc init_design {} {

    # Set device id
    set_part $::device

    # Set the clock frequency
    create_clock -period $::period -name default

    # HLS pragmas to reshape/partition the input memory read/write port
    set_directive_array_reshape -type block -factor $::inp_reshape_factor -dim 2 "load" inp_mem
    set_directive_array_reshape -type block -factor $::inp_reshape_factor -dim 2 "compute" inp_mem
    if {$::inp_partition_factor > 1} {
        set_directive_array_partition -type block -factor $::inp_partition_factor -dim 2 "load" inp_mem
        set_directive_array_partition -type block -factor $::inp_partition_factor -dim 2 "compute" inp_mem
    }
    # HLS pragmas to reshape/partition the weight memory read/write port
    set_directive_array_reshape -type block -factor $::wgt_reshape_factor -dim 2 "load" wgt_mem
    set_directive_array_reshape -type block -factor $::wgt_reshape_factor -dim 2 "compute" wgt_mem
    if {$::wgt_partition_factor >1} {
        set_directive_array_partition -type block -factor $::wgt_partition_factor -dim 2 "load" wgt_mem
        set_directive_array_partition -type block -factor $::wgt_partition_factor -dim 2 "compute" wgt_mem
    }
    # HLS pragmas to reshape/partition the output memory read/write port
    set_directive_array_reshape -type block -factor $::out_reshape_factor -dim 2 "compute" out_mem
    set_directive_array_reshape -type block -factor $::out_reshape_factor -dim 2 "store" out_mem
    if {$::out_partition_factor > 1} {
        set_directive_array_partition -type block -factor $::out_partition_factor -dim 2 "compute" out_mem
        set_directive_array_partition -type block -factor $::out_partition_factor -dim 2 "store" out_mem
    }
}

# HLS behavioral sim
open_project vta_sim
set_top vta
add_files $src_dir/vta.cc -cflags $cflags
add_files -tb $sim_dir/vta_test.cc -cflags $cflags
add_files -tb $test_dir/test_lib.cc -cflags $cflags
open_solution "soln"
init_design
csim_design -clean
close_project

# Generate fetch stage
open_project vta_fetch
set_top fetch
add_files $src_dir/vta.cc -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

# Generate load stage
open_project vta_load
set_top load
add_files $src_dir/vta.cc -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

# Generate compute stage
open_project vta_compute
set_top compute
add_files $src_dir/vta.cc -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

# Generate store stage
open_project vta_store
set_top store
add_files $src_dir/vta.cc -cflags $cflags
open_solution "soln"
init_design
csynth_design
export_design -format ip_catalog
close_project

exit

