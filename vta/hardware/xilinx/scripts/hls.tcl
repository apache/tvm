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
# Arg 1: target (FPGA)
# Arg 2: path to design sources
# Arg 3: path to sim sources
# Arg 4: path to test sources
# Arg 5: path to include sources
# Arg 6: mode
# Arg 7: debug
# Arg 8: alu_ena
# Arg 9: mul_ena
# Arg 10: target clock period
# Arg 11: target II for GEMM
# Arg 12: target II for tensor ALU
# Arg 13: input type width (log)
# Arg 14: weight type width (log)
# Arg 15: accum type width (log)
# Arg 16: output type width (log)
# Arg 17: batch size (log)
# Arg 18: in block size (log)
# Arg 19: out block size (log)
# Arg 20: bus width in b (log)
# Arg 21: uop buffer size in B (log)
# Arg 22: inp buffer size in B (log)
# Arg 23: wgt buffer size in B (log)
# Arg 24: acc buffer size in B (log)
# Arg 25: out buffer size in B (log)

if { [llength $argv] eq 27 } {
	set target [lindex $argv 2]
	set src_dir [lindex $argv 3]
	set sim_dir [lindex $argv 4]
	set test_dir [lindex $argv 5]
	set include_dir [lindex $argv 6]
	set mode [lindex $argv 7]
	set debug [lindex $argv 8]
	set alu_ena [lindex $argv 9]
	set mul_ena [lindex $argv 10]
	set target_period [lindex $argv 11]
	set target_gemm_ii [lindex $argv 12]
	set target_alu_ii [lindex $argv 13]
	set inp_width [lindex $argv 14]
	set wgt_width [lindex $argv 15]
	set acc_width [lindex $argv 16]
	set out_width [lindex $argv 17]
	set batch [lindex $argv 18]
	set block_in [lindex $argv 19]
	set block_out [lindex $argv 20]
	set bus_width [lindex $argv 21]
	set uop_buff_size [lindex $argv 22]
	set inp_buff_size [lindex $argv 23]
	set wgt_buff_size [lindex $argv 24]
	set acc_buff_size [lindex $argv 25]
	set out_buff_size [lindex $argv 26]
} else {
	puts "Not enough arguments provided!"
	exit
}

puts "about to start doing some stuff"


# Initializes the HLS design and sets HLS pragmas for memory partitioning.
# This is necessary because of a Vivado restriction that doesn't allow for
# buses wider than 1024 bits.
proc init_design {target per g_ii a_ii bus_width inp_width wgt_width out_width acc_width batch block_in block_out alu_ena} {

	# Set device number
	if {$target=="pynq"} {
		set_part {xc7z020clg484-1}
	} elseif {$target=="ultra96"} {
		set_part {xczu3eg-sbva484-1-e}
	} elseif {$target=="zcu102"} {
		set_part {xczu9eg-ffvb1156-2-e}
	} elseif {$target=="f1"} {
		set_part {xcvu9p-flgb2104-2-i}
		# config_interface -m_axi_addr64
	}

	# Max bus width (supported by Vivado)
	set max_width 1024

	# Set axi width
	set axi_width [expr {1 << $bus_width}]

	# Set the clock frequency
	create_clock -period $per -name default

	# Set pipeline directive
	set_directive_pipeline -II $g_ii "gemm/READ_GEMM_UOP"

	if {$alu_ena=="True"} {
		set_directive_pipeline -II $a_ii "alu/READ_ALU_UOP"
	}

	# Set input partition factor to (INP_VECTOR_WIDTH*BATCH/(1024*g_ii)
	set inp_bus_width [expr {(1 << ($inp_width + $block_in + $batch)) / $g_ii}]
	set inp_partition_factor [expr {$inp_bus_width / $max_width}]
	if {$inp_partition_factor == 0} {
		set inp_reshape_factor [expr {$inp_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "load" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "compute" inp_mem
	} else {
		set inp_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "load" inp_mem
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "compute" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "load" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "compute" inp_mem
	}
	# Set weight partition factor to (WGT_VECTOR_WIDTH*BLOCK_OUT/(1024*g_ii))
	set wgt_bus_width [expr {(1 << ($wgt_width + $block_in + $block_out)) / $g_ii}]
	set wgt_partition_factor [expr {$wgt_bus_width / $max_width}]
	if {$wgt_partition_factor == 0} {
		set wgt_reshape_factor [expr {$wgt_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "load" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "compute" wgt_mem
	} else {
		set wgt_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "load" wgt_mem
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "compute" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "load" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "compute" wgt_mem
	}
	# Set output partition factor to (OUT_VECTOR_WIDTH*BATCH/(1024*g_ii))
	set out_bus_width [expr {(1 << ($out_width + $block_out + $batch)) / $g_ii}]
	set out_partition_factor [expr {$out_bus_width / $max_width}]
	if {$out_partition_factor == 0} {
		set out_reshape_factor [expr {$out_bus_width / $axi_width}]
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "compute" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "store" out_mem
	} else {
		set out_reshape_factor [expr {$max_width / $axi_width}]
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "compute" out_mem
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "store" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "compute" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "store" out_mem
	}
	# Set accumulator partition factor
	# set acc_bus_width [expr {(1 << ($acc_width + $block_out + $batch)) / $g_ii}]
	# set acc_reshape_factor [expr {$acc_bus_width / $axi_width}]
	# set_directive_array_partition -type block -factor $acc_reshape_factor -dim 2 "compute" acc_mem
}

# C define flags to pass to compiler
set cflags "-I $include_dir -I $src_dir -I $test_dir \
	-DVTA_LOG_WGT_WIDTH=$wgt_width -DVTA_LOG_INP_WIDTH=$inp_width \
	-DVTA_LOG_ACC_WIDTH=$acc_width -DVTA_LOG_OUT_WIDTH=$out_width \
	-DVTA_LOG_BATCH=$batch -DVTA_LOG_BLOCK_OUT=$block_out -DVTA_LOG_BLOCK_IN=$block_in \
	-DVTA_LOG_UOP_BUFF_SIZE=$uop_buff_size -DVTA_LOG_INP_BUFF_SIZE=$inp_buff_size \
	-DVTA_LOG_WGT_BUFF_SIZE=$wgt_buff_size -DVTA_LOG_ACC_BUFF_SIZE=$acc_buff_size \
	-DVTA_LOG_OUT_BUFF_SIZE=$out_buff_size -DVTA_LOG_BUS_WIDTH=$bus_width \
	-DVTA_GEMM_II=$target_gemm_ii"
if {$debug=="True"} {
	append cflags " -DVTA_DEBUG=1"
}
if {$alu_ena=="True"} {
	append cflags " -DALU_EN"
}
if {$mul_ena=="True"} {
	append cflags " -DMUL_EN"
}


# HLS behavioral sim
if {$mode=="all" || $mode=="sim"} {
	open_project vta_sim
	set_top vta
	add_files $src_dir/vta.cc -cflags $cflags
	add_files -tb $sim_dir/vta_test.cc -cflags $cflags
	add_files -tb $test_dir/test_lib.cc -cflags $cflags
	open_solution "solution0"
	init_design $target $target_period $target_gemm_ii $target_alu_ii $bus_width $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $alu_ena
	csim_design -clean
	close_project
}

# Generate fetch stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="fetch"} {
	open_project vta_fetch
	set_top fetch
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target $target_period $target_gemm_ii $target_alu_ii $bus_width $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $alu_ena
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate load stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="load"} {
	open_project vta_load
	set_top load
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target $target_period $target_gemm_ii $target_alu_ii $bus_width $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $alu_ena
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate compute stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="compute"} {
	open_project vta_compute
	set_top compute
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target $target_period $target_gemm_ii $target_alu_ii $bus_width $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $alu_ena
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

# Generate store stage
if {$mode=="all" || $mode=="skip_sim" || $mode=="store"} {
	open_project vta_store
	set_top store
	add_files $src_dir/vta.cc -cflags $cflags
	open_solution "solution0"
	init_design $target $target_period $target_gemm_ii $target_alu_ii $bus_width $inp_width $wgt_width $out_width $acc_width $batch $block_in $block_out $alu_ena
	csynth_design
	if {$mode=="all" || $mode=="skip_sim"} {
		export_design -format ip_catalog
	}
	close_project
}

exit

