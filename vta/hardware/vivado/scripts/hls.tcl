#
#  Copyright (c) 2018 by Contributors
#  file: hls.tcl
#  brief: HLS generation script.
#

# Command line arguments:
# Arg 1: path to design sources
# Arg 2: path to sim sources
# Arg 3: path to test sources
# Arg 4: path to include sources
# Arg 5: target clock period
# Arg 6: input type width (log)
# Arg 7: weight type width (log)
# Arg 8: accum type width (log)
# Arg 9: output type width (log)
# Arg 10: batch size (log)
# Arg 11: in block size (log)
# Arg 12: out block size (log)
# Arg 13: uop buffer size in B (log)
# Arg 14: inp buffer size in B (log)
# Arg 15: wgt buffer size in B (log)
# Arg 16: acc buffer size in B (log)
# Arg 17: out buffer size in B (log)

if { [llength $argv] eq 19 } {
	set src_dir [lindex $argv 2]
	set sim_dir [lindex $argv 3]
	set test_dir [lindex $argv 4]
	set include_dir [lindex $argv 5]
	set target_period [lindex $argv 6]
	set inp_width [lindex $argv 7]
	set wgt_width [lindex $argv 8]
	set acc_width [lindex $argv 9]
	set out_width [lindex $argv 10]
	set batch [lindex $argv 11]
	set block_in [lindex $argv 12]
	set block_out [lindex $argv 13]
	set uop_buff_size [lindex $argv 14]
	set inp_buff_size [lindex $argv 15]
	set wgt_buff_size [lindex $argv 16]
	set acc_buff_size [lindex $argv 17]
	set out_buff_size [lindex $argv 18]
} else {
	set src_dir "../src/"
	set sim_dir "../sim/"
	set test_dir "../../src/test/"
	set include_dir "../../include"
	set target_period 10
	set inp_width 3
	set wgt_width 3
	set acc_width 5
	set out_width 3
	set batch 1
	set block_out 4
	set block_in 4
	set uop_buff_size 15
	set inp_buff_size 15
	set wgt_buff_size 15
	set acc_buff_size 17
	set out_buff_size 15
}

# C define flags to pass to compiler
set cflags "-I $include_dir -I $src_dir -I $test_dir \
	-DDEBUG=0 -DLOG_WGT_WIDTH=$wgt_width -DLOG_INP_WIDTH=$inp_width \
	-DLOG_ACC_WIDTH=$acc_width -DLOG_OUT_WIDTH=$out_width \
	-DLOG_BATCH=$batch -DLOG_BLOCK_OUT=$block_out -DLOG_BLOCK_IN=$block_in \
	-DLOG_UOP_BUFF_SIZE=$uop_buff_size -DLOG_INP_BUFF_SIZE=$inp_buff_size \
	-DLOG_WGT_BUFF_SIZE=$wgt_buff_size -DLOG_ACC_BUFF_SIZE=$acc_buff_size \
	-DLOG_OUT_BUFF_SIZE=$out_buff_size"

# Initializes the HLS design and sets HLS pragmas for memory partitioning.
# This is necessary because of a Vivado restriction that doesn't allow for
# buses wider than 1024 bits.
proc init_design {per inp_width wgt_width out_width batch block_in block_out} {

	# Set device number
	set_part {xc7z020clg484-1}

	# Set the clock frequency
	create_clock -period $per -name default

	# Set input partition factor to (INP_VECTOR_WIDTH*BATCH/1024)
	set inp_partition_factor [expr {(1 << ($inp_width + $block_in + $batch)) / 1024}]
	if {$inp_partition_factor == 0} {
		set_directive_array_reshape -type complete -dim 2 "load" inp_mem
		set_directive_array_reshape -type complete -dim 2 "compute" inp_mem
	} else {
		# Set input reshaping factor below to (1024/INP_VECTOR_WIDTH)
		set inp_reshape_factor [expr {1024 / (1 << ($inp_width + $block_in))}]
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "load" inp_mem
		set_directive_array_partition -type block -factor $inp_partition_factor -dim 2 "compute" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "load" inp_mem
		set_directive_array_reshape -type block -factor $inp_reshape_factor -dim 2 "compute" inp_mem
	}
	# Set weight partition factor to (WGT_VECTOR_WIDTH*BLOCK_OUT/1024)
	set wgt_partition_factor [expr {(1 << ($wgt_width + $block_in + $block_out)) / 1024}]
	if {$wgt_partition_factor == 0} {
		set_directive_array_reshape -type complete -dim 2 "load" wgt_mem
		set_directive_array_reshape -type complete -dim 2 "compute" wgt_mem
	} else {
		# Set weight reshaping factor below to (1024/WGT_VECTOR_WIDTH)
		set wgt_reshape_factor [expr {1024 / (1 << ($wgt_width + $block_in))}]
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "load" wgt_mem
		set_directive_array_partition -type block -factor $wgt_partition_factor -dim 2 "compute" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "load" wgt_mem
		set_directive_array_reshape -type block -factor $wgt_reshape_factor -dim 2 "compute" wgt_mem
	}
	# Set output partition factor to (OUT_VECTOR_WIDTH*BATCH/1024)
	set out_partition_factor [expr {(1 << ($out_width + $block_out + $batch)) / 1024}]
	if {$out_partition_factor == 0} {
		set_directive_array_reshape -type complete -dim 2 "compute" out_mem
		set_directive_array_reshape -type complete -dim 2 "store" out_mem
	} else {
		# Set output reshaping factor below to (1024/OUT_VECTOR_WIDTH)
		set out_reshape_factor [expr {1024 / (1 << ($out_width + $block_out))}]
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "compute" out_mem
		set_directive_array_partition -type block -factor $out_partition_factor -dim 2 "store" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "compute" out_mem
		set_directive_array_reshape -type block -factor $out_reshape_factor -dim 2 "store" out_mem
	}
}

# HLS behavioral sim
open_project vta_sim
set_top vta
add_files $src_dir/vta.cc -cflags $cflags
add_files -tb $sim_dir/vta_test.cc -cflags $cflags
add_files -tb $test_dir/test_lib.cc -cflags $cflags
open_solution "solution0"
init_design $target_period $inp_width $wgt_width $out_width $batch $block_in $block_out
csim_design -clean
close_project

# Generate fetch stage
open_project vta_fetch
set_top fetch
add_files $src_dir/vta.cc -cflags $cflags
open_solution "solution0"
init_design $target_period $inp_width $wgt_width $out_width $batch $block_in $block_out
csynth_design
export_design -format ip_catalog
close_project

# Generate load stage
open_project vta_load
set_top load
add_files $src_dir/vta.cc -cflags $cflags
open_solution "solution0"
init_design $target_period $inp_width $wgt_width $out_width $batch $block_in $block_out
csynth_design
export_design -format ip_catalog
close_project

# Generate compute stage
open_project vta_compute
set_top compute
add_files $src_dir/vta.cc -cflags $cflags
open_solution "solution0"
init_design $target_period $inp_width $wgt_width $out_width $batch $block_in $block_out
csynth_design
export_design -format ip_catalog
close_project

# Generate store stage
open_project vta_store
set_top store
add_files $src_dir/vta.cc -cflags $cflags
open_solution "solution0"
init_design $target_period $inp_width $wgt_width $out_width $batch $block_in $block_out
csynth_design
export_design -format ip_catalog
close_project

exit

