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
#  Copyright (c) 2018 by Xilinx, Contributors
#  file: vivado.tcl
#  brief: Vivado compilation script. Partially automatically generated
#   by Vivado.
#

# Check if script is running in correct Vivado version.
set scripts_vivado_version 2018.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_msg_id "BD_TCL-109" "ERROR" "This script was generated using Vivado \
    <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. \
    Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado \
    <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP \
    Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

# Parse argument list, derive the clock to utilize
set clock_id 0
if { [llength $argv] eq 12 } {
  set ip_path [lindex $argv 0]
  set num_threads [lindex $argv 1]
  set clock_freq [lindex $argv 2]
  set inp_width [expr 1 << [lindex $argv 3]]
  set wgt_width [expr 1 << [lindex $argv 4]]
  set out_width [expr 1 << [lindex $argv 5]]
  set batch [expr 1 << [lindex $argv 6]]
  set out_block [expr 1 << [lindex $argv 7]]
  set in_block [expr 1 << [lindex $argv 8]]
  set inp_mem_size [expr 1 << [lindex $argv 9]]
  set wgt_mem_size [expr 1 << [lindex $argv 10]]
  set out_mem_size [expr 1 << [lindex $argv 11]]
  if {$clock_freq eq 100} {
    set clock_id 0
    puts "Setting clock frequency to 100MHz"
  } elseif {$clock_freq eq 142} {
    set clock_id 1
    puts "Setting clock frequency to 142MHz"
  } elseif {$clock_freq eq 167} {
    set clock_id 3
    puts "Setting clock frequency to 167MHz"
  } elseif {$clock_freq eq 200} {
    set clock_id 2
    puts "Setting clock frequency to 200MHz"
  } else {
    set clock_id 0
    puts "Unrecognized clock frequency, setting clock to 100MHz"
  }
} else {
  puts "Arg list incomplete: <path to ip dir> <num threads> <clock freq> \
    <inp width> <wgt_width> <out_width> <batch> <batch> <out_block> <in_block
    <inp_mem_size> <wgt_mem_size> <out_mem_size>"
  return 1
}

# Derive input mem parameters
set inp_mem_width [expr $inp_width * $batch * $in_block]
set inp_bus_width 1024
set inp_part [expr $inp_mem_width / $inp_bus_width]
if {[expr $inp_part == 0]} {
  set inp_part 1
  set inp_bus_width $inp_mem_width
}
set inp_mem_depth [expr $inp_mem_size * 8 / ($inp_mem_width * $inp_part)]

# Derive weight mem parameters
set wgt_mem_width [expr $wgt_width * $out_block * $in_block]
set wgt_bus_width 1024
set wgt_part [expr $wgt_mem_width / $wgt_bus_width]
if {[expr $wgt_part == 0]} {
  set wgt_part 1
  set wgt_bus_width $wgt_mem_width
}
set wgt_mem_depth [expr $wgt_mem_size * 8 / ($wgt_mem_width * $wgt_part)]

# Derive output mem parameters
set out_mem_width [expr $out_width * $batch * $out_block]
set out_bus_width 1024
set out_part [expr $out_mem_width / $out_bus_width]
if {[expr $out_part == 0]} {
  set out_part 1
  set out_bus_width $out_mem_width
}
set out_mem_depth [expr $out_mem_size * 8 / ($out_mem_width * $out_part)]

# User defined paths
set proj_name vta
set proj_path "."
set ip_lib "ip_lib"
set fetch_ip "${ip_path}/vta_fetch/solution0/impl/ip/xilinx_com_hls_fetch_1_0.zip"
set load_ip "${ip_path}/vta_load/solution0/impl/ip/xilinx_com_hls_load_1_0.zip"
set compute_ip "${ip_path}/vta_compute/solution0/impl/ip/xilinx_com_hls_compute_1_0.zip"
set store_ip "${ip_path}/vta_store/solution0/impl/ip/xilinx_com_hls_store_1_0.zip"

# Create custom project
create_project -force $proj_name $proj_path -part xc7z020clg484-1

# Update IP repository with generated IP
file mkdir $ip_lib
set_property ip_repo_paths $ip_lib [current_project]
update_ip_catalog
update_ip_catalog -add_ip $fetch_ip -repo_path $ip_lib
update_ip_catalog -add_ip $load_ip -repo_path $ip_lib
update_ip_catalog -add_ip $compute_ip -repo_path $ip_lib
update_ip_catalog -add_ip $store_ip -repo_path $ip_lib

# CHANGE DESIGN NAME HERE
set design_name $proj_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_msg_id "BD_TCL-001" "INFO" "Changing value of <design_name> from <$design_name> \
        to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_msg_id "BD_TCL-002" "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable \
    <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES:
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable \
    <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_msg_id "BD_TCL-003" "INFO" "Currently there is no design <$design_name> in \
    project, so creating one..."

   create_bd_design $design_name

   common::send_msg_id "BD_TCL-004" "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_msg_id "BD_TCL-005" "INFO" "Currently the variable <design_name> is equal \
  to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_msg_id "BD_TCL-114" "ERROR" $errMsg}
   return $nRet
}

##################################################################
# DESIGN PROCs
##################################################################



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell clk inp_part wgt_part out_part inp_bus_width inp_mem_depth wgt_bus_width wgt_mem_depth out_bus_width out_mem_depth} {

  variable script_folder

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_msg_id "BD_TCL-100" "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_msg_id "BD_TCL-101" "ERROR" "Parent <$parentObj> has TYPE = \
      <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
  set FIXED_IO [ create_bd_intf_port -mode Master \
    -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]

  # Create ports

  # Create instance: axi_interconnect_1, and set properties
  set axi_interconnect_1 \
    [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1 ]
  set_property -dict [ list \
    CONFIG.NUM_MI {5} \
  ] $axi_interconnect_1

  # Create instance: axi_smc, and set properties
  set axi_smc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc ]
  set_property -dict [ list \
    CONFIG.NUM_SI {5} \
  ] $axi_smc

  # Create instance: axi_timer_1, and set properties
  set axi_timer_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_timer:2.0 axi_timer_1 ]

  # Create instance: compute_0, and set properties
  set compute_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:compute:1.0 compute_0 ]
  set_property -dict [ list \
    CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE {"1111"} \
    CONFIG.C_M_AXI_DATA_PORT_DATA_WIDTH {64} \
    CONFIG.C_M_AXI_UOP_PORT_CACHE_VALUE {"1111"} \
  ] $compute_0

  # Create instance: fetch_0, and set properties
  set fetch_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:fetch:1.0 fetch_0 ]
  set_property -dict [ list \
    CONFIG.C_M_AXI_INS_PORT_CACHE_VALUE {"1111"} \
    CONFIG.C_M_AXI_INS_PORT_DATA_WIDTH {64} \
  ] $fetch_0

  # Create instance: g2l_queue, and set properties
  set g2l_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 g2l_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {1022} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {1023} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {1024} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TUSER_WIDTH {0} \
  ] $g2l_queue

  # Create instance: g2s_queue, and set properties
  set g2s_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 g2s_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {1022} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {1023} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {1024} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TUSER_WIDTH {0} \
  ] $g2s_queue

  # Create instance: gemm_queue, and set properties
  set gemm_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 gemm_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {510} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {511} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {512} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TDATA_NUM_BYTES {16} \
    CONFIG.TKEEP_WIDTH {16} \
    CONFIG.TSTRB_WIDTH {16} \
    CONFIG.TUSER_WIDTH {0} \
  ] $gemm_queue

  # Create instance: l2g_queue, and set properties
  set l2g_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 l2g_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {1022} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {1023} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {1024} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TUSER_WIDTH {0} \
  ] $l2g_queue

  # Create instance: load_0, and set properties
  set load_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:load:1.0 load_0 ]
  set_property -dict [ list \
    CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE {"1111"} \
  ] $load_0

  # Create instance: load_queue, and set properties
  set load_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 load_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {510} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {511} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {512} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TDATA_NUM_BYTES {16} \
    CONFIG.TKEEP_WIDTH {16} \
    CONFIG.TSTRB_WIDTH {16} \
    CONFIG.TUSER_WIDTH {0} \
  ] $load_queue

  # Create instance: proc_sys_reset, and set properties
  set proc_sys_reset \
    [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset ]

  # Create instance: processing_system7_1, and set properties
  set processing_system7_1 \
    [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_1 ]
  set_property -dict [ list \
    CONFIG.PCW_CAN0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_CLK1_PORT {1} \
    CONFIG.PCW_EN_CLK2_PORT {1} \
    CONFIG.PCW_EN_CLK3_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {142.86} \
    CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {200} \
    CONFIG.PCW_FPGA3_PERIPHERAL_FREQMHZ {167} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {0} \
    CONFIG.PCW_I2C0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_IMPORT_BOARD_PRESET {None} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {0} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_USE_DEFAULT_ACP_USER_VAL {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_USE_HIGH_OCM {1} \
    CONFIG.PCW_USE_S_AXI_ACP {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {0} \
    CONFIG.PCW_USE_S_AXI_HP1 {0} \
    CONFIG.PCW_USE_S_AXI_HP2 {0} \
    CONFIG.PCW_USE_S_AXI_HP3 {0} \
    CONFIG.preset {ZC702} \
  ] $processing_system7_1

  # Create instance: s2g_queue, and set properties
  set s2g_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 s2g_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {1022} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {1023} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {1024} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TUSER_WIDTH {0} \
  ] $s2g_queue

  # Create instance: store_0, and set properties
  set store_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:store:1.0 store_0 ]
  set_property -dict [ list \
CONFIG.C_M_AXI_DATA_PORT_CACHE_VALUE {"1111"} \
  ] $store_0

  # Create instance: store_queue, and set properties
  set store_queue [ create_bd_cell -type ip -vlnv xilinx.com:ip:fifo_generator:13.2 store_queue ]
  set_property -dict [ list \
    CONFIG.Empty_Threshold_Assert_Value_axis {510} \
    CONFIG.Empty_Threshold_Assert_Value_rach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wach {14} \
    CONFIG.Empty_Threshold_Assert_Value_wrch {14} \
    CONFIG.FIFO_Implementation_rach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wach {Common_Clock_Distributed_RAM} \
    CONFIG.FIFO_Implementation_wrch {Common_Clock_Distributed_RAM} \
    CONFIG.Full_Flags_Reset_Value {1} \
    CONFIG.Full_Threshold_Assert_Value_axis {511} \
    CONFIG.Full_Threshold_Assert_Value_rach {15} \
    CONFIG.Full_Threshold_Assert_Value_wach {15} \
    CONFIG.Full_Threshold_Assert_Value_wrch {15} \
    CONFIG.INTERFACE_TYPE {AXI_STREAM} \
    CONFIG.Input_Depth_axis {512} \
    CONFIG.Reset_Type {Asynchronous_Reset} \
    CONFIG.TDATA_NUM_BYTES {16} \
    CONFIG.TKEEP_WIDTH {16} \
    CONFIG.TSTRB_WIDTH {16} \
    CONFIG.TUSER_WIDTH {0} \
  ] $store_queue

  # Create instance: xlconcat_1, and set properties
  set xlconcat_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_1 ]
  set_property -dict [ list \
CONFIG.NUM_PORTS {5} \
  ] $xlconcat_1

  # Create and connect inp_mem partitions
  if {${inp_part} > 1} {
    for {set i 0} {$i < ${inp_part}} {incr i} {
      # Create instance: inp_mem, and set properties
      set inp_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 inp_mem_${i} ]
      set_property -dict [ list \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $inp_bus_width \
        CONFIG.Read_Width_B $inp_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $inp_mem_depth \
        CONFIG.Write_Width_A $inp_bus_width \
        CONFIG.Write_Width_B $inp_bus_width \
        CONFIG.use_bram_block {BRAM_Controller} \
      ] $inp_mem
      # Create interface connections
      connect_bd_intf_net -intf_net load_0_inp_mem_${i}_V_PORTA \
        [get_bd_intf_pins $inp_mem/BRAM_PORTA] \
        [get_bd_intf_pins load_0/inp_mem_${i}_V_PORTA]
      connect_bd_intf_net -intf_net compute_0_inp_mem_${i}_V_PORTA \
        [get_bd_intf_pins compute_0/inp_mem_${i}_V_PORTA] \
        [get_bd_intf_pins $inp_mem/BRAM_PORTB]
    }
  } else {
      # Create instance: inp_mem, and set properties
      set inp_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 inp_mem ]
      set_property -dict [ list \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $inp_bus_width \
        CONFIG.Read_Width_B $inp_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $inp_mem_depth \
        CONFIG.Write_Width_A $inp_bus_width \
        CONFIG.Write_Width_B $inp_bus_width \
        CONFIG.use_bram_block {BRAM_Controller} \
      ] $inp_mem
      # Create interface connections
      connect_bd_intf_net -intf_net load_0_inp_mem_V_PORTA \
        [get_bd_intf_pins $inp_mem/BRAM_PORTA] \
        [get_bd_intf_pins load_0/inp_mem_V_PORTA]
      connect_bd_intf_net -intf_net compute_0_inp_mem_V_PORTA \
        [get_bd_intf_pins compute_0/inp_mem_V_PORTA] \
        [get_bd_intf_pins $inp_mem/BRAM_PORTB]
  }

  # Create and connect wgt_mem partitions
  if {${wgt_part} > 1} {
    for {set i 0} {$i < ${wgt_part}} {incr i} {
      # Create instance: wgt_mem, and set properties
      set wgt_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 wgt_mem_${i} ]
      set_property -dict [ list \
        CONFIG.Assume_Synchronous_Clk {true} \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $wgt_bus_width \
        CONFIG.Read_Width_B $wgt_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $wgt_mem_depth \
        CONFIG.Write_Width_A $wgt_bus_width \
        CONFIG.Write_Width_B $wgt_bus_width \
      ] $wgt_mem
      # Create interface connections
      connect_bd_intf_net -intf_net load_0_wgt_mem_${i}_V_PORTA \
        [get_bd_intf_pins load_0/wgt_mem_${i}_V_PORTA] \
        [get_bd_intf_pins $wgt_mem/BRAM_PORTA]
      connect_bd_intf_net -intf_net compute_0_wgt_mem_${i}_V_PORTA \
        [get_bd_intf_pins compute_0/wgt_mem_${i}_V_PORTA] \
        [get_bd_intf_pins $wgt_mem/BRAM_PORTB]
    }
  } else {
      # Create instance: wgt_mem, and set properties
      set wgt_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 wgt_mem ]
      set_property -dict [ list \
        CONFIG.Assume_Synchronous_Clk {true} \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $wgt_bus_width \
        CONFIG.Read_Width_B $wgt_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $wgt_mem_depth \
        CONFIG.Write_Width_A $wgt_bus_width \
        CONFIG.Write_Width_B $wgt_bus_width \
      ] $wgt_mem
      # Create interface connections
      connect_bd_intf_net -intf_net load_0_wgt_mem_V_PORTA \
        [get_bd_intf_pins load_0/wgt_mem_V_PORTA] \
        [get_bd_intf_pins $wgt_mem/BRAM_PORTA]
      connect_bd_intf_net -intf_net compute_0_wgt_mem_V_PORTA \
        [get_bd_intf_pins compute_0/wgt_mem_V_PORTA] \
        [get_bd_intf_pins $wgt_mem/BRAM_PORTB]
  }

  # Create and connect out_mem partitions
  if {${out_part} > 1} {
    for {set i 0} {$i < ${out_part}} {incr i} {
      # Create instance: out_mem, and set properties
      set out_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 out_mem_${i} ]
      set_property -dict [ list \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $out_bus_width \
        CONFIG.Read_Width_B $out_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $out_mem_depth \
        CONFIG.Write_Width_A $out_bus_width \
        CONFIG.Write_Width_B $out_bus_width \
        CONFIG.use_bram_block {BRAM_Controller} \
      ] $out_mem
      # Create interface connections
      connect_bd_intf_net -intf_net compute_0_out_mem_${i}_V_PORTA \
        [get_bd_intf_pins compute_0/out_mem_${i}_V_PORTA] \
        [get_bd_intf_pins $out_mem/BRAM_PORTA]
      connect_bd_intf_net -intf_net store_0_out_mem_${i}_V_PORTA \
        [get_bd_intf_pins $out_mem/BRAM_PORTB] \
        [get_bd_intf_pins store_0/out_mem_${i}_V_PORTA]
    }
  } else {
      # Create instance: out_mem, and set properties
      set out_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 out_mem ]
      set_property -dict [ list \
        CONFIG.Byte_Size {8} \
        CONFIG.Enable_32bit_Address {true} \
        CONFIG.Enable_B {Use_ENB_Pin} \
        CONFIG.Memory_Type {True_Dual_Port_RAM} \
        CONFIG.Read_Width_A $out_bus_width \
        CONFIG.Read_Width_B $out_bus_width \
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
        CONFIG.Use_Byte_Write_Enable {true} \
        CONFIG.Use_RSTA_Pin {true} \
        CONFIG.Use_RSTB_Pin {true} \
        CONFIG.Write_Depth_A $out_mem_depth \
        CONFIG.Write_Width_A $out_bus_width \
        CONFIG.Write_Width_B $out_bus_width \
        CONFIG.use_bram_block {BRAM_Controller} \
      ] $out_mem
      # Create interface connections
      connect_bd_intf_net -intf_net compute_0_out_mem_V_PORTA \
        [get_bd_intf_pins compute_0/out_mem_V_PORTA] \
        [get_bd_intf_pins $out_mem/BRAM_PORTA]
      connect_bd_intf_net -intf_net store_0_out_mem_V_PORTA \
        [get_bd_intf_pins $out_mem/BRAM_PORTB] \
        [get_bd_intf_pins store_0/out_mem_V_PORTA]
  }

  # Create interface connections
  connect_bd_intf_net -intf_net axi_interconnect_1_M01_AXI \
    [get_bd_intf_pins axi_interconnect_1/M01_AXI] \
    [get_bd_intf_pins fetch_0/s_axi_CONTROL_BUS]
  connect_bd_intf_net -intf_net axi_interconnect_1_M02_AXI \
    [get_bd_intf_pins axi_interconnect_1/M02_AXI] \
    [get_bd_intf_pins load_0/s_axi_CONTROL_BUS]
  connect_bd_intf_net -intf_net axi_interconnect_1_M03_AXI \
    [get_bd_intf_pins axi_interconnect_1/M03_AXI] \
    [get_bd_intf_pins compute_0/s_axi_CONTROL_BUS]
  connect_bd_intf_net -intf_net axi_interconnect_1_M04_AXI \
    [get_bd_intf_pins axi_interconnect_1/M04_AXI] \
    [get_bd_intf_pins store_0/s_axi_CONTROL_BUS]
  connect_bd_intf_net -intf_net axi_smc_M00_AXI \
    [get_bd_intf_pins axi_smc/M00_AXI] \
    [get_bd_intf_pins processing_system7_1/S_AXI_ACP]
  connect_bd_intf_net -intf_net compute_0_g2l_dep_queue_V \
    [get_bd_intf_pins compute_0/g2l_dep_queue_V] \
    [get_bd_intf_pins g2l_queue/S_AXIS]
  connect_bd_intf_net -intf_net compute_0_g2s_dep_queue_V \
    [get_bd_intf_pins compute_0/g2s_dep_queue_V] \
    [get_bd_intf_pins g2s_queue/S_AXIS]
  connect_bd_intf_net -intf_net compute_0_m_axi_data_port \
    [get_bd_intf_pins axi_smc/S02_AXI] \
    [get_bd_intf_pins compute_0/m_axi_data_port]
  connect_bd_intf_net -intf_net compute_0_m_axi_uop_port \
    [get_bd_intf_pins axi_smc/S01_AXI] \
    [get_bd_intf_pins compute_0/m_axi_uop_port]
  connect_bd_intf_net -intf_net fetch_0_gemm_queue_V_V \
    [get_bd_intf_pins fetch_0/gemm_queue_V_V] \
    [get_bd_intf_pins gemm_queue/S_AXIS]
  connect_bd_intf_net -intf_net fetch_0_l2g_dep_queue_V \
    [get_bd_intf_pins l2g_queue/S_AXIS] \
    [get_bd_intf_pins load_0/l2g_dep_queue_V]
  connect_bd_intf_net -intf_net fetch_0_load_queue_V_V \
    [get_bd_intf_pins fetch_0/load_queue_V_V] \
    [get_bd_intf_pins load_queue/S_AXIS]
  connect_bd_intf_net -intf_net fetch_0_m_axi_ins_port \
    [get_bd_intf_pins axi_smc/S00_AXI] \
    [get_bd_intf_pins fetch_0/m_axi_ins_port]
  connect_bd_intf_net -intf_net fetch_0_store_queue_V_V \
    [get_bd_intf_pins fetch_0/store_queue_V_V] \
    [get_bd_intf_pins store_queue/S_AXIS]
  connect_bd_intf_net -intf_net g2l_queue_M_AXIS \
    [get_bd_intf_pins g2l_queue/M_AXIS] \
    [get_bd_intf_pins load_0/g2l_dep_queue_V]
  connect_bd_intf_net -intf_net g2s_queue_M_AXIS \
    [get_bd_intf_pins g2s_queue/M_AXIS] \
    [get_bd_intf_pins store_0/g2s_dep_queue_V]
  connect_bd_intf_net -intf_net gemm_queue_M_AXIS \
    [get_bd_intf_pins compute_0/gemm_queue_V_V] \
    [get_bd_intf_pins gemm_queue/M_AXIS]
  connect_bd_intf_net -intf_net l2g_queue_M_AXIS \
    [get_bd_intf_pins compute_0/l2g_dep_queue_V] \
    [get_bd_intf_pins l2g_queue/M_AXIS]
  connect_bd_intf_net -intf_net load_0_m_axi_data_port \
    [get_bd_intf_pins axi_smc/S03_AXI] \
    [get_bd_intf_pins load_0/m_axi_data_port]
  connect_bd_intf_net -intf_net load_queue_M_AXIS \
    [get_bd_intf_pins load_0/load_queue_V_V] \
    [get_bd_intf_pins load_queue/M_AXIS]
  connect_bd_intf_net -intf_net processing_system7_1_axi_periph_m00_axi \
    [get_bd_intf_pins axi_interconnect_1/M00_AXI] \
    [get_bd_intf_pins axi_timer_1/S_AXI]
  connect_bd_intf_net -intf_net processing_system7_1_ddr \
    [get_bd_intf_ports DDR] \
    [get_bd_intf_pins processing_system7_1/DDR]
  connect_bd_intf_net -intf_net processing_system7_1_fixed_io \
    [get_bd_intf_ports FIXED_IO] \
    [get_bd_intf_pins processing_system7_1/FIXED_IO]
  connect_bd_intf_net -intf_net processing_system7_1_m_axi_gp0 \
    [get_bd_intf_pins axi_interconnect_1/S00_AXI] \
    [get_bd_intf_pins processing_system7_1/M_AXI_GP0]
  connect_bd_intf_net -intf_net s2g_queue_M_AXIS \
    [get_bd_intf_pins compute_0/s2g_dep_queue_V] \
    [get_bd_intf_pins s2g_queue/M_AXIS]
  connect_bd_intf_net -intf_net store_0_m_axi_data_port \
    [get_bd_intf_pins axi_smc/S04_AXI] \
    [get_bd_intf_pins store_0/m_axi_data_port]
  connect_bd_intf_net -intf_net store_0_s2g_dep_queue_V \
    [get_bd_intf_pins s2g_queue/S_AXIS] \
    [get_bd_intf_pins store_0/s2g_dep_queue_V]
  connect_bd_intf_net -intf_net store_queue_M_AXIS \
    [get_bd_intf_pins store_0/store_queue_V_V] \
    [get_bd_intf_pins store_queue/M_AXIS]

  # Create port connections
  connect_bd_net -net axi_timer_1_interrupt \
    [get_bd_pins axi_timer_1/interrupt] \
    [get_bd_pins xlconcat_1/In0]
  connect_bd_net -net compute_0_interrupt \
    [get_bd_pins compute_0/interrupt] \
    [get_bd_pins xlconcat_1/In3]
  connect_bd_net -net fetch_0_interrupt \
    [get_bd_pins fetch_0/interrupt] \
    [get_bd_pins xlconcat_1/In1]
  connect_bd_net -net load_0_interrupt \
    [get_bd_pins load_0/interrupt] \
    [get_bd_pins xlconcat_1/In2]
  connect_bd_net -net proc_sys_reset_interconnect_aresetn \
    [get_bd_pins axi_interconnect_1/ARESETN] \
    [get_bd_pins proc_sys_reset/interconnect_aresetn]
  connect_bd_net -net proc_sys_reset_peripheral_aresetn \
    [get_bd_pins axi_interconnect_1/M00_ARESETN] \
    [get_bd_pins axi_interconnect_1/M01_ARESETN] \
    [get_bd_pins axi_interconnect_1/M02_ARESETN] \
    [get_bd_pins axi_interconnect_1/M03_ARESETN] \
    [get_bd_pins axi_interconnect_1/M04_ARESETN] \
    [get_bd_pins axi_interconnect_1/S00_ARESETN] \
    [get_bd_pins axi_smc/aresetn] \
    [get_bd_pins axi_timer_1/s_axi_aresetn] \
    [get_bd_pins compute_0/ap_rst_n] \
    [get_bd_pins fetch_0/ap_rst_n] \
    [get_bd_pins g2l_queue/s_aresetn] \
    [get_bd_pins g2s_queue/s_aresetn] \
    [get_bd_pins gemm_queue/s_aresetn] \
    [get_bd_pins l2g_queue/s_aresetn] \
    [get_bd_pins load_0/ap_rst_n] \
    [get_bd_pins load_queue/s_aresetn] \
    [get_bd_pins proc_sys_reset/peripheral_aresetn] \
    [get_bd_pins s2g_queue/s_aresetn] \
    [get_bd_pins store_0/ap_rst_n] \
    [get_bd_pins store_queue/s_aresetn]
  connect_bd_net -net processing_system7_1_FCLK_CLK \
    [get_bd_pins axi_interconnect_1/ACLK] \
    [get_bd_pins axi_interconnect_1/M00_ACLK] \
    [get_bd_pins axi_interconnect_1/M01_ACLK] \
    [get_bd_pins axi_interconnect_1/M02_ACLK] \
    [get_bd_pins axi_interconnect_1/M03_ACLK] \
    [get_bd_pins axi_interconnect_1/M04_ACLK] \
    [get_bd_pins axi_interconnect_1/S00_ACLK] \
    [get_bd_pins axi_smc/aclk] \
    [get_bd_pins axi_timer_1/s_axi_aclk] \
    [get_bd_pins compute_0/ap_clk] \
    [get_bd_pins fetch_0/ap_clk] \
    [get_bd_pins g2l_queue/s_aclk] \
    [get_bd_pins g2s_queue/s_aclk] \
    [get_bd_pins gemm_queue/s_aclk] \
    [get_bd_pins l2g_queue/s_aclk] \
    [get_bd_pins load_0/ap_clk] \
    [get_bd_pins load_queue/s_aclk] \
    [get_bd_pins proc_sys_reset/slowest_sync_clk] \
    [get_bd_pins processing_system7_1/FCLK_CLK${clk}] \
    [get_bd_pins processing_system7_1/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_1/S_AXI_ACP_ACLK] \
    [get_bd_pins s2g_queue/s_aclk] \
    [get_bd_pins store_0/ap_clk] \
    [get_bd_pins store_queue/s_aclk]
  connect_bd_net -net processing_system7_1_fclk_reset0_n \
    [get_bd_pins proc_sys_reset/ext_reset_in] \
    [get_bd_pins processing_system7_1/FCLK_RESET0_N]
  connect_bd_net -net store_0_interrupt \
    [get_bd_pins store_0/interrupt] \
    [get_bd_pins xlconcat_1/In4]
  connect_bd_net -net xlconcat_1_dout \
    [get_bd_pins processing_system7_1/IRQ_F2P] \
    [get_bd_pins xlconcat_1/dout]

  # Create address segments
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_uop_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces compute_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces fetch_0/Data_m_axi_ins_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces load_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces load_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces load_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces load_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0
  create_bd_addr_seg -range 0x00010000 -offset 0x42800000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs axi_timer_1/S_AXI/Reg] SEG_axi_timer_1_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x43C10000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs compute_0/s_axi_CONTROL_BUS/Reg] SEG_compute_0_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs fetch_0/s_axi_CONTROL_BUS/Reg] SEG_fetch_0_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x43C20000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs load_0/s_axi_CONTROL_BUS/Reg] SEG_load_0_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x43C30000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs store_0/s_axi_CONTROL_BUS/Reg] SEG_store_0_Reg
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces store_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces store_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces store_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces store_0/Data_m_axi_data_port] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0


  # Restore current instance
  current_bd_instance $oldCurInst

  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design "" $clock_id $inp_part $wgt_part $out_part $inp_bus_width \
  $inp_mem_depth $wgt_bus_width $wgt_mem_depth $out_bus_width $out_mem_depth

# Create top-level wrapper file
make_wrapper -files \
  [get_files $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/$proj_name.bd] -top
add_files -norecurse $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/hdl/${proj_name}_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Run bistream generation on 8 threads with performance oriented P&R strategy
# create_run impl_1 -parent_run synth_1 -flow {Vivado Implementation 2017} \
#   -strategy "Performance_ExplorePostRoutePhysOpt"
launch_runs impl_1 -to_step write_bitstream -jobs $num_threads
wait_on_run impl_1

# Export hardware description file and bitstream files to export/ dir
if {[file exist $proj_path/$proj_name.runs/impl_1/${proj_name}_wrapper.bit]} {
  file mkdir $proj_path/export
  file copy -force $proj_path/$proj_name.runs/impl_1/${proj_name}_wrapper.sysdef \
    $proj_path/export/vta.hdf
  file copy -force $proj_path/$proj_name.runs/impl_1/${proj_name}_wrapper.bit \
    $proj_path/export/vta.bit
}

exit
