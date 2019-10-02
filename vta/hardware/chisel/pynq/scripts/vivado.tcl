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

set proj_path [lindex $argv 0]
set num_threads [lindex $argv 1]
set target [lindex $argv 2]
set testbench_file [lindex $argv 3]
set sim_dir [lindex $argv 4]
set test_name [lindex $argv 5]

################################################################
# This is a generated script based on design: tutorial_1
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2018.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_msg_id "BD_TCL-109" "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

set clock_id 0

################################################################
# START
################################################################

set proj_name "vta"

# Create custom project
create_project -force $proj_name $proj_path -part xc7z020clg484-1

set_property  ip_repo_paths $proj_path/ip [current_project]
update_ip_catalog

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
proc create_root_design { parentCell clock_id} {

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
    CONFIG.NUM_MI {1} \
  ] $axi_interconnect_1

  # Create instance: axi_smc, and set properties
  set axi_smc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc ]
  set_property -dict [ list \
    CONFIG.NUM_SI {1} \
  ] $axi_smc

  # Create instance: vta_rtl, and set properties
  set vta_rtl [create_bd_cell -type ip -vlnv xilinx.com:RTLKernel:XilinxShell:1.0 vta_rtl]

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
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50} \
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

  # Create interface connections
  connect_bd_intf_net -intf_net axi_interconnect_1_M00_AXI \
    [get_bd_intf_pins axi_interconnect_1/M00_AXI] \
    [get_bd_intf_pins vta_rtl/s_axi_control]
  connect_bd_intf_net -intf_net axi_smc_M00_AXI \
    [get_bd_intf_pins axi_smc/M00_AXI] \
    [get_bd_intf_pins processing_system7_1/S_AXI_ACP]
  connect_bd_intf_net -intf_net kernel_rtl_m_axi_gmem \
    [get_bd_intf_pins axi_smc/S00_AXI] \
    [get_bd_intf_pins vta_rtl/m_axi_gmem]
  connect_bd_intf_net -intf_net processing_system7_1_ddr \
    [get_bd_intf_ports DDR] \
    [get_bd_intf_pins processing_system7_1/DDR]
  connect_bd_intf_net -intf_net processing_system7_1_fixed_io \
    [get_bd_intf_ports FIXED_IO] \
    [get_bd_intf_pins processing_system7_1/FIXED_IO]
  connect_bd_intf_net -intf_net processing_system7_1_m_axi_gp0 \
    [get_bd_intf_pins axi_interconnect_1/S00_AXI] \
    [get_bd_intf_pins processing_system7_1/M_AXI_GP0]

  # Create port connections
  connect_bd_net -net ps_reset_net \
    [get_bd_pins proc_sys_reset/ext_reset_in] \
    [get_bd_pins processing_system7_1/FCLK_RESET0_N]
  connect_bd_net -net reset_net \
    [get_bd_pins axi_interconnect_1/ARESETN] \
    [get_bd_pins axi_interconnect_1/M00_ARESETN] \
    [get_bd_pins axi_interconnect_1/S00_ARESETN] \
    [get_bd_pins axi_smc/aresetn] \
    [get_bd_pins vta_rtl/ap_rst_n] \
    [get_bd_pins proc_sys_reset/interconnect_aresetn]
  connect_bd_net -net clock_net \
    [get_bd_pins axi_interconnect_1/ACLK] \
    [get_bd_pins axi_interconnect_1/M00_ACLK] \
    [get_bd_pins axi_interconnect_1/S00_ACLK] \
    [get_bd_pins axi_smc/aclk] \
    [get_bd_pins vta_rtl/ap_clk] \
    [get_bd_pins proc_sys_reset/slowest_sync_clk] \
    [get_bd_pins processing_system7_1/FCLK_CLK${clock_id}] \
    [get_bd_pins processing_system7_1/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_1/S_AXI_ACP_ACLK]

  # Create address segments
  create_bd_addr_seg -range 0x40000000 -offset 0x00000000 \
    [get_bd_addr_spaces vta_rtl/m_axi_gmem] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_DDR_LOWOCM] \
    SEG_processing_system7_1_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x00040000 -offset 0xFFFC0000 \
    [get_bd_addr_spaces vta_rtl/m_axi_gmem] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_HIGH_OCM] \
    SEG_processing_system7_1_ACP_HIGH_OCM
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 \
    [get_bd_addr_spaces vta_rtl/m_axi_gmem] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_IOP] \
    SEG_processing_system7_1_ACP_IOP
  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 \
    [get_bd_addr_spaces vta_rtl/m_axi_gmem] \
    [get_bd_addr_segs processing_system7_1/S_AXI_ACP/ACP_M_AXI_GP0] \
    SEG_processing_system7_1_ACP_M_AXI_GP0
  create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 \
    [get_bd_addr_spaces processing_system7_1/Data] \
    [get_bd_addr_segs vta_rtl/s_axi_control/reg0] SEG_kernel_rtl_reg0

  # Restore current instance
  current_bd_instance $oldCurInst

  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design "" $clock_id

# Create top-level wrapper file
make_wrapper -files \
  [get_files $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/$proj_name.bd] -top
add_files -norecurse $proj_path/$proj_name.srcs/sources_1/bd/$proj_name/hdl/${proj_name}_wrapper.v

if {$target == "hw"} {

  puts "TARGET set to HW"
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

} elseif {$target == "hw_emu"} {

  puts "TARGET set to HW-EMU"
  set_property SOURCE_SET sources_1 [get_filesets sim_1]
  add_files -fileset sim_1 -norecurse $testbench_file

  if {$test_name == "heartbeat"} {
    set mem_list [join "
      $sim_dir/$test_name/ins.mem
    "]
  } else {
    set mem_list [join "
      $sim_dir/$test_name/ins.mem
      $sim_dir/$test_name/uop.mem
      $sim_dir/$test_name/acc.mem
    "]
  }


  foreach mem_file $mem_list {
    add_files $mem_file; \
    set_property file_type {Memory Initialization Files} [get_files $mem_file]
  }
  update_compile_order -fileset sources_1
  update_compile_order -fileset sim_1
  set_property -name {xsim.simulate.runtime} -value {100ms} -objects [get_filesets sim_1]
  set_property -name {xsim.simulate.log_all_signals} -value {1} -objects [get_filesets sim_1]
  launch_simulation
  exit

} else {

  puts ""
  catch {common::send_msg_id "ERROR TARGET not supported"}

}
