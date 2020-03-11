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

# Load Quartus Prime Tcl Project package
package require ::quartus::project

set DEVICE [lindex $argv 0]
set PROJECT_NAME [lindex $argv 1]

set need_to_close_project 0
set make_assignments 1

# Check that the right project is open
if {[is_project_open]} {
  if {[string compare $quartus(project) "${PROJECT_NAME}"]} {
    puts "Project ${PROJECT_NAME} is not open"
    set make_assignments 0
  }
} else {
  # Only open if not already open
  if {[project_exists ${PROJECT_NAME}]} {
    project_open -revision ${PROJECT_NAME} ${PROJECT_NAME}
  } else {
    project_new -revision ${PROJECT_NAME} ${PROJECT_NAME}
  }
  set need_to_close_project 1
}

# Make assignments
if {$make_assignments} {
  set_global_assignment -name FAMILY "Cyclone V"
  set_global_assignment -name DEVICE $DEVICE
  set_global_assignment -name ORIGINAL_QUARTUS_VERSION 18.1.0
  set_global_assignment -name PROJECT_CREATION_TIME_DATE "14:21:53  JUNE 17, 2019"
  set_global_assignment -name LAST_QUARTUS_VERSION "18.1.0 Lite Edition"
  set_global_assignment -name MIN_CORE_JUNCTION_TEMP "-40"
  set_global_assignment -name MAX_CORE_JUNCTION_TEMP 100
  set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "23 MM HEAT SINK WITH 200 LFPM AIRFLOW"
  set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"
  set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top
  set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top
  set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top
  set_global_assignment -name DEVICE_MIGRATION_LIST $DEVICE
  set_global_assignment -name USE_DLL_FREQUENCY_FOR_DQS_DELAY_CHAIN ON
  set_global_assignment -name UNIPHY_SEQUENCER_DQS_CONFIG_ENABLE ON
  set_global_assignment -name ECO_REGENERATE_REPORT ON
  set_global_assignment -name ENABLE_SIGNALTAP OFF
  set_global_assignment -name ALLOW_REGISTER_RETIMING ON
  set_global_assignment -name OPTIMIZATION_MODE BALANCED
  set_global_assignment -name VERILOG_FILE ip/vta/VTAShell.v
  set_global_assignment -name QSYS_FILE soc_system.qsys
  set_global_assignment -name SDC_FILE set_clocks.sdc
  set_global_assignment -name VERILOG_FILE ${PROJECT_NAME}.v
  set_global_assignment -name SIGNALTAP_FILE ${PROJECT_NAME}.stp
  set_global_assignment -name USE_SIGNALTAP_FILE ${PROJECT_NAME}.stp

  set_location_assignment PIN_V11 -to FPGA_CLK1_50
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK1_50
  set_location_assignment PIN_Y13 -to FPGA_CLK2_50
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK2_50
  set_location_assignment PIN_E11 -to FPGA_CLK3_50
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK3_50
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_CONV_USB_N
  set_location_assignment PIN_W15 -to LED[0]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[0]
  set_location_assignment PIN_AA24 -to LED[1]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[1]
  set_location_assignment PIN_V16 -to LED[2]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[2]
  set_location_assignment PIN_V15 -to LED[3]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[3]
  set_location_assignment PIN_AF26 -to LED[4]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[4]
  set_location_assignment PIN_AE26 -to LED[5]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[5]
  set_location_assignment PIN_Y16 -to LED[6]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[6]
  set_location_assignment PIN_AA23 -to LED[7]
  set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LED[7]

  for {set i 0} {$i < 32} {incr i} {
    set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[$i]
    set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[$i] -tag __hps_sdram_p0
    set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[$i] -tag __hps_sdram_p0
    set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[$i] -tag __hps_sdram_p0
  }

  for {set i 0} {$i < 15} {incr i} {
    set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[$i] -tag __hps_sdram_p0
    set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[$i]
    set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[$i]
  }

  for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DM[$i]
    set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_N[$i]
    set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_P[$i]
    set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[$i] -tag __hps_sdram_p0
    set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[$i] -tag __hps_sdram_p0
    set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[$i] -tag __hps_sdram_p0
    set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[$i] -tag __hps_sdram_p0
    set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DM[$i] -tag __hps_sdram_p0
    set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DM[$i] -tag __hps_sdram_p0
    set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_P[$i] -tag __hps_sdram_p0
    set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_N[$i] -tag __hps_sdram_p0
  }

  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_BA[0]
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_BA[0]
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_BA[1]
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_BA[1]
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_BA[2]
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_BA[2]
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_CAS_N
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_CAS_N
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_CKE
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_CKE
  set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_CK_N
  set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_CK_P
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_CS_N
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_CS_N

  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ODT
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ODT
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RAS_N
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_RAS_N
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RESET_N
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_RESET_N
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RZQ
  set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_WE_N
  set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_WE_N

  set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITHOUT CALIBRATION" -to HPS_DDR3_CK_P -tag __hps_sdram_p0
  set_instance_assignment -name D5_DELAY 2 -to HPS_DDR3_CK_P -tag __hps_sdram_p0
  set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITHOUT CALIBRATION" -to HPS_DDR3_CK_N -tag __hps_sdram_p0
  set_instance_assignment -name D5_DELAY 2 -to HPS_DDR3_CK_N -tag __hps_sdram_p0

  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_BA[0] -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_BA[1] -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_BA[2] -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_CAS_N -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_CKE -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_CS_N -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ODT -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_RAS_N -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_WE_N -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_RESET_N -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_CK_P -tag __hps_sdram_p0
  set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_CK_N -tag __hps_sdram_p0

  set_instance_assignment -name PARTITION_HIERARCHY root_partition -to | -section_id Top

  # Commit assignments
  export_assignments

  load_package flow
  execute_flow -compile

  # Close project
  if {$need_to_close_project} {
    project_close
  }
}
