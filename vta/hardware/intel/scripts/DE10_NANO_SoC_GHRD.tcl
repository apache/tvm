# Copyright (C) 2018  Intel Corporation. All rights reserved.
# Your use of Intel Corporation's design tools, logic functions 
# and other software and tools, and its AMPP partner logic 
# functions, and any output files from any of the foregoing 
# (including device programming or simulation files), and any 
# associated documentation or information are expressly subject 
# to the terms and conditions of the Intel Program License 
# Subscription Agreement, the Intel Quartus Prime License Agreement,
# the Intel FPGA IP License Agreement, or other applicable license
# agreement, including, without limitation, that your use is for
# the sole purpose of programming logic devices manufactured by
# Intel and sold by Intel or its authorized distributors.  Please
# refer to the applicable agreement for further details.

# Quartus Prime: Generate Tcl File for Project
# File: DE10_NANO_SoC_GHRD.tcl
# Generated on: Tue Jul 23 10:06:45 2019

# Load Quartus Prime Tcl Project package
package require ::quartus::project

set need_to_close_project 0
set make_assignments 1

# Check that the right project is open
if {[is_project_open]} {
	if {[string compare $quartus(project) "DE10_NANO_SoC_GHRD"]} {
		puts "Project DE10_NANO_SoC_GHRD is not open"
		set make_assignments 0
	}
} else {
	# Only open if not already open
	if {[project_exists DE10_NANO_SoC_GHRD]} {
		project_open -revision DE10_NANO_SoC_GHRD DE10_NANO_SoC_GHRD
	} else {
		project_new -revision DE10_NANO_SoC_GHRD DE10_NANO_SoC_GHRD
	}
	set need_to_close_project 1
}

# Make assignments
if {$make_assignments} {
	set_global_assignment -name FAMILY "Cyclone V"
	set_global_assignment -name DEVICE 5CSEBA6U23I7
	set_global_assignment -name ORIGINAL_QUARTUS_VERSION 18.0.0
	set_global_assignment -name PROJECT_CREATION_TIME_DATE "14:21:53  JUNE 17, 2019"
	set_global_assignment -name LAST_QUARTUS_VERSION "18.0.0 Lite Edition"
	set_global_assignment -name MIN_CORE_JUNCTION_TEMP "-40"
	set_global_assignment -name MAX_CORE_JUNCTION_TEMP 100
	set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "23 MM HEAT SINK WITH 200 LFPM AIRFLOW"
	set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"
	set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top
	set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top
	set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top
	set_global_assignment -name DEVICE_MIGRATION_LIST 5CSEBA6U23I7
	set_global_assignment -name USE_DLL_FREQUENCY_FOR_DQS_DELAY_CHAIN ON
	set_global_assignment -name UNIPHY_SEQUENCER_DQS_CONFIG_ENABLE ON
	set_global_assignment -name ECO_REGENERATE_REPORT ON
	set_global_assignment -name ENABLE_SIGNALTAP OFF
	set_global_assignment -name USE_SIGNALTAP_FILE DE10_NANO_SoC_GHRD.stp
	set_global_assignment -name ALLOW_REGISTER_RETIMING ON
	set_global_assignment -name OPTIMIZATION_MODE BALANCED
	set_global_assignment -name VERILOG_FILE ip/vta/VTAShell.v
	set_global_assignment -name QSYS_FILE soc_system.qsys
	set_global_assignment -name SDC_FILE DE10_NANO_SOC_GHRD.sdc
	set_global_assignment -name VERILOG_FILE DE10_NANO_SoC_GHRD.v
	set_global_assignment -name SIGNALTAP_FILE DE10_NANO_SoC_GHRD.stp
	set_location_assignment PIN_V11 -to FPGA_CLK1_50
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK1_50
	set_location_assignment PIN_Y13 -to FPGA_CLK2_50
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK2_50
	set_location_assignment PIN_E11 -to FPGA_CLK3_50
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to FPGA_CLK3_50
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_CONV_USB_N
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[0]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[0]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[1]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[1]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[2]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[2]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[3]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[3]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[4]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[4]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[5]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[5]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[6]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[6]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[7]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[7]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[8]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[8]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[9]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[9]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[10]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[10]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[11]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[11]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[12]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[12]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[13]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[13]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ADDR[14]
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ADDR[14]
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
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DM[0]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DM[1]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DM[2]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DM[3]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[0]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[1]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[2]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[3]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[4]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[5]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[6]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[7]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[8]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[9]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[10]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[11]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[12]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[13]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[14]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[15]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[16]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[17]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[18]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[19]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[20]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[21]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[22]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[23]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[24]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[25]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[26]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[27]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[28]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[29]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[30]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_DQ[31]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_N[0]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_N[1]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_N[2]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_N[3]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_P[0]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_P[1]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_P[2]
	set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 1.5-V SSTL CLASS I" -to HPS_DDR3_DQS_P[3]
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_ODT
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_ODT
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RAS_N
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_RAS_N
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RESET_N
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_RESET_N
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_RZQ
	set_instance_assignment -name IO_STANDARD "SSTL-15 CLASS I" -to HPS_DDR3_WE_N
	set_instance_assignment -name CURRENT_STRENGTH_NEW "MAXIMUM CURRENT" -to HPS_DDR3_WE_N
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_GTX_CLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_INT_N
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_MDC
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_MDIO
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_CLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_DATA[0]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_DATA[1]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_DATA[2]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_DATA[3]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_RX_DV
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_TX_DATA[0]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_TX_DATA[1]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_TX_DATA[2]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_TX_DATA[3]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_ENET_TX_EN
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_GSENSOR_INT
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_I2C0_SCLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_I2C0_SDAT
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_I2C1_SCLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_I2C1_SDAT
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_KEY
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_LED
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_LTC_GPIO
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_CLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_CMD
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_DATA[0]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_DATA[1]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_DATA[2]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SD_DATA[3]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SPIM_CLK
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SPIM_MISO
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SPIM_MOSI
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_SPIM_SS
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_UART_RX
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_UART_TX
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_CLKOUT
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[0]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[1]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[2]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[3]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[4]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[5]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[6]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DATA[7]
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_DIR
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_NXT
	set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to HPS_USB_STP
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
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[0] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[0] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[1] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[1] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[2] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[2] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[3] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[3] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[4] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[4] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[5] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[5] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[6] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[6] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[7] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[7] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[8] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[8] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[9] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[9] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[10] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[10] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[11] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[11] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[12] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[12] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[13] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[13] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[14] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[14] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[15] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[15] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[16] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[16] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[17] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[17] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[18] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[18] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[19] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[19] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[20] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[20] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[21] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[21] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[22] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[22] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[23] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[23] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[24] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[24] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[25] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[25] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[26] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[26] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[27] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[27] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[28] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[28] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[29] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[29] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[30] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[30] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[31] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQ[31] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[0] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[0] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[1] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[1] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[2] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[2] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[3] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_P[3] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[0] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[0] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[1] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[1] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[2] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[2] -tag __hps_sdram_p0
	set_instance_assignment -name INPUT_TERMINATION "PARALLEL 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[3] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DQS_N[3] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITHOUT CALIBRATION" -to HPS_DDR3_CK_P -tag __hps_sdram_p0
	set_instance_assignment -name D5_DELAY 2 -to HPS_DDR3_CK_P -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITHOUT CALIBRATION" -to HPS_DDR3_CK_N -tag __hps_sdram_p0
	set_instance_assignment -name D5_DELAY 2 -to HPS_DDR3_CK_N -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DM[0] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DM[1] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DM[2] -tag __hps_sdram_p0
	set_instance_assignment -name OUTPUT_TERMINATION "SERIES 50 OHM WITH CALIBRATION" -to HPS_DDR3_DM[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[0] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[1] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[2] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[4] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[5] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[6] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[7] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[8] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[9] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[10] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[11] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[12] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[13] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[14] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[15] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[16] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[17] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[18] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[19] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[20] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[21] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[22] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[23] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[24] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[25] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[26] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[27] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[28] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[29] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[30] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQ[31] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DM[0] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DM[1] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DM[2] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DM[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_P[0] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_P[1] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_P[2] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_P[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_N[0] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_N[1] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_N[2] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_DQS_N[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[0] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[10] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[11] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[12] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[13] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[14] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[1] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[2] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[3] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[4] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[5] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[6] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[7] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[8] -tag __hps_sdram_p0
	set_instance_assignment -name PACKAGE_SKEW_COMPENSATION OFF -to HPS_DDR3_ADDR[9] -tag __hps_sdram_p0
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
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|ureset|phy_reset_mem_stable_n -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|ureset|phy_reset_n -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uio_pads|dq_ddio[0].read_capture_clk_buffer -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_write_side[0] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_wraddress[0] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uio_pads|dq_ddio[1].read_capture_clk_buffer -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_write_side[1] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_wraddress[1] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uio_pads|dq_ddio[2].read_capture_clk_buffer -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_write_side[2] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_wraddress[2] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uio_pads|dq_ddio[3].read_capture_clk_buffer -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_write_side[3] -tag __hps_sdram_p0
	set_instance_assignment -name GLOBAL_SIGNAL OFF -to u0|hps_0|hps_io|border|hps_sdram_inst|p0|umemphy|uread_datapath|reset_n_fifo_wraddress[3] -tag __hps_sdram_p0
	set_instance_assignment -name ENABLE_BENEFICIAL_SKEW_OPTIMIZATION_FOR_NON_GLOBAL_CLOCKS ON -to u0|hps_0|hps_io|border|hps_sdram_inst -tag __hps_sdram_p0
	set_instance_assignment -name PLL_COMPENSATION_MODE DIRECT -to u0|hps_0|hps_io|border|hps_sdram_inst|pll0|fbout -tag __hps_sdram_p0
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
