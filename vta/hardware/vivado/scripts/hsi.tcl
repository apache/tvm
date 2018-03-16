#
#  Copyright (c) 2018 by Contributors
#  file: hsi.tcl
#  brief: Driver generation script for ARMv7 driver libraries.
#

open_hw_design export/vta.hdf
create_sw_design swdesign -proc ps7_cortexa9_0 -os standalone
generate_bsp -dir bsp

exit
