sed -r -i -e 's/reg\ \[31:0\]\ uop_mem\ \[0:1023\];/reg\ \[31:0\]\ uop_mem\ \[0:1023\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' /home/liangfu/workspace/chisel-vta/chisel/ip/vta/compute/Compute.v
sed -r -i -e 's/reg\ \[511:0\]\ acc_mem\ \[0:255\];/reg\ \[511:0\]\ acc_mem\ \[0:255\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' /home/liangfu/workspace/chisel-vta/chisel/ip/vta/compute/Compute.v
# sed -r -i -e 's/reg\ \[31:0\]\ mem\ \[0:1023\];/reg\ \[31:0\]\ mem\ \[0:1023\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' /home/liangfu/workspace/chisel-vta/chisel/ip/vta/compute/Compute.v
cp /home/liangfu/workspace/chisel-vta/chisel/ip/vta/compute/Compute.v /home/liangfu/workspace/tvm/vta/hardware/intel/ip/compute/compute.v
cp /home/liangfu/workspace/chisel-vta/chisel/ip/vta/compute/Compute.v /home/liangfu/workspace/tvm/vta/hardware/intel/scripts/soc_system/synthesis/submodules/compute.v
