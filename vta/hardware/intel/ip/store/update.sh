sed -r -i -e 's/reg\ \[127:0\]\ _T_35\ \[0:7\];/reg\ \[127:0\]\ _T_35\ \[0:7\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' /home/liangfu/workspace/chisel-vta/chisel/ip/vta/store/Store.v
cp /home/liangfu/workspace/chisel-vta/chisel/ip/vta/store/Store.v /home/liangfu/workspace/tvm/vta/hardware/intel/ip/store/store.v
cp /home/liangfu/workspace/chisel-vta/chisel/ip/vta/store/Store.v /home/liangfu/workspace/tvm/vta/hardware/intel/scripts/soc_system/synthesis/submodules/store.v
