set -o verbose
export PATH=/home/liangfu/Downloads/altera/linux-socfpga-rel_socfpga-4.9.78-ltsi_18.08.02_pr/scripts/dtc:$PATH
# dtc -f -@ -I dts -O dtb -o soc_system.dtb soc_system.dts
# dtc -f -@ -I dts -O dtb -i /home/liangfu/Downloads/altera/linux-socfpga-rel_socfpga-4.9.78-ltsi_18.08.02_pr/arch/arm/boot/dts -o ALTERA_CV_SOC.dtb ALTERA_CV_SOC.dts
dtc -I dts -O dtb -o socfpga.dtb -@ -b 0 socfpga.dts
dtc -I dts -O dtb -o overlay.dtb -@ -b 0 overlay.dts
scp socfpga.dtb root@socfpga:/mnt
scp overlay.dtb root@socfpga:/lib/firmware
set +o verbose
