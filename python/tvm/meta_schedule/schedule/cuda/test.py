# Read step
print("read")
for ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused in range(2):
    for ax0_1_0_ax0_1_1_fused_1_fused_0_ax0_1_0_ax0_1_1_fused_1_fused_1_fused in range(3):
        for ax0_0_1_fused_0_ax0_0_1_fused_1_fused in range(3):
            v0 = (
                ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused * (3) + ax0_0_1_fused_0_ax0_0_1_fused_1_fused
            ) * (2) + ax0_1_0_ax0_1_1_fused_1_fused_0_ax0_1_0_ax0_1_1_fused_1_fused_1_fused
            print(
                f"block:{ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused} thread:{ax0_1_0_ax0_1_1_fused_1_fused_0_ax0_1_0_ax0_1_1_fused_1_fused_1_fused} -- index: {v0}"
            )


# Write step
print("write")
for ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused in range(2):
    for ax0_ax1_fused_0 in range(3):
        for ax0_ax1_fused_1 in range(3):
            v_ax0 = (ax0_ax1_fused_0 * 3 + ax0_ax1_fused_1) // (4)
            v_ax1 = ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused * (3) + (
                ax0_ax1_fused_0 * (3) + ax0_ax1_fused_1
            ) % (4)
            print(
                f"block:{ax0_0_0_ax0_1_0_ax0_1_1_fused_0_fused} thread:{ax0_ax1_fused_1}-- index: {v_ax0}, {v_ax1}"
            )
