from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(data: T.Buffer((1, 512, 7, 7), "float32"), kernel: T.Buffer((512, 512, 3, 3), "float32"), bias: T.Buffer((1, 512, 1, 1), "float32"), compute: T.Buffer((1, 512, 7, 7), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        blockIdx_x = T.launch_thread("blockIdx.x", 64)
        conv2d_nchw = T.allocate([8], "float32", "local")
        pad_temp_shared = T.allocate([4032], "float32", "shared")
        kernel_shared = T.allocate([1536], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 49)
        conv2d_nchw_1 = T.Buffer((8,), data=conv2d_nchw, scope="local", align=32)
        conv2d_nchw_1[0] = T.float32(0.0)
        conv2d_nchw_1[1] = T.float32(0.0)
        conv2d_nchw_1[2] = T.float32(0.0)
        conv2d_nchw_1[3] = T.float32(0.0)
        conv2d_nchw_1[4] = T.float32(0.0)
        conv2d_nchw_1[5] = T.float32(0.0)
        conv2d_nchw_1[6] = T.float32(0.0)
        conv2d_nchw_1[7] = T.float32(0.0)
        for rc_outer_outer, rx_outer_outer in T.grid(8, 3):
            cse_var_2: T.int32 = rc_outer_outer * 3136
            cse_var_1: T.int32 = rc_outer_outer * 576
            threadIdx_x_1 = T.env_thread("threadIdx.x")
            pad_temp_shared_1 = T.Buffer((4032,), data=pad_temp_shared, scope="shared")
            data_1 = T.Buffer((25088,), data=data.data)
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 49] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 49) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 98] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 98) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 147] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 147) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 196] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 196) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 245] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 245) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 294] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 294) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 343] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 343) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 392] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 392) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 441] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 335], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 490] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 490) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 539] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 539) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 588] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 588) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 637] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 637) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 686] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 686) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 735] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 735) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 784] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 784) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 833] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 833) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 882] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 678], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 931] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 931) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 980] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 980) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1029] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1029) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1078] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1078) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1127] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1127) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1176] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1176) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1225] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1225) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1274] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1274) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1323] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 1021], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1372] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1372) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1421] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1421) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1470] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1470) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1519] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1519) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1568] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1568) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1617] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1617) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1666] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1666) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1715] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1715) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1764] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 1364], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1813] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1813) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1862] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1862) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1911] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1911) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 1960] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 1960) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2009] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2009) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2058] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2058) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2107] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2107) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2156] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2156) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2205] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 1707], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2254] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2254) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2303] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2303) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2352] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2352) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2401] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2401) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2450] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2450) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2499] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2499) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2548] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2548) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2597] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2597) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2646] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 2050], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2695] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2695) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2744] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2744) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2793] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2793) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2842] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2842) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2891] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2891) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2940] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2940) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 2989] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 2989) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3038] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3038) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3087] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 2393], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3136] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3136) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3185] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3185) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3234] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3234) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3283] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3283) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3332] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3332) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3381] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3381) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3430] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3430) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3479] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3479) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3528] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 2736], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3577] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 7) % 9 and (threadIdx_x_1 // 7 + 7) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3577) // 63 * 49 + (threadIdx_x_1 // 7 + 7) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3626] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 5) % 9 and (threadIdx_x_1 // 7 + 5) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3626) // 63 * 49 + (threadIdx_x_1 // 7 + 5) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3675] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 3) % 9 and (threadIdx_x_1 // 7 + 3) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3675) // 63 * 49 + (threadIdx_x_1 // 7 + 3) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3724] = T.if_then_else(1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3724) // 63 * 49 + threadIdx_x_1 + rx_outer_outer - 1], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3773] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 8) % 9 and (threadIdx_x_1 // 7 + 8) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3773) // 63 * 49 + (threadIdx_x_1 // 7 + 8) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3822] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 6) % 9 and (threadIdx_x_1 // 7 + 6) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3822) // 63 * 49 + (threadIdx_x_1 // 7 + 6) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3871] = T.if_then_else(1 <= (threadIdx_x_1 // 7 + 4) % 9 and (threadIdx_x_1 // 7 + 4) % 9 < 8 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3871) // 63 * 49 + (threadIdx_x_1 // 7 + 4) % 9 * 7 + rx_outer_outer + threadIdx_x_1 % 7 - 8], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3920] = T.if_then_else(threadIdx_x_1 < 42 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 3920) // 63 * 49 + threadIdx_x_1 + rx_outer_outer + 6], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                pad_temp_shared_1[threadIdx_x_1 + 3969] = T.if_then_else(7 <= threadIdx_x_1 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + threadIdx_x_1 + rx_outer_outer + 3079], T.float32(0.0))
            with T.launch_thread(threadIdx_x_1, 49):
                if T.likely(threadIdx_x_1 < 14):
                    pad_temp_shared_1[threadIdx_x_1 + 4018] = T.if_then_else(threadIdx_x_1 < 7 and 1 <= rx_outer_outer + threadIdx_x_1 % 7 and rx_outer_outer + threadIdx_x_1 % 7 < 8, data_1[cse_var_2 + (threadIdx_x_1 + 4018) // 63 * 49 + rx_outer_outer + threadIdx_x_1 + 41], T.float32(0.0))
            threadIdx_x_2 = T.env_thread("threadIdx.x")
            kernel_shared_1 = T.Buffer((1536,), data=kernel_shared, scope="shared")
            kernel_1 = T.Buffer((2359296,), data=kernel.data)
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2] = kernel_1[blockIdx_x * 36864 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 49] = kernel_1[blockIdx_x * 36864 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 147]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 98] = kernel_1[blockIdx_x * 36864 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 294]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 147] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 147) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 147) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 196] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 196) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 12]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 245] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 245) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 159]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 294] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 294) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 306]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 343] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 343) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 151) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 392] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 392) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 24]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 441] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 441) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 171]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 490] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 490) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 318]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 539] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 539) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 155) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 588] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 588) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 36]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 637] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 637) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 183]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 686] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 686) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 330]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 735] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 735) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 159) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 784] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 784) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 48]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 833] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 833) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 195]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 882] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 882) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 342]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 931] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 931) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 163) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 980] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 980) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 60]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1029] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1029) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 207]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1078] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1078) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 354]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1127] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1127) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 167) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1176] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1176) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 72]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1225] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1225) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 219]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1274] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1274) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 366]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1323] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1323) // 192 * 4608 + cse_var_1 + (threadIdx_x_2 + 171) % 192 * 3 + rx_outer_outer]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1372] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1372) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 84]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1421] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1421) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 231]
            with T.launch_thread(threadIdx_x_2, 49):
                kernel_shared_1[threadIdx_x_2 + 1470] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1470) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 378]
            with T.launch_thread(threadIdx_x_2, 49):
                if T.likely(threadIdx_x_2 < 17):
                    kernel_shared_1[threadIdx_x_2 + 1519] = kernel_1[blockIdx_x * 36864 + (threadIdx_x_2 + 1519) // 192 * 4608 + cse_var_1 + threadIdx_x_2 * 3 + rx_outer_outer + 525]
            for rc_outer_inner in range(8):
                cse_var_3: T.int32 = rc_outer_inner * 24
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 192]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 384]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 576]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 3]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 195]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 387]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 579]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 6]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 198]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 390]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 582]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 9]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 201]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 393]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 585]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 12]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 204]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 396]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 588]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 15]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 207]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 399]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 591]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 18]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 210]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 402]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 594]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 21]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 213]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 405]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 597]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 768]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 960]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 1152]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x] * kernel_shared_1[cse_var_3 + 1344]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 771]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 963]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 1155]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 63] * kernel_shared_1[cse_var_3 + 1347]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 774]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 966]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 1158]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 126] * kernel_shared_1[cse_var_3 + 1350]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 777]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 969]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 1161]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 189] * kernel_shared_1[cse_var_3 + 1353]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 780]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 972]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 1164]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 252] * kernel_shared_1[cse_var_3 + 1356]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 783]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 975]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 1167]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 315] * kernel_shared_1[cse_var_3 + 1359]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 786]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 978]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 1170]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 378] * kernel_shared_1[cse_var_3 + 1362]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 789]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 981]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 1173]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 441] * kernel_shared_1[cse_var_3 + 1365]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 1]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 193]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 385]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 577]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 4]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 196]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 388]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 580]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 7]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 199]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 391]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 583]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 10]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 202]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 394]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 586]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 13]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 205]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 397]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 589]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 16]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 208]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 400]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 592]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 19]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 211]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 403]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 595]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 22]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 214]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 406]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 598]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 769]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 961]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 1153]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 7] * kernel_shared_1[cse_var_3 + 1345]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 772]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 964]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 1156]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 70] * kernel_shared_1[cse_var_3 + 1348]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 775]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 967]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 1159]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 133] * kernel_shared_1[cse_var_3 + 1351]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 778]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 970]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 1162]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 196] * kernel_shared_1[cse_var_3 + 1354]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 781]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 973]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 1165]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 259] * kernel_shared_1[cse_var_3 + 1357]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 784]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 976]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 1168]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 322] * kernel_shared_1[cse_var_3 + 1360]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 787]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 979]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 1171]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 385] * kernel_shared_1[cse_var_3 + 1363]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 790]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 982]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 1174]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 448] * kernel_shared_1[cse_var_3 + 1366]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 2]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 194]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 386]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 578]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 5]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 197]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 389]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 581]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 8]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 200]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 392]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 584]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 11]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 203]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 395]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 587]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 14]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 206]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 398]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 590]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 17]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 209]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 401]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 593]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 20]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 212]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 404]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 596]
                conv2d_nchw_1[0] = conv2d_nchw_1[0] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 23]
                conv2d_nchw_1[1] = conv2d_nchw_1[1] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 215]
                conv2d_nchw_1[2] = conv2d_nchw_1[2] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 407]
                conv2d_nchw_1[3] = conv2d_nchw_1[3] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 599]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 770]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 962]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 1154]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 14] * kernel_shared_1[cse_var_3 + 1346]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 773]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 965]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 1157]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 77] * kernel_shared_1[cse_var_3 + 1349]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 776]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 968]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 1160]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 140] * kernel_shared_1[cse_var_3 + 1352]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 779]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 971]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 1163]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 203] * kernel_shared_1[cse_var_3 + 1355]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 782]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 974]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 1166]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 266] * kernel_shared_1[cse_var_3 + 1358]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 785]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 977]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 1169]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 329] * kernel_shared_1[cse_var_3 + 1361]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 788]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 980]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 1172]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 392] * kernel_shared_1[cse_var_3 + 1364]
                conv2d_nchw_1[4] = conv2d_nchw_1[4] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 791]
                conv2d_nchw_1[5] = conv2d_nchw_1[5] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 983]
                conv2d_nchw_1[6] = conv2d_nchw_1[6] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 1175]
                conv2d_nchw_1[7] = conv2d_nchw_1[7] + pad_temp_shared_1[rc_outer_inner * 504 + threadIdx_x + 455] * kernel_shared_1[cse_var_3 + 1367]
        for i1_inner in range(8):
            compute_1 = T.Buffer((25088,), data=compute.data)
            bias_1 = T.Buffer((512,), data=bias.data)
            compute_1[blockIdx_x * 392 + i1_inner * 49 + threadIdx_x] = T.max(conv2d_nchw_1[i1_inner] + bias_1[blockIdx_x * 8 + i1_inner], T.float32(0.0))
            
import tvm
from tvm import te, tir, IRModule

mod: IRModule = Module
print(mod.script())

# get stmt
from tvm.tir.stmt_functor import ir_transform, post_order_visit
stmt = mod["main"]
print(stmt)

# use post_order_visit to get all the stmt
# make a function to visit the stmt
def visit_stmt(stmt):
    print("visit stmt")
    print(type(stmt))


tvm.tir.round()