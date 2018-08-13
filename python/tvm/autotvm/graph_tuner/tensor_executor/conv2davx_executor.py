# pylint: disable=invalid-name,no-member,unused-argument,too-many-locals,unused-variable
"""Conv2d executor class for intel AVX CPU."""
from nnvm import symbol as sym
from topi.nn.conv2d import _get_schedule_NCHWc, _get_alter_layout_schedule
from topi.x86.conv2d_avx_common import AVXConvCommonFwd
from topi.x86.conv2d_avx_1x1 import AVXConv1x1Fwd
from .base_tensor_executor import BaseTensorExecutor
from ..utils import get_factor

class Conv2dAVXExecutor(BaseTensorExecutor):
    """Executor class to benchmark 2D convolution for Intel CPU with
    AVX instruction set.
    """
    def _get_op_symbol(self):
        """Get conv2d_NCHWc operator symbol.
        """
        return sym.contrib.conv2d_NCHWc

    def _workload2params(self, workload, schedule):
        """Convert an input workload and schedule to
        conv2d_NCHWc parameters.
        """
        ic_bn = schedule.ic_bn
        oc_bn = schedule.oc_bn
        is_unit_kernel = workload.hkernel == 1 and workload.wkernel == 1
        data_layout = "NCHW%dc" % ic_bn
        kernel_layout = "OIHW%di%do" % (ic_bn, oc_bn) \
            if not is_unit_kernel else "OI%di%doHW" % (ic_bn, oc_bn)
        out_layout = "NCHW%dc" % oc_bn
        param_dict = {"channels": workload.out_filter,
                      "kernel_size": (workload.hkernel, workload.wkernel),
                      "padding": (workload.hpad, workload.wpad),
                      "strides": (workload.hstride, workload.wstride),
                      "layout": data_layout, "out_layout": out_layout,
                      "kernel_layout": kernel_layout}
        return param_dict

    def _workload2ishapes(self, workload, schedule):
        """Convert an input workload and schedule to
        conv2d_NCHWc input shapes.
        """
        batch_size = 1
        in_channel = workload.in_filter
        in_height = workload.height
        in_width = workload.width
        ic_bn = schedule.ic_bn
        data_shape = (batch_size, in_channel // ic_bn, in_height,
                      in_width, ic_bn)
        return {"data": data_shape}

    def _load_schedule(self, schedule):
        """Load schedule for conv2d_NCHWc.
        """
        @_get_schedule_NCHWc.register("cpu", override=True)
        def _get_schedule_NCHWc_x86(wkl, layout, out_layout):
            return schedule

        @_get_alter_layout_schedule.register("cpu", override=True)
        def _get_alter_layout_schedule_x86(wkl):
            return schedule

    def _create_search_space(self, workload):
        """Create search space for conv2d_NCHWc workload.
        """
        ih, iw = workload.height, workload.width
        ic, oc = workload.in_filter, workload.out_filter
        hk, wk = workload.hkernel, workload.wkernel
        hp, wp = workload.hpad, workload.wpad
        hs, ws = workload.hstride, workload.wstride
        oh = (ih - hk + 2 * hp) // hs + 1
        ow = (iw - wk + 2 * wp) // ws + 1
        ic_bn = get_factor(ic)
        oc_bn = get_factor(oc)
        ow_bn = get_factor(ow)
        ow_bn_max = 64
        tmp = []
        for ow_bn_candidate in ow_bn:
            if ow_bn_candidate <= ow_bn_max:
                tmp.append(ow_bn_candidate)
        ow_bn = tmp
        if len(ow_bn) > 2:
            ow_bn.remove(1)
        oh_bn = [1, 2] if oh > 1 else [1]
        unroll_kw = [True, False]
        is_unit_kernel = hk == 1 and wk == 1
        search_space_dict = {}
        if is_unit_kernel:
            search_space_dict["schedule_template_name"] = AVXConv1x1Fwd
            search_space_dict["oh_factor"] = oh_bn
            search_space_dict["ow_factor"] = ow_bn
        else:
            search_space_dict["schedule_template_name"] = AVXConvCommonFwd
            search_space_dict["reg_n"] = ow_bn
            search_space_dict["unroll_kw"] = unroll_kw
        search_space_dict["ic_bn"] = ic_bn
        search_space_dict["oc_bn"] = oc_bn
        return search_space_dict

    def _get_layout_related_fields(self):
        """Get layout transform related fields for conv2d_NCHWc schedule.
        """
        return "ic_bn", "oc_bn"
