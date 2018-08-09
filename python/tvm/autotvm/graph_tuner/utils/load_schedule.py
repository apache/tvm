# pylint: disable=invalid-name,global-statement
"""Helper utility function for loading schedules"""
from topi.nn.conv2d import _get_alter_layout_schedule, _get_schedule_NCHWc


global_idx = 0
global_schedule_list = []
global_sch_dict = {}

def load_conv_sch_avx(schedule_list):
    """Load convolution schedules for Intel CPU with AVX instruction set.

    Parameters
    ----------
    schedule_list : list of namedtuple
        Input schedule list. The order should be ascending order of node index
        in a graph. Otherwise performance would be lower than expected.
    """
    global global_idx, global_schedule_list
    global_idx = 0
    global_schedule_list = list(schedule_list)

    @_get_alter_layout_schedule.register("cpu", override=True)
    def _get_alter_layout_schedule_avx(wkl):
        global global_idx, global_schedule_list, global_sch_dict
        sch = global_schedule_list[global_idx]
        layout = "NCHW%dc" % sch.ic_bn
        out_layout = "NCHW%dc" % sch.oc_bn
        sch_key = "%s, %s, %s" % (wkl, layout, out_layout)
        global_sch_dict[sch_key] = sch
        global_idx += 1
        return sch

    @_get_schedule_NCHWc.register("cpu", override=True)
    def _get_schedule_NCHWc_avx(wkl, layout, out_layout):
        global global_sch_dict
        sch_key = "%s, %s, %s" % (wkl, layout, out_layout)
        sch = global_sch_dict[sch_key]
        return sch
