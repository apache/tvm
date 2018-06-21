# pylint: disable=invalid-name
"""Extract feature of iter vars

There are two types of feature
1) Itervar feature
   This feature is extracted based on loop variables.
   Different loop structures will result in different shapes of feature
2) Curve sample feature (relation feature)
   This feature is extracted by sampling relation curve.
   This feature is invariant of loop structure.
"""

import struct
import numpy as np

from ... import schedule, ir_pass, build_module, get_global_func, target as _target

def ana_lower(sch, args,
              binds=None,
              simple_mode=True):
    """Do lower while keeping all axes in IR
    i.e. Do not eliminate loop with extent of 1, do not vectorize, unroll or inject virtual threads
    """
    binds, _ = build_module.get_binds(args, binds)
    # cfg = current_build_config()
    # add_lower_pass = cfg.add_lower_pass if cfg.add_lower_pass else []
    # lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    # lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    # lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    # lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]
    # normalize schedule first
    sch = sch.normalize()
    # Phase 0
    bounds = schedule.InferBound(sch)
    stmt = schedule.ScheduleOps(sch, bounds, True)
    # stmt = ir_pass.InjectPrefetch(stmt)
    # for f in lower_phase0:
    #     stmt = f(stmt)
    # Phase 1
    stmt = ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = ir_pass.CanonicalSimplify(stmt)
    # for f in lower_phase1:
    #     stmt = f(stmt)
    # Phase 2
    # if not simple_mode:
    #     stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)
    # stmt = ir_pass.VectorizeLoop(stmt)
    # stmt = ir_pass.InjectVirtualThread(stmt)
    # stmt = ir_pass.InjectDoubleBuffer(stmt, cfg.double_buffer_split_loop)
    # stmt = ir_pass.StorageRewrite(stmt)
    # stmt = ir_pass.UnrollLoop(
    #     stmt,
    #     cfg.auto_unroll_max_step,
    #     cfg.auto_unroll_max_depth,
    #     cfg.auto_unroll_max_extent,
    #     cfg.unroll_explicit)
    # for f in lower_phase2:
    #     stmt = f(stmt)
    # Phase 3
    # stmt = ir_pass.CanonicalSimplify(stmt)
    # stmt = ir_pass.LowerStorageAccessInfo(stmt)
    # stmt = ir_pass.RemoveNoOp(stmt)
    # stmt = ir_pass.RewriteUnsafeSelect(stmt)
    # for f in lower_phase3:
    #     stmt = f(stmt)
    assert simple_mode
    return stmt

_get_itervar_feature = get_global_func("autotvm.feature.GetItervarFeature")

def get_itervar_feature(sch, args, take_log=False):
    """get features of iter vars

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    features of every axis in the IR, see doc/features.md for detail
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature(stmt, take_log)

    # convert tvm node to python type
    ret = []
    for row in feas:
        tmp = []
        tmp.append([row[0][0].value, row[0][1]])
        for item in row[1:]:
            tmp.append([item[0].value] + [x.value for x in item[1:]])
        ret.append(tmp)
    return ret

def flatten_itervar_feature(fea):
    """flatten features into one-dimensional feature vectors

    Parameters
    ----------
    fea: list
        return value of get_itervar_feature

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    flatten = []
    for axis in fea:
        for pair in axis[1:]:
            flatten.append(pair[1:])
    return np.concatenate(flatten)

_get_itervar_feature_flatten = get_global_func("autotvm.feature.GetItervarFeatureFlatten")

def get_itervar_feature_flatten(sch, args, take_log=True):
    """get flatten features of iter vars
    this is equivalent to get_itervar_feature + flatten_itervar_feature, but much faster.

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    take_log: bool
        whether take log of numerical statics

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature_flatten(stmt, take_log)
    feas = struct.unpack('%df' % (len(feas)//4), feas)
    return feas

def get_flatten_name(fea):
    """ Get names of feature after flatten.

    Parameters
    ----------
    fea: list or str
        return value of get_itervar_feature or a line of logfile

    Returns
    -------
    feature_names: Array of str
    """

    feature_name = {
        "_attr_": ["length", "nest_level", "topdown", "bottomup"] +
                  ["ann_%d" % i for i in range(20)],
        "_arith_": ["add", "mul", "div"],
        "buf_touch": ["stride", "mod", "count", "reuse", "T_count", "T_reuse"],
    }

    if isinstance(fea, str):
        from ..record import decode
        # flatten line to feature
        line = fea
        inp, _ = decode(line)
        target = _target.create(inp.target)
        with target:
            s, args = inp.template.instantiate(inp.config)
        fea = get_itervar_feature(s, args)

    names = []
    ct = 0
    for row in fea:
        var_name = str(row[0][1])
        for pair in row[1:]:
            key = pair[0]
            if key in feature_name:
                name_list = feature_name[key]
            else:
                name_list = feature_name["buf_touch"]

            for i in range(len((pair[1:]))):
                names.append(".".join(["f%d" % ct, var_name, key, name_list[i]]))
                ct += 1
    return names


_get_buffer_curve_sample_flatten = get_global_func(
    "autotvm.feature.GetCurveSampleFeatureFlatten")

def get_buffer_curve_sample_flatten(sch, args, sample_n=30):
    """
    Get flatten curve sample feature (relation feature)

    Parameters
    ----------
    sch: tvm.schedule.Schedule
    args: Array of tvm.tensor.Tensor
        the buffer args for lower
    sample_n: int
        number of sample points along one dimension

    Returns
    -------
    flatten_feature: np.ndarray
        one-dimensional vector
    """
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_buffer_curve_sample_flatten(stmt, sample_n, False)
    feas = struct.unpack('%df' % (len(feas)//4), feas)
    return feas
