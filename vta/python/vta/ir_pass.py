"""Additional IR Pass for VTA"""
# pylint: disable=len-as-condition
from __future__ import absolute_import as _abs

import tvm
from topi import util as util

from .environment import get_env


def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return ((stmt.attr_key == "pragma_" + key) or
            (stmt.attr_key == "pragma_scope" and stmt.value.value == key))


def fold_uop_loop(stmt_in):
    """Detect and fold uop loop.

    VTA support uop programming model
    that recognizes loop structure.
    This pass detect the loop structure
    and extract that into uop loop AST.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Output statement.
    """
    env = get_env()

    def _fold_outermost_loop(body):
        stmt = body
        while not isinstance(stmt, tvm.stmt.For):
            if isinstance(stmt, (tvm.stmt.ProducerConsumer,)):
                stmt = stmt.body
            else:
                return None, body, None

        loop_var = stmt.loop_var
        gemm_offsets = [None, None, None]
        fail = [False]

        def _post_order(op):
            assert isinstance(op, tvm.expr.Call)
            base_args = 2
            if op.name == "VTAUopPush":
                args = []
                args += op.args[:base_args]
                for i in range(3):
                    m = tvm.arith.DetectLinearEquation(
                        op.args[i + base_args], [loop_var])
                    if not m:
                        fail[0] = True
                        return op
                    if gemm_offsets[i] is not None:
                        if not tvm.ir_pass.Equal(m[0], gemm_offsets[i]):
                            fail[0] = True
                            return op
                        args.append(m[1])
                    else:
                        gemm_offsets[i] = m[0]
                        args.append(m[1])
                args += op.args[base_args+3:]
                return tvm.call_extern("int32", "VTAUopPush", *args)
            else:
                if op.name not in ("VTATLSCommandHandle", "tvm_thread_context"):
                    raise RuntimeError("unexpected op %s" % op)
                return op

        ret = tvm.ir_pass.IRTransform(
            stmt.body, None, _post_order, ["Call"])

        if not fail[0] and all(x is not None for x in gemm_offsets):
            def _visit(op):
                if op.same_as(loop_var):
                    fail[0] = True
            tvm.ir_pass.PostOrderVisit(ret, _visit)
            if not fail[0]:
                begin = tvm.call_extern(
                    "int32", "VTAUopLoopBegin", stmt.extent, *gemm_offsets)
                end = tvm.call_extern("int32", "VTAUopLoopEnd")
                return [begin, ret, end]
        raise ValueError("Failed to fold the GEMM instructions..")

    def _do_fold(stmt):
        if (stmt.attr_key == "coproc_uop_scope" and
                isinstance(stmt.value, tvm.expr.StringImm) and
                stmt.value.value == env.dev.vta_push_uop.value):
            body = stmt.body
            begins = []
            ends = []
            try:
                begin, body, end = _fold_outermost_loop(body)
                if begin is not None:
                    begins.append(begin)
                if end is not None:
                    ends.append(end)
                begin, body, end = _fold_outermost_loop(body)
                if begin is not None:
                    begins.append(begin)
                if end is not None:
                    ends.append(end)
            except ValueError:
                pass
            if body == stmt.body:
                return stmt
            ends = list(reversed(ends))
            body = tvm.make.stmt_seq(*(begins + [body] + ends))
            return tvm.make.AttrStmt(
                stmt.node, stmt.attr_key, stmt.value, body)
        return None
    out = tvm.ir_pass.IRTransform(
        stmt_in, _do_fold, None, ["AttrStmt"])
    return out


def cpu_access_rewrite(stmt_in):
    """Detect CPU access to VTA buffer and get address correctly.

    VTA's buffer is an opaque handle that do not
    correspond to address in CPU.
    This pass detect CPU access and rewrite to use pointer
    returned VTABufferCPUPtr for CPU access.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    rw_info = {}
    def _post_order(op):
        if isinstance(op, tvm.stmt.Allocate):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                return None
            new_var = rw_info[buffer_var]
            let_stmt = tvm.make.LetStmt(
                new_var, tvm.call_extern(
                    "handle", "VTABufferCPUPtr",
                    env.dev.command_handle,
                    buffer_var), op.body)
            alloc = tvm.make.Allocate(
                buffer_var, op.dtype, op.extents,
                op.condition, let_stmt)
            del rw_info[buffer_var]
            return alloc
        elif isinstance(op, tvm.expr.Load):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                rw_info[buffer_var] = tvm.var(
                    buffer_var.name + "_ptr", "handle")
            new_var = rw_info[buffer_var]
            return tvm.make.Load(op.dtype, new_var, op.index)
        elif isinstance(op, tvm.stmt.Store):
            buffer_var = op.buffer_var
            if not buffer_var in rw_info:
                rw_info[buffer_var] = tvm.var(
                    buffer_var.name + "_ptr", "handle")
            new_var = rw_info[buffer_var]
            return tvm.make.Store(new_var, op.value, op.index)
        else:
            raise RuntimeError("not reached")
    stmt = tvm.ir_pass.IRTransform(
        stmt_in, None, _post_order, ["Allocate", "Load", "Store"])
    for buffer_var, new_var in rw_info.items():
        stmt = tvm.make.LetStmt(
            new_var, tvm.call_extern(
                "handle", "VTABufferCPUPtr",
                env.dev.command_handle,
                buffer_var), stmt)
    return stmt


def lift_alloc_to_scope_begin(stmt_in):
    """Lift allocate to beginning of the current scope.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    lift_stmt = [[]]
    def _merge_block(slist, body):
        for op in slist:
            if op.body == body:
                body = op
            elif isinstance(op, tvm.stmt.Allocate):
                body = tvm.make.Allocate(
                    op.buffer_var, op.dtype,
                    op.extents, op.condition, body)
            elif isinstance(op, tvm.stmt.AttrStmt):
                body = tvm.make.AttrStmt(
                    op.node, op.attr_key, op.value, body)
            elif isinstance(op, tvm.stmt.For):
                body = tvm.make.For(
                    op.loop_var, op.min, op.extent, op.for_type,
                    op.device_api, body)
            else:
                raise RuntimeError("unexpected op")
        del slist[:]
        return body

    def _pre_order(op):
        if isinstance(op, tvm.stmt.For):
            lift_stmt.append([])
        elif isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "virtual_thread":
                lift_stmt.append([])

        return None

    def _post_order(op):
        if isinstance(op, tvm.stmt.Allocate):
            lift_stmt[-1].append(op)
            return op.body
        elif isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == "storage_scope":
                lift_stmt[-1].append(op)
                return op.body
            elif op.attr_key == "virtual_thread":
                return _merge_block(lift_stmt.pop() + [op], op.body)
            return op
        elif isinstance(op, tvm.stmt.For):
            return _merge_block(lift_stmt.pop() + [op], op.body)
        else:
            raise RuntimeError("not reached")
    stmt = tvm.ir_pass.IRTransform(
        stmt_in, _pre_order, _post_order, ["Allocate", "AttrStmt", "For"])
    assert len(lift_stmt) == 1
    return _merge_block(lift_stmt[0], stmt)


def inject_skip_copy(stmt_in):
    """Pass to inject skip copy stmt, used for debug purpose.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    def _do_fold(stmt):
        if _match_pragma(stmt, "skip_dma_copy"):
            return tvm.make.Evaluate(0)
        return None
    return tvm.ir_pass.IRTransform(
        stmt_in, _do_fold, None, ["AttrStmt"])


def inject_coproc_sync(stmt_in):
    """Pass to inject skip copy stmt, used in debug.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    success = [False]
    def _do_fold(stmt):
        if _match_pragma(stmt, "coproc_sync"):
            success[0] = True
            sync = tvm.make.Call(
                "int32", "vta.coproc_sync", [], tvm.expr.Call.Intrinsic, None, 0)
            return tvm.make.Block(stmt.body, tvm.make.Evaluate(sync))
        elif _match_pragma(stmt, "trim_loop"):
            op = stmt.body
            assert isinstance(op, tvm.stmt.For)
            return tvm.make.For(
                op.loop_var, op.min, 2, op.for_type,
                op.device_api, op.body)
        return None
    stmt = tvm.ir_pass.IRTransform(
        stmt_in, None, _do_fold, ["AttrStmt"])
    stmt = tvm.ir_pass.CoProcSync(stmt)
    return stmt


def inject_dma_intrin(stmt_in):
    """Pass to inject DMA copy intrinsics.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    def _check_compact(buf):
        ndim = len(buf.shape)
        size = tvm.const(1, buf.shape[0].dtype)
        for i in reversed(range(ndim)):
            if not util.equal_const_int(size - buf.strides[i], 0):
                raise RuntimeError(
                    "Cannot prove compact: shape=%s, strides=%s" % (buf.shape, buf.strides))
            size = size * buf.shape[i]

    def _fold_buffer_dim(buf, scope, elem_block):
        ndim = len(buf.shape)
        x_size = 1
        base = 0
        for i in range(1, ndim + 1):
            if not util.equal_const_int(buf.strides[ndim - i] - x_size, 0):
                raise RuntimeError("scope %s needs to have block=%d" % (scope, elem_block))
            x_size = x_size * buf.shape[ndim - i]
            if util.equal_const_int(x_size - elem_block, 0):
                base = i + 1
                break
        if base == 0:
            raise RuntimeError("scope %s need to have block=%d, shape=%s" % (
                scope, elem_block, buf.shape))
        shape = [elem_block]
        strides = [1]

        if base < ndim + 1 and not util.equal_const_int(buf.strides[ndim - base], elem_block):
            shape.append(1)
            strides.append(elem_block)

        while base < ndim + 1:
            x_size = 1
            x_stride = buf.strides[ndim - base]
            next_base = base
            if not util.equal_const_int(x_stride % elem_block, 0):
                raise RuntimeError(
                    "scope %s need to have block=%d, shape=%s, strides=%s" % (
                        scope, elem_block, buf.shape, buf.strides))
            for i in range(base, ndim + 1):
                k = ndim - i
                if not util.equal_const_int(x_size * x_stride - buf.strides[k], 0):
                    break
                x_size = x_size * buf.shape[k]
                next_base = i + 1
            shape.append(tvm.ir_pass.Simplify(x_size))
            strides.append(x_stride)
            assert next_base != base
            base = next_base

        strides = list(reversed(strides))
        shape = list(reversed(shape))
        return shape, strides

    def _get_2d_pattern(buf, elem_width, elem_bytes, dtype, scope, allow_fold):
        elem_block = elem_bytes * 8 // elem_width
        if buf.dtype != dtype:
            raise RuntimeError("Expect buffer type to be %s instead of %s" %
                               (dtype, buf.dtype))
        shape, strides = buf.shape, buf.strides
        if not util.equal_const_int(buf.elem_offset % elem_block, 0):
            raise RuntimeError("scope %s need to have block=%d" % (scope, elem_block))
        if allow_fold:
            shape, strides = _fold_buffer_dim(buf, scope, elem_block)
        else:
            shape = list(x for x in shape)
            strides = list(x for x in strides)

        def raise_error():
            """Internal function to raise error """
            raise RuntimeError(
                ("Scope[%s]: cannot detect 2d pattern with elem_block=%d:" +
                 " shape=%s, strides=%s") % (scope, elem_block, buf.shape, buf.strides))

        ndim = len(shape)

        # Check if the inner-tensor is already flat
        flat = util.equal_const_int(shape[-1], elem_block)

        if flat:
            if not util.equal_const_int(strides[-1], 1):
                raise_error()

            if ndim == 1:
                x_size = 1
                x_stride = 1
                y_size = 1
                return x_size, y_size, x_stride, buf.elem_offset / elem_block
            if not util.equal_const_int(strides[-2] - elem_block, 0):
                raise_error()

            if ndim == 2:
                x_size = shape[-2]
                x_stride = shape[-2]
                y_size = 1
                return x_size, y_size, x_stride, buf.elem_offset / elem_block
            if not util.equal_const_int(strides[-3] % elem_block, 0):
                raise_error()

            if ndim == 3:
                x_size = shape[-2]
                x_stride = strides[-3] / elem_block
                y_size = shape[-3]
                return x_size, y_size, x_stride, buf.elem_offset / elem_block

        else:
            if not util.equal_const_int(strides[-1], 1):
                raise_error()
            if not util.equal_const_int(strides[-2] - shape[-1], 0):
                raise_error()
            if not util.equal_const_int(shape[-1] * shape[-2], elem_block):
                raise_error()

            if ndim == 2:
                x_size = 1
                x_stride = 1
                y_size = 1
                return x_size, y_size, x_stride, buf.elem_offset / elem_block
            if not util.equal_const_int(strides[-3], elem_block):
                raise_error()

            if ndim == 3:
                x_size = shape[-3]
                x_stride = shape[-3]
                y_size = 1
                return x_size, y_size, x_stride, buf.elem_offset / elem_block
            if not util.equal_const_int(strides[-4] % elem_block, 0):
                raise_error()

            if ndim == 4:
                x_size = shape[-3]
                x_stride = strides[-4] / elem_block
                y_size = shape[-4]
                return x_size, y_size, x_stride, buf.elem_offset / elem_block

        raise_error()


    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # FIXME: pad_value is ignored...
        _ = pad_value
        if dst.scope == "global":
            # Store
            if pad_before or pad_after:
                raise RuntimeError("Do not support copy into DRAM with pad")
            if src.scope == env.acc_scope:
                elem_width = env.OUT_WIDTH
                elem_bytes = env.OUT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_OUT
                data_type = "int%d" % env.OUT_WIDTH
                task_qid = env.dev.QID_STORE_OUT
            else:
                raise RuntimeError("Do not support copy %s->dram" % (src.scope))
            _check_compact(src)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                dst, elem_width, elem_bytes, data_type, src.scope, allow_fold=True)
            irb = tvm.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope",
                           env.dev.get_task_qid(task_qid))
            irb.emit(tvm.call_extern(
                "int32", "VTAStoreBuffer2D",
                env.dev.command_handle,
                src.access_ptr("r", "int32"),
                mem_type, dst.data, offset, x_size, y_size, x_stride))
            return irb.get()
        elif src.scope == "global":
            if dst.scope == env.acc_scope:
                elem_width = env.ACC_WIDTH
                elem_bytes = env.ACC_ELEM_BYTES
                mem_type = env.dev.MEM_ID_ACC
                data_type = "int%d" % env.ACC_WIDTH
                task_qid = env.dev.QID_LOAD_OUT
            elif dst.scope == env.inp_scope:
                elem_width = env.INP_WIDTH
                elem_bytes = env.INP_ELEM_BYTES
                mem_type = env.dev.MEM_ID_INP
                data_type = "int%d" % env.INP_WIDTH
                task_qid = env.dev.QID_LOAD_INP
            elif dst.scope == env.wgt_scope:
                elem_width = env.WGT_WIDTH
                elem_bytes = env.WGT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_WGT
                data_type = "int%d" % env.WGT_WIDTH
                task_qid = env.dev.QID_LOAD_WGT
            else:
                raise RuntimeError("Do not support copy dram->%s" % (dst.scope))
            # collect pad statistics
            if pad_before:
                assert pad_after
                ndim = len(pad_before)
                if ndim <= 2 or ndim > 4:
                    raise ValueError("Limitation of 2D pad load forbid ndim=%d" % ndim)
                if ndim > 2:
                    if not util.equal_const_int(pad_before[ndim - 1], 0):
                        raise ValueError("Do not support pad on the innermost block")
                    if not util.equal_const_int(pad_after[ndim - 1], 0):
                        raise ValueError("Do not support pad on the innermost block")
                if ndim > 3:
                    if not util.equal_const_int(pad_before[ndim - 2], 0):
                        raise ValueError("Do not support pad on the innermost block")
                    if not util.equal_const_int(pad_after[ndim - 2], 0):
                        raise ValueError("Do not support pad on the innermost block")
                y_pad_before = pad_before[0]
                x_pad_before = pad_before[1]
                y_pad_after = pad_after[0]
                x_pad_after = pad_after[1]
                allow_fold = False
            else:
                x_pad_before = 0
                y_pad_before = 0
                x_pad_after = 0
                y_pad_after = 0
                allow_fold = True

            _check_compact(dst)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                src, elem_width, elem_bytes, data_type,
                dst.scope, allow_fold=allow_fold)

            irb = tvm.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope",
                           env.dev.get_task_qid(task_qid))

            irb.emit(tvm.call_extern(
                "int32", "VTALoadBuffer2D",
                env.dev.command_handle,
                src.data, offset, x_size, y_size, x_stride,
                x_pad_before, y_pad_before,
                x_pad_after, y_pad_after,
                dst.access_ptr("r", "int32"), mem_type))
            return irb.get()

        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope, dst.scope))

    return tvm.ir_pass.InjectCopyIntrin(stmt_in, "dma_copy", _inject_copy)


def annotate_alu_coproc_scope(stmt_in):
    """Pass to insert ALU instruction.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    def _do_fold(stmt):
        if _match_pragma(stmt, "alu"):
            irb = tvm.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope",
                           env.dev.get_task_qid(env.dev.QID_COMPUTE))
            irb.scope_attr(env.dev.vta_axis, "coproc_uop_scope",
                           tvm.make.StringImm("VTAPushALUOp"))
            irb.emit(stmt)
            return irb.get()
        elif _match_pragma(stmt, "skip_alu"):
            return tvm.make.Evaluate(0)
        return stmt

    stmt_out = tvm.ir_pass.IRTransform(
        stmt_in, None, _do_fold, ["AttrStmt"])

    return stmt_out


def inject_alu_intrin(stmt_in):
    """Pass to inject ALU micro-ops.

    Parameters
    ----------
    stmt_in : Stmt
        Input statement

    Returns
    -------
    stmt_out : Stmt
        Transformed statement
    """
    env = get_env()
    def _do_fold(stmt):
        def _equal(x, y):
            return tvm.ir_pass.Equal(tvm.ir_pass.Simplify(x - y), 0)

        def _flatten_loop(src_coeff, dst_coeff, extents):
            src_coeff = list(src_coeff)
            dst_coeff = list(dst_coeff)
            extents = list(extents)
            rev_src_coeff = [src_coeff.pop()]
            rev_dst_coeff = [dst_coeff.pop()]
            rev_extents = []
            assert src_coeff
            vsrc = src_coeff.pop()
            vdst = dst_coeff.pop()
            vext = extents.pop()
            while src_coeff:
                next_src = src_coeff.pop()
                next_dst = dst_coeff.pop()
                next_ext = extents.pop()

                if _equal(next_src, vsrc * vext) and _equal(next_dst, vdst * vext):
                    vext = tvm.ir_pass.Simplify(vext * next_ext)
                else:
                    rev_src_coeff.append(vsrc)
                    rev_dst_coeff.append(vdst)
                    rev_extents.append(vext)
                    vsrc = next_src
                    vdst = next_dst
                    vext = next_ext
            rev_src_coeff.append(vsrc)
            rev_dst_coeff.append(vdst)
            rev_extents.append(vext)
            rev_src_coeff.reverse()
            rev_dst_coeff.reverse()
            rev_extents.reverse()

            return rev_src_coeff, rev_dst_coeff, rev_extents

        if _match_pragma(stmt, "alu"):
            # Get to the innermost loop body
            loop_body = stmt.body
            nest_size = 0
            while isinstance(loop_body, tvm.stmt.For):
                loop_body = loop_body.body
                nest_size += 1
            # Get the src/dst arguments
            dst_var = loop_body.buffer_var
            dst_idx = loop_body.index
            # Derive loop variables and extents
            tmp_body = stmt.body
            indices = []
            extents = []
            for _ in range(nest_size):
                indices.append(tmp_body.loop_var)
                extents.append(tmp_body.extent)
                tmp_body = tmp_body.body
            # Derive opcode
            if isinstance(loop_body.value, tvm.expr.Add):
                alu_opcode = env.dev.ALU_OPCODE_ADD
                lhs = loop_body.value.a
                rhs = loop_body.value.b
            elif isinstance(loop_body.value, tvm.expr.Sub):
                alu_opcode = env.dev.ALU_OPCODE_SUB
                lhs = loop_body.value.a
                rhs = loop_body.value.b
            elif isinstance(loop_body.value, tvm.expr.Mul):
                alu_opcode = env.dev.ALU_OPCODE_MUL
                lhs = loop_body.value.a
                rhs = loop_body.value.b
            elif isinstance(loop_body.value, tvm.expr.Min):
                alu_opcode = env.dev.ALU_OPCODE_MIN
                lhs = loop_body.value.a
                rhs = loop_body.value.b
            elif isinstance(loop_body.value, tvm.expr.Max):
                alu_opcode = env.dev.ALU_OPCODE_MAX
                lhs = loop_body.value.a
                rhs = loop_body.value.b
            elif isinstance(loop_body.value, tvm.expr.Call):
                if loop_body.value.name == 'shift_left':
                    alu_opcode = env.dev.ALU_OPCODE_SHR
                    lhs = loop_body.value.args[0]
                    rhs = tvm.ir_pass.Simplify(-loop_body.value.args[1])
                elif loop_body.value.name == 'shift_right':
                    alu_opcode = env.dev.ALU_OPCODE_SHR
                    lhs = loop_body.value.args[0]
                    rhs = loop_body.value.args[1]
                else:
                    raise RuntimeError(
                        "Function call not recognized %s" % (loop_body.value.name))
            elif isinstance(loop_body.value, tvm.expr.Load):
                alu_opcode = env.dev.ALU_OPCODE_SHR
                lhs = loop_body.value
                rhs = tvm.const(0)
            else:
                raise RuntimeError(
                    "Expression not recognized %s, %s, %s" % (
                        type(loop_body.value), str(loop_body.value), str(stmt)))

            # Derive array index coefficients
            dst_coeff = tvm.arith.DetectLinearEquation(dst_idx, indices)
            # Check if lhs/rhs is immediate
            use_imm = False
            imm_val = None
            if isinstance(rhs, tvm.expr.IntImm):
                assert lhs.buffer_var.same_as(dst_var)
                src_coeff = tvm.arith.DetectLinearEquation(lhs.index, indices)
                use_imm = True
                imm_val = rhs
            if isinstance(lhs, tvm.expr.IntImm):
                assert rhs.buffer_var.same_as(dst_var)
                src_coeff = tvm.arith.DetectLinearEquation(rhs.index, indices)
                use_imm = True
                imm_val = lhs
            if imm_val is None:
                imm_val = 0
                assert lhs.buffer_var.same_as(dst_var) and rhs.buffer_var.same_as(dst_var)
                src_lhs_coeff = tvm.arith.DetectLinearEquation(lhs.index, indices)
                src_rhs_coeff = tvm.arith.DetectLinearEquation(rhs.index, indices)
                # Determine which side has the same coefficients
                lhs_equal = True
                rhs_equal = True
                for i, coef in enumerate(dst_coeff):
                    if not tvm.ir_pass.Equal(coef, src_lhs_coeff[i]):
                        lhs_equal = False
                    if not tvm.ir_pass.Equal(coef, src_rhs_coeff[i]):
                        rhs_equal = False
                # Make sure at least one of the source is identical to the
                # destination (in-place computation)
                assert lhs_equal or rhs_equal
                # Assign the source coefficients
                if lhs_equal:
                    src_coeff = src_rhs_coeff
                else:
                    src_coeff = src_lhs_coeff

            # Ensure that we have the proper tensor dimensions in the
            # innermost loop (pattern match)
            src_coeff = list(src_coeff)
            dst_coeff = list(dst_coeff)
            extents = list(extents)
            assert len(src_coeff) > 1
            assert len(dst_coeff) > 1
            assert len(extents) != 0
            assert tvm.ir_pass.Equal(
                tvm.ir_pass.Simplify(
                    src_coeff[-1] % (env.BATCH * env.BLOCK_OUT)), 0)
            assert tvm.ir_pass.Equal(
                tvm.ir_pass.Simplify(
                    dst_coeff[-1] % (env.BATCH * env.BLOCK_OUT)), 0)
            assert tvm.ir_pass.Equal(src_coeff[-2], 1)
            assert tvm.ir_pass.Equal(dst_coeff[-2], 1)
            if env.BATCH > 1:
                assert len(src_coeff) > 2
                assert len(dst_coeff) > 2
                assert len(extents) > 1
                assert tvm.ir_pass.Equal(src_coeff[-3], env.BLOCK_OUT)
                assert tvm.ir_pass.Equal(dst_coeff[-3], env.BLOCK_OUT)

            # Apply tensorization of the loop coefficients
            src_offset = src_coeff[-1]
            dst_offset = dst_coeff[-1]
            if env.BATCH == 1:
                src_coeff = src_coeff[:-2]
                dst_coeff = dst_coeff[:-2]
                extents = extents[:-1]
            else:
                src_coeff = src_coeff[:-3]
                dst_coeff = dst_coeff[:-3]
                extents = extents[:-2]
            src_coeff.append(src_offset)
            dst_coeff.append(dst_offset)
            src_coeff = [
                tvm.ir_pass.Simplify(c // (env.BATCH * env.BLOCK_OUT)) for c in src_coeff]
            dst_coeff = [
                tvm.ir_pass.Simplify(c // (env.BATCH * env.BLOCK_OUT)) for c in dst_coeff]

            # Flatten the outer loops
            if extents:
                src_coeff, dst_coeff, extents = _flatten_loop(src_coeff, dst_coeff, extents)

            # Insert ALU micro-ops
            irb = tvm.ir_builder.create()
            for idx, extent in enumerate(extents):
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopLoopBegin",
                    extent, dst_coeff[idx], src_coeff[idx], 0))
            use_imm = int(use_imm)
            irb.emit(tvm.call_extern(
                "int32", "VTAUopPush",
                1, 0,
                dst_coeff[len(dst_coeff)-1],
                src_coeff[len(src_coeff)-1],
                0,
                alu_opcode, use_imm, imm_val))
            for extent in extents:
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopLoopEnd"))
            return irb.get()
        return stmt

    stmt_out = tvm.ir_pass.IRTransform(
        stmt_in, None, _do_fold, ["AttrStmt"])
    return stmt_out


def debug_print(stmt):
    """A debug pass that print the stmt

    Parameters
    ----------
    stmt : Stmt
        The input statement

    Returns
    -------
    stmt : Stmt
        The
    """
    # pylint: disable=superfluous-parens
    print(stmt)
    return stmt
