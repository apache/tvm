import tvm
from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt

def visit_stmts_internal(stmt, ws=""): # tir.stmt.____
    if isinstance(stmt, (_stmt.For, _stmt.While, _stmt.BufferRealize)):
        print(ws+" "+str(type(stmt)) + " Start")
        visit_stmts_internal(stmt.body, ws+"|")
        print(ws+" "+str(type(stmt)) + " End")
    elif isinstance(stmt, _stmt.SeqStmt):
        print(ws+" "+str(type(stmt)))
        for st in stmt.seq:
            visit_stmts_internal(st, ws+"|")
    elif isinstance(stmt, _stmt.BufferStore):
        print(ws+" "+str(type(stmt)))
        # print(ws+str(type(stmt))+" --> "+str(stmt))
        print(ws+" "+"@Location to be stored:  ############################################")
        for idx in range(0, len(stmt.indices)):
            # print(str(idx))
            visit_stmts_internal(stmt.indices[idx], ws+str(idx))
        print(ws+" "+"@Value to be stored:  ############################################")
        visit_exprs_internal(stmt.value, ws+"|")
    elif isinstance(stmt, _stmt.IfThenElse):
        print(ws+" "+str(type(stmt)))
        # print(ws+str(type(stmt))+" --> "+str(stmt))
        print(ws+" "+"@IfThen Case: ############################################")
        visit_stmts_internal(stmt.then_case, ws+"|")
        # print("")
        print(ws+" "+"@Else   Case: ############################################")
        visit_stmts_internal(stmt.else_case, ws+"|")
        # print("")
    elif isinstance(stmt, tvm.ir.PrimExpr):
        visit_exprs_internal(stmt, ws)
    elif stmt is None:
        print(ws+" "+str(type(stmt)))
        return
    else:
        assert 0, str(type(stmt))

def visit_exprs_internal(expr, ws=""): # tir.expr.____
    # print("WS: ", str(ws))
    if hasattr(expr, "indices"): # BufferLoad
        print(ws+" "+str(type(expr))+" --> "+str(expr))
        for idx in range(0, len(expr.indices)):
            visit_exprs_internal(expr.indices[idx], ws+str(idx))
    elif all(hasattr(expr, attr) for attr in ["a", "b"]): # Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EA, NE, LT, LE, GT, GE, And, Or
        print(ws+" "+str(type(expr))+" --> "+str(expr))
        visit_exprs_internal(expr.a, ws+"L")
        visit_exprs_internal(expr.b, ws+"R")
    elif isinstance(expr, _expr.Cast): # Cast
        print(ws+" "+str(type(expr))+" --> "+str(expr))
        visit_exprs_internal(expr.value, ws+"*")
    elif isinstance(expr, _expr.Call): # Call
        print(ws+" "+str(type(expr.op))+" --> "+str(expr.op))
        for idx in range(0, len(expr.args)):
            print(ws+" "+str(type(expr.args[idx]))+" --> "+str(expr.args[idx]))
            visit_exprs_internal(expr.args[idx], ws+"*")
    elif isinstance(expr, (_expr.Var, _expr.SizeVar)): # Var, SizeVar
        print(ws+" "+str(type(expr))+" --> "+str(expr))
    elif isinstance(expr, (_expr.FloatImm, _expr.IntImm)): # FloatImm, IntImm
        print(ws+" "+str(type(expr))+" --> "+str(expr))
    else:
        print("##"*10)
        print(type(expr))
        print(str(expr))
        print(str(expr.name))
        print("Currently Not Supported!!!!!")
        assert 0
    return

def visit_stmts(primfn):
    visit_stmts_internal(primfn.body)
