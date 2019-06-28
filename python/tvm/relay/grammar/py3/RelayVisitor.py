# Generated from /home/marisa/Work/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .RelayParser import RelayParser
else:
    from RelayParser import RelayParser

# This class defines a complete generic visitor for a parse tree produced by RelayParser.

class RelayVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by RelayParser#opIdent.
    def visitOpIdent(self, ctx:RelayParser.OpIdentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#prog.
    def visitProg(self, ctx:RelayParser.ProgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#identExpr.
    def visitIdentExpr(self, ctx:RelayParser.IdentExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#call.
    def visitCall(self, ctx:RelayParser.CallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#neg.
    def visitNeg(self, ctx:RelayParser.NegContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tuple.
    def visitTuple(self, ctx:RelayParser.TupleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#parens.
    def visitParens(self, ctx:RelayParser.ParensContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcExpr.
    def visitFuncExpr(self, ctx:RelayParser.FuncExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarExpr.
    def visitScalarExpr(self, ctx:RelayParser.ScalarExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#let.
    def visitLet(self, ctx:RelayParser.LetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tensor.
    def visitTensor(self, ctx:RelayParser.TensorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#ifElse.
    def visitIfElse(self, ctx:RelayParser.IfElseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#graph.
    def visitGraph(self, ctx:RelayParser.GraphContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#binOp.
    def visitBinOp(self, ctx:RelayParser.BinOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#func.
    def visitFunc(self, ctx:RelayParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#defn.
    def visitDefn(self, ctx:RelayParser.DefnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#argList.
    def visitArgList(self, ctx:RelayParser.ArgListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#varList.
    def visitVarList(self, ctx:RelayParser.VarListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#var.
    def visitVar(self, ctx:RelayParser.VarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attrList.
    def visitAttrList(self, ctx:RelayParser.AttrListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attr.
    def visitAttr(self, ctx:RelayParser.AttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeParamSeq.
    def visitTypeParamSeq(self, ctx:RelayParser.TypeParamSeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tupleType.
    def visitTupleType(self, ctx:RelayParser.TupleTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeIdentType.
    def visitTypeIdentType(self, ctx:RelayParser.TypeIdentTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tensorType.
    def visitTensorType(self, ctx:RelayParser.TensorTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcType.
    def visitFuncType(self, ctx:RelayParser.FuncTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#incompleteType.
    def visitIncompleteType(self, ctx:RelayParser.IncompleteTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#intType.
    def visitIntType(self, ctx:RelayParser.IntTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#shapeSeq.
    def visitShapeSeq(self, ctx:RelayParser.ShapeSeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#parensShape.
    def visitParensShape(self, ctx:RelayParser.ParensShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#intShape.
    def visitIntShape(self, ctx:RelayParser.IntShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeIdent.
    def visitTypeIdent(self, ctx:RelayParser.TypeIdentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#body.
    def visitBody(self, ctx:RelayParser.BodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarFloat.
    def visitScalarFloat(self, ctx:RelayParser.ScalarFloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarInt.
    def visitScalarInt(self, ctx:RelayParser.ScalarIntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarBool.
    def visitScalarBool(self, ctx:RelayParser.ScalarBoolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#ident.
    def visitIdent(self, ctx:RelayParser.IdentContext):
        return self.visitChildren(ctx)



del RelayParser