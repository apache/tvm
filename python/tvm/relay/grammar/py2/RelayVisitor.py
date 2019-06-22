# Generated from /home/sslyu/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
from antlr4 import *

# This class defines a complete generic visitor for a parse tree produced by RelayParser.

class RelayVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by RelayParser#opIdent.
    def visitOpIdent(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#prog.
    def visitProg(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#identExpr.
    def visitIdentExpr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#call.
    def visitCall(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#neg.
    def visitNeg(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tuple.
    def visitTuple(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#parens.
    def visitParens(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcExpr.
    def visitFuncExpr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarExpr.
    def visitScalarExpr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#let.
    def visitLet(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tensor.
    def visitTensor(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#ifElse.
    def visitIfElse(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#graph.
    def visitGraph(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#binOp.
    def visitBinOp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#func.
    def visitFunc(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#defn.
    def visitDefn(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#argList.
    def visitArgList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#varList.
    def visitVarList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#var.
    def visitVar(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attrList.
    def visitAttrList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attr.
    def visitAttr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeParamSeq.
    def visitTypeParamSeq(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tupleType.
    def visitTupleType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeIdentType.
    def visitTypeIdentType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tensorType.
    def visitTensorType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcType.
    def visitFuncType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#incompleteType.
    def visitIncompleteType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#intType.
    def visitIntType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#shapeSeq.
    def visitShapeSeq(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#parensShape.
    def visitParensShape(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#intShape.
    def visitIntShape(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeIdent.
    def visitTypeIdent(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#body.
    def visitBody(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarFloat.
    def visitScalarFloat(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarInt.
    def visitScalarInt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarBool.
    def visitScalarBool(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#ident.
    def visitIdent(self, ctx):
        return self.visitChildren(ctx)


