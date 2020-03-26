# Generated from /Users/doobs/Code/repo/sampl/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .RelayParser import RelayParser
else:
    from RelayParser import RelayParser

# This class defines a complete generic visitor for a parse tree produced by RelayParser.

class RelayVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by RelayParser#prog.
    def visitProg(self, ctx:RelayParser.ProgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#generalIdent.
    def visitGeneralIdent(self, ctx:RelayParser.GeneralIdentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#globalVar.
    def visitGlobalVar(self, ctx:RelayParser.GlobalVarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#localVar.
    def visitLocalVar(self, ctx:RelayParser.LocalVarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#graphVar.
    def visitGraphVar(self, ctx:RelayParser.GraphVarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#exprList.
    def visitExprList(self, ctx:RelayParser.ExprListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#callNoAttr.
    def visitCallNoAttr(self, ctx:RelayParser.CallNoAttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#callWithAttr.
    def visitCallWithAttr(self, ctx:RelayParser.CallWithAttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcExpr.
    def visitFuncExpr(self, ctx:RelayParser.FuncExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#metaExpr.
    def visitMetaExpr(self, ctx:RelayParser.MetaExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#match.
    def visitMatch(self, ctx:RelayParser.MatchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tensor.
    def visitTensor(self, ctx:RelayParser.TensorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#graph.
    def visitGraph(self, ctx:RelayParser.GraphContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#identExpr.
    def visitIdentExpr(self, ctx:RelayParser.IdentExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#stringExpr.
    def visitStringExpr(self, ctx:RelayParser.StringExprContext):
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


    # Visit a parse tree produced by RelayParser#paren.
    def visitParen(self, ctx:RelayParser.ParenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#scalarExpr.
    def visitScalarExpr(self, ctx:RelayParser.ScalarExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#let.
    def visitLet(self, ctx:RelayParser.LetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#projection.
    def visitProjection(self, ctx:RelayParser.ProjectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#ifElse.
    def visitIfElse(self, ctx:RelayParser.IfElseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#binOp.
    def visitBinOp(self, ctx:RelayParser.BinOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#func.
    def visitFunc(self, ctx:RelayParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#funcDefn.
    def visitFuncDefn(self, ctx:RelayParser.FuncDefnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#externAdtDefn.
    def visitExternAdtDefn(self, ctx:RelayParser.ExternAdtDefnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtDefn.
    def visitAdtDefn(self, ctx:RelayParser.AdtDefnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#constructorName.
    def visitConstructorName(self, ctx:RelayParser.ConstructorNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtConsDefnList.
    def visitAdtConsDefnList(self, ctx:RelayParser.AdtConsDefnListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtConsDefn.
    def visitAdtConsDefn(self, ctx:RelayParser.AdtConsDefnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#matchClauseList.
    def visitMatchClauseList(self, ctx:RelayParser.MatchClauseListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#matchClause.
    def visitMatchClause(self, ctx:RelayParser.MatchClauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#matchType.
    def visitMatchType(self, ctx:RelayParser.MatchTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#patternList.
    def visitPatternList(self, ctx:RelayParser.PatternListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#wildcardPattern.
    def visitWildcardPattern(self, ctx:RelayParser.WildcardPatternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#varPattern.
    def visitVarPattern(self, ctx:RelayParser.VarPatternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#constructorPattern.
    def visitConstructorPattern(self, ctx:RelayParser.ConstructorPatternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tuplePattern.
    def visitTuplePattern(self, ctx:RelayParser.TuplePatternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtCons.
    def visitAdtCons(self, ctx:RelayParser.AdtConsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtConsParamList.
    def visitAdtConsParamList(self, ctx:RelayParser.AdtConsParamListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#adtConsParam.
    def visitAdtConsParam(self, ctx:RelayParser.AdtConsParamContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#argNoAttr.
    def visitArgNoAttr(self, ctx:RelayParser.ArgNoAttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#argWithAttr.
    def visitArgWithAttr(self, ctx:RelayParser.ArgWithAttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#varList.
    def visitVarList(self, ctx:RelayParser.VarListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#var.
    def visitVar(self, ctx:RelayParser.VarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attrSeq.
    def visitAttrSeq(self, ctx:RelayParser.AttrSeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#attr.
    def visitAttr(self, ctx:RelayParser.AttrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#tupleType.
    def visitTupleType(self, ctx:RelayParser.TupleTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeParen.
    def visitTypeParen(self, ctx:RelayParser.TypeParenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#typeCallType.
    def visitTypeCallType(self, ctx:RelayParser.TypeCallTypeContext):
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


    # Visit a parse tree produced by RelayParser#typeParamList.
    def visitTypeParamList(self, ctx:RelayParser.TypeParamListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#shapeList.
    def visitShapeList(self, ctx:RelayParser.ShapeListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#meta.
    def visitMeta(self, ctx:RelayParser.MetaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#metaShape.
    def visitMetaShape(self, ctx:RelayParser.MetaShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#parensShape.
    def visitParensShape(self, ctx:RelayParser.ParensShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RelayParser#intShape.
    def visitIntShape(self, ctx:RelayParser.IntShapeContext):
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